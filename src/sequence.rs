// Copyright (C) 2026 arnu515
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Julia's input layout (`julia/data.py: sequence`), from a request to token ids and the position
//! of each option's marker:
//!
//! `[CLS] {type} question: {question} [SEP] <mask> option… <mask> option… [SEP] {state} [SEP]`

use std::{
    borrow::Cow,
    fmt::{self, Write},
};

use eyre::ensure;
use serde::{
    Deserialize, Deserializer,
    de::{
        MapAccess, SeqAccess, Visitor,
        value::{MapAccessDeserializer, SeqAccessDeserializer},
    },
};
use serde_json::Value;

use crate::{
    Model,
    head::{Answer, QType, answer},
    tokenizer::Tokenizer,
};

/// `TransformerEngine`'s default question/option token budget.
pub const DEFAULT_HEAD_LENGTH: usize = 256;

// Layout constants hard-coded in `data.py` (not in any config).
/// Option tokens kept after each marker.
const OPTION_TOKENS: usize = 48;
/// Below this question budget, options are cut to share `head_length - MIN_BUDGET`…
const MIN_BUDGET: isize = 16;
/// …but each keeps at least this many tokens (marker included).
const MIN_OPTION: isize = 4;
/// Question tokens kept however small the budget.
const MIN_QUESTION: isize = 8;

/// One question about a state: a row of Julia's JSONL. Other fields (`target`, …) are ignored.
/// Deserializing borrows from the input where it can (strings without escapes).
#[derive(Debug, Deserialize)]
pub struct Request<'a> {
    /// Text. A JSON row may also give an object or array, which becomes its `python_json`.
    #[serde(borrow, deserialize_with = "state_text")]
    pub state: Cow<'a, str>,
    #[serde(borrow)]
    pub question: Cow<'a, str>,
    #[serde(default, rename = "type")]
    pub qtype: QType,
    #[serde(borrow, deserialize_with = "borrowed_strs")]
    pub options: Vec<Cow<'a, str>>,
}

/// `Vec<Cow<str>>` that borrows each element where it can. (`#[serde(borrow)]` only reaches a
/// `Cow` field itself; serde's own `Cow` impl always copies.)
fn borrowed_strs<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Cow<'de, str>>, D::Error> {
    #[derive(Deserialize)]
    struct Text<'a>(#[serde(borrow)] Cow<'a, str>);
    Ok(Vec::<Text>::deserialize(d)?
        .into_iter()
        .map(|t| t.0)
        .collect())
}

/// A request's `state`: text as it is, an object or array as `python_json`.
fn state_text<'de, D: Deserializer<'de>>(d: D) -> Result<Cow<'de, str>, D::Error> {
    struct State;
    impl<'de> Visitor<'de> for State {
        type Value = Cow<'de, str>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("text or a JSON object/array")
        }
        fn visit_borrowed_str<E>(self, s: &'de str) -> Result<Self::Value, E> {
            Ok(s.into())
        }
        fn visit_str<E>(self, s: &str) -> Result<Self::Value, E> {
            Ok(s.to_owned().into())
        }
        fn visit_string<E>(self, s: String) -> Result<Self::Value, E> {
            Ok(s.into())
        }
        fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
            Ok(python_json(&Value::deserialize(MapAccessDeserializer::new(map))?).into())
        }
        fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
            Ok(python_json(&Value::deserialize(SeqAccessDeserializer::new(seq))?).into())
        }
    }
    d.deserialize_any(State)
}

pub struct Sequence {
    pub ids: Vec<u32>,
    /// Position of each option's marker, in option order.
    pub markers: Vec<usize>,
    pub qtype: QType,
    /// The state was cut to fit `max_length`.
    pub truncated: bool,
}

/// `data.py: sequence` (non-strict, as `Collator` calls it), with `validate_row`'s checks.
pub fn sequence(
    tok: &Tokenizer,
    req: &Request,
    max_length: usize,
    head_length: usize,
) -> eyre::Result<Sequence> {
    ensure!(
        head_length + 4 < max_length,
        "max_length {max_length} must leave room beyond head_length {head_length}"
    );
    ensure!(
        (2..=20).contains(&req.options.len()) && req.options.iter().all(|o| !o.is_empty()),
        "options must contain 2–20 nonempty descriptions"
    );
    ensure!(
        req.qtype != QType::Noul || req.options.len() == 2,
        "noul options must be ordered [false, true]"
    );
    let sp = tok.special;
    // `<mask>` in user text would read as an option marker.
    let clean = |t: &str| t.replace(&*tok.mask_token, " ");

    let mut head = Vec::new();
    tok.encode(
        &format!("{} question: {}", req.qtype.name(), clean(&req.question)),
        &mut head,
    );
    let mut options: Vec<Vec<u32>> = req
        .options
        .iter()
        .map(|o| {
            let mut ids = vec![sp.mask];
            tok.encode(&format!(" {}", clean(o)), &mut ids);
            ids.truncate(1 + OPTION_TOKENS);
            ids
        })
        .collect();
    let hl = head_length as isize;
    let used = |opts: &[Vec<u32>]| opts.iter().map(Vec::len).sum::<usize>() as isize;
    let mut budget = hl - used(&options);
    if budget < MIN_BUDGET {
        let per = MIN_OPTION.max((hl - MIN_BUDGET).div_euclid(options.len() as isize));
        options.iter_mut().for_each(|o| o.truncate(per as usize));
        budget = hl - used(&options);
    }

    let mut ids = vec![sp.cls];
    ids.extend(&head[..head.len().min(budget.max(MIN_QUESTION) as usize)]);
    ids.push(sp.sep);
    let mut markers = Vec::with_capacity(options.len());
    for o in &options {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(sp.sep);
    let room = max_length.saturating_sub(ids.len() + 1);
    ensure!(
        room >= 1,
        "question/options exceed the sequence budget; shorten descriptions"
    );
    let start = ids.len();
    tok.encode(&clean(&req.state), &mut ids);
    let truncated = ids.len() - start > room;
    ids.truncate(start + room);
    ids.push(sp.sep);
    Ok(Sequence {
        ids,
        markers,
        qtype: req.qtype,
        truncated,
    })
}

/// Python's `json.dumps(v, ensure_ascii=False)`: `", "` and `": "` separators, keys in input
/// order, `repr` floats. (Integers beyond 64 bits reach here as floats; Python keeps them exact.)
pub fn python_json(v: &Value) -> String {
    let mut out = String::new();
    write_python_json(v, &mut out);
    out
}

fn write_python_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => python_float(f, out),
            _ => write!(out, "{n}").unwrap(),
        },
        Value::String(s) => python_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_json(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_str(k, out);
                out.push_str(": ");
                write_python_json(x, out);
            }
            out.push('}');
        }
    }
}

/// `repr(float)`: the shortest digits that round-trip, fixed notation for decimal exponents
/// -4..16 (always with a `.`), otherwise `1.5e-07` / `1e+16`.
fn python_float(f: f64, out: &mut String) {
    // `{:e}` gives the shortest digit count, but when two such strings round-trip it doesn't always
    // pick the one nearest `f` as Python does (…009.3 vs …009.2); `{:.N$e}` rounds exactly.
    let shortest = format!("{f:e}");
    let digits = shortest
        .split_once('e')
        .unwrap()
        .0
        .bytes()
        .filter(u8::is_ascii_digit)
        .count();
    let sci = format!("{:.*e}", digits - 1, f.abs());
    let (m, e) = sci.split_once('e').unwrap();
    let (d, e) = (m.replace('.', ""), e.parse::<i32>().unwrap());
    if f.is_sign_negative() {
        out.push('-');
    }
    if (-4..16).contains(&e) {
        if e < 0 {
            out.push_str("0.");
            (1..-e).for_each(|_| out.push('0'));
            out.push_str(&d);
        } else {
            let int = e as usize + 1;
            if d.len() > int {
                out.push_str(&d[..int]);
                out.push('.');
                out.push_str(&d[int..]);
            } else {
                out.push_str(&d);
                (d.len()..int).for_each(|_| out.push('0'));
                out.push_str(".0");
            }
        }
    } else {
        out.push_str(&d[..1]);
        if d.len() > 1 {
            out.push('.');
            out.push_str(&d[1..]);
        }
        write!(out, "e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()).unwrap();
    }
}

/// A JSON string as Python escapes it with `ensure_ascii=False`.
fn python_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub struct Prediction {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub answer: Answer,
    /// The state was cut to fit `max_length`.
    pub truncated: bool,
}

impl Model {
    /// A request end to end: sequence, encoder, head, answer.
    pub fn predict(
        &self,
        req: &Request,
        max_length: usize,
        head_length: usize,
    ) -> eyre::Result<Prediction> {
        ensure!(
            max_length <= self.config.max_positions,
            "max_length {max_length} is over the model's {}",
            self.config.max_positions
        );
        let seq = sequence(&self.tokenizer, req, max_length, head_length)?;
        let logits = self.forward(&seq.ids, seq.qtype, &seq.markers);
        ensure!(
            logits.iter().all(|z| z.is_finite()),
            "inference returned nonfinite logits"
        );
        let (answer, probabilities) = answer(seq.qtype, &logits);
        Ok(Prediction {
            logits,
            probabilities,
            answer,
            truncated: seq.truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{Request, python_json};

    #[test]
    fn json_like_python() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"b": [1, -2, 3.0, 0.1, 1e16, 1.5e-7, 0.0001, -0.0, 123456789.125, -1541722455959009.2, 18446744073709551615],
                "a": {"s": "q\"\\\n\t\u0001é😀", "e": [], "o": {}, "n": null, "t": true}}"#,
        )
        .unwrap();
        let s = python_json(&v);
        // json.dumps(json.loads(same text), ensure_ascii=False)
        let want = r#"{"b": [1, -2, 3.0, 0.1, 1e+16, 1.5e-07, 0.0001, -0.0, 123456789.125, -1541722455959009.2, 18446744073709551615], "a": {"s": "q\"\\\n\t\u0001é😀", "e": [], "o": {}, "n": null, "t": true}}"#;
        assert_eq!(s, want);
    }

    #[test]
    fn request_borrows_from_the_row() {
        let row =
            r#"{"state": {"n": 1, "s": "a\nb"}, "question": "plain", "options": ["x", "y\"z"]}"#;
        let req: Request = serde_json::from_str(row).unwrap();
        assert_eq!(req.state, r#"{"n": 1, "s": "a\nb"}"#);
        assert!(matches!(req.question, Cow::Borrowed("plain")));
        // An escape has to be decoded, so that string can't point into the row.
        assert!(matches!(&req.options[..], [Cow::Borrowed("x"), Cow::Owned(o)] if o == "y\"z"));

        let req: Request =
            serde_json::from_str(r#"{"state": "s", "question": "q", "options": []}"#).unwrap();
        assert!(matches!(req.state, Cow::Borrowed("s")));

        let err =
            serde_json::from_str::<Request>(r#"{"state": 42, "question": "q", "options": []}"#);
        assert!(
            err.unwrap_err()
                .to_string()
                .contains("text or a JSON object/array")
        );
    }
}
