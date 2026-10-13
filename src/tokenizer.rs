//! Gemma-style BPE tokenizer, matching Hugging Face `tokenizer(text, add_special_tokens=False)`
//! for Julia-1's `tokenizer.json`:
//! added-token split -> `' '` to `'▁'` -> Metaspace (prepend `▁` to every segment, split before
//! each `▁`) -> BPE over ranked merges with byte fallback.

use std::collections::HashMap;

use color_eyre::eyre::{ensure, eyre};
use serde::Deserialize;

const SPACE: char = '▁';

#[derive(Clone, Copy, Debug)]
pub struct SpecialIds {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub unk: u32,
}

struct AddedToken {
    content: Box<str>,
    id: u32,
    lstrip: bool,
    rstrip: bool,
}

pub struct Tokenizer {
    /// Single-char vocab entries: the starting symbols of BPE.
    char_ids: HashMap<char, u32>,
    /// `<0xNN>` byte-fallback ids.
    byte_ids: [Option<u32>; 256],
    /// `(left << 32 | right) -> (rank, merged id)`.
    merges: HashMap<u64, (u32, u32)>,
    /// Added tokens bucketed by first byte, longest first within each bucket.
    added: Vec<Vec<AddedToken>>,
    space_id: u32,
    pub special: SpecialIds,
    /// Number of ids (max vocab id + 1).
    pub vocab_size: usize,
}

#[derive(Deserialize)]
struct TokenizerJson {
    added_tokens: Vec<AddedTokenJson>,
    model: BpeJson,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    lstrip: bool,
    rstrip: bool,
}

#[derive(Deserialize)]
struct BpeJson {
    #[serde(rename = "type")]
    kind: String,
    byte_fallback: bool,
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct ConfigJson {
    cls_token: String,
    sep_token: String,
    mask_token: String,
    pad_token: String,
    unk_token: String,
}

impl Tokenizer {
    pub fn load(tokenizer_json: &str, config_json: &str) -> color_eyre::Result<Self> {
        let json: TokenizerJson = serde_json::from_str(&std::fs::read_to_string(tokenizer_json)?)?;
        let config: ConfigJson = serde_json::from_str(&std::fs::read_to_string(config_json)?)?;
        let model = json.model;
        ensure!(
            model.kind == "BPE" && model.byte_fallback,
            "expected a BPE tokenizer with byte_fallback"
        );

        let added_id = |name: &str| {
            json.added_tokens
                .iter()
                .find(|t| t.content == name)
                .map(|t| t.id)
                .ok_or_else(|| eyre!("special token `{name}` is not an added token"))
        };
        let special = SpecialIds {
            cls: added_id(&config.cls_token)?,
            sep: added_id(&config.sep_token)?,
            mask: added_id(&config.mask_token)?,
            pad: added_id(&config.pad_token)?,
            unk: added_id(&config.unk_token)?,
        };

        let char_ids: HashMap<char, u32> = model
            .vocab
            .iter()
            .filter_map(|(tok, &id)| {
                let mut chars = tok.chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => Some((c, id)),
                    _ => None,
                }
            })
            .collect();
        let vocab_size = model.vocab.values().max().map_or(0, |&m| m as usize + 1);
        let space_id = *char_ids
            .get(&SPACE)
            .ok_or_else(|| eyre!("`{SPACE}` missing from vocab"))?;

        let byte_ids = std::array::from_fn(|b| model.vocab.get(&format!("<0x{b:02X}>")).copied());

        let mut merges = HashMap::with_capacity(model.merges.len());
        let mut joined = String::new();
        for (rank, (a, b)) in model.merges.iter().enumerate() {
            joined.clear();
            joined.push_str(a);
            joined.push_str(b);
            let id = |s: &str| {
                model
                    .vocab
                    .get(s)
                    .copied()
                    .ok_or_else(|| eyre!("merge `{a} {b}`: `{s}` missing from vocab"))
            };
            let key = (id(a)? as u64) << 32 | id(b)? as u64;
            merges.entry(key).or_insert((rank as u32, id(&joined)?));
        }

        let mut added: Vec<Vec<AddedToken>> = (0..256).map(|_| Vec::new()).collect();
        for t in json.added_tokens {
            ensure!(!t.content.is_empty(), "empty added token {}", t.id);
            added[t.content.as_bytes()[0] as usize].push(AddedToken {
                content: t.content.into(),
                id: t.id,
                lstrip: t.lstrip,
                rstrip: t.rstrip,
            });
        }
        for bucket in &mut added {
            bucket.sort_by_key(|t| std::cmp::Reverse(t.content.len()));
        }

        Ok(Self {
            char_ids,
            byte_ids,
            merges,
            added,
            space_id,
            vocab_size,
            special,
        })
    }

    /// Appends the ids of `text` to `out`, without any special tokens.
    pub fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let bytes = text.as_bytes();
        let mut scratch = Scratch::default();
        let mut seg_start = 0;
        let mut i = 0;
        while i < bytes.len() {
            let Some(tok) = self.added[bytes[i] as usize]
                .iter()
                .find(|t| text[i..].starts_with(&*t.content))
            else {
                i += 1;
                continue;
            };
            let segment = &text[seg_start..i];
            self.encode_segment(
                if tok.lstrip {
                    segment.trim_end()
                } else {
                    segment
                },
                &mut scratch,
                out,
            );
            out.push(tok.id);
            i += tok.content.len();
            if tok.rstrip {
                i = bytes.len() - text[i..].trim_start().len();
            }
            seg_start = i;
        }
        self.encode_segment(&text[seg_start..], &mut scratch, out);
    }

    /// Normalize + Metaspace + BPE for text containing no added tokens.
    fn encode_segment(&self, segment: &str, s: &mut Scratch, out: &mut Vec<u32>) {
        let Some(first) = segment.chars().next() else {
            return;
        };
        s.word.clear();
        if first != ' ' && first != SPACE {
            s.word.push(self.space_id);
        }
        for c in segment.chars() {
            if c == ' ' || c == SPACE {
                if !s.word.is_empty() {
                    self.bpe(s, out);
                }
                s.word.push(self.space_id);
            } else {
                self.push_char(c, &mut s.word);
            }
        }
        self.bpe(s, out);
    }

    fn push_char(&self, c: char, word: &mut Vec<u32>) {
        if let Some(&id) = self.char_ids.get(&c) {
            word.push(id);
            return;
        }
        let mut buf = [0; 4];
        let utf8 = c.encode_utf8(&mut buf).as_bytes();
        if utf8.iter().all(|&b| self.byte_ids[b as usize].is_some()) {
            word.extend(utf8.iter().map(|&b| self.byte_ids[b as usize].unwrap()));
        } else if word.last() != Some(&self.special.unk) {
            // fuse_unk: consecutive unknowns collapse into one <unk>
            word.push(self.special.unk);
        }
    }

    fn rank(&self, a: u32, b: u32) -> (u32, u32) {
        self.merges
            .get(&((a as u64) << 32 | b as u64))
            .copied()
            .unwrap_or((u32::MAX, 0))
    }

    /// Merges `s.word` by lowest rank (leftmost on ties) and drains it into `out`.
    fn bpe(&self, s: &mut Scratch, out: &mut Vec<u32>) {
        let word = &mut s.word;
        let pairs = &mut s.pairs;
        pairs.clear();
        pairs.extend(word.windows(2).map(|w| self.rank(w[0], w[1])));
        while let Some((i, &(_, merged))) = pairs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.0 != u32::MAX)
            .min_by_key(|(_, p)| p.0)
        {
            word[i] = merged;
            word.remove(i + 1);
            pairs.remove(i);
            if i > 0 {
                pairs[i - 1] = self.rank(word[i - 1], merged);
            }
            if i < pairs.len() {
                pairs[i] = self.rank(merged, word[i + 1]);
            }
        }
        out.append(word);
    }
}

#[derive(Default)]
struct Scratch {
    word: Vec<u32>,
    pairs: Vec<(u32, u32)>,
}
