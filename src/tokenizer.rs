//! llama2.c's byte-pair tokenizer: its `tokenizer.bin`, encoded and decoded as `run.c` does.
//!
//! `tokenizer.bin` is `max_token_length: i32`, then per token `score: f32, len: i32, bytes`. Encoding needs the
//! vocabulary sorted, which takes one `u32` per token of the caller's memory; nothing else is allocated.

use crate::model::ModelError;

pub const BOS: u32 = 1;
pub const EOS: u32 = 2;
/// Tokens 3..259 are the 256 bytes, for text no other token covers.
const BYTE_BASE: u32 = 3;

/// A parsed tokenizer: `offsets[i]` is where token `i`'s record starts in `bytes`.
pub struct Tokenizer<'a> {
    bytes: &'a [u8],
    offsets: &'a [u32],
    sorted: &'a [u32],
    max_token_length: usize,
}

fn i32_at(b: &[u8], at: usize) -> Option<i32> {
    Some(i32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

impl<'a> Tokenizer<'a> {
    /// Parses `bytes` for a vocabulary of `vocab` tokens, using `offsets` and `sorted` (each `vocab` long) as its index.
    pub fn new(bytes: &'a [u8], vocab: usize, offsets: &'a mut [u32], sorted: &'a mut [u32]) -> Result<Self, ModelError> {
        if offsets.len() < vocab || sorted.len() < vocab || bytes.len() > u32::MAX as usize {
            return Err(ModelError::Tokenizer);
        }
        let max_token_length = i32_at(bytes, 0).filter(|&m| m > 0 && m <= 1024).ok_or(ModelError::Tokenizer)? as usize;
        let mut at = 4usize;
        for offset in offsets[..vocab].iter_mut() {
            *offset = at as u32;
            let len = i32_at(bytes, at + 4).filter(|&l| l >= 0 && l as usize <= max_token_length).ok_or(ModelError::Tokenizer)?;
            at = at.checked_add(8 + len as usize).filter(|&end| end <= bytes.len()).ok_or(ModelError::Tokenizer)?;
        }
        let offsets = &offsets[..vocab];
        let sorted = &mut sorted[..vocab];
        for (i, s) in sorted.iter_mut().enumerate() {
            *s = i as u32;
        }
        let piece = |i: u32| {
            let at = offsets[i as usize] as usize;
            let len = i32_at(bytes, at + 4).unwrap_or(0) as usize;
            &bytes[at + 8..at + 8 + len]
        };
        // Ties sort by id, so a duplicated piece finds its lowest id, as a stable qsort would.
        sorted.sort_unstable_by(|&a, &b| piece(a).cmp(piece(b)).then(a.cmp(&b)));
        Ok(Tokenizer { bytes, offsets, sorted, max_token_length })
    }

    pub fn vocab(&self) -> usize {
        self.offsets.len()
    }

    /// Token `id`'s bytes.
    pub fn piece(&self, id: u32) -> &'a [u8] {
        let at = self.offsets[id as usize] as usize;
        let len = i32_at(self.bytes, at + 4).unwrap_or(0) as usize;
        &self.bytes[at + 8..at + 8 + len]
    }

    pub fn score(&self, id: u32) -> f32 {
        f32::from_le_bytes(self.bytes[self.offsets[id as usize] as usize..][..4].try_into().unwrap())
    }

    /// The token whose piece is exactly `text`, if there is one.
    pub fn lookup(&self, text: &[u8]) -> Option<u32> {
        let found = self.sorted.partition_point(|&id| self.piece(id) < text);
        self.sorted.get(found).copied().filter(|&id| self.piece(id) == text)
    }

    /// Encodes `text` into `out` as `run.c` does, with `BOS` first and a leading space token. Returns how many tokens,
    /// or `None` if `out` is too short.
    pub fn encode(&self, text: &[u8], out: &mut [u32]) -> Option<usize> {
        let mut n = 0usize;
        let push = |out: &mut [u32], n: &mut usize, t: u32| -> Option<()> {
            *out.get_mut(*n)? = t;
            *n += 1;
            Some(())
        };
        push(out, &mut n, BOS)?;
        if !text.is_empty() {
            push(out, &mut n, self.lookup(b" ")?)?;
        }
        let mut buffer = [0u8; 4];
        let mut len = 0usize;
        for (i, &c) in text.iter().enumerate() {
            if c & 0xc0 != 0x80 {
                len = 0;
            }
            if len < 4 {
                buffer[len] = c;
                len += 1;
            }
            if text.get(i + 1).is_some_and(|&next| next & 0xc0 == 0x80) && len < 4 {
                continue;
            }
            match self.lookup(&buffer[..len]) {
                Some(id) => push(out, &mut n, id)?,
                None => {
                    for &b in &buffer[..len] {
                        push(out, &mut n, b as u32 + BYTE_BASE)?;
                    }
                }
            }
            len = 0;
        }
        // Merge the best-scoring adjacent pair until none merges.
        let mut pair = [0u8; 2048];
        loop {
            let mut best: Option<(f32, u32, usize)> = None;
            for i in 0..n.saturating_sub(1) {
                let (a, b) = (self.piece(out[i]), self.piece(out[i + 1]));
                if a.len() + b.len() > pair.len() || a.len() + b.len() > 2 * self.max_token_length {
                    continue;
                }
                pair[..a.len()].copy_from_slice(a);
                pair[a.len()..a.len() + b.len()].copy_from_slice(b);
                if let Some(id) = self.lookup(&pair[..a.len() + b.len()]) {
                    let score = self.score(id);
                    if best.is_none_or(|(s, _, _)| score > s) {
                        best = Some((score, id, i));
                    }
                }
            }
            let Some((_, id, i)) = best else { break };
            out[i] = id;
            out.copy_within(i + 2..n, i + 1);
            n -= 1;
        }
        Some(n)
    }

    /// Token `next`'s text after `prev`, as `run.c`'s `decode`: the space after `BOS` dropped, and `<0xXX>` as its byte.
    /// The byte, if it is one, is written to `byte`.
    pub fn decode<'b>(&'b self, prev: u32, next: u32, byte: &'b mut [u8; 1]) -> &'b [u8] {
        let mut piece = self.piece(next);
        if prev == BOS && piece.first() == Some(&b' ') {
            piece = &piece[1..];
        }
        if piece.len() == 6 && piece.starts_with(b"<0x") && piece[5] == b'>' {
            let hex = |c: u8| (c as char).to_digit(16);
            if let (Some(h), Some(l)) = (hex(piece[3]), hex(piece[4])) {
                byte[0] = (h * 16 + l) as u8;
                return &byte[..];
            }
        }
        piece
    }
}
