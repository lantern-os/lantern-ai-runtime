//! Host tests on small synthetic models. `lantern-reference` checks a real one against an independent f32 forward pass.

use crate::convert::{convert, quantize};
use crate::engine::{forward, generate, GenerateError, Request, Sampler, Workspace};

use crate::model::{self, ModelError};
use crate::tokenizer::{Tokenizer, BOS};

/// A deterministic stream of small floats.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    }
}

/// A llama2.c tokenizer: the 3 specials, the 256 bytes, then `extra`.
fn tokenizer_bin(extra: &[(&str, f32)]) -> Vec<u8> {
    let mut pieces: Vec<(Vec<u8>, f32)> = vec![(b"<unk>".to_vec(), 0.0), (b"\n<s>\n".to_vec(), 0.0), (b"\n</s>\n".to_vec(), 0.0)];
    for b in 0..=255u8 {
        pieces.push((format!("<0x{b:02X}>").into_bytes(), 0.0));
    }
    pieces.extend(extra.iter().map(|(p, s)| (p.as_bytes().to_vec(), *s)));
    let mut out = (pieces.iter().map(|(p, _)| p.len()).max().unwrap() as i32).to_le_bytes().to_vec();
    for (p, s) in pieces {
        out.extend_from_slice(&s.to_le_bytes());
        out.extend_from_slice(&(p.len() as i32).to_le_bytes());
        out.extend_from_slice(&p);
    }
    out
}

const EXTRA: &[(&str, f32)] = &[(" ", -1.0), ("a", -2.0), ("b", -2.0), ("ab", -0.5), (" ab", -0.1), ("c", -3.0), ("é", -1.0)];

/// A legacy f32 checkpoint: `[dim, hidden, layers, heads, kv_heads, vocab, seq_len]`, a negative vocab for an unshared
/// classifier.
fn checkpoint(config: [i32; 7], seed: u64) -> Vec<u8> {
    checkpoint_with_attention(config, seed, 1.0)
}

/// As [`checkpoint`], with the query and key weights scaled by `qk`: large enough, attention picks out positions sharply,
/// so a mistake in how positions are encoded or looked up shows in the logits.
fn checkpoint_with_attention(config: [i32; 7], seed: u64, qk: f32) -> Vec<u8> {
    let [dim, hidden, layers, heads, kv_heads, vocab, seq_len] = config.map(|v| v.unsigned_abs() as usize);
    let (hs, kv) = (dim / heads, dim * kv_heads / heads);
    let mut n = vocab * dim + layers * (dim + 2 * dim * dim + 2 * dim * kv + dim + 3 * dim * hidden) + dim + seq_len * hs;
    if config[5] < 0 {
        n += vocab * dim;
    }
    let mut rng = Rng(seed);
    let mut out: Vec<u8> = config.iter().flat_map(|v| v.to_le_bytes()).collect();
    // `wq` then `wk`, after the embeddings and the attention norms.
    let q_start = vocab * dim + layers * dim;
    let k_end = q_start + layers * (dim * dim + dim * kv);
    for i in 0..n {
        let v = rng.next() * if (q_start..k_end).contains(&i) { qk } else { 1.0 };
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// The model file, copied into memory aligned for its f32s, as the service's heap is.
fn aligned(bytes: &[u8]) -> Vec<u32> {
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    // SAFETY: a u32 buffer at least `bytes.len()` long, viewed as bytes.
    unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) }.copy_from_slice(bytes);
    words
}

fn bytes(words: &[u32], len: usize) -> &[u8] {
    // SAFETY: as in `aligned`.
    unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), len) }
}

const VOCAB: i32 = 259 + EXTRA.len() as i32;
const SMALL: [i32; 7] = [96, 160, 2, 4, 2, VOCAB, 32];

fn file(config: [i32; 7]) -> Vec<u8> {
    convert(&checkpoint(config, 7), &tokenizer_bin(EXTRA)).unwrap()
}

/// Runs `prompt` greedily for up to `max` tokens on a model file, by `forward` alone, returning the generated ids.
fn run(file: &[u8], prompt: &[u8], max: usize, sampler: Sampler) -> Result<Vec<u32>, GenerateError> {
    let words = aligned(file);
    let model = model::parse(bytes(&words, file.len())).unwrap();
    let c = model.config;
    let (nf, nb) = Workspace::sizes(&c);
    let (mut f, mut b) = (vec![0f32; nf], vec![0i8; nb]);
    let mut ws = Workspace::new(&c, &mut f, &mut b).unwrap();
    let (mut offsets, mut sorted) = (vec![0u32; c.vocab_size], vec![0u32; c.vocab_size]);
    let tok = Tokenizer::new(model.tokenizer, c.vocab_size, &mut offsets, &mut sorted).unwrap();
    let mut tokens = vec![0u32; c.seq_len];
    let mut out = Vec::new();
    let n = tok.encode(prompt, &mut tokens).unwrap_or(c.seq_len);
    if n >= c.seq_len {
        return Err(GenerateError::PromptTooLong);
    }
    let _ = sampler;
    let mut token = tokens[0];
    for pos in 0..(n - 1 + max).min(c.seq_len) {
        let logits = forward(&model, &mut ws, token, pos);
        let next = if pos < n - 1 {
            tokens[pos + 1]
        } else {
            let mut best = 0;
            for (i, &l) in logits.iter().enumerate() {
                if l > logits[best] {
                    best = i;
                }
            }
            best as u32
        };
        if pos >= n - 1 {
            out.push(next);
        }
        token = next;
    }
    Ok(out)
}

#[test]
fn a_converted_checkpoint_parses_back_with_its_shape() {
    let f = file(SMALL);
    let words = aligned(&f);
    let m = model::parse(bytes(&words, f.len())).unwrap();
    let c = m.config;
    assert_eq!((c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.seq_len), (96, 160, 2, 4, 2, 32));
    assert_eq!(c.vocab_size, VOCAB as usize);
    assert_eq!(c.group_size, 32, "64 doesn't divide 96, so export.py's rule halves it");
    assert!(c.shared_classifier);
    assert_eq!(m.tokenizer, tokenizer_bin(EXTRA).as_slice());
    assert_eq!(model::file_len(&f), Ok(f.len()));
    // An unshared classifier is read after the layers.
    let unshared = file([96, 160, 2, 4, 2, -VOCAB, 32]);
    let words = aligned(&unshared);
    let m = model::parse(bytes(&words, unshared.len())).unwrap();
    assert!(!m.config.shared_classifier);
    assert_ne!(m.weights.wcls.q.as_ptr(), m.weights.tokens.q.as_ptr());
}

#[test]
fn quantization_is_within_half_a_step_of_every_weight() {
    let mut rng = Rng(3);
    let w: Vec<f32> = (0..256).map(|_| rng.next()).collect();
    let (q, s) = quantize(&w, 32);
    for (i, &v) in w.iter().enumerate() {
        let back = q[i] as f32 * s[i / 32];
        assert!((back - v).abs() <= s[i / 32] / 2.0 + 1e-7, "{i}: {v} vs {back}");
    }
    assert!(q.iter().any(|&v| v == 127 || v == -127), "each group's largest weight is a full step");
}

#[test]
fn a_file_that_isnt_a_model_or_is_cut_short_is_refused() {
    let f = file(SMALL);
    let words = aligned(&f);
    let whole = bytes(&words, f.len());
    assert_eq!(model::parse(&whole[..whole.len() - 1]).err(), Some(ModelError::Format));
    let mut wrong = f.clone();
    wrong[0] = b'X';
    assert_eq!(model::file_len(&wrong), Err(ModelError::Format));
    // A checkpoint whose header promises more weights than it has.
    let mut short = f.clone();
    short[model::HEADER + 8 + 4 * 2..][..4].copy_from_slice(&9i32.to_le_bytes()); // 9 layers, not 2
    let words = aligned(&short);
    assert_eq!(model::parse(bytes(&words, short.len())).err(), Some(ModelError::Truncated));
    // A shape that doesn't divide.
    let mut odd = f.clone();
    odd[model::HEADER + 8 + 4 * 3..][..4].copy_from_slice(&5i32.to_le_bytes()); // 5 heads into 96
    let words = aligned(&odd);
    assert_eq!(model::parse(bytes(&words, odd.len())).err(), Some(ModelError::Checkpoint));
    // Lengths that overflow.
    let mut huge = f.clone();
    huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(model::file_len(&huge), Err(ModelError::Format));
}

#[test]
fn the_tokenizer_merges_by_score_and_falls_back_to_bytes() {
    let bin = tokenizer_bin(EXTRA);
    let vocab = VOCAB as usize;
    let (mut offsets, mut sorted) = (vec![0u32; vocab], vec![0u32; vocab]);
    let tok = Tokenizer::new(&bin, vocab, &mut offsets, &mut sorted).unwrap();
    let id = |p: &str| tok.lookup(p.as_bytes()).unwrap();
    let mut out = [0u32; 32];
    // " " + "ab" merges to " ab", the best pair; "c" stays; "z" is a byte.
    let n = tok.encode(b"ab cz", &mut out).unwrap();
    assert_eq!(&out[..n], &[BOS, id(" ab"), id(" "), id("c"), b'z' as u32 + 3]);
    // A two-byte character is one piece when the vocabulary has it.
    let n = tok.encode("é".as_bytes(), &mut out).unwrap();
    assert_eq!(&out[..n], &[BOS, id(" "), id("é")]);
    // Decoding drops the space after BOS, and a byte token is its byte.
    let mut byte = [0u8; 1];
    assert_eq!(tok.decode(BOS, id(" ab"), &mut byte), b"ab");
    assert_eq!(tok.decode(id("c"), b'z' as u32 + 3, &mut byte), b"z");
    assert_eq!(tok.encode(b"ab cz", &mut out[..3]), None, "too little room");
    assert!(Tokenizer::new(&bin[..bin.len() - 1], vocab, &mut vec![0; vocab], &mut vec![0; vocab]).is_err());
}

#[test]
fn generation_is_deterministic_and_greedy_follows_the_argmax() {
    let f = file(SMALL);
    let a = run(&f, b"ab c", 12, Sampler::GREEDY).unwrap();
    let b = run(&f, b"ab c", 12, Sampler::GREEDY).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 12);
    // `generate`'s own emitted text matches decoding the same ids.
    let words = aligned(&f);
    let model = model::parse(bytes(&words, f.len())).unwrap();
    let c = model.config;
    let (nf, nb) = Workspace::sizes(&c);
    let (mut fl, mut by) = (vec![0f32; nf], vec![0i8; nb]);
    let mut ws = Workspace::new(&c, &mut fl, &mut by).unwrap();
    let (mut offsets, mut sorted) = (vec![0u32; c.vocab_size], vec![0u32; c.vocab_size]);
    let tok = Tokenizer::new(model.tokenizer, c.vocab_size, &mut offsets, &mut sorted).unwrap();
    let mut tokens = vec![0u32; c.seq_len];
    let mut text = Vec::new();
    let stops_at_bos = a.iter().position(|&t| t == BOS).unwrap_or(a.len());
    let n = generate(&model, &tok, &mut ws, &mut tokens, Request { prompt: b"ab c", max_tokens: 12, sampler: Sampler::GREEDY }, |p| {
        text.extend_from_slice(p);
        true
    }).unwrap();
    assert_eq!(n, stops_at_bos);
    let mut expected = Vec::new();
    let mut prev = tok.lookup(b"c").unwrap();
    for &t in &a[..stops_at_bos] {
        let mut byte = [0u8; 1];
        expected.extend_from_slice(tok.decode(prev, t, &mut byte));
        prev = t;
    }
    assert_eq!(text, expected);
    // A seeded sample is repeatable, and a different seed can differ.
    let sample = |seed| {
        let (mut fl, mut by) = (vec![0f32; nf], vec![0i8; nb]);
        let mut ws = Workspace::new(&c, &mut fl, &mut by).unwrap();
        let mut text = Vec::new();
        let sampler = Sampler { temperature: 1.0, seed };
        generate(&model, &tok, &mut ws, &mut tokens.clone(), Request { prompt: b"ab", max_tokens: 12, sampler }, |p| {
        text.extend_from_slice(p);
        true
    }).unwrap();
        text
    };
    assert_eq!(sample(42), sample(42));
    assert!((1..20).any(|seed| sample(seed) != sample(42)), "sampling uses its seed");
}

#[test]
fn a_prompt_that_fills_the_context_is_refused_and_generation_stops_at_its_end() {
    let f = file(SMALL);
    assert_eq!(run(&f, &[b'z'; 31], 1, Sampler::GREEDY).err(), Some(GenerateError::PromptTooLong));
    // 2 + 20 tokens of prompt leaves 10 positions of a 32-token context.
    let out = run(&f, &[b'z'; 20], 100, Sampler::GREEDY).unwrap();
    assert_eq!(out.len(), 32 - 21);
}

#[test]
fn grouped_query_attention_and_multi_query_both_run() {
    for kv_heads in [1, 2, 4] {
        let f = file([96, 160, 2, 4, kv_heads, VOCAB, 32]);
        let out = run(&f, b"ab", 4, Sampler::GREEDY).unwrap();
        assert_eq!(out.len(), 4);
    }
}

/// The engine's logits against [`crate::reference`]'s, an independent f32 forward pass over the original checkpoint, at
/// each position of a fixed token sequence: int8 rounding moves them a little, never a lot. A slip in the layout or the
/// arithmetic (a missing rotation, a wrong head's cache, the wrong layer's weights) moves them a lot.
#[test]
fn the_engine_tracks_an_independent_f32_forward_pass() {
    use crate::reference::Ref;
    for kv_heads in [1, 2, 4] {
        let config = [96, 160, 2, 4, kv_heads, VOCAB, 32];
        let ck = checkpoint_with_attention(config, 11, 6.0);
        let f = convert(&ck, &tokenizer_bin(EXTRA)).unwrap();
        let words = aligned(&f);
        let model = model::parse(bytes(&words, f.len())).unwrap();
        let c = model.config;
        let (nf, nb) = Workspace::sizes(&c);
        let (mut fl, mut by) = (vec![0f32; nf], vec![0i8; nb]);
        let mut ws = Workspace::new(&c, &mut fl, &mut by).unwrap();
        let tokens: Vec<u32> = (0..20).map(|i| (i * 37 % VOCAB) as u32).collect();
        let reference = Ref::new(&ck);
        let mut worst = 1f32;
        reference.each_logits(&tokens, 0, |pos, want| {
            let got = forward(&model, &mut ws, tokens[pos], pos).to_vec();
            let dot: f32 = got.iter().zip(want).map(|(a, b)| a * b).sum();
            let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cosine = dot / (norm(&got) * norm(want));
            assert!(cosine > 0.999, "kv_heads {kv_heads}, position {pos}: cosine {cosine}");
            worst = worst.min(cosine);
            tokens.get(pos + 1).copied().unwrap_or(0)
        });
        eprintln!("kv_heads {kv_heads}: worst cosine {worst}");
    }
}

#[test]
fn a_workspace_too_small_is_refused() {
    let f = file(SMALL);
    let words = aligned(&f);
    let c = model::parse(bytes(&words, f.len())).unwrap().config;
    let (nf, nb) = Workspace::sizes(&c);
    assert!(Workspace::new(&c, &mut vec![0f32; nf - 1], &mut vec![0i8; nb]).is_none());
    assert!(Workspace::new(&c, &mut vec![0f32; nf], &mut vec![0i8; nb - 1]).is_none());
}
