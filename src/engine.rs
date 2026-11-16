//! The transformer's forward pass and generation, after llama2.c's `runq.c`: int8 weights in groups with an f32 scale
//! per group, activations quantized the same way before each matrix multiply, f32 everywhere else.
//!
//! All the memory a forward pass needs is carved from one buffer the caller gives ([`Workspace::floats`] f32s), so the
//! engine allocates nothing and its size is known before the model runs.

use crate::model::{Config, Model, QTensor};
use crate::tokenizer::{Tokenizer, BOS};
use libm::{cosf, expf, powf, roundf, sinf, sqrtf};

/// The forward pass's buffers.
pub struct Workspace<'a> {
    x: &'a mut [f32],
    xb: &'a mut [f32],
    xb2: &'a mut [f32],
    hb: &'a mut [f32],
    hb2: &'a mut [f32],
    q: &'a mut [f32],
    att: &'a mut [f32],
    logits: &'a mut [f32],
    key_cache: &'a mut [f32],
    value_cache: &'a mut [f32],
    xq: (&'a mut [i8], &'a mut [f32]),
    hq: (&'a mut [i8], &'a mut [f32]),
}

fn split<'a, T>(rest: &mut &'a mut [T], n: usize) -> &'a mut [T] {
    let (head, tail) = core::mem::take(rest).split_at_mut(n);
    *rest = tail;
    head
}

impl<'a> Workspace<'a> {
    /// How many f32s a workspace for `c` needs, and how many i8s.
    pub fn sizes(c: &Config) -> (usize, usize) {
        let cache = c.n_layers * c.seq_len * c.kv_dim();
        let floats = 3 * c.dim
            + 2 * c.hidden_dim
            + c.dim
            + c.n_heads * c.seq_len
            + c.vocab_size
            + 2 * cache
            + c.dim / c.group_size
            + c.hidden_dim / c.group_size;
        (floats, c.dim + c.hidden_dim)
    }

    /// Carves a workspace for `c` from `floats` and `bytes`, at least [`sizes`](Self::sizes) long.
    pub fn new(c: &Config, floats: &'a mut [f32], bytes: &'a mut [i8]) -> Option<Self> {
        let (need_f, need_b) = Self::sizes(c);
        if floats.len() < need_f || bytes.len() < need_b {
            return None;
        }
        let cache = c.n_layers * c.seq_len * c.kv_dim();
        let mut f = floats;
        let mut b = bytes;
        Some(Workspace {
            x: split(&mut f, c.dim),
            xb: split(&mut f, c.dim),
            xb2: split(&mut f, c.dim),
            hb: split(&mut f, c.hidden_dim),
            hb2: split(&mut f, c.hidden_dim),
            q: split(&mut f, c.dim),
            att: split(&mut f, c.n_heads * c.seq_len),
            logits: split(&mut f, c.vocab_size),
            key_cache: split(&mut f, cache),
            value_cache: split(&mut f, cache),
            xq: (split(&mut b, c.dim), split(&mut f, c.dim / c.group_size)),
            hq: (split(&mut b, c.hidden_dim), split(&mut f, c.hidden_dim / c.group_size)),
        })
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / sqrtf(ss);
    for ((o, &w), &v) in o.iter_mut().zip(weight).zip(x) {
        *o = w * (ss * v);
    }
}

fn rmsnorm_in_place(x: &mut [f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / sqrtf(ss);
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = w * (ss * *v);
    }
}

fn softmax(x: &mut [f32]) {
    let mut max = x[0];
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - max);
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn quantize(q: &mut [i8], s: &mut [f32], x: &[f32], gs: usize) {
    for ((group, q), scale) in x.chunks_exact(gs).zip(q.chunks_exact_mut(gs)).zip(s.iter_mut()) {
        let mut wmax = 0.0f32;
        for &v in group {
            let a = libm::fabsf(v);
            if a > wmax {
                wmax = a;
            }
        }
        let sc = wmax / 127.0;
        *scale = sc;
        for (q, &v) in q.iter_mut().zip(group) {
            *q = roundf(v / sc) as i8;
        }
    }
}

/// `out = W x`, W of `out.len()` rows of `n`, both quantized in groups of `gs`.
fn matmul(out: &mut [f32], xq: &[i8], xs: &[f32], w: QTensor<'_>, n: usize, gs: usize) {
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w.q[i * n..(i + 1) * n];
        let scales = &w.s[i * n / gs..(i + 1) * n / gs];
        let mut val = 0.0f32;
        for ((wg, xg), (&ws, &xs)) in row.chunks_exact(gs).zip(xq.chunks_exact(gs)).zip(scales.iter().zip(xs)) {
            let mut ival = 0i32;
            for (&a, &b) in wg.iter().zip(xg) {
                ival += a as i32 * b as i32;
            }
            val += ival as f32 * ws * xs;
        }
        *o = val;
    }
}

/// The model's logits for `token` at position `pos`, after every position before it has been run.
pub fn forward<'w>(model: &Model<'_>, ws: &'w mut Workspace<'_>, token: u32, pos: usize) -> &'w mut [f32] {
    let c = &model.config;
    let w = &model.weights;
    let (dim, hidden, gs, hs, kv_dim) = (c.dim, c.hidden_dim, c.group_size, c.head_size(), c.kv_dim());
    let kv_mul = c.n_heads / c.n_kv_heads;
    // The token's embedding, dequantized from its row.
    let row = token as usize * dim;
    for (i, x) in ws.x.iter_mut().enumerate() {
        *x = w.tokens.q[row + i] as f32 * w.tokens.s[(row + i) / gs];
    }
    for l in 0..c.n_layers {
        rmsnorm(ws.xb, ws.x, &w.rms_att[l * dim..(l + 1) * dim]);
        quantize(ws.xq.0, ws.xq.1, ws.xb, gs);
        let loff = l * c.seq_len * kv_dim;
        let at = loff + pos * kv_dim;
        matmul(ws.q, ws.xq.0, ws.xq.1, w.wq.get(l), dim, gs);
        matmul(&mut ws.key_cache[at..at + kv_dim], ws.xq.0, ws.xq.1, w.wk.get(l), dim, gs);
        matmul(&mut ws.value_cache[at..at + kv_dim], ws.xq.0, ws.xq.1, w.wv.get(l), dim, gs);
        // Rotary position embedding, on q and k.
        for i in (0..dim).step_by(2) {
            let head_dim = i % hs;
            let freq = 1.0 / powf(10000.0, head_dim as f32 / hs as f32);
            let val = pos as f32 * freq;
            let (fcr, fci) = (cosf(val), sinf(val));
            let rotate = |v: &mut [f32]| {
                let (v0, v1) = (v[i], v[i + 1]);
                v[i] = v0 * fcr - v1 * fci;
                v[i + 1] = v0 * fci + v1 * fcr;
            };
            rotate(ws.q);
            if i < kv_dim {
                rotate(&mut ws.key_cache[at..at + kv_dim]);
            }
        }
        // Attention, head by head.
        for h in 0..c.n_heads {
            let q = &ws.q[h * hs..(h + 1) * hs];
            let att = &mut ws.att[h * c.seq_len..h * c.seq_len + pos + 1];
            for (t, a) in att.iter_mut().enumerate() {
                let k = &ws.key_cache[loff + t * kv_dim + (h / kv_mul) * hs..][..hs];
                let mut score = 0.0f32;
                for (&qi, &ki) in q.iter().zip(k) {
                    score += qi * ki;
                }
                *a = score / sqrtf(hs as f32);
            }
            softmax(att);
            let xb = &mut ws.xb[h * hs..(h + 1) * hs];
            xb.fill(0.0);
            for (t, &a) in att.iter().enumerate() {
                let v = &ws.value_cache[loff + t * kv_dim + (h / kv_mul) * hs..][..hs];
                for (x, &vi) in xb.iter_mut().zip(v) {
                    *x += a * vi;
                }
            }
        }
        quantize(ws.xq.0, ws.xq.1, ws.xb, gs);
        matmul(ws.xb2, ws.xq.0, ws.xq.1, w.wo.get(l), dim, gs);
        for (x, &d) in ws.x.iter_mut().zip(ws.xb2.iter()) {
            *x += d;
        }
        // The feed-forward network: w2(silu(w1 x) * w3 x).
        rmsnorm(ws.xb, ws.x, &w.rms_ffn[l * dim..(l + 1) * dim]);
        quantize(ws.xq.0, ws.xq.1, ws.xb, gs);
        matmul(ws.hb, ws.xq.0, ws.xq.1, w.w1.get(l), dim, gs);
        matmul(ws.hb2, ws.xq.0, ws.xq.1, w.w3.get(l), dim, gs);
        for (h, &h2) in ws.hb.iter_mut().zip(ws.hb2.iter()) {
            let mut val = *h;
            val *= 1.0 / (1.0 + expf(-val));
            val *= h2;
            *h = val;
        }
        quantize(ws.hq.0, ws.hq.1, ws.hb, gs);
        matmul(ws.xb, ws.hq.0, ws.hq.1, w.w2.get(l), hidden, gs);
        for (x, &d) in ws.x.iter_mut().zip(ws.xb.iter()) {
            *x += d;
        }
    }
    rmsnorm_in_place(ws.x, w.rms_final);
    quantize(ws.xq.0, ws.xq.1, ws.x, gs);
    matmul(ws.logits, ws.xq.0, ws.xq.1, w.wcls, dim, gs);
    ws.logits
}

/// How tokens are chosen: greedy at temperature 0, else sampled with llama2.c's xorshift from `seed`.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub temperature: f32,
    pub seed: u64,
}

impl Sampler {
    pub const GREEDY: Sampler = Sampler { temperature: 0.0, seed: 0 };

    fn random_f32(state: &mut u64) -> f32 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let u = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32;
        (u >> 8) as f32 / 16_777_216.0
    }

    fn sample(&self, state: &mut u64, logits: &mut [f32]) -> u32 {
        if self.temperature == 0.0 {
            let mut best = 0;
            for (i, &l) in logits.iter().enumerate() {
                if l > logits[best] {
                    best = i;
                }
            }
            return best as u32;
        }
        for l in logits.iter_mut() {
            *l /= self.temperature;
        }
        softmax(logits);
        let coin = Self::random_f32(state);
        let mut cdf = 0.0f32;
        for (i, &p) in logits.iter().enumerate() {
            cdf += p;
            if coin < cdf {
                return i as u32;
            }
        }
        logits.len() as u32 - 1
    }
}

/// Why a generation didn't run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenerateError {
    /// The prompt leaves no room in the model's context to generate anything.
    PromptTooLong,
}

/// What to generate: continue `prompt` by up to `max_tokens` tokens, chosen by `sampler`.
#[derive(Clone, Copy, Debug)]
pub struct Request<'p> {
    pub prompt: &'p [u8],
    pub max_tokens: usize,
    pub sampler: Sampler,
}

/// Continues `prompt` by up to `max_tokens` tokens, calling `emit` with each one's text, as `run.c`'s `generate` does: it
/// stops early at `BOS`, which ends a text, when the context is full, or when `emit` returns `false`. `tokens` holds the prompt's tokens, and must be
/// at least the model's context long. Returns how many tokens it generated and emitted.
pub fn generate(
    model: &Model<'_>,
    tokenizer: &Tokenizer<'_>,
    ws: &mut Workspace<'_>,
    tokens: &mut [u32],
    request: Request<'_>,
    mut emit: impl FnMut(&[u8]) -> bool,
) -> Result<usize, GenerateError> {
    let Request { prompt, max_tokens, sampler } = request;
    let seq_len = model.config.seq_len;
    let n = tokenizer.encode(prompt, &mut tokens[..seq_len]).ok_or(GenerateError::PromptTooLong)?;
    if n >= seq_len {
        return Err(GenerateError::PromptTooLong);
    }
    let steps = (n - 1 + max_tokens).min(seq_len);
    let mut rng = sampler.seed;
    let mut token = tokens[0];
    let mut generated = 0;
    let mut pos = 0;
    while pos < steps {
        let logits = forward(model, ws, token, pos);
        let next = if pos < n - 1 { tokens[pos + 1] } else { sampler.sample(&mut rng, logits) };
        pos += 1;
        if next == BOS {
            break;
        }
        if pos >= n {
            let mut byte = [0u8; 1];
            if !emit(tokenizer.decode(token, next, &mut byte)) {
                break;
            }
            generated += 1;
        }
        token = next;
    }
    Ok(generated)
}
