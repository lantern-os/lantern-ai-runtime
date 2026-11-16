//! A llama2.c checkpoint (`run.c`'s f32 "version 0" format, as Karpathy's TinyStories models are published) and its
//! `tokenizer.bin`, to an `LMODEL01` file: the weights quantized to int8 in groups, as llama2.c's `export.py
//! --version 2` lays them out, which is what `runq.c` reads.

use crate::model::{CHECKPOINT_HEADER, HEADER, MAGIC};

/// What a checkpoint's header says, and the f32 weights after it.
fn read_legacy(bytes: &[u8]) -> Result<([i32; 7], Vec<f32>), String> {
    if bytes.len() < 28 || !(bytes.len() - 28).is_multiple_of(4) {
        return Err("not a llama2.c checkpoint".into());
    }
    let mut config = [0i32; 7];
    for (i, c) in config.iter_mut().enumerate() {
        *c = i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    }
    let floats = bytes[28..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok((config, floats))
}

/// Quantizes `w` in groups of `gs`, as `runq.c`'s `quantize`: one scale per group, `max |w| / 127`.
pub fn quantize(w: &[f32], gs: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; w.len()];
    let mut s = vec![0f32; w.len() / gs];
    for ((group, q), scale) in w.chunks_exact(gs).zip(q.chunks_exact_mut(gs)).zip(s.iter_mut()) {
        let wmax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        *scale = wmax / 127.0;
        for (q, &v) in q.iter_mut().zip(group) {
            *q = libm::roundf(v / *scale) as i8;
        }
    }
    (q, s)
}

/// The `LMODEL01` file for a legacy checkpoint and tokenizer.
pub fn convert(checkpoint: &[u8], tokenizer: &[u8]) -> Result<Vec<u8>, String> {
    let (config, floats) = read_legacy(checkpoint)?;
    let [dim, hidden, layers, heads, kv_heads, vocab, seq_len] = config;
    let shared = vocab > 0;
    let vocab = vocab.unsigned_abs() as usize;
    if [dim, hidden, layers, heads, kv_heads, seq_len].iter().any(|&v| v <= 0) || dim % heads != 0 {
        return Err("a checkpoint header that doesn't make sense".into());
    }
    let (dim, hidden, layers, heads, kv_heads, seq_len) =
        (dim as usize, hidden as usize, layers as usize, heads as usize, kv_heads as usize, seq_len as usize);
    let hs = dim / heads;
    let kv = dim * kv_heads / heads;
    let mut gs = 64;
    while dim % gs != 0 {
        gs /= 2;
    }
    if gs < 4 || hidden % gs != 0 {
        return Err("no group size divides the model's dimensions".into());
    }
    let mut at = 0usize;
    let mut next = |n: usize| -> Result<&[f32], String> {
        let slice = floats.get(at..at + n).ok_or("the checkpoint is shorter than its header says")?;
        at += n;
        Ok(slice)
    };
    let tokens = next(vocab * dim)?;
    let rms_att = next(layers * dim)?;
    let wq = next(layers * dim * dim)?;
    let wk = next(layers * dim * kv)?;
    let wv = next(layers * dim * kv)?;
    let wo = next(layers * dim * dim)?;
    let rms_ffn = next(layers * dim)?;
    let w1 = next(layers * dim * hidden)?;
    let w2 = next(layers * hidden * dim)?;
    let w3 = next(layers * dim * hidden)?;
    let rms_final = next(dim)?;
    next(seq_len * hs / 2)?;
    next(seq_len * hs / 2)?;
    let wcls = if shared { None } else { Some(next(vocab * dim)?) };

    let mut out = Vec::new();
    out.extend_from_slice(&0x616b_3432u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    for v in [dim, hidden, layers, heads, kv_heads, vocab, seq_len] {
        out.extend_from_slice(&(v as i32).to_le_bytes());
    }
    out.push(shared as u8);
    out.extend_from_slice(&(gs as u32).to_le_bytes());
    out.resize(CHECKPOINT_HEADER, 0);
    for w in [rms_att, rms_ffn, rms_final] {
        w.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    }
    let mut tensor = |w: &[f32]| {
        let (q, s) = quantize(w, gs);
        out.extend(q.iter().map(|&b| b as u8));
        s.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    };
    tensor(tokens);
    for (all, each) in [(wq, dim * dim), (wk, dim * kv), (wv, dim * kv), (wo, dim * dim), (w1, dim * hidden), (w2, hidden * dim), (w3, dim * hidden)] {
        for layer in all.chunks_exact(each) {
            tensor(layer);
        }
    }
    if let Some(wcls) = wcls {
        tensor(wcls);
    }
    let mut file = vec![0u8; HEADER];
    file[..8].copy_from_slice(MAGIC);
    file[8..16].copy_from_slice(&(out.len() as u64).to_le_bytes());
    file[16..24].copy_from_slice(&(tokenizer.len() as u64).to_le_bytes());
    file.extend_from_slice(&out);
    file.extend_from_slice(tokenizer);
    Ok(file)
}

/// The version-2 checkpoint inside an `LMODEL01` file, as `runq.c` reads it.
pub fn checkpoint(file: &[u8]) -> Option<&[u8]> {
    let len = u64::from_le_bytes(file.get(8..16)?.try_into().ok()?) as usize;
    file.get(HEADER..HEADER + len)
}
