//! `lantern-reference <checkpoint.bin> <model.lmodel> <max-tokens> <prompt>`: an independent check on the engine. It runs
//! the ORIGINAL f32 checkpoint through `lantern_ai_runtime::reference`, greedily, and the int8 model file through the
//! engine, and says how many tokens they agree on. int8 drifts from f32 eventually, so the two should agree for a long
//! prefix, not forever.

use lantern_ai_runtime::engine::{forward, Workspace};
use lantern_ai_runtime::reference::Ref;
use lantern_ai_runtime::{model, tokenizer::Tokenizer};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [ck, lm, max, prompt] = args.as_slice() else {
        eprintln!("usage: lantern-reference <checkpoint.bin> <model.lmodel> <max-tokens> <prompt>");
        std::process::exit(2);
    };
    let max: usize = max.parse().unwrap();
    let r = Ref::new(&std::fs::read(ck).unwrap());
    let bytes = std::fs::read(lm).unwrap();
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    // SAFETY: a u32 buffer at least `bytes.len()` long, viewed as bytes.
    let file = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) };
    file.copy_from_slice(&bytes);
    let model = model::parse(file).unwrap();
    let c = model.config;
    let (mut offsets, mut sorted) = (vec![0u32; c.vocab_size], vec![0u32; c.vocab_size]);
    let tok = Tokenizer::new(model.tokenizer, c.vocab_size, &mut offsets, &mut sorted).unwrap();
    let mut prompt_ids = vec![0u32; c.seq_len];
    let n = tok.encode(prompt.as_bytes(), &mut prompt_ids).unwrap();
    let prompt_ids = &prompt_ids[..n];
    let reference = r.run(prompt_ids, max);
    let (nf, nb) = Workspace::sizes(&c);
    let (mut f, mut b) = (vec![0f32; nf], vec![0i8; nb]);
    let mut ws = Workspace::new(&c, &mut f, &mut b).unwrap();
    let mut engine = Vec::new();
    let mut token = prompt_ids[0];
    for pos in 0..(n - 1 + max).min(c.seq_len) {
        let logits = forward(&model, &mut ws, token, pos);
        let next = if pos + 1 < n { prompt_ids[pos + 1] } else { (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b }) as u32 };
        if pos + 1 >= n {
            engine.push(next);
        }
        token = next;
    }
    let agree = reference.iter().zip(&engine).take_while(|(a, b)| a == b).count();
    println!("prompt {n} tokens; f32 reference and int8 engine agree on the first {agree} of {} generated", engine.len());
}
