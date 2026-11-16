//! `lantern-generate <model.lmodel> <max-tokens> <prompt>`: the engine on the host, greedy, printing the prompt and its
//! continuation, and how fast it ran.

use lantern_ai_runtime::engine::{generate, Request, Sampler, Workspace};
use lantern_ai_runtime::{model, tokenizer::Tokenizer};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, max, prompt] = args.as_slice() else {
        eprintln!("usage: lantern-generate <model.lmodel> <max-tokens> <prompt>");
        std::process::exit(2);
    };
    let bytes = std::fs::read(path).expect("read the model");
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    // SAFETY: a u32 buffer at least `bytes.len()` long, viewed as bytes, so the model's f32s are aligned.
    let file = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) };
    file.copy_from_slice(&bytes);
    let model = model::parse(file).expect("parse the model");
    let c = model.config;
    let (nf, nb) = Workspace::sizes(&c);
    let (mut f, mut b) = (vec![0f32; nf], vec![0i8; nb]);
    let mut ws = Workspace::new(&c, &mut f, &mut b).unwrap();
    let (mut offsets, mut sorted) = (vec![0u32; c.vocab_size], vec![0u32; c.vocab_size]);
    let tok = Tokenizer::new(model.tokenizer, c.vocab_size, &mut offsets, &mut sorted).expect("parse the tokenizer");
    let mut tokens = vec![0u32; c.seq_len];
    let mut out = std::io::stdout();
    out.write_all(prompt.as_bytes()).unwrap();
    let start = std::time::Instant::now();
    let request = Request { prompt: prompt.as_bytes(), max_tokens: max.parse().expect("a number"), sampler: Sampler::GREEDY };
    let n = generate(&model, &tok, &mut ws, &mut tokens, request, |piece| out.write_all(piece).is_ok()).expect("generate");
    println!();
    eprintln!("{n} tokens, {:.1} tok/s", n as f64 / start.elapsed().as_secs_f64());
}
