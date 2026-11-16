//! An independent f32 forward pass over a legacy llama2.c checkpoint: matrices as rows, no code shared with
//! [`crate::engine`]. What the engine is checked against, by the host tests and `lantern-reference`. Not for the
//! service: it allocates, and it's slow.

/// A legacy f32 checkpoint's shape and weights.
pub struct Ref {
    pub dim: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub seq: usize,
    pub w: Vec<f32>,
}

fn mat(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows).map(|r| w[r * cols..(r + 1) * cols].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

fn norm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ss = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5).sqrt().recip();
    x.iter().zip(w).map(|(v, w)| w * v * ss).collect()
}

impl Ref {
    /// Reads a legacy checkpoint with a shared classifier.
    pub fn new(checkpoint: &[u8]) -> Self {
        let h: Vec<i32> = (0..7).map(|i| i32::from_le_bytes(checkpoint[i * 4..i * 4 + 4].try_into().unwrap())).collect();
        assert!(h[5] > 0, "this reference handles shared classifiers only");
        Ref {
            dim: h[0] as usize,
            hidden: h[1] as usize,
            layers: h[2] as usize,
            heads: h[3] as usize,
            kv_heads: h[4] as usize,
            vocab: h[5] as usize,
            seq: h[6] as usize,
            w: checkpoint[28..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        }
    }

    /// Greedy decoding after `prompt`, for up to `max` tokens.
    pub fn run(&self, prompt: &[u32], max: usize) -> Vec<u32> {
        let mut out = Vec::new();
        self.each_logits(prompt, max, |pos, logits| {
            let next = if pos + 1 < prompt.len() {
                prompt[pos + 1]
            } else {
                (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b }) as u32
            };
            if pos + 1 >= prompt.len() {
                out.push(next);
            }
            next
        });
        out
    }

    /// Runs `prompt` then up to `max` more positions, calling `step` with each position's logits; `step` returns the
    /// next token.
    pub fn each_logits(&self, prompt: &[u32], max: usize, mut step: impl FnMut(usize, &[f32]) -> u32) {
        let (d, hd, hs) = (self.dim, self.hidden, self.dim / self.heads);
        let kv = d * self.kv_heads / self.heads;
        let mut off = 0usize;
        let mut part = |n: usize| {
            let s = off;
            off += n;
            s
        };
        let (l, v) = (self.layers, self.vocab);
        let emb = part(v * d);
        let ratt = part(l * d);
        let wq = part(l * d * d);
        let wk = part(l * d * kv);
        let wv = part(l * d * kv);
        let wo = part(l * d * d);
        let rffn = part(l * d);
        let w1 = part(l * d * hd);
        let w2 = part(l * hd * d);
        let w3 = part(l * d * hd);
        let rfin = part(d);
        let w = &self.w;
        let mut kc: Vec<Vec<Vec<f32>>> = (0..l).map(|_| Vec::new()).collect();
        let mut vc: Vec<Vec<Vec<f32>>> = (0..l).map(|_| Vec::new()).collect();
        let mut token = prompt[0];
        for pos in 0..(prompt.len() - 1 + max).min(self.seq) {
            let mut x = w[emb + token as usize * d..][..d].to_vec();
            for i in 0..l {
                let xb = norm(&x, &w[ratt + i * d..][..d]);
                let mut q = mat(&w[wq + i * d * d..], d, d, &xb);
                let mut k = mat(&w[wk + i * d * kv..], kv, d, &xb);
                let val = mat(&w[wv + i * d * kv..], kv, d, &xb);
                for j in (0..d).step_by(2) {
                    let f = 1.0 / 10000f32.powf((j % hs) as f32 / hs as f32);
                    let (c, s) = ((pos as f32 * f).cos(), (pos as f32 * f).sin());
                    let r = |v: &mut Vec<f32>| {
                        let (a, b) = (v[j], v[j + 1]);
                        v[j] = a * c - b * s;
                        v[j + 1] = a * s + b * c;
                    };
                    r(&mut q);
                    if j < kv {
                        r(&mut k);
                    }
                }
                kc[i].push(k);
                vc[i].push(val);
                let mut att_out = vec![0f32; d];
                for h in 0..self.heads {
                    let g = h / (self.heads / self.kv_heads);
                    let qh = &q[h * hs..(h + 1) * hs];
                    let mut sc: Vec<f32> = kc[i]
                        .iter()
                        .map(|k| qh.iter().zip(&k[g * hs..(g + 1) * hs]).map(|(a, b)| a * b).sum::<f32>() / (hs as f32).sqrt())
                        .collect();
                    let m = sc.iter().cloned().fold(f32::MIN, f32::max);
                    sc.iter_mut().for_each(|s| *s = (*s - m).exp());
                    let sum: f32 = sc.iter().sum();
                    for (t, s) in sc.iter().enumerate() {
                        for e in 0..hs {
                            att_out[h * hs + e] += s / sum * vc[i][t][g * hs + e];
                        }
                    }
                }
                let o = mat(&w[wo + i * d * d..], d, d, &att_out);
                x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
                let xb = norm(&x, &w[rffn + i * d..][..d]);
                let a = mat(&w[w1 + i * d * hd..], hd, d, &xb);
                let b = mat(&w[w3 + i * d * hd..], hd, d, &xb);
                let hmix: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
                let o = mat(&w[w2 + i * hd * d..], d, hd, &hmix);
                x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            }
            let x = norm(&x, &w[rfin..][..d]);
            let logits = mat(&w[emb..], v, d, &x); // the TinyStories models share their classifier
            token = step(pos, &logits);
        }
    }
}

