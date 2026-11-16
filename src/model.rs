//! The model file, `LMODEL01`: what `lantern-mkmodel` writes and the inference service reads from its partition.
//!
//! ```text
//! 0     "LMODEL01"
//! 8     checkpoint length, u64 LE
//! 16    tokenizer length, u64 LE
//! 24    zero, to HEADER
//! 512   the checkpoint: llama2.c's version-2 int8 export, as `runq.c` reads it (its own 256-byte header first)
//! ...   the tokenizer: llama2.c's `tokenizer.bin`
//! ```
//!
//! The file's identity is the BLAKE3 digest of all of it, which the service checks before it parses anything. Parsing
//! still checks every length and offset: a digest says which file, not that its writer was careful.

/// The model file's own header, a sector, so the checkpoint starts aligned.
pub const HEADER: usize = 512;
pub const MAGIC: &[u8; 8] = b"LMODEL01";
/// The checkpoint's own header, in llama2.c's version 2.
pub const CHECKPOINT_HEADER: usize = 256;
const CHECKPOINT_MAGIC: u32 = 0x616b_3432;
const CHECKPOINT_VERSION: u32 = 2;

/// The transformer's shape, from the checkpoint's header.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// The quantization group: every quantized row is in groups of this many, each with one f32 scale.
    pub group_size: usize,
    /// Whether the classifier is the token embedding table (as in the TinyStories models).
    pub shared_classifier: bool,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
    pub fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }
}

/// What can be wrong with a model file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelError {
    /// Not an `LMODEL01` file, or its lengths don't fit what was read.
    Format,
    /// A checkpoint header this engine doesn't run: wrong magic or version, or a shape that doesn't divide.
    Checkpoint,
    /// The tokenizer doesn't parse, or doesn't have the checkpoint's vocabulary.
    Tokenizer,
    /// The weights the header promises are longer than the checkpoint.
    Truncated,
}

/// An int8 tensor in groups: `q` the values, `s` one scale per group.
#[derive(Clone, Copy)]
pub struct QTensor<'a> {
    pub q: &'a [i8],
    pub s: &'a [f32],
}

/// One tensor per layer, stored as runq.c stores them: each layer's values, then its scales.
#[derive(Clone, Copy)]
pub struct Layers<'a> {
    span: &'a [u8],
    each: usize,
    group_size: usize,
}

impl<'a> Layers<'a> {
    fn per(&self) -> usize {
        self.each + self.each / self.group_size * 4
    }

    /// Layer `l`'s tensor.
    pub fn get(&self, l: usize) -> QTensor<'a> {
        let per = self.per();
        let (q, s) = self.span[l * per..(l + 1) * per].split_at(self.each);
        QTensor { q: i8s(q), s: f32s(s).expect("checked when parsed") }
    }
}

/// The weights, as views into the model file: nothing is copied.
pub struct Weights<'a> {
    pub rms_att: &'a [f32],
    pub rms_ffn: &'a [f32],
    pub rms_final: &'a [f32],
    pub tokens: QTensor<'a>,
    pub wq: Layers<'a>,
    pub wk: Layers<'a>,
    pub wv: Layers<'a>,
    pub wo: Layers<'a>,
    pub w1: Layers<'a>,
    pub w2: Layers<'a>,
    pub w3: Layers<'a>,
    pub wcls: QTensor<'a>,
}

/// A parsed model file.
pub struct Model<'a> {
    pub config: Config,
    pub weights: Weights<'a>,
    pub tokenizer: &'a [u8],
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// The file's total length, from its first [`HEADER`] bytes: what a reader must read before it can check the digest.
pub fn file_len(header: &[u8]) -> Result<usize, ModelError> {
    if header.get(..8) != Some(MAGIC.as_slice()) {
        return Err(ModelError::Format);
    }
    let checkpoint = u64_at(header, 8).ok_or(ModelError::Format)?;
    let tokenizer = u64_at(header, 16).ok_or(ModelError::Format)?;
    let total = (HEADER as u64).checked_add(checkpoint).and_then(|n| n.checked_add(tokenizer)).ok_or(ModelError::Format)?;
    usize::try_from(total).map_err(|_| ModelError::Format)
}

/// Splits `n` bytes off the front of `rest`.
fn take<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8], ModelError> {
    if rest.len() < n {
        return Err(ModelError::Truncated);
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}

fn f32s(bytes: &[u8]) -> Result<&[f32], ModelError> {
    // SAFETY: every bit pattern is a valid f32, and `align_to` only puts correctly aligned, whole f32s in the middle.
    let (head, floats, tail) = unsafe { bytes.align_to::<f32>() };
    if head.is_empty() && tail.is_empty() {
        Ok(floats)
    } else {
        Err(ModelError::Format)
    }
}

fn i8s(bytes: &[u8]) -> &[i8] {
    // SAFETY: i8 and u8 have the same size, alignment and validity.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) }
}

fn floats<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [f32], ModelError> {
    f32s(take(rest, n.checked_mul(4).ok_or(ModelError::Checkpoint)?)?)
}

/// `layers` tensors of `each` values, each checked to be whole groups with aligned scales.
fn layers<'a>(rest: &mut &'a [u8], layers: usize, each: usize, gs: usize) -> Result<Layers<'a>, ModelError> {
    if !each.is_multiple_of(gs) {
        return Err(ModelError::Checkpoint);
    }
    let per = each.checked_add(each / gs * 4).ok_or(ModelError::Checkpoint)?;
    let span = take(rest, per.checked_mul(layers).ok_or(ModelError::Checkpoint)?)?;
    let tensors = Layers { span, each, group_size: gs };
    for l in 0..layers {
        f32s(&span[l * per + each..(l + 1) * per])?;
    }
    Ok(tensors)
}

/// Parses a model file that has already been read whole (and, by the service, checked against its digest).
pub fn parse(file: &[u8]) -> Result<Model<'_>, ModelError> {
    let total = file_len(file)?;
    if file.len() < total {
        return Err(ModelError::Format);
    }
    let checkpoint_len = u64_at(file, 8).ok_or(ModelError::Format)? as usize;
    let checkpoint = &file[HEADER..HEADER + checkpoint_len];
    let tokenizer = &file[HEADER + checkpoint_len..total];
    if u32_at(checkpoint, 0) != Some(CHECKPOINT_MAGIC) || u32_at(checkpoint, 4) != Some(CHECKPOINT_VERSION) {
        return Err(ModelError::Checkpoint);
    }
    let field = |i: usize| u32_at(checkpoint, 8 + 4 * i).map(|v| v as i32).ok_or(ModelError::Checkpoint);
    let positive = |v: i32| if v > 0 && v < (1 << 24) { Ok(v as usize) } else { Err(ModelError::Checkpoint) };
    let shared = *checkpoint.get(36).ok_or(ModelError::Checkpoint)?;
    let gs = u32_at(checkpoint, 37).ok_or(ModelError::Checkpoint)? as usize;
    let config = Config {
        dim: positive(field(0)?)?,
        hidden_dim: positive(field(1)?)?,
        n_layers: positive(field(2)?)?,
        n_heads: positive(field(3)?)?,
        n_kv_heads: positive(field(4)?)?,
        vocab_size: positive(field(5)?)?,
        seq_len: positive(field(6)?)?,
        group_size: gs,
        shared_classifier: shared == 1,
    };
    check_shape(&config)?;
    let weights = read_weights(&config, checkpoint)?;
    Ok(Model { config, weights, tokenizer })
}

/// The shape the engine needs: everything divides, and every quantized row is whole groups of 4-byte-aligned scales.
fn check_shape(c: &Config) -> Result<(), ModelError> {
    let ok = c.group_size >= 4
        && c.group_size.is_multiple_of(4)
        && c.dim.is_multiple_of(c.n_heads)
        && c.n_heads.is_multiple_of(c.n_kv_heads)
        && c.head_size().is_multiple_of(2)
        && c.dim.is_multiple_of(c.group_size)
        && c.hidden_dim.is_multiple_of(c.group_size)
        && c.seq_len <= 4096
        && c.vocab_size <= u32::MAX as usize;
    ok.then_some(()).ok_or(ModelError::Checkpoint)
}

fn read_weights<'a>(c: &Config, checkpoint: &'a [u8]) -> Result<Weights<'a>, ModelError> {
    let mut rest = checkpoint.get(CHECKPOINT_HEADER..).ok_or(ModelError::Truncated)?;
    let (dim, hidden, n, gs) = (c.dim, c.hidden_dim, c.n_layers, c.group_size);
    let (qdim, kv) = (c.n_heads * c.head_size(), c.kv_dim());
    let rms_att = floats(&mut rest, n * dim)?;
    let rms_ffn = floats(&mut rest, n * dim)?;
    let rms_final = floats(&mut rest, dim)?;
    let tokens = layers(&mut rest, 1, c.vocab_size * dim, gs)?.get(0);
    let wq = layers(&mut rest, n, dim * qdim, gs)?;
    let wk = layers(&mut rest, n, dim * kv, gs)?;
    let wv = layers(&mut rest, n, dim * kv, gs)?;
    let wo = layers(&mut rest, n, qdim * dim, gs)?;
    let w1 = layers(&mut rest, n, dim * hidden, gs)?;
    let w2 = layers(&mut rest, n, hidden * dim, gs)?;
    let w3 = layers(&mut rest, n, dim * hidden, gs)?;
    let wcls = if c.shared_classifier { tokens } else { layers(&mut rest, 1, dim * c.vocab_size, gs)?.get(0) };
    Ok(Weights { rms_att, rms_ffn, rms_final, tokens, wq, wk, wv, wo, w1, w2, w3, wcls })
}
