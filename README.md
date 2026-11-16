# lantern-ai-runtime

AI as a **first-class operating-system capability**: local inference, model management, and
**capability-scoped, auditable agents**. An agent never receives unrestricted access — it
receives explicit capabilities, and everything it does is logged and attributable.

- **Layer:** runtime (on top of [`lantern-runtime`](https://github.com/lantern-os/lantern-runtime); confined user space).
- **System context:** [wiki/AI](https://github.com/lantern-os/lantern-docs/blob/main/wiki/AI.md).

> **Phase 4** ([RFC-0036](../lantern-rfcs/rfcs/0036-phase-3-to-phase-4-transition.md)): hardware and assurance. The local inference engine is built
> ([RFC-0034](../lantern-rfcs/rfcs/0034-local-inference-and-a-file-assistant.md)): `lantern-boot`'s confined
> `inference-service` runs it behind a capability. Larger models and accelerators wait for real hardware. See [`STATUS.md`](./STATUS.md).

## In this repo
- **`lantern-ai-runtime`**, a `no_std` Rust crate with no allocation: a Llama-family transformer over int8 weights, after
  Andrej Karpathy's llama2.c `runq.c` (MIT).
  - `model`: the `LMODEL01` model file, its weights viewed in place.
  - `tokenizer`: llama2.c's byte-pair tokenizer.
  - `engine`: the forward pass, sampling and generation.
  - `convert` (feature `std`): a llama2.c f32 checkpoint to an `LMODEL01` file, for `lantern-boot`'s `lantern-mkmodel`.
- Host tools, with `--features std`: `lantern-generate <model> <tokens> <prompt>` runs the engine; `lantern-reference
  <checkpoint> <model> <tokens> <prompt>` compares it with an independent f32 forward pass on the original checkpoint.
- `cargo test` runs the host tests, on small synthetic models.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md), [`THREAT_MODEL.md`](./THREAT_MODEL.md), [`STATUS.md`](./STATUS.md).

## The non-negotiable rule
> An AI agent acts **only** within its granted capabilities, and every consequential action
> is recorded in a tamper-evident audit log. No "the AI can do anything" mode exists.
