# lantern-ai-runtime — Architecture

Companion to [wiki/AI](https://github.com/lantern-os/lantern-docs/blob/main/wiki/AI.md). Builds on the WASM runtime and capability
model.

## As built ([RFC-0034](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0034-local-inference-and-a-file-assistant.md))

```
 file assistant (Wasm, confined)          inference-service (native, confined)
   open(0) file 1 ── read ──► store        model partition ──► volume ──► heap: LMODEL01, digest-checked
   open(1) file 2 ── write ─► store        GENERATE: badge live? → audit log → engine → text
   inference.open(0) ── generate ────────► (this crate's engine)
```

- **The engine** (this crate): int8 weights in groups with an f32 scale each, activations quantized the same way before
  every matrix multiply, f32 elsewhere. Every buffer comes from the caller ([`engine::Workspace::sizes`]), so the service
  knows its memory before the model runs. Greedy by default, or sampled from a caller's seed: deterministic either way.
- **The model file**, `LMODEL01`: a sector of header, then llama2.c's version-2 int8 checkpoint, then its tokenizer. Its
  identity is the BLAKE3 digest of all of it, which the service checks before parsing anything.
- **The service** (`lantern-boot/inference-service`) is a separate confined program, not part of the agent's runtime:
  the weights and the work live outside the agent's process, and the capability is the boundary. It holds the model's
  partition (read-only), its own endpoint, and an append-only badge on the audit log. Nothing else.
- **The agent** is an app, granted through the same binder and console as any other.

## Three jobs
1. **Local inference** — run models on-device (CPU/GPU/NPU); cloud models are an optional
   peer accessed under explicit, audited capabilities. Local-first by default.
2. **Model management** — models are content-addressed artefacts with provenance/integrity
   ([`lantern-filesystem`](https://github.com/lantern-os/lantern-filesystem)); accelerator scheduling shares scarce
   resources fairly across agents.
3. **Capability-scoped, auditable agency** — the heart of the design.

## Agent model
- An agent is a confined component ([`lantern-runtime`](https://github.com/lantern-os/lantern-runtime)) holding a
  **capability set**. It can read *this* folder or call *that* tool only because it was
  granted those caps — never ambiently.
- Capabilities are **attenuable and revocable**: "read this folder", "email this address for
  the next hour", revocable instantly ([RFC-0003](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0003-capability-model.md)).
- The **shell** ([`lantern-shell`](https://github.com/lantern-os/lantern-shell)) is the locus of consent: granting a new
  capability is a visible, user-mediated act.

## Auditability
Every consequential action → a **tamper-evident audit log** entry: which agent, under which
capability, did what, to which object, when. Enables accountability, provenance, and
containment review. Making the log trustworthy without bloating the TCB is an open question.

## Why capabilities defeat prompt injection
An induced/injected agent cannot exceed its granted capabilities, no matter what it is
tricked into "wanting." No network capability ⇒ no exfiltration, regardless of the prompt.
This is why AI sits *on top of* the capability model (system threat T4).

## Hardware
GPU/NPU via confined drivers behind the [HAL](https://github.com/lantern-os/lantern-hal) and an IOMMU; the long-term
[open-hardware](https://github.com/lantern-os/lantern-docs/blob/main/wiki/Hardware.md) AI accelerator path is also
capability-mediated and auditable.

## Open questions
- Model/agent scheduling across CPU/GPU/NPU under fairness/energy constraints.
- Audit-log tamper-evidence and storage cost; TCB footprint.
- Standard capability *shapes* for common agent powers (comprehensible consent).
- Sandboxing inference of large models that may embed behaviours.
- Consent UX that is meaningful, not click-through.
