# lantern-ai-runtime — Architecture

Companion to [wiki/AI](https://github.com/lantern-os/lantern-docs/blob/main/wiki/AI.md). Builds on the WASM runtime and capability
model.

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
