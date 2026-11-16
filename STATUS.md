# lantern-ai-runtime — Status

**Phase:** 3 (Privacy, identity, networking, and AI) — opened per [RFC-0017](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0017-phase-2-to-phase-3-transition.md)/[ADR-0021](https://github.com/lantern-os/lantern-rfcs/blob/main/adr/0021-phase-2-complete-phase-3-opened.md). The inference engine is built, and runs confined (RFC-0034).

## Done
- Three-jobs model, agent capability model, and audit approach drafted and reviewed ([ARCHITECTURE.md](./ARCHITECTURE.md)).
- Prompt-injection-as-capability-containment argument documented.
- Threat model drafted and reviewed.

- **A first audit log exists (2026-11-05)**, in `lantern-shell/policy-store` for now:
  [RFC-0024](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0024-audit-log.md)/[ADR-0029](https://github.com/lantern-os/lantern-rfcs/blob/main/adr/0029-audit-log.md). It's a hash-chained
  record of the launch binder's grants, declines and revocations, with none of it in the TCB. It doesn't yet record what an agent
  does with a grant.
- **A person decides what an agent gets (2026-11-11)**: [RFC-0026](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0026-trusted-path-and-consent.md)/
  [ADR-0032](https://github.com/lantern-os/lantern-rfcs/blob/main/adr/0032-trusted-path-and-consent.md). In `lantern-boot-system`, each grant within the system's
  policy is asked at a console only the shell reaches; the person can list, revoke, and read a verified audit log there. With
  the audit log, that is the exit criterion's "visible, revocable capability set and faithful audit log", for one app, with no
  model yet: local inference is still to come.
- **It records what an agent does with a grant (2026-11-10)**: [RFC-0025](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0025-audit-log-service-operations.md)/[ADR-0030](https://github.com/lantern-os/lantern-rfcs/blob/main/adr/0030-audit-log-service-operations.md). The keystore and filesystem services record each
  use before acting, under the grant's badge, so the log answers "under which grant, did what, to which object, in what order".
  The runtime hosting the agent records nothing: it's the party being audited. The log retires its oldest history when full,
  visibly, and a per-grant quota stops one agent pushing everyone else's out. "When" is still order only.

- **Local inference and a file assistant (2026-11-16)**: [RFC-0034](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0034-local-inference-and-a-file-assistant.md)/
  [ADR-0039](https://github.com/lantern-os/lantern-rfcs/blob/main/adr/0039-local-inference-and-a-file-assistant.md). Phase 3's exit criterion "a local AI agent
  with a visible, revocable capability set and a faithful audit log" is met.
  - This crate: a `no_std`, allocation-free Llama-family transformer over int8 weights (llama2.c `runq.c`'s scheme), its
    tokenizer, and the `LMODEL01` model file. 9 host tests on synthetic models: shape parsing, quantization error, hostile
    and truncated files, tokenizer merges and byte fallback, determinism, seeded sampling, the context limit, and the
    engine's logits against an independent f32 forward pass (`reference`) at every position, for multi-query,
    grouped-query and full attention. The test model's attention is sharpened so a mistake in positions shows: leaving
    the keys unrotated fails it.
  - Checked on the real `stories15M`: an independent f32 reference (`lantern-reference`) agrees with the int8 engine for
    the first 50, 64 and 71 tokens of three prompts. llama2.c's own `runq.c` wasn't run.
  - `lantern-boot/inference-service` runs it confined, its model read from the disk and digest-checked. Under QEMU's
    emulation, 64 tokens take about 2 seconds, and the words match the host's token for token.
  - The file assistant (`lantern-runtime/file-assistant-guest`) reads file 1, asks the model to continue it, and writes
    file 2. `console-matrix.sh` runs it: granted, declined, given text that tells it what to do, and with a changed or
    missing model.

## Next
- Define standard capability shapes for common agent powers.
- Measure the cost per operation on real flash (Phase 4), and export retired history before it's erased.
- A person-facing view of the log, with names rather than badges and hashes, as part of the consent surface.
- Models as signed packages a person chooses; more than one.
- Record how many tokens a call generated, after it.
- A deadline on a call, and fair scheduling across agents.
- An instruction-following model on the RISC-V board (SmolLM2-135M or similar: the engine already runs grouped-query
  attention, and would need its rotary base, fixed at 10000 now, read from the model), and accelerators behind the same `inference` interface.

## Blocked on
- Accelerator access via HAL/IOMMU ([`lantern-hal`](https://github.com/lantern-os/lantern-hal)), for anything beyond the CPU.
