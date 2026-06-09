# lantern-ai-runtime

AI as a **first-class operating-system capability**: local inference, model management, and
**capability-scoped, auditable agents**. An agent never receives unrestricted access — it
receives explicit capabilities, and everything it does is logged and attributable.

- **Layer:** runtime (on top of [`lantern-runtime`](https://github.com/lantern-os/lantern-runtime); confined user space).
- **System context:** [wiki/AI](https://github.com/lantern-os/lantern-docs/blob/main/wiki/AI.md).

> ⚠️ **Phase 0.** Design only; no code. See [`STATUS.md`](./STATUS.md).

## In this repo
- [`ARCHITECTURE.md`](./ARCHITECTURE.md), [`THREAT_MODEL.md`](./THREAT_MODEL.md), [`STATUS.md`](./STATUS.md).

## The non-negotiable rule
> An AI agent acts **only** within its granted capabilities, and every consequential action
> is recorded in a tamper-evident audit log. No "the AI can do anything" mode exists.
