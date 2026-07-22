# lantern-ai-runtime — Status

**Phase:** 0 (Foundations) — design only.

## Done
- Three-jobs model, agent capability model, and audit approach drafted and reviewed ([ARCHITECTURE.md](./ARCHITECTURE.md)).
- Prompt-injection-as-capability-containment argument documented.
- Threat model drafted and reviewed.

## Next
- Define standard capability shapes for common agent powers.
- Design the tamper-evident audit log (and its TCB footprint).
- Phase 3: local inference + a capability-scoped agent with a working audit log.

## Blocked on
- WASM runtime + capability brokering ([`lantern-runtime`](https://github.com/lantern-os/lantern-runtime), [`lantern-capabilities`](https://github.com/lantern-os/lantern-capabilities)).
- Accelerator access via HAL/IOMMU ([`lantern-hal`](https://github.com/lantern-os/lantern-hal)).
