# lantern-ai-runtime — Status

**Phase:** 3 (Privacy, identity, networking, and AI) — opened per [RFC-0017](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0017-phase-2-to-phase-3-transition.md)/[ADR-0021](https://github.com/lantern-os/lantern-rfcs/blob/main/adr/0021-phase-2-complete-phase-3-opened.md). Still design only — see "Blocked on".

## Done
- Three-jobs model, agent capability model, and audit approach drafted and reviewed ([ARCHITECTURE.md](./ARCHITECTURE.md)).
- Prompt-injection-as-capability-containment argument documented.
- Threat model drafted and reviewed.

## Next
- Define standard capability shapes for common agent powers.
- Design the tamper-evident audit log (and its TCB footprint).
- Phase 3: local inference + a capability-scoped agent with a working audit log.

## Blocked on
- **The confined-execution port carried forward from Phase 2 (ADR-0021).** A "capability-scoped agent with a faithful audit log" cannot be faithful while the runtime and the services it calls share an address space. The WIT-handle mapping (RFC-0014/RFC-0016) and `lantern-sdk`'s manifest/plan pipeline (RFC-0015) an agent's capability set is built on already exist. The services and `lantern-runtime` must run as separate confined processes on the kernel, not host-target stand-ins. That port is Phase 3's first work; this crate follows it.
- WASM runtime + capability brokering ([`lantern-runtime`](https://github.com/lantern-os/lantern-runtime), [`lantern-capabilities`](https://github.com/lantern-os/lantern-capabilities)).
- Accelerator access via HAL/IOMMU ([`lantern-hal`](https://github.com/lantern-os/lantern-hal)).
