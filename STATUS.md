# lantern-ai-runtime — Status

**Phase:** 3 (Privacy, identity, networking, and AI) — opened per [RFC-0017](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0017-phase-2-to-phase-3-transition.md)/[ADR-0021](https://github.com/lantern-os/lantern-rfcs/blob/main/adr/0021-phase-2-complete-phase-3-opened.md). Still design only — see "Blocked on".

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

## Next
- Define standard capability shapes for common agent powers.
- Measure the cost per operation on real flash (Phase 4), and export retired history before it's erased.
- A person-facing view of the log, with names rather than badges and hashes, as part of the consent surface.
- Phase 3: local inference + a capability-scoped agent with a working audit log.

## Blocked on
- **The confined-execution port carried forward from Phase 2 (ADR-0021).** A "capability-scoped agent with a faithful audit log" cannot be faithful while the runtime and the services it calls share an address space. The WIT-handle mapping (RFC-0014/RFC-0016) and `lantern-sdk`'s manifest/plan pipeline (RFC-0015) an agent's capability set is built on already exist. The services and `lantern-runtime` must run as separate confined processes on the kernel, not host-target stand-ins. That port is Phase 3's first work; this crate follows it.
- WASM runtime + capability brokering ([`lantern-runtime`](https://github.com/lantern-os/lantern-runtime), [`lantern-capabilities`](https://github.com/lantern-os/lantern-capabilities)).
- Accelerator access via HAL/IOMMU ([`lantern-hal`](https://github.com/lantern-os/lantern-hal)).
