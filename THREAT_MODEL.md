# lantern-ai-runtime — Threat Model

Inherits the [system threat model](https://github.com/lantern-os/lantern-docs/blob/main/wiki/Threat-Model.md). Introduces a
genuinely new adversary: the **autonomous (and prompt-injectable) agent** (system threat T4).

## Assets
- The boundary of an agent's granted authority.
- Faithfulness and tamper-evidence of the audit log.
- Integrity/provenance of models and their outputs.
- User consent integrity.

## Threats and mitigations
| # | Threat | Mitigation |
| --- | --- | --- |
| A1 | Agent exceeds user intent (incl. prompt injection) | Hard capability scoping — the agent literally cannot act beyond its caps. |
| A2 | Data exfiltration by a compromised agent | No network capability ⇒ no exfiltration; network caps are explicit, scoped, revocable. |
| A3 | Tampered/unfaithful audit log | Tamper-evident log; provenance bound to the capability used. |
| A4 | Malicious model weights / supply chain | Content-addressed models with integrity + provenance. |
| A5 | Accelerator/driver reads arbitrary memory | IOMMU confinement behind the HAL. |
| A6 | Consent fatigue / click-through grants | Comprehensible standard capability shapes; meaningful consent UX in the shell. |
| A7 | Silent telemetry of user interactions | Local-first by default; no telemetry pipeline; cloud access is explicit + audited. |

## Non-goals
- Guaranteeing a model is "aligned" or won't *try* to misbehave — we make misbehaviour
  *ineffective* via capabilities, not impossible to attempt.
- Defending against the *user* intentionally granting broad authority (we make it visible).
