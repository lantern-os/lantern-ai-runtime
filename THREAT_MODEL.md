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

## As built (RFC-0034)
- **A4, the model:** the service accepts only the file whose BLAKE3 digest it was built with, and checks it before parsing;
  the parser still checks every length. The model is identified, not signed: a new model means a new system image.
  `scripts/get-model.sh` pins the upstream checkpoint and tokenizer by commit and SHA-256.
- **The model is untrusted input to the agent, and the agent is untrusted to the system.** Prompt injection changes the
  words an agent writes, never what it can reach: `console-matrix.sh` gives the file assistant text that tells it to
  write elsewhere, and its grants and uses are the same as for any other text.
- **The service sees every prompt.** It holds no store, network or console capability, so it can only answer the caller.
  A compromised service could answer with any text.
- **Resource use:** one call at a time, at most 128 tokens. A caller can hold the service for that long, and a
  `REVOKE` waits for the call in progress. No deadline cuts a call short yet.
- **The audit log** records each call before it runs: which grant, which model, how many tokens asked for, how long a
  prompt. Not the prompt itself, and not how many tokens came back.

## Non-goals
- Guaranteeing a model is "aligned" or won't *try* to misbehave — we make misbehaviour
  *ineffective* via capabilities, not impossible to attempt.
- Defending against the *user* intentionally granting broad authority (we make it visible).
