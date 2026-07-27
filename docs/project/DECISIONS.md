# DECISIONS

Provenance ledger: **who recommended what, when, and who decided** — distinct from the technical
rationale, which lives in the linked source. This file does not re-explain the *why*; it records the
*who/when/status* so a reader can trace any decision back to its origin and its owner.

Actors: **Denisa** (project owner, decides) · **Claude** (Claude Code) · **Codex** · **ChatGPT-doc**
(the environment-setup record). "Recommended by" ≠ "Decided by" ≠ implementer, on purpose.

## Protocol & experiment decisions

Detail and rationale: [../experiments/README.md](../experiments/README.md) §7 and the arm documents.

| ID | Decision | Recommended by | Decided by | Date | Status |
|----|----------|----------------|------------|------|--------|
| D1 | Three config subdirectories (`foundation`/`generation`/`language`), not two | Denisa (asked for `foundation/`); Claude flagged it belongs to both arms | Denisa | 2026-07-27 | Accepted |
| D2 | Robustness perturbations evaluate against **every** arm, not just the encoder | Claude | Denisa | 2026-07-27 | Accepted |
| D3 | Generative readout (L5) **deferred** beyond v1 | Claude | Denisa | 2026-07-27 | Deferred |
| D4 | Text-summary feature set: drop `rms` (redundant), add `waveform_length` + `zero_crossings` (temporal); per-channel, `%.2f`, versioned template | Claude | Denisa | 2026-07-27 | Accepted |
| D5 | Adaptation baseline **in v1**, scoped to head-only fine-tuning on a shared `calibration_strategy` axis | Claude | Denisa | 2026-07-27 | Accepted |
| — | Discriminative readout held identical across L2/L3/L4 (readout ≠ input path) | Claude | Denisa | 2026-07-27 | Accepted (spec §4) |

## Project & workflow decisions

| Decision | Recommended by | Decided by | Date | Status |
|----------|----------------|------------|------|--------|
| Keep `planning-documents/` as a separate **sibling git repo, no remote** (not nested + gitignored) | Claude | Denisa | 2026-07-27 | Accepted |
| Both agents rooted on `…/temporallens`; never the parent or a ChatGPT mirror | Codex; Claude | Denisa | 2026-07-27 | In progress |
| Concurrent agents use **git worktrees / separate branches**, never one shared working copy | Codex; Claude | Denisa | 2026-07-27 | Adopted (AGENTS.md) |
| Governance kept **lean (Tier-1)**: AGENTS/CLAUDE/STATUS/DECISIONS now; GitHub Issues/Project deferred to Tier-2 | Claude (right-sizing Codex's fuller proposal) | Denisa | 2026-07-27 | Adopted |
| Decision record kept **in-repo** (this file + experiments spec), **not** a parallel `docs/adr/` system | Claude (vs Codex's ADR proposal) | Denisa | 2026-07-27 | Adopted |
| Checkpoint contract: every checkpoint stores `{model_state, model_config}` | Claude | Denisa | 2026-07-27 | Accepted (not yet implemented) |

## Pending — proposed, not yet decided

| Proposal | Proposed by | Date | Notes |
|----------|-------------|------|-------|
| GitHub Issues + Project, issue/PR templates, milestones | Codex | 2026-07-27 | Tier-2; revisit when a second contributor is active. Needs `gh` or the GitHub connector. |
| `CHANGELOG.md` with an Unreleased section | Codex | 2026-07-27 | Low priority; add at first release. |
| Whether the eventual GitHub Project is public or private | Codex | 2026-07-27 | Decide before any sync. |
