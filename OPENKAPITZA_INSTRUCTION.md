# OpenKapitza — agent instruction

**This file has moved to [`docs/agent-instruction-v2.md`](docs/agent-instruction-v2.md).**

That is the canonical copy. This stub exists so the old path still resolves.

## Why

For a while there were two copies of the instruction: this one at the repo
root and a newer one under `docs/`. They had already drifted — the root copy
was 344 lines and missing the gap-model limitation, hard rule 6 on digitised
data, the Y/σ domain note and the per-component validation-status table.

Two copies of a specification is one copy and one trap. Anything that reads
the stale one acts on superseded physics while believing it is current, and
nothing in the repo flags the disagreement. So there is now exactly one.

The canonical instruction is also mirrored into the Claude project knowledge
base at `claude/agent-instruction-v2.md`. If those two disagree, the file
under `docs/` is current.
