# CAREWSM Manager Contract

The manager owns the research loop.

## Loop

1. Read owner instruction + active docs.
2. Select exactly one atomic task.
3. Write a complete Codex prompt.
4. Codex implements on `codex/<task>`, commits, pushes and stops.
5. Owner returns Codex handoff.
6. Manager independently inspects branch, commit, diff, tests/evidence and scope.
7. PASS -> manager opens and merges PR.
8. FAIL -> manager issues a corrective prompt; no merge.
9. Only after accepted merge is the next task authorized.

## Manager responsibilities

- project sequencing;
- experiment design;
- feature-combination strategy;
- choice of models;
- evaluation discipline;
- independent GitHub review;
- PR/merge;
- keeping requirements/plan/progress/next-task current;
- preventing TEST-driven iteration;
- stopping combinatorial experiments that do not answer a useful question.

## Codex responsibilities

- execute only the active prompt;
- inspect local CARE/WSM data where required;
- produce code/tests/evidence;
- push task branch normally;
- never merge;
- never start the next experiment from its own recommendation.

## Research principle

Fast does not mean unverifiable.

Feature audit should be optimized for one-pass metadata/schema checks and minimal necessary loading, but must not mark a feature valid based on filename existence alone.

Large experiment campaigns should use staged screening and promotion so compute is spent on informative candidates.
