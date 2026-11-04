# CAREWSM Manager Contract

The manager owns sequencing and scientific control. Codex owns execution on the machine with the actual data.

## Loop

1. Manager reads owner instruction and current repository state.
2. Manager selects exactly one atomic task.
3. Manager writes a complete Codex prompt.
4. Codex executes on `codex/<task>`, commits, pushes and stops.
5. Owner returns the Codex handoff.
6. Manager independently inspects GitHub branch, commit, diff, tests and evidence.
7. PASS -> manager opens/merges PR and updates the plan.
8. FAIL -> manager issues a corrective task; no merge.
9. Only then is the next task authorized.

## Manager responsibilities

- methodology and task sequencing;
- feature experiment design;
- three-model research programme;
- Mantis temporalization contract;
- multitask loss study;
- cross-corpus protocol;
- TEST discipline;
- independent review;
- PR/merge;
- active documentation.

## Codex responsibilities

- inspect real CARE/WSM/external data when authorized;
- implement only the current task;
- run tests/experiments;
- produce reproducible evidence;
- push normally;
- never merge;
- never start the next task autonomously.

## Important separation

The manager does not claim local feature validity without Codex evidence.

Codex does not change scientific policy merely because an implementation is convenient.

## Durable manager gates

The manager authorizes and reviews exactly one atomic task at a time. The active refactor sequence is REFACTOR-000 through REFACTOR-005, followed by a final integration review.

`docs/NEXT_TASK_EN.md` identifies the currently proposed task, but Codex still requires explicit manager authorization before implementation. REFACTOR-001 is not authorized until REFACTOR-000 passes independent review and is merged. TASK-011 remains paused until the approved refactor sequence is completed and reviewed.

TASK-001 is historical; its feature-audit evidence remains preserved.
