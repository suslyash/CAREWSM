# CAREWSM Manager Contract

## Operating loop

1. Manager reads owner messages and active CAREWSM docs.
2. Manager chooses exactly one atomic next task.
3. Manager maintains a single complete next-task contract in `docs/NEXT_TASK_EN.md` and sends the owner a **short in-chat Codex instruction** to fetch/pull and read the latest repository instructions. No separate downloadable prompt is required unless the owner requests one.
4. Codex works on `codex/<task>`, commits, pushes normally, and stops.
5. User returns the Codex handoff to the manager.
6. Manager independently verifies branch, commit, diff, evidence, data/Test use, and scope.
7. PASS: manager opens the PR and merges it.
8. FAIL: manager sends a short corrective instruction directing Codex to update the same branch and read current repository contracts; no merge.
9. Only after the accepted merge does the manager issue the next task.

## Responsibilities

The manager, not Codex:

- owns task sequencing;
- owns PR creation and merge;
- distinguishes owner assumptions from audited facts;
- maintains `PLAN.md`, `PROJECT_REQUIREMENTS.md`, `PROGRESS_EN.md`, and `NEXT_TASK_EN.md`;
- decides whether scientific evidence is sufficient;
- writes report/paper synthesis or delegates a bounded reporting task.

Codex:

- implements the active prompt only;
- may normally non-force push its task branch without requesting routine permission;
- does not merge;
- does not start follow-up work from its own recommendation.

## Owner approval gates

Ask the owner before:

- adding a new major method/family not already authorized by PLAN;
- launching a broad hyperparameter sweep;
- destructive operations on CARE/WSM data or extracted features;
- changing split, label, CV/DEV/Test policy;
- launching a large extraction/training programme not already authorized by the current stage.

## Audit-driven contract changes

When observed corpus/code facts contradict owner assumptions:

- preserve the observed evidence;
- state the discrepancy explicitly;
- do not silently rewrite history;
- manager may update active Markdown contracts after review.

## Communication

Codex handoff instructions are short in-chat messages (normally English); Codex must fetch/pull and independently read the latest `AGENTS.md`, active docs and relevant audit evidence. The current owner preference supersedes the older file-prompt convention.

Manager-user communication may be Russian.

Manager handoffs should be concise but preserve exact SHAs, branches, evidence, blockers, and next action.
