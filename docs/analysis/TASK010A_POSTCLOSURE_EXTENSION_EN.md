# POST-CLOSURE EXPERIMENTAL EXTENSION — DOES NOT REVISE FROZEN WSM EVIDENCE

## TASK-010A-FIX1 corrective status — complete

The original TASK-010A evidence is superseded due to an invalid gradient
ownership implementation. It classified task-specific zero gradients as
shared. FIX1 corrects this by binding the loss to structural named-parameter
ownership: 14 fully shared tensors, 18 Depression-only tensors, 18
Parkinson-only tensors, and one row-partitioned task_queries tensor.

The previous 12 selected results are retained only as superseded procedural
history and are not scientific evidence. The original TASK-010A evidence
commit da8abe5507ac3f2410af02762c723ef92e201bd0 is explicitly superseded and
invalid as scientific evidence. The corrected firewall commit was
858efb026bbfb8c5a49469218380ea28fe2bbf5a, and the corrected final evidence
commit is 4828412f62299abdab908596802f77f50d49a30c.

Exactly twelve FIX1 production runs completed after the corrected firewall,
with structural ownership 14 shared / 18 Depression-only / 18
Parkinson-only / 1 row-partitioned. The authoritative comparison and CSV
are complete. Selection was DEV-only by maximum dev/mean_score; no Test
loader, Test metric, or Final-Test result was inspected. No frozen method
role, paper conclusion, or promotion decision changed.

The complete post-closure evidence remains descriptive and DEV-only. It
records the corrected mechanism diagnostics, paired D/P/Mean deltas, and
the scalar/loss-balancing versus explicit shared-gradient method-family
comparison without significance, causal, superiority, or universality
claims. No further experiment is authorized by this extension.
