# TASK-010A-FIX1 corrected gradient-aware MTL comparator evidence

## Scope and correction

This is a manager-authorized, DEV-only post-closure extension. It does not
revise Stage-3, Stage-6, Stage-7, paper, Final-Test, or frozen method roles.
The original TASK-010A comparison is superseded because it inferred parameter
ownership from autograd `None`/non-`None` behavior and therefore reported all
51 trainable tensors as shared. FIX1 uses explicit structural R4 ownership:
14 fully shared tensors, 18 Depression-only tensors, 18 Parkinson-only
tensors, and one row-partitioned `task_queries` tensor.

The corrected loss/callback implementation and all twelve FIX1 configs were
frozen and the corrective firewall was pushed at commit
`858efb026bbfb8c5a49469218380ea28fe2bbf5a` before production. Scientific
settings, model, optimizer, warm-up, semantic pseudo cache, reliability, and
the fixed R4 composition were unchanged. The semantic cache SHA256 was
`17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`, with
accepted D/P counts `376/1801` and class counts D `376/0`, P `212/1589`.

Exactly twelve FIX1 production invocations completed in this order:

```text
01_pcgrad_seed42, 02_pcgrad_seed43, 03_pcgrad_seed44,
04_cagrad_seed42, 05_cagrad_seed43, 06_cagrad_seed44,
07_gradnorm_seed42, 08_gradnorm_seed43, 09_gradnorm_seed44,
10_dbmtl_seed42, 11_dbmtl_seed43, 12_dbmtl_seed44
```

Checkpoint selection was exclusively the maximum DEV `dev/mean_score`. The
machine-readable selected-checkpoint ledger is
[task010a_mtl_runs.csv](task010a_mtl_runs.csv). The DEV-only data module
returned only `dev`; no Test loader was iterated and no Test metric or
Final-Test result was inspected.

## Selected DEV evidence

| Method | Seed | Epoch | D UAR/MF1/Score | P UAR/MF1/Score | Mean | Checkpoint SHA256 |
|---|---:|---:|---|---|---:|---|
| PCGrad | 42 | 16 | 0.699113/0.698860/0.698987 | 0.835542/0.856957/0.846249 | 0.772618 | `7d29ff30e0fe3a325b8d06d5d15ce0dfea15a37d93c8b99b4b5fa30b38358486` |
| PCGrad | 43 | 7 | 0.704248/0.703233/0.703741 | 0.832919/0.846219/0.839569 | 0.771655 | `f16d6c9b54c893bc4f69ddc1ae148943a21ab01b907dd097614042974c0dab26` |
| PCGrad | 44 | 7 | 0.691970/0.691891/0.691931 | 0.809317/0.831574/0.820445 | 0.756188 | `63d3fb5ed8ab77f241510bba23e141a13318a6ca0e37769d7787505398e6f396` |
| CAGrad | 42 | 3 | 0.706116/0.705641/0.705878 | 0.806625/0.819444/0.813035 | 0.759457 | `27e5d2d402f6d61830d8762f37c96631b483be561ad373716873fcb35831fa3e` |
| CAGrad | 43 | 8 | 0.707796/0.707439/0.707618 | 0.816218/0.830236/0.823227 | 0.765422 | `a8879586ef88c6ca0d9cb5c4d97df1b54abecd409a919b6d645b48515263de07` |
| CAGrad | 44 | 7 | 0.694911/0.694508/0.694710 | 0.823602/0.844861/0.834232 | 0.764471 | `6fa694b2f8d492f0461a0328b15da765d48c6b0efdc39dc50b60d97d47b89d28` |
| GradNorm | 42 | 10 | 0.695331/0.695336/0.695334 | 0.864044/0.879630/0.871837 | 0.783585 | `a0439abf2d915961bd0ce5c0b49d32fbc83bc53c4aa79776c90905429e97dc54` |
| GradNorm | 43 | 7 | 0.701167/0.700450/0.700808 | 0.832919/0.846219/0.839569 | 0.770189 | `23ccc998630935db3ba7a00aceb520214d94a61ce271ad30c8b673203322a257` |
| GradNorm | 44 | 15 | 0.703315/0.703297/0.703306 | 0.825673/0.836353/0.831013 | 0.767159 | `5a50a0af3dcf7c53ff7364ca2bf77c25e54498464bfe0d946a018564ccf7e875` |
| DB-MTL | 42 | 4 | 0.719841/0.717404/0.718623 | 0.818150/0.817404/0.817777 | 0.768200 | `0c24c9fdfdac09070445fdd4f8dafbf63966478dd3210c4cc07b98fe696745c1` |
| DB-MTL | 43 | 7 | 0.714146/0.713666/0.713906 | 0.772740/0.773379/0.773059 | 0.743483 | `01a4398d48d5d0b2dea982cb687ee652489b646e186b89befc3e820ea462a8aa` |
| DB-MTL | 44 | 6 | 0.727918/0.725678/0.726798 | 0.784886/0.790833/0.787860 | 0.757329 | `ad6af68e768c60e21cb6b0032b86ed7e8191abaa93b58f44a161a2c5c70861b6` |

Three-seed DEV summaries are arithmetic mean / sample standard deviation /
range:

| Method | D Score | P Score | Mean Score |
|---|---|---|---|
| Equal | 0.707925/0.004600/0.009199 | 0.850421/0.006475/0.012701 | 0.779173/0.001092/0.002007 |
| Static-STCH | 0.708698/0.014271/0.026132 | 0.852097/0.002273/0.004137 | 0.780397/0.006028/0.011217 |
| Progress | 0.708271/0.011546/0.020352 | 0.861231/0.013176/0.026146 | 0.784751/0.009157/0.017774 |
| RA-STCH | 0.693150/0.011767/0.021372 | 0.851964/0.021825/0.038064 | 0.772557/0.016775/0.029452 |
| PCGrad FIX1 | 0.698220/0.005942/0.011810 | 0.835421/0.013393/0.025804 | 0.766820/0.009220/0.016430 |
| CAGrad FIX1 | 0.702735/0.007004/0.012908 | 0.823498/0.010601/0.021197 | 0.763117/0.003205/0.005965 |
| GradNorm FIX1 | 0.699816/0.004078/0.007972 | 0.847473/0.021529/0.040824 | 0.773644/0.008741/0.016426 |
| DB-MTL FIX1 | 0.719776/0.006523/0.012892 | 0.792899/0.022781/0.044718 | 0.756337/0.012388/0.024717 |

The frozen full R4 contextual three-seed reference remains
D/P/Mean `0.7352563333/0.8385606667/0.7869088179`. These are descriptive
DEV comparisons only.

## Paired DEV deltas

Each tuple is (D Score, P Score, Mean Score) for seeds 42/43/44; the final
tuple is the three-seed mean delta. wins counts positive Mean deltas across
the three paired seeds. The three explicit regression columns flag each
per-seed D, P, or Mean delta below -0.010.

| Method | Reference | Seed42 | Seed43 | Seed44 | Mean delta | wins | D regressions >0.010 | P regressions >0.010 | Mean regressions >0.010 |
|---|---|---|---|---|---|---:|---|---|---|
| PCGrad | Equal | (-.013511,+.002907,-.005302) | (+.000442,-.016474,-.008016) | (-.016046,-.031433,-.023739) | (-.009705,-.015000,-.012352) | 0/3 | 42,44 | 43,44 | 44 |
| PCGrad | Progress | (-.002258,-.027105,-.014681) | (-.017856,-.023562,-.020709) | (-.010040,-.026763,-.018402) | (-.010051,-.025810,-.017931) | 0/3 | 43,44 | 42,43,44 | 42,43,44 |
| PCGrad | RA-STCH | (-.002005,-.018049,-.010027) | (+.004902,-.025260,-.010179) | (+.012311,-.006320,+.002995) | (+.005069,-.016543,-.005737) | 1/3 | none | 42,43 | 42,43 |
| CAGrad | Equal | (-.006620,-.030307,-.018463) | (+.004319,-.032816,-.014249) | (-.013267,-.017646,-.015456) | (-.005189,-.026923,-.016056) | 0/3 | 44 | 42,43,44 | 42,43,44 |
| CAGrad | Progress | (+.004633,-.060319,-.027842) | (-.013979,-.039904,-.026942) | (-.007261,-.012976,-.010119) | (-.005536,-.037733,-.021634) | 0/3 | 43 | 42,43,44 | 42,43,44 |
| CAGrad | RA-STCH | (+.004886,-.051263,-.023188) | (+.008779,-.041602,-.016412) | (+.015090,+.007467,+.011278) | (+.009585,-.028466,-.009441) | 1/3 | none | 42,43 | 42,43 |
| GradNorm | Equal | (-.017164,+.028495,+.005665) | (-.002491,-.016474,-.009482) | (-.004671,-.020865,-.012768) | (-.008109,-.002948,-.005528) | 1/3 | 42 | 43,44 | 44 |
| GradNorm | Progress | (-.005911,-.001517,-.003714) | (-.020789,-.023562,-.022175) | (+.001335,-.016195,-.007431) | (-.008455,-.013758,-.011107) | 0/3 | 43 | 43,44 | 43 |
| GradNorm | RA-STCH | (-.005658,+.007539,+.000940) | (+.001969,-.025260,-.011645) | (+.023686,+.004248,+.013966) | (+.006666,-.004491,+.001087) | 2/3 | none | 43 | 43 |
| DB-MTL | Equal | (+.006125,-.025565,-.009720) | (+.010607,-.082984,-.036188) | (+.018821,-.064018,-.022598) | (+.011851,-.057522,-.022835) | 0/3 | none | 42,43,44 | 43,44 |
| DB-MTL | Progress | (+.017378,-.055577,-.019099) | (-.007691,-.090072,-.048881) | (+.024827,-.059348,-.017261) | (+.011505,-.068332,-.028414) | 0/3 | none | 42,43,44 | 42,43,44 |
| DB-MTL | RA-STCH | (+.017631,-.046521,-.014445) | (+.015067,-.091770,-.038351) | (+.047178,-.038905,+.004136) | (+.026625,-.059065,-.016220) | 1/3 | none | 42,43,44 | 42,43 |

No paired delta is a significance test. No method is promoted, demoted, or
selected from this descriptive extension.

## Corrected mechanism diagnostics

All selected rows report finite diagnostics and exact structural counts
`shared=14`, `depression_only=18`, `parkinson_only=18`,
`row_partitioned=1`. Selected diagnostics, by seed 42/43/44, were:

- PCGrad: conflict fraction `0.454545/0.454545/0.484848`; pre-cosine
  `0.008775/0.014798/0.006349`; post-cosine
  `0.047676/0.113929/0.065431`; pre shared norms D/P
  `1.763565/1.488407`, `4.631876/2.640287`, `3.969351/2.289870`.
- CAGrad: conflict fraction `0.525253/0.479798/0.439394`; pre-cosine
  `0.000475/0.008052/0.010065`; simplex D/P weights
  `0.328580/0.671420`, `0.388126/0.611874`, `0.275474/0.724526`; solver
  converged on all three seeds. Adjustment norms were
  `0.890868/1.184406/0.908473`.
- GradNorm: final weights D/P
  `1.021491/0.978509`, `0.847924/1.152076`, `1.274562/0.725438`; selected
  norm pairs D/P `2.851972/2.413785`, `4.713192/2.672934`,
  `2.528154/1.704965`; all finite.
- DB-MTL: conflict fraction `0.313131/0.373737/0.348485`; pre/post cosine
  `0.041469/0.093908`, `0.037238/0.107022`, `0.030250/0.084364`; EMA norm
  pairs D/P `3.532724/2.121199`, `1.192614/0.974719`,
  `1.631090/1.068212`. Normalized D/P norms were equal within each selected
  row (`3.647751`, `1.270405`, `1.673461`).

The corrected implementation demonstrates distinct operations on the actual
14-tensor shared trunk while task-local and row-partitioned tensors remain
structurally isolated. The diagnostics do not establish that conflict
handling reduced disease-task conflict relative to an unmodified shared
trunk, nor that any reduction caused a better DEV Mean.

## Descriptive synthesis and boundaries

### Method-family synthesis

Scalar/loss balancing comprises Equal, Static-STCH, Progress, RA-STCH, and
GradNorm. Explicit shared-gradient handling comprises PCGrad, CAGrad, and
DB-MTL. GradNorm is therefore classified with scalar/loss balancing here;
it is not described as an explicit gradient-conflict method.

Among scalar/loss-balancing methods, Progress had the highest three-seed DEV
Mean point estimate (0.7847510000), while GradNorm reached 0.7736443333.
Among explicit shared-gradient handling methods, PCGrad was highest
(0.7668203333), followed by CAGrad (0.7631166667) and DB-MTL
(0.7563373333). The contextual full R4 mean is 0.7869088179.

Under the fixed R4 composition, scalar/loss-balancing methods were
descriptively stronger than the tested explicit shared-gradient handling
methods. Progress had the highest three-seed DEV Mean among the balancing
methods, while PCGrad was the strongest explicit gradient-handling method but
remained below Equal and Progress.

The eight-method main comparison is complete for Score/Mean; historical
UAR/MF1 aggregates are omitted where the committed evidence does not fully
ground every method/seed. These are descriptive research observations, not
method-promotion decisions.

This extension does not claim causality, statistical significance, pseudo-label
correctness, missing-label recovery, comorbidity recovery, corpus-shortcut
causality, or final-method superiority. Unknown labels remain masked and
observed truth overrides pseudo supervision. Stage 5 remains closed, the
Stage-7/Final-Test results remain frozen, and this post-closure comparator
does not authorize another experiment or revise the paper.

## Reproducibility and audit boundary

- Firewall commit before production: `858efb026bbfb8c5a49469218380ea28fe2bbf5a`.
- FIX1 source/config settings stayed frozen after the firewall; no post-firewall
  source/config/script/dependency edit occurred.
- The corrected test suite was `13 passed`; all twelve configs were validated
  before the firewall; the TRAIN-only firewall reported model parameters
  `403079`, cache SHA above, accepted counts above, detached pseudo tensors,
  finite diagnostics, and zero optimizer steps.
- No Test loader iteration, Test metric inspection, Final-Test invocation,
  threshold search, sweep, metric-driven retry, or extra seed occurred.
- `src/audio` stayed unchanged. The superseded invalid comparison remains at
  [TASK010A_MTL_COMPARISON_SUPERSEDED_EN.md](TASK010A_MTL_COMPARISON_SUPERSEDED_EN.md);
  it is procedural history only.

Corrected FIX1 evidence is complete as a descriptive DEV-only extension. It
does not reopen the closed research programme or authorize a follow-up task.
