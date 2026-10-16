# POST-CLOSURE DEV-ONLY FULL ABLATION/MTL SUITE — NOT PART OF THE FROZEN STAGE-7/FINAL-TEST EVIDENCE

This is descriptive DEV-only evidence around TASK-011A Shared+Progress trial-28. It does not revise Stage-7, Final-Test, or paper roles.

## POST-PRODUCTION REPORTING-ONLY COLLECTION RECOVERY

Collection integrity: manifest rows 52/52; scientific production results 52/52; referenced MLflow run IDs 53; duplicate-alias rows 1.
The accepted resolver uses anchored run-name prefixes, FINISHED status, frozen config/artifact identity, and the lexicographically smallest ID for an equivalent alias set.
Row 34 aliases 44455703f7de44f6b032b4cde21ddd5d and b43e09ce179a49f199993628ad3c063d were equivalent FINISHED aliases for one frozen result; canonical ID is 44455703f7de44f6b032b4cde21ddd5d and alias ID is b43e09ce179a49f199993628ad3c063d.
Equivalence included config SHA 660cac8293e60e28648c4225edf78486dbb06087a9459d730bb49ada157d5be8, selected epoch 5, checkpoint SHA 4215e95623afe66cd42706f42f0f72e2662a679561cff844e89a7fde836e40f1, and identical DEV-only selected metrics D 0.746078/0.745256/0.745667, P 0.863906/0.874235/0.869071, Mean 0.807369.
Canonicalization is bookkeeping only and does not count the alias as another production run.

## 1. Five-seed tuned parent confirmation

Only allowlisted DEV metric keys were collected.
seed42 frozen tuned parent: D UAR/F1/Score unavailable/unavailable/0.749256; P UAR/F1/Score unavailable/unavailable/0.918143; Mean 0.833699; epoch 7; checkpoint SHA ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346.
seed43: D 0.710738/0.709874/0.710306; P 0.790131/0.809336/0.799734; Mean 0.755020; epoch 9; checkpoint SHA ba280153b2a1dca9808f25c05e581fec72ab3404a3602233154ef26c9f62b926.
seed44: D 0.750980/0.750329/0.750655; P 0.875845/0.885640/0.880743; Mean 0.815699; epoch 5; checkpoint SHA 775826bf903fced08e24eb9bfff267e414c1ea7d5e94201a466bc40642e93100.
seed45: D 0.730439/0.730270/0.730355; P 0.878054/0.880722/0.879388; Mean 0.804871; epoch 19; checkpoint SHA e4a6776902ca57885ceb8f1c5444736d16fb8257618181c027b2fc8ad996e3d9.
seed46: D 0.741223/0.740314/0.740769; P 0.854313/0.863506/0.858910; Mean 0.799839; epoch 3; checkpoint SHA 374efc91f7ecf153f85ce42bee6af92475a229795d1221e6c8dfa4c0bb267478.
Tuned parent five-seed Mean mean/std/min/max/range/95% t CI(df=4): 0.801826/0.029208/0.755020/0.833699/0.078680/[0.765559,0.838092].
Frozen Shared trial012 five-seed Mean mean/std/95% t CI(df=4) from committed Stage-7 DEV evidence: 0.790914/0.019857/[0.766258,0.815571].
Tuned-minus-frozen Shared Mean deltas by seed (42/43/44/45/46): +0.0092344328, -0.0163041676, +0.0307886787, +0.0176691360, +0.0131680720.
Paired Mean delta mean/sample SD/95% exploratory t CI(df=4)/wins: +0.0109112304/0.0172435866/[-0.0104995103,+0.0323219710]/4/5.
Parent task-score comparison versus frozen Shared: D Score mean delta -0.0014224191, wins 2/5; P Score mean delta +0.0232447067, wins 4/5.
This is descriptive post-closure DEV evidence, not a Final-Test superiority claim.

## 2. Core component ablations

Each row reports three-seed DEV task-level aggregates and paired Mean deltas against the same-seed tuned Shared+Progress parent.
no_direct_pseudo: D UAR/F1/Score mean 0.734189/0.732839/0.733514; P UAR/F1/Score mean 0.855924/0.864836/0.860380; Mean mean/sample SD 0.796947/0.033728; paired Mean deltas (42/43/44) -0.037435, +0.008545, +0.015312; wins 2/3.
uniform_reliability: D UAR/F1/Score mean 0.727622/0.726978/0.727300; P UAR/F1/Score mean 0.854474/0.868389/0.861432; Mean mean/sample SD 0.794366/0.034104; paired Mean deltas (42/43/44) -0.020514, -0.000021, -0.000786; wins 0/3.
no_semantic_depression: D UAR/F1/Score mean 0.739029/0.737702/0.738365; P UAR/F1/Score mean 0.840051/0.849939/0.844995; Mean mean/sample SD 0.791680/0.035551; paired Mean deltas (42/43/44) -0.037435, -0.000961, +0.009018; wins 1/3.
no_video_online: D UAR/F1/Score mean 0.719810/0.719188/0.719499; P UAR/F1/Score mean 0.783000/0.802811/0.792905; Mean mean/sample SD 0.756202/0.008773; paired Mean deltas (42/43/44) -0.086421, +0.001493, -0.050884; wins 1/3.
no_audio_online: D UAR/F1/Score mean 0.694460/0.694418/0.694439; P UAR/F1/Score mean 0.790798/0.805915/0.798357; Mean mean/sample SD 0.746398/0.011929; paired Mean deltas (42/43/44) -0.079864, -0.002300, -0.083060; wins 0/3.

## 3. Architecture and shuffled-pseudo controls

task_aware_r4_progress: D UAR/F1/Score mean 0.730984/0.729788/0.730386; P UAR/F1/Score mean 0.840902/0.853897/0.847399; Mean mean/sample SD 0.788893/0.027406; paired Mean deltas (42/43/44) -0.037036, +0.003420, -0.004125; wins 1/3.
shuffled_pseudo: D UAR/F1/Score mean 0.732477/0.730692/0.731585; P UAR/F1/Score mean 0.841684/0.853196/0.847440; Mean mean/sample SD 0.789512/0.032606; paired Mean deltas (42/43/44) -0.037435, -0.000961, +0.002514; wins 1/3.

## 4. Single-task vs joint-MTL controls

Single-task controls report only their selected task metrics; joint Mean is not reported or used for these comparisons.
depression_only: selected D UAR/F1/Score by seed (42/43/44): 0.721662/0.719959/0.720811, 0.720121/0.718612/0.719367, 0.734174/0.731339/0.732756; D Score mean/sample SD 0.724311/0.007349; paired D-only D Score minus joint-parent D Score deltas -0.028445, +0.009061, -0.017898; wins 1/3; joint Mean —.
parkinson_only: selected P UAR/F1/Score by seed (42/43/44): 0.892616/0.903827/0.898221, 0.828295/0.846811/0.837553, 0.911594/0.916456/0.914025; P Score mean/sample SD 0.883267/0.040370; paired P-only P Score minus joint-parent P Score deltas -0.019922, +0.037820, +0.033282; wins 2/3; joint Mean —.

## 5. Scalarization / balancing substitutions

equal: D UAR/F1/Score mean 0.733287/0.732355/0.732821; P UAR/F1/Score mean 0.834599/0.849162/0.841880; Mean mean/sample SD 0.787350/0.034362; paired Mean deltas (42/43/44) -0.026690, -0.007347, -0.008330; wins 0/3.
static_stch: D UAR/F1/Score mean 0.726813/0.725791/0.726302; P UAR/F1/Score mean 0.835358/0.849405/0.842381; Mean mean/sample SD 0.784342/0.025601; paired Mean deltas (42/43/44) -0.036527, -0.000157, -0.014709; wins 0/3.
ra_stch: D UAR/F1/Score mean 0.725226/0.724241/0.724733; P UAR/F1/Score mean 0.852887/0.866568/0.859727; Mean mean/sample SD 0.792230/0.034402; paired Mean deltas (42/43/44) -0.028081, -0.001873, +0.002227; wins 1/3.

## 6. Gradient-MTL substitutions

Common hyperparameters were tuned under Progress. These are frozen-recipe substitution audits, not equal-budget globally tuned rankings of balancing/gradient-MTL methods.

gradnorm: D UAR/F1/Score mean 0.728462/0.727555/0.728009; P UAR/F1/Score mean 0.835404/0.850054/0.842729; Mean mean/sample SD 0.785369/0.036314; paired Mean deltas (42/43/44) -0.023585, -0.011340, -0.013387; wins 0/3.
pcgrad: D UAR/F1/Score mean 0.727358/0.726209/0.726784; P UAR/F1/Score mean 0.833034/0.848663/0.840849; Mean mean/sample SD 0.783816/0.034048; paired Mean deltas (42/43/44) -0.029554, -0.010512, -0.012904; wins 0/3.
cagrad: D UAR/F1/Score mean 0.734143/0.732665/0.733404; P UAR/F1/Score mean 0.805820/0.816056/0.810938; Mean mean/sample SD 0.772171/0.031727; paired Mean deltas (42/43/44) -0.053194, -0.017912, -0.016800; wins 0/3.
dbmtl: D UAR/F1/Score mean 0.729505/0.724856/0.727180; P UAR/F1/Score mean 0.857350/0.860307/0.858828; Mean mean/sample SD 0.793004/0.011199; paired Mean deltas (42/43/44) -0.045936, +0.030367, -0.009835; wins 1/3.

## 7. Scope / limitations / Test quarantine

- All reported metrics and comparisons are DEV-only; checkpoint selection remains by each run's configured DEV selector.
- No Test loader, Test metric, mixed raw summary, or Final-Test artifact was read by this collection or synthesis.
- D-only/P-only are not deployable two-task models; their joint Mean is deliberately absent and no fake Mean comparison is made.
- Common Progress-tuned hyperparameters make Groups F/G frozen-recipe audits, not globally tuned rankings.
- Online modality removals retain historical offline pseudo evidence.
- All three-seed statistics and any paired t intervals are descriptive/exploratory; no multiplicity-corrected significance or universal usefulness/uselessness claim is authorized.
- No final-method promotion, demotion, or paper-role revision is authorized.
