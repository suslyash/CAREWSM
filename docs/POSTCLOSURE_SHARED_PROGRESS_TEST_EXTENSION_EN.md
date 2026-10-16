# POST-CLOSURE COMPARATIVE TEST EXTENSION — NOT A PRISTINE FIRST-USE FINAL TEST

TASK-012A evaluates the frozen Shared + Progress parent on reconstructed DEV and on the three canonical Test protocols. The candidate recipe and checkpoints were frozen from DEV before this extension; no Test-based tuning or follow-up run occurred. The Test set had been exposed historically in the project, so this is not a pristine held-out evaluation. Frozen Stage-7 evidence remains unchanged.

## Frozen ledger and invariants

The five entries below were resolved from the committed TASK-011A/TASK-011B identity evidence. Selected epochs are 7/9/5/19/3 and trainable parameters are 552775 for every entry. The parent model is `wsm_av_r3_disease_query_model`, `hidden_dim=256`, `gate_hidden_dim=160`, `task_aware_fusion=false`, with Progress balancing. The canonical pseudo-cache SHA is `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`, accepted counts D/P `376/1801`.

| seed | MLflow run ID | epoch | config SHA256 | checkpoint SHA256 |
|---:|---|---:|---|---|
| 42 | `90a5adebf43c46808814f29374e092d0` | 7 | `a5bcb56f48b223f6156281183b72c8b53200cff320b3895c4cb3247fd86878d6` | `ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346` |
| 43 | `009b4d3c9be44220b6d3169989c518e4` | 9 | `dda16afd9478f935107a00ddf901956b3318eb37d8385ce169e2b2e2dbb25a52` | `ba280153b2a1dca9808f25c05e581fec72ab3404a3602233154ef26c9f62b926` |
| 44 | `28b2b383cc244c8b98011c60c73ac14e` | 5 | `44c7cad4879a3635c7eff6d9a1f9a1a5cbbdbaf123ba4d19f3f9a3f2effe775b` | `775826bf903fced08e24eb9bfff267e414c1ea7d5e94201a466bc40642e93100` |
| 45 | `ff2b0e3033ef4688ab46841585e49bcf` | 19 | `f411e86536681dc69903d01d23961714e9045df9d92e431281d688c8f0145a88` | `e4a6776902ca57885ceb8f1c5444736d16fb8257618181c027b2fc8ad996e3d9` |
| 46 | `aa76a8669b09431ba9c6fbe3f4f98b0c` | 3 | `bb3db27f3ce216403cb5390ba235a8f8a0fa8f410bb6424891b5187cdcf56c60` | `374efc91f7ecf153f85ce42bee6af92475a229795d1221e6c8dfa4c0bb267478` |

The resolved snapshot identities are the five run directories named by the corresponding run names; seed42 snapshot `code.zip` SHA256 is `b2b92c58f6209b1df19aa7ef0de1249f7e987c23248b278e7adc9a67726568f2`, and seeds43–46 share `52de31a38c779b052c8ef06163727c7cf6d1cb2186dd7ced39f16d8ea51a93de`.

## DEV reconstruction

Raw logits were accumulated as CPU float32 and classified at logit `>= 0`. Unknown labels remained masked and observed truth was used unchanged.

| seed | D UAR | D macro-F1 | D Score | P UAR | P macro-F1 | P Score | Mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.751447245565 | 0.747064351335 | 0.749255798450 | 0.918633540373 | 0.917652594012 | 0.918143067192 | 0.833699432821 |
| 43 | 0.710737628385 | 0.709874432650 | 0.710306030517 | 0.790131124914 | 0.809336143572 | 0.799733634243 | 0.755019832380 |
| 44 | 0.750980392157 | 0.750328947368 | 0.750654669763 | 0.875845410628 | 0.885639964529 | 0.880742687578 | 0.815698678670 |
| 45 | 0.730438842204 | 0.730270361653 | 0.730354601928 | 0.878053830228 | 0.880721509749 | 0.879387669988 | 0.804871135958 |
| 46 | 0.741223155929 | 0.740314048550 | 0.740768602240 | 0.854313319531 | 0.863505764115 | 0.858909541823 | 0.799839072031 |

DEV gate: all D Score, P Score, and Mean differences from the accepted frozen values were within `0.0005`; maximum absolute difference was `2.02e-7`. Five-seed aggregate (mean / sample SD / min / max / range / Student-t 95% CI, df=4): D UAR `0.736965452848 / 0.016990911558 / 0.710737628385 / 0.751447245565 / 0.040709617180 / [0.715868449670,0.758062456026]`; D F1 `0.735570428311 / 0.016284969940 / 0.709874432650 / 0.750328947368 / 0.040454514718 / [0.715349967425,0.755790889198]`; D Score `0.736267940580 / 0.016620068683 / 0.710306030517 / 0.750654669763 / 0.040348639245 / [0.715631399648,0.756904481511]`; P UAR `0.863395445135 / 0.047093690349 / 0.790131124914 / 0.918633540373 / 0.128502415459 / [0.804920905286,0.921869984984]`; P F1 `0.871371195195 / 0.039822125361 / 0.809336143572 / 0.917652594012 / 0.108316450441 / [0.821925495803,0.920816894588]`; P Score `0.867383320165 / 0.043437926848 / 0.799733634243 / 0.918143067192 / 0.118409432950 / [0.813448010241,0.921318630089]`; Mean `0.801825630372 / 0.029207897030 / 0.755019832380 / 0.833699432821 / 0.078679600441 / [0.765559236167,0.838092024578]`.

## Test results

The canonical membership counts were verified as TEST_NONE=1364, TEST_SOFT=1208, TEST_HARD=1014. The single Phase-B invocation evaluated exactly 15 pairs (five seeds × three protocols), once each. Each row is D UAR/F1/Score, P UAR/F1/Score, Mean.

### TEST_NONE

| seed | D UAR/F1/Score | P UAR/F1/Score | Mean |
|---:|---|---|---:|
| 42 | 0.748892731465/0.726546771853/0.737719751659 | 0.774501228318/0.739130976354/0.756816102336 | 0.747267926997 |
| 43 | 0.777699915059/0.771810348813/0.774755131936 | 0.745337973647/0.735520094563/0.740429034105 | 0.757592083020 |
| 44 | 0.786973668244/0.783840866211/0.785407267227 | 0.814105188714/0.797913835011/0.806009511862 | 0.795708389545 |
| 45 | 0.723489261012/0.712264707270/0.717876984141 | 0.779684359413/0.733197342112/0.756440850763 | 0.737158917452 |
| 46 | 0.794976337823/0.791462536023/0.793219436923 | 0.822722027842/0.814699792961/0.818710910401 | 0.805965173662 |

Aggregate mean / SD / min / max / range / 95% CI: D UAR `0.766406382721/0.029645432857/0.723489261012/0.794976337823/0.071487076811/[0.729596716031,0.803216049410]`; D F1 `0.757185046034/0.035552493689/0.712264707270/0.791462536023/0.079197828753/[0.713040794573,0.801329297494]`; D Score `0.761795714377/0.032486185974/0.717876984141/0.793219436923/0.075342452782/[0.721458786815,0.802132641940]`; P UAR `0.787270155587/0.031447781872/0.745337973647/0.822722027842/0.077384054195/[0.748222577013,0.826317734161]`; P F1 `0.764092408200/0.039047892385/0.733197342112/0.814699792961/0.081502450848/[0.715608046020,0.812576770380]`; P Score `0.775681281894/0.034424412219/0.740429034105/0.818710910401/0.078281876296/[0.732937728582,0.818424835205]`; Mean `0.768738498135/0.030396127732/0.737158917452/0.805965173662/0.068806256210/[0.730996720645,0.806480275625]`.

### TEST_SOFT

| seed | D UAR/F1/Score | P UAR/F1/Score | Mean |
|---:|---|---|---:|
| 42 | 0.727808136005/0.704742935936/0.716275535971 | 0.774546564353/0.743025215866/0.758785890109 | 0.737530713040 |
| 43 | 0.765857113945/0.760427751966/0.763142432955 | 0.745313044995/0.734552675585/0.739932860290 | 0.751537646623 |
| 44 | 0.789698441611/0.785413051354/0.787555746482 | 0.814440181374/0.803341354128/0.808890767751 | 0.798223257117 |
| 45 | 0.707022869864/0.695656315489/0.701339592677 | 0.783179281479/0.737714407725/0.760446844602 | 0.730893218639 |
| 46 | 0.794616474398/0.789841681267/0.792229077832 | 0.820980118591/0.817448680352/0.819214399471 | 0.805721738652 |

Aggregate mean / SD / min / max / range / 95% CI: D UAR `0.757000607165/0.038446437272/0.707022869864/0.794616474398/0.087593604533/[0.709263050145,0.804738164184]`; D F1 `0.747216347202/0.044477174098/0.695656315489/0.789841681267/0.094185365778/[0.691990641383,0.802442053021]`; D Score `0.752108477183/0.041380547928/0.701339592677/0.792229077832/0.090889485156/[0.700727740596,0.803489213771]`; P UAR `0.787691838158/0.030873560615/0.745313044995/0.820980118591/0.075667073596/[0.749357249447,0.826026426870]`; P F1 `0.767216466731/0.039845934958/0.734552675585/0.817448680352/0.082896004767/[0.717741203819,0.816691729643]`; P Score `0.777454152445/0.034560739710/0.739932860290/0.819214399471/0.079281539181/[0.734541326195,0.820366978694]`; Mean `0.764781314814/0.034859987506/0.730893218639/0.805721738652/0.074828520013/[0.721496923351,0.808065706278]`.

### TEST_HARD

| seed | D UAR/F1/Score | P UAR/F1/Score | Mean |
|---:|---|---|---:|
| 42 | 0.724678549430/0.698465859224/0.711572204327 | 0.746893994419/0.714836754374/0.730865374396 | 0.721218789362 |
| 43 | 0.763801201109/0.757236585452/0.760518893281 | 0.761248518674/0.736672806268/0.748960662471 | 0.754739777876 |
| 44 | 0.790008084386/0.784474154016/0.787241119201 | 0.806988034711/0.790545454545/0.798766744628 | 0.793003931914 |
| 45 | 0.703842008007/0.690564305323/0.697203156665 | 0.738579456401/0.695474836473/0.717027146437 | 0.707115151551 |
| 46 | 0.806802433015/0.800263739178/0.803533086096 | 0.800699568026/0.794380097192/0.797539832609 | 0.800536459353 |

Aggregate mean / SD / min / max / range / 95% CI: D UAR `0.757826455189/0.043252472312/0.703842008007/0.806802433015/0.102960425008/[0.704121417487,0.811531492892]`; D F1 `0.746200928639/0.049707855746/0.690564305323/0.800263739178/0.109699433855/[0.684480473731,0.807921383546]`; D Score `0.752013691914/0.046387678194/0.697203156665/0.803533086096/0.106329929431/[0.694415781940,0.809611601888]`; P UAR `0.770881914446/0.031242662349/0.738579456401/0.806988034711/0.068408578310/[0.732089025399,0.809674803493]`; P F1 `0.746381989770/0.044539696660/0.695474836473/0.794380097192/0.098905260719/[0.691078651936,0.801685327604]`; P Score `0.758631952108/0.037815675558/0.717027146437/0.798766744628/0.081739598191/[0.711677589200,0.805586315017]`; Mean `0.755322822011/0.041687958453/0.707115151551/0.800536459353/0.093421307802/[0.703560384842,0.807085259180]`.

## Historical same-seed Shared comparison

The reference values are the committed Stage-7 Final-Test dossier, sections `TEST_NONE`, `TEST_SOFT`, and `TEST_HARD`, rows for equal-parameter shared fusion. They are rounded to six decimals; the comparison therefore preserves that precision and is descriptive only. New minus historical Shared Mean deltas (seeds 42–46), mean / sample SD / exploratory Student-t CI / wins: TEST_NONE `-0.053856073003, -0.061537916980, -0.012506610455, -0.056810082548, -0.001614826338`; `-0.037265101865 / 0.027974815410 / [-0.072000423015,-0.002529780714] / 0/5`; TEST_SOFT `-0.073471286960, -0.074698353377, -0.019853742883, -0.067013781361, -0.017802261348`; `-0.050567885186 / 0.029130128097 / [-0.086737716507,-0.014398053865] / 0/5`; TEST_HARD `-0.094327210638, -0.084811222124, -0.038844068086, -0.102580848449, -0.035236540647`; `-0.071159977989 / 0.031800784656 / [-0.110645867410,-0.031674088567] / 0/5`. No superiority claim is made.

## Commands, artifacts, and limitations

Firewall command: `python3 -m py_compile scripts/common/evaluate_postclosure_shared_progress_test.py`; synthetic checks passed; `git diff --check` passed; firewall commit `00e4569d988d6fbd95194d9d4c0e1ff4c20f1a1b` was pushed and the remote task ref matched it. DEV command used the evaluator with `--mode dev-preflight` and wrote `logs/task012a_postclosure_shared_progress_test/dev_preflight.json`; its SHA256 and schema are recorded in the external artifact directory. The single Test command used `--mode test`, `--firewall-sha 00e4569d988d6fbd95194d9d4c0e1ff4c20f1a1b`, and wrote `logs/task012a_postclosure_shared_progress_test/test_results.json`; marker status is `COMPLETED`, count 15. External artifacts contain no raw logits, predictions, probabilities, labels, sample metadata, caches, checkpoints, or MLflow databases.

External artifact SHA256: `dev_preflight.json` `67a7c70754070dd0594262ab706aac8d3d853d9b2a84f349adf97e08e8ac2e64`; `test_results.json` `ddcae2835c04a346c1a27d3e9fa9d7c2b2afc96980b46d206dd2e79120baf285`; `test_invocation.marker` `3a37730e932f29c5928b930263d6c745c6c9f9171b9c42b34990320de329e12d`; evaluator `7157f32253b6f3e831a00992c180f39fdd0d86d47c4718517b0b7e6584d2a5b7`.

The original artifacts record Python 3.12.3, torch 2.10.0+cu128, NumPy 2.5.1, CUDA 12.8, cuDNN 91002, CUDA device, `autocast_enabled=true`, model parameters `torch.float32`, and CPU output accumulation `torch.float32`; the exact autocast dtype was not retained and is not inferred here. No Test result was used for selection, tuning, thresholding, or a follow-up run. The historical TASK-011A raw-summary Test-column exposure remains a documented procedural caveat; strict quarantine compliance is not claimed. `src/audio`, model/data/loss semantics, production configs, the historical Stage-7 evaluator, Stage-7 evidence, and README/NEXT_TASK/requirements/plans were unchanged.

## TASK-012A-C1 read-only audit and closure guard

This section audits the original TASK-012A evidence without inference, dataset access, model loading, or evaluator execution. The original evaluator at the immutable firewall commit was SHA256 `7157f32253b6f3e831a00992c180f39fdd0d86d47c4718517b0b7e6584d2a5b7`. It generated the reported results. The post-evaluation closure revision is a separate code revision and is not attributed to the historical evaluation.

### Original artifact audit

| artifact | schema/status | bytes | SHA256 | audit result |
|---|---|---:|---|---|
| `dev_preflight.json` | `task012a-dev-preflight-v1`, `test_iteration=false` | 11291 | `67a7c70754070dd0594262ab706aac8d3d853d9b2a84f349adf97e08e8ac2e64` | seeds 42–46 exactly once; complete scalar metrics; finite arithmetic; all DEV differences ≤0.0005 |
| `test_results.json` | `task012a-test-v1`, `test_iteration=true` | 12875 | `ddcae2835c04a346c1a27d3e9fa9d7c2b2afc96980b46d206dd2e79120baf285` | exactly 15 unique seed/protocol pairs; scalar metrics and counts complete |
| `test_invocation.marker` | `status=COMPLETED`, `evaluation_count=15` | 48 | `3a37730e932f29c5928b930263d6c745c6c9f9171b9c42b34990320de329e12d` | completed marker; does not by itself prove absence of every prior invocation |

The DEV ledger matches the immutable firewall `LEDGER`: config SHA, checkpoint SHA, epochs `7/9/5/19/3`, model dimensions 256/160, shared fusion, Progress balancing, 552775 trainable parameters, and canonical pseudo-cache SHA. The literal invariant counts are not treated as independent proof: the unchanged RAMPS DataModule validation path checks the cache and canonical TRAIN alignment in `src/fusion/data/wsm_ramps_semantic_datamodule.py` lines 114–163, including accepted D/P counts. This audit did not instantiate that production DataModule.

DEV observed counts were D/P `621/312` for every seed. Test observed counts were identical across seeds: NONE D/P `827/537`, SOFT `710/498`, HARD `654/360`; protocol membership was respectively `1364/1208/1014`. Score and Mean arithmetic, finite values, reproduction differences, and the complete Cartesian pair set were independently checked from the saved scalar JSON.

The JSON field `invocation_count=15` counts checkpoint/protocol evaluations. The authorized Phase-B CLI invocation count was one; the completed marker records `evaluation_count=15` and is not rewritten.

### Original command and provenance record

The exact commands retained in the preceding Codex execution context were run from `/media/maxim/Programs/Projects/WSM` using the direct `.venv/bin/python` executable, with no explicit `PYTHONPATH` or other environment prefix recorded:

```text
.venv/bin/python scripts/common/evaluate_postclosure_shared_progress_test.py --mode dev-preflight --data-root /media/maxim/Databases/WSM_NEW --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache --pseudo-cache-path /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --output logs/task012a_postclosure_shared_progress_test/dev_preflight.json

.venv/bin/python scripts/common/evaluate_postclosure_shared_progress_test.py --mode test --data-root /media/maxim/Databases/WSM_NEW --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache --pseudo-cache-path /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --preflight logs/task012a_postclosure_shared_progress_test/dev_preflight.json --marker logs/task012a_postclosure_shared_progress_test/test_invocation.marker --firewall-sha 00e4569d988d6fbd95194d9d4c0e1ff4c20f1a1b --output logs/task012a_postclosure_shared_progress_test/test_results.json
```

The retained execution context records the firewall commit `00e4569d988d6fbd95194d9d4c0e1ff4c20f1a1b`, its ordinary push, and a pre-Test remote-ref check matching that SHA. Exact UTC timestamps and a complete original shell transcript were not retained; no precise chronology is inferred from commit or filesystem timestamps. The original evidence establishes the completed metric pass and artifact contents, while absence of metric-bearing retries or checkpoint changes is supported by the retained task execution assertions and immutable identities, not by the marker alone.

Original runtime fields record `device=cuda`, `autocast_enabled=true`, model dtype `torch.float32`, and output accumulation dtype `torch.float32` for every saved row. The exact autocast dtype was not recorded.

### Historical guard limitation and post-evaluation closure

The original `test_pass` checked an existing output only inside `atomic_json`, after all Test evaluations. Its preflight gate checked only `test_iteration=false` and five result rows; firewall SHA was optional and local-HEAD-only. These are historical procedural limitations, not claims about the original result values. The original synthetic checks also did not cover zero-boundary logits, Student-t CI construction, or paired aggregation.

The new post-evaluation closure revision adds one unconditional first statement in `test_pass`: every Test-mode request raises `TASK-012A is closed: Test execution is permanently disabled`, before preflight reads, marker creation, output writes, DataModule construction, model loading, or inference. It has no CLI bypass. The original inference/metric/LEDGER/scientific code is otherwise unchanged. Original evaluator SHA256: `7157f32253b6f3e831a00992c180f39fdd0d86d47c4718517b0b7e6584d2a5b7`; new closure evaluator SHA256: `108079eb382ac580b3293f8716a58848e4d6e1123f952141f48aa3041a247317`. Post-evaluation synthetic checks use only fabricated inputs and verify the closed path with spies, zero-boundary/masked metrics, sample SD/Student-t CI, and paired arithmetic: `4 passed`.

### Numeric preservation

All original per-seed DEV/Test scalar rows, protocol counts, aggregate values, paired historical Shared deltas, negative results, and artifact hashes remain unchanged. The original task remains a completed comparative extension, not a pristine first-use final evaluation.

### Appendix A — complete original `dev_preflight.json` bytes

The following fenced block is the complete original file content, including its trailing newline:

```json
{
  "invariants": {
    "accepted_pseudo_counts": [
      376,
      1801
    ],
    "model": "wsm_av_r3_disease_query_model",
    "progress_balancing": true,
    "pseudo_cache_sha256": "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945",
    "task_aware_fusion": false,
    "trainable_parameters": 552775
  },
  "ledger": [
    {
      "checkpoint": "logs/wsm_mm_pd_dep_v1/postclosure_shared_progress_optuna_seed42_2026-10-08_15-14_wsm_av_r3_disease_query_model_task011a-shared-progress-8a53-029_f5dbef29/checkpoints/epoch=7_dev_mean_score=0.8337.pt",
      "checkpoint_sha256": "ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346",
      "checkpoint_sha256_actual": "ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346",
      "config": "logs/wsm_mm_pd_dep_v1/postclosure_shared_progress_optuna_seed42_2026-10-08_15-14_wsm_av_r3_disease_query_model_task011a-shared-progress-8a53-029_f5dbef29/task011a-shared-progress-8a53-029.yaml",
      "config_sha256": "a5bcb56f48b223f6156281183b72c8b53200cff320b3895c4cb3247fd86878d6",
      "config_sha256_actual": "a5bcb56f48b223f6156281183b72c8b53200cff320b3895c4cb3247fd86878d6",
      "epoch": 7,
      "expected": [
        0.749256,
        0.918143,
        0.8336994328
      ],
      "mlflow_run_id": "90a5adebf43c46808814f29374e092d0",
      "run_name": "task011a-shared-progress-8a53-029",
      "seed": 42,
      "trainable_parameters": 552775
    },
    {
      "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed43_2026-10-08_20-35_wsm_av_r3_disease_query_model_967a8e52/checkpoints/epoch=9_dev_mean_score=0.7550.pt",
      "checkpoint_sha256": "ba280153b2a1dca9808f25c05e581fec72ab3404a3602233154ef26c9f62b926",
      "checkpoint_sha256_actual": "ba280153b2a1dca9808f25c05e581fec72ab3404a3602233154ef26c9f62b926",
      "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/01_A_shared_progress_parent_seed43.yaml",
      "config_sha256": "dda16afd9478f935107a00ddf901956b3318eb37d8385ce169e2b2e2dbb25a52",
      "config_sha256_actual": "dda16afd9478f935107a00ddf901956b3318eb37d8385ce169e2b2e2dbb25a52",
      "epoch": 9,
      "expected": [
        0.7103060305,
        0.7997336342,
        0.7550198324
      ],
      "mlflow_run_id": "009b4d3c9be44220b6d3169989c518e4",
      "run_name": "task011b_shared_progress_parent_seed43",
      "seed": 43,
      "trainable_parameters": 552775
    },
    {
      "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed44_2026-10-08_20-38_wsm_av_r3_disease_query_model_d47b97e1/checkpoints/epoch=5_dev_mean_score=0.8157.pt",
      "checkpoint_sha256": "775826bf903fced08e24eb9bfff267e414c1ea7d5e94201a466bc40642e93100",
      "checkpoint_sha256_actual": "775826bf903fced08e24eb9bfff267e414c1ea7d5e94201a466bc40642e93100",
      "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/02_A_shared_progress_parent_seed44.yaml",
      "config_sha256": "44c7cad4879a3635c7eff6d9a1f9a1a5cbbdbaf123ba4d19f3f9a3f2effe775b",
      "config_sha256_actual": "44c7cad4879a3635c7eff6d9a1f9a1a5cbbdbaf123ba4d19f3f9a3f2effe775b",
      "epoch": 5,
      "expected": [
        0.7506546698,
        0.8807426876,
        0.8156986787
      ],
      "mlflow_run_id": "28b2b383cc244c8b98011c60c73ac14e",
      "run_name": "task011b_shared_progress_parent_seed44",
      "seed": 44,
      "trainable_parameters": 552775
    },
    {
      "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed45_2026-10-08_20-41_wsm_av_r3_disease_query_model_87fac8f5/checkpoints/epoch=19_dev_mean_score=0.8049.pt",
      "checkpoint_sha256": "e4a6776902ca57885ceb8f1c5444736d16fb8257618181c027b2fc8ad996e3d9",
      "checkpoint_sha256_actual": "e4a6776902ca57885ceb8f1c5444736d16fb8257618181c027b2fc8ad996e3d9",
      "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/03_A_shared_progress_parent_seed45.yaml",
      "config_sha256": "f411e86536681dc69903d01d23961714e9045df9d92e431281d688c8f0145a88",
      "config_sha256_actual": "f411e86536681dc69903d01d23961714e9045df9d92e431281d688c8f0145a88",
      "epoch": 19,
      "expected": [
        0.7303546019,
        0.87938767,
        0.804871136
      ],
      "mlflow_run_id": "ff2b0e3033ef4688ab46841585e49bcf",
      "run_name": "task011b_shared_progress_parent_seed45",
      "seed": 45,
      "trainable_parameters": 552775
    },
    {
      "checkpoint": "logs/wsm_mm_pd_dep_v1/task011b_shared_progress_parent_seed46_2026-10-08_20-47_wsm_av_r3_disease_query_model_b26025ee/checkpoints/epoch=3_dev_mean_score=0.7998.pt",
      "checkpoint_sha256": "374efc91f7ecf153f85ce42bee6af92475a229795d1221e6c8dfa4c0bb267478",
      "checkpoint_sha256_actual": "374efc91f7ecf153f85ce42bee6af92475a229795d1221e6c8dfa4c0bb267478",
      "config": "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite/04_A_shared_progress_parent_seed46.yaml",
      "config_sha256": "bb3db27f3ce216403cb5390ba235a8f8a0fa8f410bb6424891b5187cdcf56c60",
      "config_sha256_actual": "bb3db27f3ce216403cb5390ba235a8f8a0fa8f410bb6424891b5187cdcf56c60",
      "epoch": 3,
      "expected": [
        0.7407686022,
        0.8589095418,
        0.799839072
      ],
      "mlflow_run_id": "aa76a8669b09431ba9c6fbe3f4f98b0c",
      "run_name": "task011b_shared_progress_parent_seed46",
      "seed": 46,
      "trainable_parameters": 552775
    }
  ],
  "protocol_metadata": {
    "test_hard": {
      "count": 1014,
      "expected_count": 1014
    },
    "test_none": {
      "count": 1364,
      "expected_count": 1364
    },
    "test_soft": {
      "count": 1208,
      "expected_count": 1208
    }
  },
  "results": [
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 7,
      "expected": {
        "d_score": 0.749256,
        "mean": 0.8336994328,
        "p_score": 0.918143
      },
      "metrics": {
        "depression": {
          "mf1": 0.7470643513354506,
          "score": 0.7492557984501715,
          "uar": 0.7514472455648926
        },
        "mean": 0.8336994328212888,
        "observed_counts": {
          "depression": 621,
          "parkinson": 312
        },
        "parkinson": {
          "mf1": 0.9176525940121409,
          "score": 0.9181430671924059,
          "uar": 0.9186335403726709
        },
        "protocol_count": 933
      },
      "mlflow_run_id": "90a5adebf43c46808814f29374e092d0",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "reproduction_difference": [
        -2.015498284935191e-07,
        6.719240586150477e-08,
        2.1288748541792302e-11
      ],
      "run_name": "task011a-shared-progress-8a53-029",
      "seed": 42
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 9,
      "expected": {
        "d_score": 0.7103060305,
        "mean": 0.7550198324,
        "p_score": 0.7997336342
      },
      "metrics": {
        "depression": {
          "mf1": 0.709874432650121,
          "score": 0.710306030517404,
          "uar": 0.7107376283846871
        },
        "mean": 0.7550198323800305,
        "observed_counts": {
          "depression": 621,
          "parkinson": 312
        },
        "parkinson": {
          "mf1": 0.80933614357158,
          "score": 0.7997336342426569,
          "uar": 0.7901311249137336
        },
        "protocol_count": 933
      },
      "mlflow_run_id": "009b4d3c9be44220b6d3169989c518e4",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "reproduction_difference": [
        1.740407817862888e-11,
        4.26568780298453e-11,
        -1.996958154393269e-11
      ],
      "run_name": "task011b_shared_progress_parent_seed43",
      "seed": 43
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 5,
      "expected": {
        "d_score": 0.7506546698,
        "mean": 0.8156986787,
        "p_score": 0.8807426876
      },
      "metrics": {
        "depression": {
          "mf1": 0.7503289473684212,
          "score": 0.750654669762642,
          "uar": 0.7509803921568627
        },
        "mean": 0.815698678670457,
        "observed_counts": {
          "depression": 621,
          "parkinson": 312
        },
        "parkinson": {
          "mf1": 0.8856399645285249,
          "score": 0.8807426875782721,
          "uar": 0.8758454106280193
        },
        "protocol_count": 933
      },
      "mlflow_run_id": "28b2b383cc244c8b98011c60c73ac14e",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "reproduction_difference": [
        -3.7358005577914355e-11,
        -2.172784174803155e-11,
        -2.9543034685275416e-11
      ],
      "run_name": "task011b_shared_progress_parent_seed44",
      "seed": 44
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 19,
      "expected": {
        "d_score": 0.7303546019,
        "mean": 0.804871136,
        "p_score": 0.87938767
      },
      "metrics": {
        "depression": {
          "mf1": 0.7302703616526003,
          "score": 0.7303546019280742,
          "uar": 0.7304388422035482
        },
        "mean": 0.8048711359581536,
        "observed_counts": {
          "depression": 621,
          "parkinson": 312
        },
        "parkinson": {
          "mf1": 0.8807215097487228,
          "score": 0.879387669988233,
          "uar": 0.8780538302277433
        },
        "protocol_count": 933
      },
      "mlflow_run_id": "ff2b0e3033ef4688ab46841585e49bcf",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "reproduction_difference": [
        2.8074209623696333e-11,
        -1.1766920771094647e-11,
        -4.184641522186894e-11
      ],
      "run_name": "task011b_shared_progress_parent_seed45",
      "seed": 45
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 3,
      "expected": {
        "d_score": 0.7407686022,
        "mean": 0.799839072,
        "p_score": 0.8589095418
      },
      "metrics": {
        "depression": {
          "mf1": 0.7403140485499822,
          "score": 0.7407686022395102,
          "uar": 0.7412231559290383
        },
        "mean": 0.7998390720311056,
        "observed_counts": {
          "depression": 621,
          "parkinson": 312
        },
        "parkinson": {
          "mf1": 0.863505764114691,
          "score": 0.858909541822701,
          "uar": 0.8543133195307109
        },
        "protocol_count": 933
      },
      "mlflow_run_id": "aa76a8669b09431ba9c6fbe3f4f98b0c",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "reproduction_difference": [
        3.9510283933452683e-11,
        2.27009522291155e-11,
        3.110556257013286e-11
      ],
      "run_name": "task011b_shared_progress_parent_seed46",
      "seed": 46
    }
  ],
  "runtime": {
    "cuda": "12.8",
    "cudnn": 91002,
    "numpy": "2.5.1",
    "python": "3.12.3",
    "torch": "2.10.0+cu128"
  },
  "schema": "task012a-dev-preflight-v1",
  "synthetic_checks": {
    "metric_arithmetic": true,
    "raw_logit_threshold": true,
    "sparse_mask": true,
    "student_t_ddof1": true
  },
  "test_iteration": false
}
```

### Appendix B — complete original `test_results.json` bytes

The following fenced block is the complete original file content, including its trailing newline:

```json
{
  "invocation_count": 15,
  "no_labels": true,
  "no_raw_logits": true,
  "no_raw_predictions": true,
  "no_raw_probabilities": true,
  "no_sample_metadata": true,
  "protocol_counts": {
    "test_hard": 1014,
    "test_none": 1364,
    "test_soft": 1208
  },
  "results": [
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 7,
      "metrics": {
        "depression": {
          "mf1": 0.7265467718526148,
          "score": 0.7377197516586215,
          "uar": 0.7488927314646281
        },
        "mean": 0.7472679269974125,
        "observed_counts": {
          "depression": 827,
          "parkinson": 537
        },
        "parkinson": {
          "mf1": 0.7391309763540866,
          "score": 0.7568161023362037,
          "uar": 0.7745012283183206
        },
        "protocol_count": 1364
      },
      "mlflow_run_id": "90a5adebf43c46808814f29374e092d0",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_none",
      "run_name": "task011a-shared-progress-8a53-029",
      "seed": 42
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 7,
      "metrics": {
        "depression": {
          "mf1": 0.7047429359364399,
          "score": 0.7162755359706486,
          "uar": 0.7278081360048574
        },
        "mean": 0.7375307130399946,
        "observed_counts": {
          "depression": 710,
          "parkinson": 498
        },
        "parkinson": {
          "mf1": 0.7430252158656989,
          "score": 0.7587858901093406,
          "uar": 0.7745465643529823
        },
        "protocol_count": 1208
      },
      "mlflow_run_id": "90a5adebf43c46808814f29374e092d0",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_soft",
      "run_name": "task011a-shared-progress-8a53-029",
      "seed": 42
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 7,
      "metrics": {
        "depression": {
          "mf1": 0.698465859224153,
          "score": 0.7115722043271981,
          "uar": 0.7246785494302432
        },
        "mean": 0.721218789361756,
        "observed_counts": {
          "depression": 654,
          "parkinson": 360
        },
        "parkinson": {
          "mf1": 0.7148367543738807,
          "score": 0.7308653743963138,
          "uar": 0.7468939944187469
        },
        "protocol_count": 1014
      },
      "mlflow_run_id": "90a5adebf43c46808814f29374e092d0",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_hard",
      "run_name": "task011a-shared-progress-8a53-029",
      "seed": 42
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 9,
      "metrics": {
        "depression": {
          "mf1": 0.7718103488134496,
          "score": 0.7747551319361509,
          "uar": 0.7776999150588522
        },
        "mean": 0.7575920830204845,
        "observed_counts": {
          "depression": 827,
          "parkinson": 537
        },
        "parkinson": {
          "mf1": 0.7355200945626478,
          "score": 0.7404290341048183,
          "uar": 0.7453379736469887
        },
        "protocol_count": 1364
      },
      "mlflow_run_id": "009b4d3c9be44220b6d3169989c518e4",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_none",
      "run_name": "task011b_shared_progress_parent_seed43",
      "seed": 43
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 9,
      "metrics": {
        "depression": {
          "mf1": 0.7604277519661536,
          "score": 0.7631424329553496,
          "uar": 0.7658571139445456
        },
        "mean": 0.7515376466226878,
        "observed_counts": {
          "depression": 710,
          "parkinson": 498
        },
        "parkinson": {
          "mf1": 0.7345526755852843,
          "score": 0.7399328602900261,
          "uar": 0.745313044994768
        },
        "protocol_count": 1208
      },
      "mlflow_run_id": "009b4d3c9be44220b6d3169989c518e4",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_soft",
      "run_name": "task011b_shared_progress_parent_seed43",
      "seed": 43
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 9,
      "metrics": {
        "depression": {
          "mf1": 0.7572365854523531,
          "score": 0.7605188932805345,
          "uar": 0.7638012011087157
        },
        "mean": 0.7547397778757527,
        "observed_counts": {
          "depression": 654,
          "parkinson": 360
        },
        "parkinson": {
          "mf1": 0.7366728062676807,
          "score": 0.748960662470971,
          "uar": 0.7612485186742612
        },
        "protocol_count": 1014
      },
      "mlflow_run_id": "009b4d3c9be44220b6d3169989c518e4",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_hard",
      "run_name": "task011b_shared_progress_parent_seed43",
      "seed": 43
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 5,
      "metrics": {
        "depression": {
          "mf1": 0.7838408662108085,
          "score": 0.7854072672274768,
          "uar": 0.786973668244145
        },
        "mean": 0.7957083895449683,
        "observed_counts": {
          "depression": 827,
          "parkinson": 537
        },
        "parkinson": {
          "mf1": 0.7979138350105592,
          "score": 0.8060095118624597,
          "uar": 0.8141051887143602
        },
        "protocol_count": 1364
      },
      "mlflow_run_id": "28b2b383cc244c8b98011c60c73ac14e",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_none",
      "run_name": "task011b_shared_progress_parent_seed44",
      "seed": 44
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 5,
      "metrics": {
        "depression": {
          "mf1": 0.7854130513539535,
          "score": 0.7875557464824816,
          "uar": 0.7896984416110099
        },
        "mean": 0.7982232571167369,
        "observed_counts": {
          "depression": 710,
          "parkinson": 498
        },
        "parkinson": {
          "mf1": 0.8033413541277254,
          "score": 0.8088907677509921,
          "uar": 0.8144401813742588
        },
        "protocol_count": 1208
      },
      "mlflow_run_id": "28b2b383cc244c8b98011c60c73ac14e",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_soft",
      "run_name": "task011b_shared_progress_parent_seed44",
      "seed": 44
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 5,
      "metrics": {
        "depression": {
          "mf1": 0.7844741540161266,
          "score": 0.7872411192008567,
          "uar": 0.7900080843855868
        },
        "mean": 0.7930039319144937,
        "observed_counts": {
          "depression": 654,
          "parkinson": 360
        },
        "parkinson": {
          "mf1": 0.7905454545454546,
          "score": 0.7987667446281308,
          "uar": 0.8069880347108069
        },
        "protocol_count": 1014
      },
      "mlflow_run_id": "28b2b383cc244c8b98011c60c73ac14e",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_hard",
      "run_name": "task011b_shared_progress_parent_seed44",
      "seed": 44
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 19,
      "metrics": {
        "depression": {
          "mf1": 0.7122647072698967,
          "score": 0.7178769841409549,
          "uar": 0.7234892610120132
        },
        "mean": 0.7371589174518668,
        "observed_counts": {
          "depression": 827,
          "parkinson": 537
        },
        "parkinson": {
          "mf1": 0.7331973421121726,
          "score": 0.7564408507627787,
          "uar": 0.7796843594133849
        },
        "protocol_count": 1364
      },
      "mlflow_run_id": "ff2b0e3033ef4688ab46841585e49bcf",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_none",
      "run_name": "task011b_shared_progress_parent_seed45",
      "seed": 45
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 19,
      "metrics": {
        "depression": {
          "mf1": 0.6956563154886646,
          "score": 0.7013395926765322,
          "uar": 0.7070228698643999
        },
        "mean": 0.7308932186391428,
        "observed_counts": {
          "depression": 710,
          "parkinson": 498
        },
        "parkinson": {
          "mf1": 0.7377144077246092,
          "score": 0.7604468446017535,
          "uar": 0.7831792814788978
        },
        "protocol_count": 1208
      },
      "mlflow_run_id": "ff2b0e3033ef4688ab46841585e49bcf",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_soft",
      "run_name": "task011b_shared_progress_parent_seed45",
      "seed": 45
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 19,
      "metrics": {
        "depression": {
          "mf1": 0.6905643053226276,
          "score": 0.6972031566650095,
          "uar": 0.7038420080073915
        },
        "mean": 0.7071151515509813,
        "observed_counts": {
          "depression": 654,
          "parkinson": 360
        },
        "parkinson": {
          "mf1": 0.6954748364726673,
          "score": 0.717027146436953,
          "uar": 0.7385794564012387
        },
        "protocol_count": 1014
      },
      "mlflow_run_id": "ff2b0e3033ef4688ab46841585e49bcf",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_hard",
      "run_name": "task011b_shared_progress_parent_seed45",
      "seed": 45
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 3,
      "metrics": {
        "depression": {
          "mf1": 0.7914625360230547,
          "score": 0.7932194369230672,
          "uar": 0.7949763378230797
        },
        "mean": 0.8059651736621697,
        "observed_counts": {
          "depression": 827,
          "parkinson": 537
        },
        "parkinson": {
          "mf1": 0.8146997929606625,
          "score": 0.8187109104012722,
          "uar": 0.8227220278418819
        },
        "protocol_count": 1364
      },
      "mlflow_run_id": "aa76a8669b09431ba9c6fbe3f4f98b0c",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_none",
      "run_name": "task011b_shared_progress_parent_seed46",
      "seed": 46
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 3,
      "metrics": {
        "depression": {
          "mf1": 0.7898416812665499,
          "score": 0.7922290778322225,
          "uar": 0.7946164743978952
        },
        "mean": 0.8057217386518032,
        "observed_counts": {
          "depression": 710,
          "parkinson": 498
        },
        "parkinson": {
          "mf1": 0.8174486803519061,
          "score": 0.8192143994713839,
          "uar": 0.8209801185908616
        },
        "protocol_count": 1208
      },
      "mlflow_run_id": "aa76a8669b09431ba9c6fbe3f4f98b0c",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_soft",
      "run_name": "task011b_shared_progress_parent_seed46",
      "seed": 46
    },
    {
      "autocast_enabled": true,
      "device": "cuda",
      "epoch": 3,
      "metrics": {
        "depression": {
          "mf1": 0.8002637391775789,
          "score": 0.8035330860963349,
          "uar": 0.8068024330150909
        },
        "mean": 0.8005364593527229,
        "observed_counts": {
          "depression": 654,
          "parkinson": 360
        },
        "parkinson": {
          "mf1": 0.7943800971919213,
          "score": 0.797539832609111,
          "uar": 0.8006995680263007
        },
        "protocol_count": 1014
      },
      "mlflow_run_id": "aa76a8669b09431ba9c6fbe3f4f98b0c",
      "model_dtype": "torch.float32",
      "output_accumulation_dtype": "torch.float32",
      "protocol": "test_hard",
      "run_name": "task011b_shared_progress_parent_seed46",
      "seed": 46
    }
  ],
  "runtime": {
    "cuda": "12.8",
    "cudnn": 91002,
    "numpy": "2.5.1",
    "python": "3.12.3",
    "torch": "2.10.0+cu128"
  },
  "schema": "task012a-test-v1",
  "test_iteration": true
}
```

### Appendix C — complete original `test_invocation.marker` bytes

```text
{"status": "COMPLETED", "evaluation_count": 15}
```
