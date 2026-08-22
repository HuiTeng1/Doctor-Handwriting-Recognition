# Known Issues

Open items found while dry-running `RUN_ORDER.md` end to end and while auditing
`final_code/` for dead code / stale references. Fixed items are listed too, so this
doubles as a changelog of what's been verified.

## Fixed and verified (re-ran the actual code, not just reasoned about it)

| Issue | File | Fix |
|---|---|---|
| `TROcrFineTune.py` checked for a `final/` subfolder that never existed on disk (the checkpoint sits flat in `checkpoint/normal/trocr/`), so re-running it would silently start real training from scratch instead of refusing to overwrite | `train_model/normal/trocr/TROcrFineTune.py` | Check now looks for `checkpoint/normal/trocr/model.safetensors` directly. Verified: running it now correctly raises `FileExistsError` and stops. |
| `results/*.csv` (`crnn_finetune_summary.csv`, `crnn_pilot_summary.csv`, `trocr_pilot_summary.csv`) stored absolute checkpoint paths from before the project got reorganized into `final_code/` (`D:\...\finetune_preprocessing_compare\...`) | `data/doctor/results/*.csv` | Paths rewritten to the current `final_code/checkpoint/doctor/...` locations. Verified: all 8 paths resolve to real files. |
| `resolve_doctor_label_conflicts.py` used `pd.concat` to append its 6 hardcoded resolutions with no idempotency check - re-running it would duplicate those 6 rows in `labels.csv` | `previous_code/B_superseded_pipeline/doctor_dedup_v1/resolve_doctor_label_conflicts.py` (archived) | Dropped any existing row for the filename before re-adding it, at the time. Verified then: ran it twice back to back, `labels.csv` stayed at 10050 rows both times. Since fully superseded by `preprocessing/doctor/discover_doctor_duplicates.py` + `resolve_doctor_duplicates.py`, which don't need this patch - they're idempotent by construction (fresh dataframe each run, `drop_duplicates(keep="last")`). |
| `final_test_eval.py` unconditionally read `crnn_pilot_summary.csv` / `trocr_pilot_summary.csv` (the 15%-pilot results) and reported that as the "final" score - the CRNN number on record (test_cer=0.395) was actually the pilot (15%-trained) model's score, not the full-finetune model's | `train_model/doctor/final_test_eval.py` | Split into a shared `run_eval()` plus two calls: pilot-decision check (`final_test_scores_pilot.csv`) and full-finetune final score (`final_test_scores_full.csv`). Verified: real run produced correct pilot-stage output. |
| `pilot_compare.py` re-used `finetune_all.py`'s module-level `PIPELINES` (already narrowed to `["baseline"]`), so re-running the pilot silently stopped checking `clhaa` at all, contradicting its own docstring ("baseline vs clhaa") | `train_model/doctor/finetune_all.py`, `train_model/doctor/pilot_compare.py` | `run_crnn_all()` / `run_trocr_all()` now take an optional `pipelines=` param; `pilot_compare.py` passes its own `PILOT_PIPELINES` list. Verified: re-run now prints a real skip-check for `clhaa` (and now `fajardo`/`benitez` too, see below). |
| `data/Doctor/train.csv` / `val.csv` / `test.csv` (produced by the old, archived `split_Doctor.py`) were never read by any current script - dead weight left over from before the clean+split pipeline moved to `Doctor_dataset_prepare/` | `doctor/data/Doctor/` | Moved to `previous_code/B_superseded_pipeline/data_Doctor_old_split/`. |
| `final_test_eval.py`'s "2/2 full-finetune final score" step required *both* `crnn_finetune_summary.csv` and `trocr_finetune_summary.csv` to exist before evaluating *either* model, so CRNN's real, ready result was being skipped just because TrOCR's full finetune hasn't finished | `train_model/doctor/final_test_eval.py` | `run_eval()` now checks each model's results file independently. Verified: `data/doctor/results/final_test_scores_full.csv` currently contains only the CRNN row (TrOCR's summary file still doesn't exist), produced by a real run. |
| `app/streamlit_app.py` interpolated OCR output directly into HTML with `unsafe_allow_html=True` and no escaping (`raw_prediction`, `best_name` going straight into f-string `<span>` tags) | `app/streamlit_app.py` | Every interpolated value now goes through `html.escape()` first. |
| `app/streamlit_app.py`'s TrOCR inference call used the raw uploaded image instead of the preprocessed one, so turning the preprocessing toggle on wouldn't actually change what TrOCR sees once its own `clhaa` full finetune finishes and the toggle becomes live for it - the exact train/inference mismatch this whole redesign exists to prevent, just currently masked because TrOCR's toggle is disabled today and `baseline` preprocessing is a no-op anyway | `app/streamlit_app.py` | Changed `run_trocr(model, processor, device, image)` to `run_trocr(model, processor, device, image_for_model)`. No visible effect today (`baseline` = `img.copy()`), but stops the bug from resurfacing later. |

## Deliberately left as-is (confirmed decisions, not bugs)

- **TrOCR doctor full finetune, `baseline`: manually finalized at epoch 9, not run to natural early-stopping.** Training was interrupted at epoch 11 for time (~2.4h/epoch on full data); `trainer_state.json`'s own `best_model_checkpoint` already pointed at epoch 9 (`checkpoint-3954`, eval CER 11.84%) - epochs 10-11 had already regressed (12.19%, 12.46%), the same signal `EarlyStoppingCallback` would eventually act on, just observed manually instead of waiting out the full patience window. `checkpoint/doctor/trocr/baseline/final/` was built from `checkpoint-3954` (the 10 model/processor files `trainer.save_model()`+`processor.save_pretrained()` would have produced, not the training-resume-only `optimizer.pt`/`scheduler.pt`/`rng_state.pth`/`trainer_state.json`), `DONE.txt` written, `trocr_finetune_summary.csv` created with this row. Verified: re-ran `finetune_all.py` for real, it now prints "baseline: result already in the summary, skipping" for TrOCR. `clhaa` is not done yet - see `finetune_all.py`'s `max_epochs = 11 if pipeline_name == "clhaa" else 1000` (hard-capped at epoch 11 for the same time-constraint reasoning, applied proactively this time instead of after the fact).
- **`checkpoint/doctor/crnn/clhaa/`** (the losing pipeline's full-finetune checkpoint) is unreferenced by any current code (`finetune_all.py`'s `PIPELINES = ["baseline"]` only). Decided to keep it in place rather than archive - it's physical proof the full 4-way comparison was actually run, and moving it wouldn't delete the CER numbers in `results/crnn_finetune_summary.csv` anyway, which stay either way. `fajardo`/`benitez`'s full-finetune checkpoints (and their two rows from this summary csv) **were** archived to `previous_code/C_decided_not_used/doctor_crnn_full_finetune/` - the decision changed to deciding CRNN's winner from the 15%-pilot comparison only, not the 100%-data one, so those two stopped being load-bearing evidence.
- **`pilot_compare.py`'s `PILOT_PIPELINES`** was expanded to `["baseline", "clhaa", "fajardo", "benitez"]` (code change applied, verified with a syntax check), but has **not been run** - would trigger real training for `fajardo`/`benitez` on both CRNN and TrOCR (4 real training runs). Deferred until explicitly requested.

## Still open - needs a decision

- **`IAM_WORDS_TXT`** in `final_code/preprocessing/normal/Iam_split.py` is a dead variable - defined, never read by any function in the project (confirmed: no code anywhere parses `words_new.txt` from scratch; the splits are all pre-computed CSVs). Candidate for deletion.
- **A fresh clone of this repo will not run end to end on its own** - resolved by handing over data separately, see below. `final_code/data/normal/I_Am_Dataset/`, `data/doctor/doctor_raw/images/`, `data/doctor/doctor_clean/images/`, `data/doctor/preprocessed_doctor/`, `data/doctor/preprocessed_doctor_test/` are all gitignored (too large for git). Scripts that only rely on cached results (`test_CrnnNormal.py`, `Test_TrOcr.py`, `finetune_all.py`'s CRNN half, `pilot_compare.py`'s `baseline`/`clhaa`) happen to skip before ever touching images, so they're fine as-is. The rest need real files handed over directly (not re-derivable from anything in git):
  - `combineDataset.py` - `SOURCE_FOLDERS` now points at 5 fixed paths right under `doctor_raw/` itself (no more per-machine editing) - **extract the 5 raw zips there, using the exact folder names `SOURCE_FOLDERS` expects, before running.** These 5 folders are gitignored, same as `doctor_raw/images/`. If you don't have those 5 raw zips handy, easier to skip this step entirely: its output, `doctor_raw/images/`, is what actually needs to be handed over (58MB) - place it at `final_code/data/doctor/doctor_raw/images/`, `labels.csv` is already in git, then skip straight to `discover_doctor_duplicates.py`.
  - `discover_doctor_duplicates.py`, `generate_doctor_preprocessed.py` - need `doctor_raw/images/` and `doctor_clean/images/` respectively, handed over directly (see table below).
  - `TROcrFineTune.py` - its "already trained, refuse to overwrite" check (see the ① fix above) now points at the right file, but that file (`checkpoint/normal/trocr/model.safetensors`, 1.27GB) is itself on the >100MB gitignore list, so a fresh clone still won't have it and the check will still (correctly, not a bug this time) conclude no model exists yet and start training from scratch unless it's handed over too.
- **`checkpoint/doctor/trocr/baseline/checkpoint-3954/model.safetensors` and similar (>100MB) files** are gitignored (GitHub's hard push limit) and need a separate transfer method (Git LFS, cloud storage, manual copy) - not `git push`. Full list and sizes:
  - `final_code/checkpoint/normal/trocr/model.safetensors` (1273.9 MB)
  - `final_code/checkpoint/doctor/trocr/baseline/checkpoint-3954/model.safetensors` (1273.9 MB)
  - `final_code/checkpoint/doctor/trocr/baseline/checkpoint-3954/optimizer.pt` (2543.5 MB)
  - `final_code/checkpoint/doctor/trocr/baseline/checkpoint-4833/model.safetensors` (1273.9 MB)
  - `final_code/checkpoint/doctor/trocr/baseline/checkpoint-4833/optimizer.pt` (2543.5 MB)
  - `final_code/checkpoint/doctor/trocr_pilot/baseline/final/model.safetensors` (1273.9 MB)
  - `final_code/checkpoint/doctor/trocr_pilot/clhaa/final/model.safetensors` (1273.9 MB)

## Images that need to be handed over separately (gitignored, not in git)

| Folder | Size | Files |
|---|---|---|
| `data/normal/I_Am_Dataset/` (IAM zip + extracted images) | 1,490.8 MB | 34,248 |
| `data/doctor/doctor_raw/images/` | 58.2 MB | 10,347 |
| `data/doctor/doctor_clean/images/` | 56.9 MB | 10,050 |
| `data/doctor/preprocessed_doctor/` | 90.2 MB | 32,160 |
| `data/doctor/preprocessed_doctor_test/` | 6.3 MB | 2,010 |

**Total ≈ 1.7 GB**, plus the checkpoint files listed above (~11 GB) if the recipient needs those too.

---

# 已知问题（中文版）

这份文档记录了完整跑一遍 `RUN_ORDER.md`、以及审查 `final_code/` 死代码/失效引用时找到的问题。已经修好的也列在这里，等于是一份"验证过什么"的变更记录。

## 已修好、且实测验证过的

| 问题 | 文件 | 修法 |
|---|---|---|
| `TROcrFineTune.py` 检查的是一个从来不存在的 `final/` 子文件夹（checkpoint文件其实是直接摊平放在 `checkpoint/normal/trocr/` 下），导致重跑这个脚本会静默地真的从头开始训练，而不是拒绝覆盖 | `train_model/normal/trocr/TROcrFineTune.py` | 判断条件改成直接检查 `checkpoint/normal/trocr/model.safetensors` 存不存在。已验证：现在跑会正确抛出 `FileExistsError` 并停止 |
| `results/*.csv`（`crnn_finetune_summary.csv`、`crnn_pilot_summary.csv`、`trocr_pilot_summary.csv`）存的是项目重组进 `final_code/` 之前的旧绝对路径（`D:\...\finetune_preprocessing_compare\...`） | `data/doctor/results/*.csv` | 路径全部改写成现在 `final_code/checkpoint/doctor/...` 的位置。已验证：全部8条路径都指向真实存在的文件 | 
| `resolve_doctor_label_conflicts.py` 用 `pd.concat` 追加6条硬编码的解决方案，没有幂等检查——重跑会在 `labels.csv` 里把这6行重复加一次 | `previous_code/B_superseded_pipeline/doctor_dedup_v1/resolve_doctor_label_conflicts.py`（已归档） | 当时改成先删掉同名的旧行，再插入新行。当时验证：连续跑两次，`labels.csv` 两次都稳定在10050行。现在这个脚本已经被 `preprocessing/doctor/discover_doctor_duplicates.py` + `resolve_doctor_duplicates.py` 完全取代——新版本不需要这个补丁，天生就是幂等的（每次都是全新算出来的dataframe，加 `drop_duplicates(keep="last")`） |
| `final_test_eval.py` 无条件读取 `crnn_pilot_summary.csv`/`trocr_pilot_summary.csv`（15%pilot的结果）当成"最终成绩"上报——存档的CRNN数字（test_cer=0.395）其实是pilot(15%训练)模型的成绩，不是全量finetune模型的 | `train_model/doctor/final_test_eval.py` | 拆成共用的 `run_eval()` 加两次调用：pilot决策验证(`final_test_scores_pilot.csv`)和全量finetune最终成绩(`final_test_scores_full.csv`)。已验证：实跑产出了正确的pilot阶段结果 |
| `pilot_compare.py` 借用了 `finetune_all.py` 模块级的 `PIPELINES`（已经收窄成`["baseline"]`），导致重跑pilot时`clhaa`完全不会被检查，跟它自己文档写的"baseline vs clhaa"对不上 | `train_model/doctor/finetune_all.py`、`train_model/doctor/pilot_compare.py` | `run_crnn_all()`/`run_trocr_all()` 现在多一个可选的 `pipelines=` 参数；`pilot_compare.py` 传自己的 `PILOT_PIPELINES` 列表。已验证：重跑现在会真的打印出对`clhaa`（以及现在的`fajardo`/`benitez`）的跳过检查 |
| `data/Doctor/train.csv`/`val.csv`/`test.csv`（旧版、已归档的 `split_Doctor.py` 产出）从来没被任何现存脚本读取——是清洗+切分流程搬去 `Doctor_dataset_prepare/` 之前留下的死重 | `doctor/data/Doctor/` | 搬进 `previous_code/B_superseded_pipeline/data_Doctor_old_split/` |
| `final_test_eval.py` 的"2/2全量finetune最终成绩"这一步，原本要求CRNN和TrOCR的summary文件同时存在才会评估任何一个模型，导致CRNN那份（其实已经真实、可用）也一起被跳过 | `train_model/doctor/final_test_eval.py` | `run_eval()` 现在对每个模型独立判断。已验证：`data/doctor/results/final_test_scores_full.csv` 目前只有CRNN那一行（TrOCR的summary文件还不存在），是真跑出来的结果 |
| `app/streamlit_app.py` 把OCR识别结果直接塞进 `unsafe_allow_html=True` 的HTML里，没做任何转义（`raw_prediction`、`best_name` 直接拼进f-string的`<span>`） | `app/streamlit_app.py` | 现在每个插值都先过一遍 `html.escape()` |
| `app/streamlit_app.py` 的TrOCR推理调用用的是原图，不是预处理过的图——一旦TrOCR自己的`clhaa`全量finetune做完、开关对它也生效了，打开开关根本不会真的改变喂给TrOCR的图，正好是这次重新设计UI要防的那种训练/推理不匹配，只是现在因为TrOCR的开关本来就是禁用的、且baseline预处理是no-op，所以还没显形 | `app/streamlit_app.py` | 把 `run_trocr(model, processor, device, image)` 改成 `run_trocr(model, processor, device, image_for_model)`。现在没有任何可见变化（baseline=`img.copy()`），但避免以后这个bug冒出来 |

## 确认过、故意保持原样的（不是bug）

- **TrOCR doctor 全量finetune，`baseline`：手动定案在epoch 9，没有等它自然触发early stopping。** 训练在epoch 11因为时间原因中断（全量数据下约2.4小时/epoch）；`trainer_state.json` 自己记录的 `best_model_checkpoint` 早就指向epoch 9（`checkpoint-3954`，eval CER 11.84%）——epoch 10、11 其实已经在回退（12.19%、12.46%），跟 `EarlyStoppingCallback` 最终会触发的信号是同一个，只是人工提前观察到、没有等完整的patience窗口。`checkpoint/doctor/trocr/baseline/final/` 是从 `checkpoint-3954` 建的（`trainer.save_model()`+`processor.save_pretrained()` 会产出的那10个model/processor文件，不含只用于续训的 `optimizer.pt`/`scheduler.pt`/`rng_state.pth`/`trainer_state.json`），写了 `DONE.txt`，`trocr_finetune_summary.csv` 也补上了这一行。已验证：真跑了一次 `finetune_all.py`，TrOCR那半现在会打印"baseline: result already in the summary, skipping"。`clhaa` 还没做——见 `finetune_all.py` 里的 `max_epochs = 11 if pipeline_name == "clhaa" else 1000`（同样因为时间原因硬顶在epoch 11，这次是提前设好，不是事后补救）
- **`checkpoint/doctor/crnn/clhaa/`**（落选pipeline的全量finetune checkpoint）没有任何现存代码在引用（`finetune_all.py` 的 `PIPELINES = ["baseline"]` 只跑baseline）。决定留在原地不归档——它是"4个方案真的全部跑过对比"的实物证据，而且搬不搬都不会影响 `results/crnn_finetune_summary.csv` 里的CER数字，那些数字始终都在。`fajardo`/`benitez` 的全量finetune checkpoint(以及汇总表里对应的两行)**已经**归档到 `previous_code/C_decided_not_used/doctor_crnn_full_finetune/`——决定改成只用15% pilot结果来选CRNN的赢家，不再靠100%全量数据的这份结果，这两个就不算数了
- **`pilot_compare.py` 的 `PILOT_PIPELINES`** 已经扩到 `["baseline", "clhaa", "fajardo", "benitez"]`（代码已改，语法检查过），但**还没有真的跑**——跑起来会对CRNN和TrOCR的`fajardo`/`benitez`触发真实训练(4段)。按你的要求先搁置，等明确要求再跑

## 还没解决，需要你决定

- **`final_code/preprocessing/normal/Iam_split.py` 里的 `IAM_WORDS_TXT`** 是个死变量——定义了，但项目里没有任何函数会读它（确认过：没有任何代码会从头解析 `words_new.txt`，切分全部是预先算好的csv）。可以考虑删掉
- **这个repo如果直接clone，自己是跑不完整个流程的**——已经通过额外传输数据解决，见下方清单。`data/normal/I_Am_Dataset/`、`data/doctor/doctor_raw/images/`、`data/doctor/doctor_clean/images/`、`data/doctor/preprocessed_doctor/`、`data/doctor/preprocessed_doctor_test/` 全部被gitignore（体积太大）。只依赖缓存结果的脚本（`test_CrnnNormal.py`、`Test_TrOcr.py`、`finetune_all.py`的CRNN半、`pilot_compare.py`的`baseline`/`clhaa`）刚好会在碰到图片之前就跳过，所以没事。其余的需要真实文件直接传过去（git里推不出这些）：
  - `combineDataset.py` —— `SOURCE_FOLDERS` 现在指向 `doctor_raw/` 下面5个固定的路径（不用再按机器改了）——**跑之前先把5个原始zip解压到那几个固定路径下,文件夹名字要跟 `SOURCE_FOLDERS` 里写的完全一致**。这5个文件夹已经跟 `doctor_raw/images/` 一样被gitignore了。如果手头没有这5个原始zip，更简单的做法是直接跳过这一步：真正需要传的是它的产出 `doctor_raw/images/`（58MB）——放到 `final_code/data/doctor/doctor_raw/images/`，`labels.csv` 本来就在git里，然后直接从 `discover_doctor_duplicates.py` 开始跑
  - `discover_doctor_duplicates.py`、`generate_doctor_preprocessed.py` —— 分别需要 `doctor_raw/images/` 和 `doctor_clean/images/`，直接传（见下方表格）
  - `TROcrFineTune.py` —— 它"已经训练过、拒绝覆盖"的判断（见上面①的修复）现在指对文件了，但那个文件（`checkpoint/normal/trocr/model.safetensors`，1.27GB）本身也在"超100MB"的gitignore名单里，所以全新clone下来还是没有这个文件，判断结果还是"没训练过"、会真的开始训练——这次不是bug了，是数据确实没给，除非也把这个文件传过去
- **`checkpoint/doctor/trocr/baseline/checkpoint-3954/model.safetensors` 等超过100MB的文件**被gitignore了（GitHub硬性推送上限），需要另外的传输方式（Git LFS、云存储、手动拷贝）——不能靠 `git push`。完整清单和大小见下方表格

## 需要额外传输的图片（被gitignore，不在git里）

| 文件夹 | 大小 | 文件数 |
|---|---|---|
| `data/normal/I_Am_Dataset/`（IAM压缩包+解压图片） | 1,490.8 MB | 34,248 |
| `data/doctor/doctor_raw/images/` | 58.2 MB | 10,347 |
| `data/doctor/doctor_clean/images/` | 56.9 MB | 10,050 |
| `data/doctor/preprocessed_doctor/` | 90.2 MB | 32,160 |
| `data/doctor/preprocessed_doctor_test/` | 6.3 MB | 2,010 |

**图片总计约 1.7 GB**，如果对方也需要checkpoint文件，再加上面列的checkpoint（约11GB）。
