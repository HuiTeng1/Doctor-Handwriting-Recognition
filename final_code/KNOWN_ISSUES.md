# Known Issues

Open items found while dry-running `RUN_ORDER.md` end to end and while auditing
`final_code/` for dead code / stale references. Fixed items are listed too, so this
doubles as a changelog of what's been verified.

## Fixed and verified (re-ran the actual code, not just reasoned about it)

| Issue | File | Fix |
|---|---|---|
| `TROcrFineTune.py` checked for a `final/` subfolder that never existed on disk (the checkpoint sits flat in `checkpoint/trocr/`), so re-running it would silently start real training from scratch instead of refusing to overwrite | `normal/trocr/TROcrFineTune.py` | Check now looks for `checkpoint/trocr/model.safetensors` directly. Verified: running it now correctly raises `FileExistsError` and stops. |
| `results/*.csv` (`crnn_finetune_summary.csv`, `crnn_pilot_summary.csv`, `trocr_pilot_summary.csv`) stored absolute checkpoint paths from before the project got reorganized into `final_code/` (`D:\...\finetune_preprocessing_compare\...`) | `doctor/results/*.csv` | Paths rewritten to the current `final_code/doctor/...` locations. Verified: all 8 paths resolve to real files. |
| `resolve_doctor_label_conflicts.py` used `pd.concat` to append its 6 hardcoded resolutions with no idempotency check - re-running it would duplicate those 6 rows in `labels.csv` | `doctor/dataset_prepare/resolve_doctor_label_conflicts.py` | Now drops any existing row for the filename before re-adding it. Verified: ran it twice back to back, `labels.csv` stayed at 10050 rows both times. |
| `final_test_eval.py` unconditionally read `crnn_pilot_summary.csv` / `trocr_pilot_summary.csv` (the 15%-pilot results) and reported that as the "final" score - the CRNN number on record (test_cer=0.395) was actually the pilot (15%-trained) model's score, not the full-finetune model's | `doctor/final_test_eval.py` | Split into a shared `run_eval()` plus two calls: pilot-decision check (`final_test_scores_pilot.csv`) and full-finetune final score (`final_test_scores_full.csv`). Verified: real run produced correct pilot-stage output. |
| `pilot_compare.py` re-used `finetune_all.py`'s module-level `PIPELINES` (already narrowed to `["baseline"]`), so re-running the pilot silently stopped checking `clhaa` at all, contradicting its own docstring ("baseline vs clhaa") | `doctor/finetune_all.py`, `doctor/pilot_compare.py` | `run_crnn_all()` / `run_trocr_all()` now take an optional `pipelines=` param; `pilot_compare.py` passes its own `PILOT_PIPELINES` list. Verified: re-run now prints a real skip-check for `clhaa` (and now `fajardo`/`benitez` too, see below). |
| `data/Doctor/train.csv` / `val.csv` / `test.csv` (produced by the old, archived `split_Doctor.py`) were never read by any current script - dead weight left over from before the clean+split pipeline moved to `Doctor_dataset_prepare/` | `doctor/data/Doctor/` | Moved to `previous_code/B_superseded_pipeline/data_Doctor_old_split/`. |

## Deliberately left as-is (confirmed decisions, not bugs)

- **TrOCR doctor full finetune never finished** (`doctor/checkpoints_trocr/baseline/` has `checkpoint-3954/` and `checkpoint-4833/` but no `DONE.txt`, no `final/`, and `results/trocr_finetune_summary.csv` doesn't exist). Confirmed this is genuinely incomplete work, not a bug - left untouched.
- **`doctor/checkpoints/{clhaa,fajardo,benitez}/`** (the losing pipelines' full-finetune checkpoints) are unreferenced by any current code (`finetune_all.py`'s `PIPELINES = ["baseline"]` only). Decided to keep them in place rather than archive - they're physical proof the full 4-way comparison was actually run, and moving them wouldn't delete the CER numbers in `results/crnn_finetune_summary.csv` anyway, which stay either way.
- **`pilot_compare.py`'s `PILOT_PIPELINES`** was expanded to `["baseline", "clhaa", "fajardo", "benitez"]` (code change applied, verified with a syntax check), but has **not been run** - would trigger real training for `fajardo`/`benitez` on both CRNN and TrOCR (4 real training runs). Deferred until explicitly requested.

## Still open - needs a decision

- **`final_test_eval.py`'s "2/2 full-finetune final score" step requires *both* `crnn_finetune_summary.csv` and `trocr_finetune_summary.csv` to exist before evaluating *either* model.** Since TrOCR's doesn't exist yet, this currently skips CRNN too, even though CRNN's full-finetune result is real and ready. Proposed fix: make the two models' checks independent so CRNN's real final score is available now, and TrOCR's fills in once its full finetune completes.
- **`IAM_WORDS_TXT`** in `normal/dataset/Iam_split.py` is a dead variable - defined, never read by any function in the project (confirmed: no code anywhere parses `words_new.txt` from scratch; the splits are all pre-computed CSVs). Candidate for deletion.
- **A fresh clone of this repo will not run end to end on its own** - resolved by handing over data separately, see below. `final_code/normal/dataset/I_Am_Dataset/`, `doctor/data/Doctor/images/`, `doctor/data/Doctor_clean/images/`, `doctor/preprocessed_doctor/`, `doctor/preprocessed_doctor_test/` are all gitignored (too large for git). Scripts that only rely on cached results (`test_CrnnNormal.py`, `Test_TrOcr.py`, `finetune_all.py`'s CRNN half, `pilot_compare.py`'s `baseline`/`clhaa`) happen to skip before ever touching images, so they're fine as-is. The rest need real files handed over directly (not re-derivable from anything in git):
  - `combineDataset.py` - its 5 raw source folders still exist (kept outside this repo, e.g. external storage), but `SOURCE_FOLDERS` is hardcoded to the author's old machine-specific path - **edit it to wherever those 5 folders actually are before running.** If you don't have those 5 raw folders handy, easier to skip this step entirely: its output, `data/Doctor/images/`, is what actually needs to be handed over (58MB) - place it at `final_code/doctor/data/Doctor/images/`, `labels.csv` is already in git, then skip straight to `clean_doctor_dataset.py`.
  - `clean_doctor_dataset.py`, `generate_doctor_preprocessed.py` - need `data/Doctor/images/` and `data/Doctor_clean/images/` respectively, handed over directly (see table below).
  - `TROcrFineTune.py` - its "already trained, refuse to overwrite" check (see the ① fix above) now points at the right file, but that file (`checkpoint/trocr/model.safetensors`, 1.27GB) is itself on the >100MB gitignore list, so a fresh clone still won't have it and the check will still (correctly, not a bug this time) conclude no model exists yet and start training from scratch unless it's handed over too.
- **`doctor/checkpoints_trocr/baseline/checkpoint-3954/model.safetensors` and similar (>100MB) files** are gitignored (GitHub's hard push limit) and need a separate transfer method (Git LFS, cloud storage, manual copy) - not `git push`. Full list and sizes:
  - `final_code/normal/checkpoint/trocr/model.safetensors` (1273.9 MB)
  - `final_code/doctor/checkpoints_trocr/baseline/checkpoint-3954/model.safetensors` (1273.9 MB)
  - `final_code/doctor/checkpoints_trocr/baseline/checkpoint-3954/optimizer.pt` (2543.5 MB)
  - `final_code/doctor/checkpoints_trocr/baseline/checkpoint-4833/model.safetensors` (1273.9 MB)
  - `final_code/doctor/checkpoints_trocr/baseline/checkpoint-4833/optimizer.pt` (2543.5 MB)
  - `final_code/doctor/checkpoints_trocr_pilot/baseline/final/model.safetensors` (1273.9 MB)
  - `final_code/doctor/checkpoints_trocr_pilot/clhaa/final/model.safetensors` (1273.9 MB)

## Images that need to be handed over separately (gitignored, not in git)

| Folder | Size | Files |
|---|---|---|
| `normal/dataset/I_Am_Dataset/` (IAM zip + extracted images) | 1,490.8 MB | 34,248 |
| `doctor/data/Doctor/images/` | 58.2 MB | 10,347 |
| `doctor/data/Doctor_clean/images/` | 56.9 MB | 10,050 |
| `doctor/preprocessed_doctor/` | 90.2 MB | 32,160 |
| `doctor/preprocessed_doctor_test/` | 6.3 MB | 2,010 |

**Total ≈ 1.7 GB**, plus the checkpoint files listed above (~11 GB) if the recipient needs those too.

---

# 已知问题（中文版）

这份文档记录了完整跑一遍 `RUN_ORDER.md`、以及审查 `final_code/` 死代码/失效引用时找到的问题。已经修好的也列在这里，等于是一份"验证过什么"的变更记录。

## 已修好、且实测验证过的

| 问题 | 文件 | 修法 |
|---|---|---|
| `TROcrFineTune.py` 检查的是一个从来不存在的 `final/` 子文件夹（checkpoint文件其实是直接摊平放在 `checkpoint/trocr/` 下），导致重跑这个脚本会静默地真的从头开始训练，而不是拒绝覆盖 | `normal/trocr/TROcrFineTune.py` | 判断条件改成直接检查 `checkpoint/trocr/model.safetensors` 存不存在。已验证：现在跑会正确抛出 `FileExistsError` 并停止 |
| `results/*.csv`（`crnn_finetune_summary.csv`、`crnn_pilot_summary.csv`、`trocr_pilot_summary.csv`）存的是项目重组进 `final_code/` 之前的旧绝对路径（`D:\...\finetune_preprocessing_compare\...`） | `doctor/results/*.csv` | 路径全部改写成现在 `final_code/doctor/...` 的位置。已验证：全部8条路径都指向真实存在的文件 | 
| `resolve_doctor_label_conflicts.py` 用 `pd.concat` 追加6条硬编码的解决方案，没有幂等检查——重跑会在 `labels.csv` 里把这6行重复加一次 | `doctor/dataset_prepare/resolve_doctor_label_conflicts.py` | 现在会先删掉同名的旧行，再插入新行。已验证：连续跑两次，`labels.csv` 两次都稳定在10050行 |
| `final_test_eval.py` 无条件读取 `crnn_pilot_summary.csv`/`trocr_pilot_summary.csv`（15%pilot的结果）当成"最终成绩"上报——存档的CRNN数字（test_cer=0.395）其实是pilot(15%训练)模型的成绩，不是全量finetune模型的 | `doctor/final_test_eval.py` | 拆成共用的 `run_eval()` 加两次调用：pilot决策验证(`final_test_scores_pilot.csv`)和全量finetune最终成绩(`final_test_scores_full.csv`)。已验证：实跑产出了正确的pilot阶段结果 |
| `pilot_compare.py` 借用了 `finetune_all.py` 模块级的 `PIPELINES`（已经收窄成`["baseline"]`），导致重跑pilot时`clhaa`完全不会被检查，跟它自己文档写的"baseline vs clhaa"对不上 | `doctor/finetune_all.py`、`doctor/pilot_compare.py` | `run_crnn_all()`/`run_trocr_all()` 现在多一个可选的 `pipelines=` 参数；`pilot_compare.py` 传自己的 `PILOT_PIPELINES` 列表。已验证：重跑现在会真的打印出对`clhaa`（以及现在的`fajardo`/`benitez`）的跳过检查 |
| `data/Doctor/train.csv`/`val.csv`/`test.csv`（旧版、已归档的 `split_Doctor.py` 产出）从来没被任何现存脚本读取——是清洗+切分流程搬去 `Doctor_dataset_prepare/` 之前留下的死重 | `doctor/data/Doctor/` | 搬进 `previous_code/B_superseded_pipeline/data_Doctor_old_split/` |

## 确认过、故意保持原样的（不是bug）

- **TrOCR doctor 全量finetune从来没跑完**（`doctor/checkpoints_trocr/baseline/` 下有 `checkpoint-3954/` 和 `checkpoint-4833/`，但没有 `DONE.txt`，没有 `final/`，`results/trocr_finetune_summary.csv` 也不存在）。确认这是真的没做完，不是bug——没有动它
- **`doctor/checkpoints/{clhaa,fajardo,benitez}/`**（落选pipeline的全量finetune checkpoint）没有任何现存代码在引用（`finetune_all.py` 的 `PIPELINES = ["baseline"]` 只跑baseline）。决定留在原地不归档——它们是"4个方案真的全部跑过对比"的实物证据，而且搬不搬都不会影响 `results/crnn_finetune_summary.csv` 里的CER数字，那些数字始终都在
- **`pilot_compare.py` 的 `PILOT_PIPELINES`** 已经扩到 `["baseline", "clhaa", "fajardo", "benitez"]`（代码已改，语法检查过），但**还没有真的跑**——跑起来会对CRNN和TrOCR的`fajardo`/`benitez`触发真实训练(4段)。按你的要求先搁置，等明确要求再跑

## 还没解决，需要你决定

- **`final_test_eval.py` 的"2/2全量finetune最终成绩"这一步，要求CRNN和TrOCR的summary文件同时存在才会评估任何一个模型。** 因为TrOCR的还不存在，导致CRNN那份（其实已经真实、可用）也一起被跳过了。建议的修法：把两个模型的判断改成互相独立，这样CRNN现在就能拿到真实的最终成绩，TrOCR等它全量finetune做完再自动补上
- **`normal/dataset/Iam_split.py` 里的 `IAM_WORDS_TXT`** 是个死变量——定义了，但项目里没有任何函数会读它（确认过：没有任何代码会从头解析 `words_new.txt`，切分全部是预先算好的csv）。可以考虑删掉
- **这个repo如果直接clone，自己是跑不完整个流程的**——已经通过额外传输数据解决，见下方清单。`I_Am_Dataset/`、`doctor/data/Doctor/images/`、`doctor/data/Doctor_clean/images/`、`doctor/preprocessed_doctor/`、`doctor/preprocessed_doctor_test/` 全部被gitignore（体积太大）。只依赖缓存结果的脚本（`test_CrnnNormal.py`、`Test_TrOcr.py`、`finetune_all.py`的CRNN半、`pilot_compare.py`的`baseline`/`clhaa`）刚好会在碰到图片之前就跳过，所以没事。其余的需要真实文件直接传过去（git里推不出这些）：
  - `combineDataset.py` —— 它需要的5个原始来源文件夹本身还在（放在这个repo之外，比如外部存储），但 `SOURCE_FOLDERS` 写死的是原作者旧电脑上的路径——**重新跑之前要先把它改成这5个文件夹在当前机器上实际的位置**。如果手头没有这5个原始文件夹，更简单的做法是直接跳过这一步：真正需要传的是它的产出 `data/Doctor/images/`（58MB）——放到 `final_code/doctor/data/Doctor/images/`，`labels.csv` 本来就在git里，然后直接从 `clean_doctor_dataset.py` 开始跑
  - `clean_doctor_dataset.py`、`generate_doctor_preprocessed.py` —— 分别需要 `data/Doctor/images/` 和 `data/Doctor_clean/images/`，直接传（见下方表格）
  - `TROcrFineTune.py` —— 它"已经训练过、拒绝覆盖"的判断（见上面①的修复）现在指对文件了，但那个文件（`checkpoint/trocr/model.safetensors`，1.27GB）本身也在"超100MB"的gitignore名单里，所以全新clone下来还是没有这个文件，判断结果还是"没训练过"、会真的开始训练——这次不是bug了，是数据确实没给，除非也把这个文件传过去
- **`doctor/checkpoints_trocr/baseline/checkpoint-3954/model.safetensors` 等超过100MB的文件**被gitignore了（GitHub硬性推送上限），需要另外的传输方式（Git LFS、云存储、手动拷贝）——不能靠 `git push`。完整清单和大小见下方表格

## 需要额外传输的图片（被gitignore，不在git里）

| 文件夹 | 大小 | 文件数 |
|---|---|---|
| `normal/dataset/I_Am_Dataset/`（IAM压缩包+解压图片） | 1,490.8 MB | 34,248 |
| `doctor/data/Doctor/images/` | 58.2 MB | 10,347 |
| `doctor/data/Doctor_clean/images/` | 56.9 MB | 10,050 |
| `doctor/preprocessed_doctor/` | 90.2 MB | 32,160 |
| `doctor/preprocessed_doctor_test/` | 6.3 MB | 2,010 |

**图片总计约 1.7 GB**，如果对方也需要checkpoint文件，再加上面列的checkpoint（约11GB）。
