# Data Provenance

`preprocessing/normal/Iam_split.py` only *reads* pre-computed IAM splits
(`data/normal/crnn/{train,val,test}.csv`, `data/normal/trocr/{train,val}.csv`) -
it never generates them from scratch. The actual generation happened in separate Google
Colab notebooks that live outside this repo. This file records which of those notebooks
have been located and verified against the current data, and which are still missing.

**Why this matters**: if any of the `crnn/*.csv` or `trocr/*.csv` files under
`data/normal/` are ever lost or corrupted, this project cannot regenerate them on its own - the
generating notebook(s) would need to be found again.

## Verification method

For each candidate script, checked whether its exact split arithmetic (fraction, seed,
truncation behavior) reproduces the row counts actually present in the current
`data/normal/crnn/` or `data/normal/trocr` CSVs. An exact match is strong evidence (not
conclusive proof) that a script is the real source, since split ratios + `int()`
truncation rarely collide by chance.

## Confirmed: TrOCR Stage-1 IAM fine-tune script

**Match confirmed.** The user located a Colab script titled "阶段一 Fine-tune 脚本 — 只用
IAM 数据微调 trocr-base-handwritten" with:
- `WORDS_TXT` = `words_new.txt`, `IAM_WORDS_ZIP` = `iam_words.zip` (matches this project's files)
- Parses IAM's `words.txt` format, keeps only `status == "ok"` rows
- `VAL_SPLIT_RATIO = 0.05` (95/5 split), `RANDOM_SEED = 42`
- Output dir named `stage1_iam_finetuned` (this project's checkpoint folder was originally
  `stage1_iam_finetuned_final` - a very close naming match, "_final" likely appended once
  this training run was chosen as the keeper)

Arithmetic check:
```
data/normal/trocr/train.csv = 36,389 rows
data/normal/trocr/val.csv   =  1,915 rows
total                   = 38,304 rows

script's train_val_split(): n_val = max(1, int(38304 * 0.05)) = 1915
                             n_train = 38304 - 1915 = 36389
```
Exact match. This is the TrOCR Normal-stage source script.

**2026-08-22 update - recovered into the repo, verified byte-exact, gap closed.** The
user provided the original Colab script's full source. Its data-prep half (parse
`words_new.txt` -> filter `status=="ok"` -> shuffle with seed 42 -> 95/5 slice) is now in
this repo at `preprocessing/normal/generate_trocr_split.py`, with only two path-level
adaptations (image existence/corruption check reads directly from `iam_words.zip` instead
of a pre-extracted local folder; writes the project's flat `i_<word_id>.png` filename
convention instead of a full local disk path) - the filtering/splitting algorithm itself
is untouched. The training half of that Colab script was deliberately not brought in,
since `train_model/normal/trocr/finetune_TrOCR.py` already covers that, adapted to local paths.

Verified stronger than the arithmetic check above: ran it against the real
`I_Am_Dataset/` on disk and compared the resulting train/val **filename sets** (not just
counts) against the existing `data/normal/trocr/train.csv` and `val.csv` - **exact match,
every filename**. This is no longer a documented gap - the split is fully reproducible
from `words_new.txt` + `iam_words.zip` again.

## Not a match: three other located scripts

The user located three more Colab scripts while searching; none of them match the
current `data/normal/crnn/*.csv` data. All three turned out to belong to a **different,
abandoned lineage** that merged the old Kaggle "Normal" handwriting dataset together
with IAM into one combined dataset (filenames prefixed `n_` for Normal, `i_` for IAM),
rather than training CRNN on IAM alone the way this project's `crnn/train_iam_only.py`
does.

| Script | What it is | Why it doesn't match |
|---|---|---|
| "全量数据准备 (Normal + IAM, 不抽样)" | Merges Normal `labels.csv` + IAM `words_new.txt` into one 70/10/20 split (`full_dataset/{train,val,test}.csv`), filenames prefixed `n_`/`i_` | Current `data/normal/crnn/train.csv` and `data/normal/trocr/train.csv` are **100% `i_`-prefixed** (verified: 0 rows with any other prefix) - no Normal data mixed in, so this can't be the source |
| CRNN training script using `train_df_full` / `IMG_DIR_FULL` / `DRIVE_CKPT_DIR = f"{DRIVE_DIR}/full_dataset/checkpoints"` | Training counterpart to the script above | Variable names (`train_df_full`, `IMG_DIR_FULL`, `full_dataset` in paths) tie back exactly to the merged-dataset script's own output naming - trains on the merged data, not pure IAM |
| `split_analysis.py` (reads `val_predictions_large.csv`, splits stats by `i_`/`n_` prefix) | Post-hoc analysis of an already-trained model's predictions | Also handles both `i_` and `n_` prefixes together - evidence of a model trained on the same merged dataset, not a split-generation script at all (doesn't touch train/val/test creation) |

## Still missing: CRNN's pure-IAM 70/10/20 split script

**Not found**, despite checking `previous_code/` (the full archive of every superseded
script in this project's history) and three candidate Colab scripts the user located.
`previous_code/B_superseded_pipeline/code/` has `split_Normal.py` and `split_Doctor.py`
but no IAM equivalent.

Reverse-engineered how close a guess gets, working only from `words_new.txt` and the
extracted `iam_words_images/` (read-only, no project files modified):

| Filter applied | Rows remaining | Diff from actual (34,056) |
|---|---|---|
| `status == "ok"` only | 38,304 | 4,248 |
| + restricted to the CRNN 70-class character dictionary (`CHARS` in `modeling.py`) | 35,925 | 1,869 |
| + image file must actually exist in `iam_words_images/` | 34,140 | 84 |

The remaining gap of 84 is most likely a small number of corrupted/unopenable IAM images
being filtered out (the TrOCR script above has an explicit `img.verify()` check for
exactly this; the CRNN script probably had the same step) - not independently confirmed,
since that would require actually opening all 34,140 candidate images with PIL.

**Conclusion**: the real generating script very likely applied `status=="ok"` +
CRNN-dictionary filtering + image-existence/corruption checks + a 70/10/20 split, and is
a sibling of the confirmed TrOCR script above (same `words_new.txt`/`iam_words.zip`
source, same era) - but the actual file has not been located. The data itself
(`data/normal/crnn/*.csv`) is intact and working; only the generating script is missing.

---

# 数据来源考古（中文版）

`preprocessing/normal/Iam_split.py` 只负责**读取**预先算好的IAM切分（`data/normal/crnn/{train,val,test}.csv`、`data/normal/trocr/{train,val}.csv`），从来不会从头生成它们。真正的生成过程是在这个repo之外、独立的Google Colab notebook里完成的。这份文档记录哪些notebook已经找到并核对过、哪些还没找到。

**为什么这件事重要**：如果 `data/normal/crnn/*.csv` 或 `data/normal/trocr/*.csv` 哪天丢了或者损坏了，这个项目自己是没办法重新生成它们的——得重新找回当初那份生成脚本才行。

## 核对方法

对每一份候选脚本，检查它的切分算法（比例、随机种子、截断行为）算出来的行数，跟现在 `data/normal/crnn/` 或 `data/normal/trocr` 里csv的实际行数对不对得上。精确匹配是很强的证据（不是绝对证明）——切分比例加上 `int()` 截断这种组合，很难是巧合撞上的。

## 已确认：TrOCR 阶段一 IAM finetune 脚本

**匹配确认。** 你找到的这份Colab脚本，标题是"阶段一 Fine-tune 脚本 — 只用 IAM 数据微调 trocr-base-handwritten"，特征是：
- `WORDS_TXT` = `words_new.txt`，`IAM_WORDS_ZIP` = `iam_words.zip`（跟项目里的文件对得上）
- 解析IAM的 `words.txt` 格式，只保留 `status == "ok"` 的行
- `VAL_SPLIT_RATIO = 0.05`（95/5切分），`RANDOM_SEED = 42`
- 输出目录叫 `stage1_iam_finetuned`（项目里的checkpoint文件夹原本叫 `stage1_iam_finetuned_final`——命名非常接近，"_final"大概率是这次训练被选定为最终版之后加上去的）

算法核对：
```
data/normal/trocr/train.csv = 36,389 行
data/normal/trocr/val.csv   =  1,915 行
总计                     = 38,304 行

脚本里的 train_val_split(): n_val = max(1, int(38304 * 0.05)) = 1915
                             n_train = 38304 - 1915 = 36389
```
精确匹配。这就是 TrOCR Normal 阶段的源头脚本。

## 不匹配：另外三份找到的脚本

你搜索过程中还找到另外三份Colab脚本，都跟现在 `data/normal/crnn/*.csv` 的数据对不上。三份全部属于**另一条被放弃的路线**——把旧版Kaggle "Normal"手写数据集跟IAM合并成一个数据集（文件名前缀 `n_` 代表Normal，`i_` 代表IAM），而不是像这个项目现在的 `crnn/train_iam_only.py` 那样只用纯IAM训练CRNN。

| 脚本 | 是什么 | 为什么不匹配 |
|---|---|---|
| "全量数据准备 (Normal + IAM, 不抽样)" | 把Normal的 `labels.csv` 和 IAM的 `words_new.txt` 合并成一份70/10/20切分（`full_dataset/{train,val,test}.csv`），文件名前缀 `n_`/`i_` | 现在 `data/normal/crnn/train.csv` 和 `data/normal/trocr/train.csv` **100%都是 `i_` 前缀**（验证过：没有一行是别的前缀）——没有混入Normal数据，所以不可能是这个来源 |
| 用 `train_df_full`/`IMG_DIR_FULL`/`DRIVE_CKPT_DIR = f"{DRIVE_DIR}/full_dataset/checkpoints"` 的CRNN训练脚本 | 上面那份脚本的训练对应版 | 变量名（`train_df_full`、`IMG_DIR_FULL`、路径里的`full_dataset`）跟合并数据集那份脚本自己的输出命名完全对得上——训练用的是合并数据，不是纯IAM |
| `split_analysis.py`（读取 `val_predictions_large.csv`，按 `i_`/`n_` 前缀拆开统计） | 对已经训练好的模型的预测结果做事后分析 | 同样是同时处理 `i_` 和 `n_` 两种前缀——证明对应的模型是在同一份合并数据集上训练出来的，而且这份脚本本身根本不涉及生成train/val/test切分 |

## 仍未找到：CRNN 纯IAM 70/10/20 切分脚本

**没找到**，查过了 `previous_code/`（这个项目历史上所有被取代脚本的完整归档）和你找到的三份候选Colab脚本。`previous_code/B_superseded_pipeline/code/` 里有 `split_Normal.py` 和 `split_Doctor.py`，但没有对应IAM的版本。

只用 `words_new.txt` 和已解压的 `iam_words_images/` 反向推导，看能猜多接近（全程只读，没有修改任何项目文件）：

| 过滤条件 | 剩余行数 | 跟实际(34,056)的差 |
|---|---|---|
| 只看 `status == "ok"` | 38,304 | 4,248 |
| + 限制在CRNN 70类字符字典内（`modeling.py` 里的 `CHARS`） | 35,925 | 1,869 |
| + 图片文件确实存在于 `iam_words_images/` | 34,140 | 84 |

剩下84条的差距，最可能是有少量损坏/打不开的IAM图片被过滤掉了（上面那份TrOCR脚本里就有专门的 `img.verify()` 检查做这件事，CRNN那份大概率也有同样一步）——没有独立验证过，因为要验证得真的用PIL挨个打开全部34,140张候选图片。

**结论**：真正的生成脚本很可能用了 `status=="ok"` + CRNN字典过滤 + 图片存在性/完整性检查 + 70/10/20切分，是上面确认的TrOCR脚本的姊妹版本（同一份 `words_new.txt`/`iam_words.zip` 来源，同一个年代）——但实际文件还没找到。数据本身（`data/normal/crnn/*.csv`）完好、能正常用，只是生成它的脚本目前找不到了。
