# Sleep Event Heart-Rate Response and Cardiovascular Survival Analysis

## 中文简介

本仓库提供一个用于研究**睡眠呼吸事件相关心率反应**及其与**心血管死亡结局**关系的两阶段分析流程。该流程包括：

1. **特征提取阶段**：从 EDF 生理信号中提取 `H.R` 通道，并结合事件标注工作簿 `NH.xlsx` 计算受试者层面的心率反应特征，包括 `HR surge`、`HR surge 30` 和 `HR surge 60`。
2. **生存分析阶段**：将特征表与生存结局和协变量表合并，完成单变量和多变量 Cox 比例风险模型分析，并输出 Kaplan–Meier 曲线、调整后生存曲线、森林图和结果汇总表。

本仓库适用于论文配套代码公开、方法复现以及派生分析。公开版本已对本地路径和环境相关信息进行了脱敏处理，但**不包含任何原始临床数据或可识别个体信息**。

---

## English Overview

This repository provides a two-stage analysis pipeline for studying the **heart-rate response to sleep-disordered breathing events** and its association with **cardiovascular mortality**. The workflow includes:

1. **Feature extraction**: the `H.R` channel is extracted from EDF physiological recordings, and subject-level HR response features are computed using respiratory-event annotations from `NH.xlsx`, including `HR surge`, `HR surge 30`, and `HR surge 60`.
2. **Survival analysis**: extracted features are merged with survival outcomes and covariates, followed by univariable and multivariable Cox proportional hazards analyses, Kaplan–Meier estimation, adjusted survival-curve plotting, and tabular summary export.

The repository is intended for manuscript-supporting code release, methodological reproducibility, and derivative analyses. The public version has been sanitized to remove local path information and machine-specific settings, but it **does not include raw clinical data or identifiable participant information**.

---

## Repository contents / 仓库内容

- `hr_feature_extraction_open_source.py`  
  从 EDF 文件提取 HR 信号，并根据 `NH.xlsx` 中的呼吸事件与睡眠分期信息生成受试者层面的 HR 反应特征。  
  Extracts HR signals from EDF files and computes subject-level HR response features using respiratory event annotations and sleep-stage intervals.

- `cox_open_source.py`  
  将特征、生存结局和协变量整合后开展 Cox 生存分析，并输出图表和结果表。  
  Merges features, survival outcomes, and covariates for Cox regression modelling and figure/table generation.

- `requirements.txt`  
  项目运行所需的 Python 依赖列表。  
  Python dependency list required to run the pipeline.

---

## Suggested repository structure / 建议目录结构

```text
.
├── README.md
├── requirements.txt
├── cox.py
├── hr_feature_extraction.py
├── data/
│   ├── features.xlsx
│   ├── sample_ids.txt
│   ├── survival.xlsx
│   ├── covariates.xlsx
│   ├── NH.xlsx
│   ├── edf/
│   └── npy_cache/
└── outputs/
```

---

## Data requirements / 数据要求

### 1. Feature extraction input / 特征提取输入

`hr_feature_extraction_open_source.py` requires:  
`hr_feature_extraction_open_source.py` 需要以下输入：

- a folder containing EDF files / 包含 EDF 文件的目录
- an annotation workbook `NH.xlsx` / 事件标注工作簿 `NH.xlsx`
- an EDF channel labelled `H.R` / EDF 中存在名为 `H.R` 的心率通道

### NH.xlsx format / NH.xlsx 格式要求

`NH.xlsx` should contain the following sheets / `NH.xlsx` 应至少包含以下工作表：

- `呼吸暂停事件`
- `低通气事件_1`
- `低通气事件_2`
- `睡眠阶段`

Required columns / 必需列名：

**Event sheets / 事件表：**
- `文件名称`
- `开始时间_秒`
- `结束时间_秒`

**Sleep-stage sheet / 睡眠分期表：**
- `文件名称`
- `开始时间_秒`
- `结束时间_秒`
- `睡眠阶段`

Additional notes / 补充说明：

- The script extracts a 6-digit subject ID from `文件名称`.  
  脚本会从 `文件名称` 中提取 6 位受试者编号。
- Only events overlapping with `N1期`, `N2期`, `N3期`, `N4期`, or `REM期` are retained.  
  仅保留与 `N1期`、`N2期`、`N3期`、`N4期` 或 `REM期` 重叠的事件。
- Events beginning within the first 100 seconds are excluded by default.  
  默认忽略起始于前 100 秒内的事件。

### 2. Survival analysis input / 生存分析输入

`cox_open_source.py` expects:  
`cox_open_source.py` 需要以下输入：

- `features.xlsx`: extracted physiological feature table / 提取后的生理特征表
- `sample_ids.txt`: subject IDs if `features.xlsx` lacks `sample_id` / 若 `features.xlsx` 中没有 `sample_id` 列，则需要该文件
- `survival.xlsx`: survival outcome table / 生存结局表
- `covariates.xlsx`: demographic and clinical covariate table / 人口学和临床协变量表

---

## Methods summary / 方法摘要

### HR response feature extraction / HR 反应特征提取

For each respiratory event, the script computes HR response relative to the 60-second pre-event baseline:  
对于每一个呼吸事件，脚本基于事件前 60 秒基线心率计算以下指标：

- `HR surge` = max HR during event − mean HR during the 60 s before event onset  
- `HR surge 30` = max HR from event onset to 30 s after event end − same baseline  
- `HR surge 60` = max HR from event onset to 60 s after event end − same baseline

Patient-level features are obtained by averaging valid event-level estimates.  
受试者层面的特征值通过对所有有效事件的事件级估计取平均得到。

### Survival modelling / 生存建模

The survival-analysis script includes / 生存分析脚本主要包括：

- quintile-based univariable Cox analysis / 基于五分位分组的单变量 Cox 分析
- multivariable Cox models with progressive covariate adjustment / 逐步协变量调整的多变量 Cox 模型
- proportional hazards assumption checks / 比例风险假设检验
- VIF-based multicollinearity assessment / 基于 VIF 的多重共线性检查
- joint grouping by hypoxic burden and HR surge / 基于 hypoxic burden 与 HR surge 的联合分组
- survival-curve plotting and absolute-risk estimation / 生存曲线绘制与绝对风险估计

---

## Main outputs / 主要输出

The pipeline can generate / 本流程可输出：

- subject-level HR feature tables (`.csv`) / 受试者层面的 HR 特征表
- Kaplan–Meier plots (`.png`) / Kaplan–Meier 曲线图
- adjusted survival curves (`.png`) / 调整后生存曲线
- forest plots (`.png`) / 森林图
- Cox regression result tables (`.csv`) / Cox 回归结果表
- resumable processing logs (`.log`) / 支持断点续跑的日志文件

---

## Installation / 安装

```bash
pip install -r requirements.txt
```

---

## Configuration / 配置方式

### Feature extraction / 特征提取

```bash
export EDF_FOLDER="data/edf"
export FEATURE_OUTPUT_PATH="outputs/hypoxia_features_all_patients.csv"
export ANNOTATION_FILE="data/NH.xlsx"
export CACHE_DIR="data/npy_cache"
python hr_feature_extraction_open_source.py
```

### Survival analysis / 生存分析

```bash
export FEATURE_EXCEL_PATH="data/features.xlsx"
export SAMPLE_ID_PATH="data/sample_ids.txt"
export SURVIVAL_DATA_PATH="data/survival.xlsx"
export COVARIATE_DATA_PATH="data/covariates.xlsx"
export OUTPUT_DIR="outputs"
python cox_open_source.py
```

For Windows PowerShell, replace `export` with `$env:`.  
若使用 Windows PowerShell，请将 `export` 替换为 `$env:`。

---

## Privacy and data-governance statement / 隐私与数据治理声明

This repository does not distribute raw EDF recordings, annotation workbooks, survival tables, or covariate tables. Public users are expected to prepare their own de-identified data in the required format.  
本仓库不公开原始 EDF 记录、标注工作簿、生存结局表或协变量表。公开使用者应自行准备符合格式要求的匿名化数据。

The open-source release has already removed or generalized:  
当前开源版本已移除或泛化以下信息：

- local absolute file paths / 本地绝对路径
- private folder naming conventions / 私有目录命名习惯
- machine-specific output locations / 机器相关的输出位置

### Checklist before publishing / 发布前核对清单

1. Do **not** upload raw EDF files or participant-level clinical tables.  
   不要上传原始 EDF 文件或受试者层面的临床数据表。
2. Do **not** publish files containing direct identifiers such as subject IDs, names, record numbers, or exact dates.  
   不要公开包含受试者编号、姓名、病历号或精确日期等直接标识信息的文件。
3. Inspect derived CSV outputs carefully before release.  
   在公开前请仔细检查派生 CSV 输出文件。
4. Remove screenshots, notebooks, and logs containing unpublished results or local paths.  
   删除包含未发表结果或本地路径的截图、notebook 和日志。

---

## Reproducibility note / 可复现性说明

For manuscript-associated release, we recommend reporting:  
对于论文配套代码发布，建议在正文或补充材料中报告：

- Python version / Python 版本
- operating system / 操作系统
- major package versions / 关键依赖版本
- data preprocessing rules / 数据预处理规则
- any cohort-specific exclusion criteria / 队列特异性的排除标准

---

## Citation / 引用

If you use this repository in a manuscript, please cite the associated paper and describe this repository as the privacy-sanitized implementation of the analysis workflow.  
如在论文中使用本仓库，请引用对应论文，并说明该仓库为经隐私脱敏后的分析流程实现。

---

## Disclaimer / 免责声明

This repository is intended for research use only and should not be interpreted as clinical decision-support software.  
本仓库仅用于科研目的，不应被解释为临床决策支持软件。
