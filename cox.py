"""
Cox survival analysis for hypoxic burden and HR surge features.

This script merges extracted physiological features with survival outcomes and
covariates, performs univariable and multivariable Cox proportional hazards
analyses, and generates publication-oriented figures and summary tables.

Open-source notes:
- Local absolute paths should be replaced by environment variables or relative paths.
- Public releases should only use de-identified input data.
"""

# Open-source sanitized version
# Local absolute paths have been replaced with environment variables or relative paths.

import numpy as np
import pandas as pd
import re
from decimal import Decimal, InvalidOperation
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os
import pickle
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False  # 避免负号乱码/显示异常
# 忽略警告
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')


# ==================== Utility functions: normalize sample_id ====================
def normalize_sample_id_series(s: pd.Series) -> pd.Series:
    """Normalize sample IDs into a stable string representation.

    This helper mitigates common merge issues caused by surrounding spaces,
    missing values, values like ``12345.0``, and scientific notation.
    """
    s = s.astype(str).str.strip()

    # Standardize common missing-value markers before ID cleaning
    s = s.replace(['nan', 'NaN', 'None', '<NA>', ''], pd.NA)

    sci_pat = re.compile(r'^\d+(?:\.\d+)?[eE][+-]?\d+$')
    float_int_pat = re.compile(r'^\d+\.0+$')

    def _fix_one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return pd.NA
        x = str(x).strip()
        if x == '' or x.lower() in ('nan', 'none', '<na>'):
            return pd.NA
        # 12345.0 -> 12345
        if float_int_pat.fullmatch(x):
            return x.split('.')[0]
        # 科学计数法 -> 整数串（仅在确实是整数时转换）
        if sci_pat.fullmatch(x):
            try:
                d = Decimal(x)
                if d == d.to_integral():
                    return str(int(d))
            except (InvalidOperation, ValueError):
                pass
        return x

    return s.map(_fix_one)


def diag_id_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str, col: str = 'sample_id') -> None:
    """Print overlap statistics for sample IDs across two tables."""
    sa = set(df_a[col].dropna().astype(str))
    sb = set(df_b[col].dropna().astype(str))
    inter = sa & sb
    print(f"\n[ID CHECK] {name_a}: rows={len(df_a):,}, unique_ids={len(sa):,}")
    print(f"[ID CHECK] {name_b}: rows={len(df_b):,}, unique_ids={len(sb):,}")
    print(f"[ID CHECK] intersection={len(inter):,}")
    if len(inter) == 0:
        # 打印少量示例帮助肉眼发现 '.0' / 空格 / 前导0等问题
        print(f"  sample {name_a} ids: {list(sorted(sa))[:10]}")
        print(f"  sample {name_b} ids: {list(sorted(sb))[:10]}")

# ==================== Data loading and integration ====================
def load_and_prepare_data():
    """Load, merge, and preprocess the feature, survival, and covariate tables."""

    # ===== 1. 加载特征数据（Excel） =====
    feature_excel_path = os.environ.get("FEATURE_EXCEL_PATH", "data/features.xlsx")
    print(f"\nLoading feature data from: {feature_excel_path}")
    feature_df_raw = pd.read_excel(feature_excel_path)

    # 你指定的特征名称
    feature_names = [
        'event_hr_mean',
        'event_hr_min',
        'event_hr_max',
        'event_hr_amp',
        'latency',
        'slope',
        'peak_lag',
        'recovery_time',
        'HR surge',
        'hypoxic_burden',
    ]

    # 如果 Excel 里已经有这些列，就按列名选；否则用前 13 列并强制改名
    if set(feature_names).issubset(feature_df_raw.columns):
        print("Detected named feature columns in Excel, using them directly.")
        feature_df = feature_df_raw[feature_names].copy()
    else:
        print("Excel does not contain all target feature column names.")
        print("Assuming the first 13 columns are the feature matrix and renaming them.")
        feature_df = feature_df_raw.iloc[:, :len(feature_names)].copy()
        feature_df.columns = feature_names

    # ===== 2. 处理 sample_id =====
    if 'sample_id' in feature_df_raw.columns:
        # 如果 Excel 里已经有 sample_id，就直接用
        print("Found 'sample_id' column in Excel, using it.")
        feature_df['sample_id'] = feature_df_raw['sample_id'].astype(str)
    else:
        # 否则仍然从原来的 txt 文件读取
        print("No 'sample_id' in Excel, loading from sample_ids.txt ...")
        sample_id_path = os.environ.get("SAMPLE_ID_PATH", "data/sample_ids.txt")
        with open(sample_id_path, 'r', encoding="utf-8") as f:
            sample_ids = np.array([line.strip() for line in f])

        if len(sample_ids) != len(feature_df):
            print(f"Warning: number of sample_ids ({len(sample_ids)}) "
                  f"!= number of feature rows ({len(feature_df)}).")
            min_len = min(len(sample_ids), len(feature_df))
            print(f"Using first {min_len} rows for alignment.")
            feature_df = feature_df.iloc[:min_len, :].copy()
            sample_ids = sample_ids[:min_len]

        feature_df['sample_id'] = sample_ids.astype(str)

    # ===== 3. 生存数据 =====
    survival_data_path = os.environ.get("SURVIVAL_DATA_PATH", "data/survival.xlsx")
    survival_data = pd.read_excel(survival_data_path, sheet_name="Sheet1")
    survival_data.rename(columns={
        'shhs1_nsrrid': 'sample_id',
        'cvd_death': 'event',
        'cvd_dthdt': 'time'
    }, inplace=True)

    # 将生存时间从天数转换为年数
    print("\nConverting survival time from days to years...")
    survival_data['time'] = survival_data['time'] / 365.25
    print(f"  Survival time range: {survival_data['time'].min():.2f} to {survival_data['time'].max():.2f} years")

    # ===== 4. 协变量数据 =====
    covariate_data_path = os.environ.get("COVARIATE_DATA_PATH", "data/covariates.xlsx")
    covariate_data = pd.read_excel(covariate_data_path)
    covariate_data.rename(columns={covariate_data.columns[0]: 'sample_id'}, inplace=True)

    # 统一并清洗 sample_id（避免 12345 vs 12345.0 / 空格 / 科学计数法 导致合并为 0 行）
    survival_data['sample_id'] = normalize_sample_id_series(survival_data['sample_id'])
    feature_df['sample_id'] = normalize_sample_id_series(feature_df['sample_id'])
    covariate_data['sample_id'] = normalize_sample_id_series(covariate_data['sample_id'])

    # Inspect ID overlap before merging the three data tables
    diag_id_overlap(survival_data, feature_df, 'survival_data', 'feature_df')
    diag_id_overlap(survival_data, covariate_data, 'survival_data', 'covariate_data')
    diag_id_overlap(feature_df, covariate_data, 'feature_df', 'covariate_data')

    # ===== 5. 合并数据 =====
    merged_data = pd.merge(survival_data, feature_df, on='sample_id', how='inner')
    print(f"After survival x feature merge: {merged_data.shape}")
    merged_data = pd.merge(merged_data, covariate_data, on='sample_id', how='inner')
    print(f"After + covariate merge: {merged_data.shape}")

    if merged_data.empty:
        raise ValueError(
            "合并后 merged_data 为 0 行，无法继续。"
            "请检查三张表的 sample_id 是否一致（常见原因：12345 vs 12345.0、科学计数法、前后空格、前导0）。"
        )

    # ===== 6. Impute missing feature values =====
    imputer = SimpleImputer(strategy='median')
    merged_data[feature_names] = imputer.fit_transform(merged_data[feature_names])

    # ===== 7. Preprocess covariates =====
    print("\nProcessing covariate data...")

    categorical_covariates = ['age', 'bmi', 'ahi', 'gender', 'race', 'smoking',
                              'TST', 'hypertension', 'diabetes', 'hyperlipidemia','asthma','tst90']
    continuous_covariates = ['arousal_index', 'REM%', 'meanspo2', 'minsat']

    # Continuous covariates: median imputation
    continuous_imputer = SimpleImputer(strategy='median')
    for cov in continuous_covariates:
        if cov in merged_data.columns:
            missing_count = merged_data[cov].isnull().sum()
            if missing_count > 0:
                print(f"  Continuous covariate {cov}: filling {missing_count} missing values")
                merged_data[cov] = continuous_imputer.fit_transform(merged_data[[cov]]).ravel()

    # Categorical covariates: mode imputation followed by integer casting
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    for cov in categorical_covariates:
        if cov in merged_data.columns:
            missing_count = merged_data[cov].isnull().sum()
            if missing_count > 0:
                print(f"  Categorical covariate {cov}: filling {missing_count} missing values")
                merged_data[cov] = categorical_imputer.fit_transform(merged_data[[cov]]).ravel()
            merged_data[cov] = merged_data[cov].astype(int)

    updated_covariate_cols = [col for col in covariate_data.columns if col != 'sample_id']

    print("\nAvailable covariates:")
    for i, col in enumerate(updated_covariate_cols):
        data_type = "Categorical" if col in categorical_covariates else "Continuous"
        missing_count = merged_data[col].isnull().sum() if col in merged_data.columns else "N/A"
        print(f"{i + 1:2d}. {col} ({data_type}, Missing: {missing_count})")

    # ===== 8. 数据整体情况 =====
    print(f"\nData Statistics:")
    print(f"Sample size: {len(merged_data)}")
    print(f"Event rate: {merged_data['event'].mean():.2%}")
    print(f"Mean survival time: {merged_data['time'].mean():.1f} years")
    print(f"Min survival time: {merged_data['time'].min():.1f} years")
    print(f"Max survival time: {merged_data['time'].max():.1f} years")

    return merged_data, feature_names, updated_covariate_cols, categorical_covariates


# ==================== Univariable quintile Cox regression ====================
def univariate_quartile_cox_analysis(data, feature,
                                     output_dir='outputs'):
    """Run univariable Cox regression using quantile-based grouping."""
    if feature not in data.columns:
        print(f"Error: Feature '{feature}' not found in dataset.")
        return None

    print(f"\nPerforming univariate quintile Cox analysis for feature: {feature}")

    # Use tertiles for AHI/ODI and quintiles for the remaining features
    if feature in ['AHI', 'ODI']:
        print("Using tertile groups for AHI/ODI: <15, 15-30, >30")

        conditions = [
            (data[feature] < 15),
            (data[feature] >= 15) & (data[feature] <= 30),
            (data[feature] > 30)
        ]
        choices = [1, 2, 3]
        data['quartile_group'] = np.select(conditions, choices, default=1)

        range_descriptions = []
        min_val = data[feature].min()
        max_val = data[feature].max()

        group1_max = min(15, max_val)
        range_descriptions.append(f"{round(min_val, 2):.2f} to {round(group1_max, 2):.2f}")
        group2_min = max(15, min_val)
        group2_max = min(30, max_val)
        range_descriptions.append(f"{round(group2_min, 2):.2f} to {round(group2_max, 2):.2f}")
        group3_min = max(30, min_val)
        range_descriptions.append(f"{round(group3_min, 2):.2f} to {round(max_val, 2):.2f}")

        groups_to_analyze = [2, 3]
        all_groups = [1, 2, 3]

    else:
        quintiles = data[feature].quantile([0.2, 0.4, 0.6, 0.8])
        q1 = quintiles[0.2]
        q2 = quintiles[0.4]
        q3 = quintiles[0.6]
        q4 = quintiles[0.8]

        print(f"Quintile cut points: Q1={q1:.4f}, Q2={q2:.4f}, Q3={q3:.4f}, Q4={q4:.4f}")

        conditions = [
            (data[feature] <= q1),
            (data[feature] > q1) & (data[feature] <= q2),
            (data[feature] > q2) & (data[feature] <= q3),
            (data[feature] > q3) & (data[feature] <= q4),
            (data[feature] > q4)
        ]
        choices = [1, 2, 3, 4, 5]
        data['quartile_group'] = np.select(conditions, choices, default=1)

        range_descriptions = []
        min_val = data[feature].min()
        max_val = data[feature].max()
        range_descriptions.append(f"{min_val:.2f} to {q1:.2f}")
        range_descriptions.append(f"{q1:.2f} to {q2:.2f}")
        range_descriptions.append(f"{q2:.2f} to {q3:.2f}")
        range_descriptions.append(f"{q3:.2f} to {q4:.2f}")
        range_descriptions.append(f"{q4:.2f} to {max_val:.2f}")

        groups_to_analyze = [2, 3, 4, 5]
        all_groups = [1, 2, 3, 4, 5]

    data['quartile_group'] = data['quartile_group'].astype(int)

    cox_data = data[['time', 'event', 'quartile_group']].copy()
    cox_data['time'] = cox_data['time'].astype(float)
    cox_data['event'] = cox_data['event'].astype(int)
    cox_data['quartile_group'] = cox_data['quartile_group'].astype(int)

    cph = CoxPHFitter(penalizer=0.0)

    try:
        cph.fit(cox_data, duration_col='time', event_col='event', formula="C(quartile_group)")

        summary = cph.summary
        results = []

        for group in groups_to_analyze:
            group_label = f"C(quartile_group)[T.{group}]"
            if group_label in summary.index:
                coef = summary.loc[group_label, 'coef']
                exp_coef = summary.loc[group_label, 'exp(coef)']
                p_value = summary.loc[group_label, 'p']
                lower_ci = summary.loc[group_label, 'exp(coef) lower 95%']
                upper_ci = summary.loc[group_label, 'exp(coef) upper 95%']

                range_desc = range_descriptions[group - 1]

                results.append({
                    'feature': feature,
                    'quartile': group,
                    'range': range_desc,
                    'coef': round(coef, 2),  # Round to 4 decimal places
                    'HR': round(exp_coef, 2),  # Round to 4 decimal places
                    'HR_lower_95': round(lower_ci, 2),  # Round to 4 decimal places
                    'HR_upper_95': round(upper_ci, 2),  # Round to 4 decimal places
                    'p_value': p_value,
                    'Significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'
                })

                group_name = f"L{group}" if feature in ['AHI', 'ODI'] else f"Q{group}"
                print(
                    f"{group_name} vs {('L1' if feature in ['AHI', 'ODI'] else 'Q1')}: "
                    f"Range = {range_desc}, HR = {exp_coef:.4f} "
                    f"(95% CI: {lower_ci:.4f}-{upper_ci:.4f}), p-value = {p_value:.4f}"
                )

        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        kmf = KaplanMeierFitter()

        for group in all_groups:
            group_data = cox_data[cox_data['quartile_group'] == group]
            if len(group_data) > 0:
                group_prefix = "L" if feature in ['AHI', 'ODI'] else "Q"
                label = f"{group_prefix}{group}"
                kmf.fit(group_data['time'],
                        event_observed=group_data['event'],
                        label=label)
                kmf.plot_survival_function(ax=ax)

        group_type = "Tertile" if feature in ['AHI', 'ODI'] else "Quintile"
        plt.title(f'Survival by {feature} {group_type} Groups')
        plt.ylabel('Survival Probability')
        plt.xlabel('Time (years)')
        plt.legend(title=f'{group_type} Group')
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'univariate_km_curve_{feature}.png'), dpi=300)
        plt.close()

        if results:
            results_df = pd.DataFrame(results)
            # 创建 'HR (95% CI)' 列，格式为 "HR (95% CI) ***"
            results_df['HR (95% CI)'] = results_df.apply(
                lambda row: f"{row['HR']} ({row['HR_lower_95']}, {row['HR_upper_95']}) {row['Significance']}", axis=1
            )
            plt.figure(figsize=(10, 6))
            group_prefix = "L" if feature in ['AHI', 'ODI'] else "Q"
            y_labels = [f"{group_prefix}{int(q)}" for q in results_df['quartile']]

            plt.errorbar(
                x=results_df['HR'],
                y=y_labels,
                xerr=[results_df['HR'] - results_df['HR_lower_95'],
                      results_df['HR_upper_95'] - results_df['HR']],
                fmt='o',
                color='black',
                capsize=5
            )

            plt.axvline(x=1, color='red', linestyle='--')
            plt.xscale('log')
            group_type = "Tertile" if feature in ['AHI', 'ODI'] else "Quintile"
            plt.title(f'Hazard Ratios for {feature} {group_type}s (vs {group_prefix}1)')
            plt.xlabel('Hazard Ratio (HR) [log scale]')
            plt.ylabel(f'{group_type} Group')
            plt.grid(True, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'univariate_forest_plot_{feature}.png'), dpi=300)
            plt.close()

            results_df.to_csv(os.path.join(output_dir, f'univariate_results_{feature}.csv'), index=False)
            return results_df

    except Exception as e:
        print(f"Error in univariate quintile Cox analysis: {str(e)}")
        return None

# ==================== Variable transformation diagnostics ====================
def test_variable_transformations(data, target_feature,
                                  output_dir='outputs'):
    """Assess alternative transformations of the target feature.

    Martingale residual plots are generated to visually inspect approximate
    linearity with respect to the survival outcome.
    """
    print(f"\nTesting different transformations for {target_feature}...")

    transformations = {
        'Original': data[target_feature],
        'Log_Transform': np.log(data[target_feature] + 1),
        'Square_Root': np.sqrt(data[target_feature]),
        'Square': data[target_feature] ** 2
    }

    os.makedirs(output_dir, exist_ok=True)

    for name, transformed_data in transformations.items():
        try:
            temp_data = data[['time', 'event']].copy()
            temp_data['transformed_feature'] = transformed_data

            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(temp_data, duration_col='time', event_col='event', formula='transformed_feature')

            residuals = cph.compute_residuals(temp_data, 'martingale')

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=temp_data['transformed_feature'], y=residuals['martingale'])
            sns.regplot(x=temp_data['transformed_feature'], y=residuals['martingale'],
                        scatter=False, lowess=True, color='red')

            plt.title(f'Martingale Residuals - {name} {target_feature}')
            plt.xlabel(f'{name} {target_feature}')
            plt.ylabel('Martingale Residuals')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'residuals_{target_feature}_{name}.png'), dpi=300)
            plt.close()

            print(f"  {name} transformation analysis completed, plot saved")

        except Exception as e:
            print(f"  {name} transformation analysis failed: {str(e)}")

    return transformations

# ==================== Variance inflation factor (VIF) check ====================
def calculate_vif(df, features):
    print("\nCalculating Variance Inflation Factor (VIF) for multicollinearity check...")

    X = df[features].copy()
    X['intercept'] = 1

    vif_data = pd.DataFrame()
    vif_data['Feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(features))]

    print(vif_data)
    return vif_data

# ==================== Multivariable quintile Cox regression ====================
def multivariable_quartile_cox_analysis(
        data,
        target_feature,
        covariates=None,
        output_dir='outputs',
        transformation_type='Original'):
    """Run multivariable Cox regression with quantile-based exposure groups."""

    os.makedirs(output_dir, exist_ok=True)

    transformations = test_variable_transformations(data, target_feature, output_dir)

    valid_transformations = list(transformations.keys())
    if transformation_type not in valid_transformations:
        print(f"Warning: Invalid transformation type '{transformation_type}'. Using default 'Original'.")
        transformation_type = 'Original'

    print(f"\nUsing {transformation_type} transformation for {target_feature}")

    if covariates is None:
        covariates = ['age', 'gender', 'bmi', 'race', 'smoking', 'TST']

    categorical_covariates = ['gender', 'race', 'smoking']

    missing_cols = []
    if target_feature not in data.columns:
        missing_cols.append(target_feature)
    for cov in covariates:
        if cov not in data.columns:
            missing_cols.append(cov)

    if missing_cols:
        print(f"Error: The following columns are missing: {missing_cols}")
        print(f"Available columns: {[col for col in data.columns if col not in ['time', 'event', 'sample_id']]}")
        return None

    print(f"\nPerforming multivariable quintile Cox analysis for feature: {target_feature}")
    print(f"Covariates: {covariates}")

    feature_values = transformations[transformation_type].copy()

    use_temp_column = transformation_type != 'Original'
    if use_temp_column:
        temp_column_name = f"{target_feature}_{transformation_type}"
        data[temp_column_name] = feature_values
        feature_values = data[temp_column_name]
        print(f"Created temporary column '{temp_column_name}' for {transformation_type}")
    else:
        feature_values = data[target_feature]

    if target_feature in ['AHI', 'ODI']:
        print("Using tertile groups for AHI/ODI: <15, 15-30, >30")

        conditions = [
            (feature_values < 15),
            (feature_values >= 15) & (feature_values <= 30),
            (feature_values > 30)
        ]
        choices = [1, 2, 3]
        data['target_quartile'] = np.select(conditions, choices, default=1)

        range_descriptions = []
        min_val = feature_values.min()
        max_val = feature_values.max()

        group1_max = min(15, max_val)
        range_descriptions.append(f"{min_val:.1f} to {group1_max:.1f}")
        group2_min = max(15, min_val)
        group2_max = min(30, max_val)
        range_descriptions.append(f"{group2_min:.1f} to {group2_max:.1f}")
        group3_min = max(30, min_val)
        range_descriptions.append(f"{group3_min:.1f} to {max_val:.1f}")

        groups_to_analyze = [2, 3]
        all_groups = [1, 2, 3]

    else:
        quintiles = feature_values.quantile([0.2, 0.4, 0.6, 0.8])
        q1 = quintiles[0.2]
        q2 = quintiles[0.4]
        q3 = quintiles[0.6]
        q4 = quintiles[0.8]

        print(f"{transformation_type} Quintile cut points: Q1={q1:.4f}, Q2={q2:.4f}, "
              f"Q3={q3:.4f}, Q4={q4:.4f}")

        conditions = [
            (feature_values <= q1),
            (feature_values > q1) & (feature_values <= q2),
            (feature_values > q2) & (feature_values <= q3),
            (feature_values > q3) & (feature_values <= q4),
            (feature_values > q4)
        ]
        choices = [1, 2, 3, 4, 5]
        data['target_quartile'] = np.select(conditions, choices, default=1)

        groups_to_analyze = [2, 3, 4, 5]
        all_groups = [1, 2, 3, 4, 5]

    data['target_quartile'] = data['target_quartile'].astype(int)
    models = {
        'Model 0': {
            'description': 'Unadjusted (only target feature quintiles)',
            'covariates': ['target_quartile']
        },
        'Model 1': {
            'description': 'Adjusted for age, gender',
            # 注意：这里保留 target_quartile，才能输出各分位数的 HR
            'covariates': ['target_quartile'] + [
                'age', 'gender'
            ]
        },
        'Model 2': {
            'description': 'Model 1 + bmi+race+smoking',
            # 在 Model 1 基础上加 AHI，同时可选 covariate 形式的 ahi
            'covariates': ['target_quartile'] + [
                'age', 'gender', 'bmi', 'race',
                'smoking'
            ] + ([] if target_feature == 'AHI' else ['ahi'])
        },
        'Model 3': {
            'description': 'Model 2 + PSG',
            'covariates': ['target_quartile'] + [
                'age', 'gender', 'bmi', 'race',
                'smoking', 'AHI',
                'arousal_index', 'TST', 'tst90','REM', 'minsat'
            ] + ([] if target_feature == 'AHI' else ['ahi'])
        },
        'Model 4': {
            'description': 'Model 3+cardio-metabolic diseases',
            # 注意：这里保留 target_quartile，才能输出各分位数的 HR
            'covariates': ['target_quartile'] + [
                'age', 'gender', 'bmi', 'race',
                'smoking', 'AHI',
                'arousal_index', 'TST', 'tst90', 'REM', 'minsat',
                'hypertension', 'diabetes', 'hyperlipidemia', 'asthma'
            ]
        },
    }
    available_cols = set(data.columns)
    for model_name, model_info in models.items():
        model_info['covariates'] = [col for col in model_info['covariates'] if col in available_cols]
        print(f"{model_name}: {model_info['covariates']}")

    all_results = []

    for model_name, model_info in models.items():
        print(f"\n--- {model_name}: {model_info['description']} ---")

        analysis_cols = ['time', 'event'] + model_info['covariates']
        analysis_data = data[analysis_cols].copy()
        analysis_data = analysis_data.dropna()

        if len(analysis_data) == 0:
            print(f"Warning: No data available for {model_name}")
            continue

        print(f"Sample size for {model_name}: {len(analysis_data)}")

        try:
            formula_terms = []
            for cov in model_info['covariates']:
                if cov in categorical_covariates or cov == 'target_quartile':
                    formula_terms.append(f"C({cov})")
                else:
                    formula_terms.append(cov)

            formula = " + ".join(formula_terms)
            print(f"  Using formula: {formula}")

            continuous_vars = [cov for cov in model_info['covariates'] if
                               cov not in categorical_covariates and cov != 'target_quartile']
            if len(continuous_vars) > 1:
                vif_df = calculate_vif(analysis_data, continuous_vars)
                high_vif_vars = vif_df[vif_df['VIF'] > 5]
                if not high_vif_vars.empty:
                    print(f"  Warning: The following variables have VIF > 5: {list(high_vif_vars['Feature'])}")

            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(analysis_data, duration_col='time', event_col='event', formula=formula)

            results_for_model = []
            for group in groups_to_analyze:
                group_label = f"C(target_quartile)[T.{group}]"
                if group_label in cph.summary.index:
                    hr_row = cph.summary.loc[group_label]

                    result = {
                        'model': model_name,
                        'description': model_info['description'],
                        'feature': target_feature,
                        'quartile': group,
                        'HR': round(hr_row['exp(coef)'], 2),  # Round HR to 4 decimal places
                        'HR_lower_95': round(hr_row['exp(coef) lower 95%'], 2),  # Round HR lower CI to 4 decimal places
                        'HR_upper_95': round(hr_row['exp(coef) upper 95%'], 2),  # Round HR upper CI to 4 decimal places
                        'p_value': hr_row['p'],
                        'coef': round(hr_row['coef'], 2),
                        'se_coef': round(hr_row['se(coef)'], 2),
                        'n_samples': len(analysis_data),
                        'concordance': round(cph.concordance_index_, 2),
                        'Significance': '***' if hr_row['p'] < 0.001 else '**' if hr_row['p'] < 0.01 else '*' if hr_row['p'] < 0.05 else 'NS'
                    }
                    results_for_model.append(result)

                    group_prefix = "L" if target_feature in ['AHI', 'ODI'] else "Q"
                    print(f"{group_prefix}{group} vs {group_prefix}1: HR = {hr_row['exp(coef)']:.3f} "
                          f"(95% CI: {hr_row['exp(coef) lower 95%']:.3f}-{hr_row['exp(coef) upper 95%']:.3f}), "
                          f"p = {hr_row['p']:.4f}")

            print("  Checking proportional hazards assumption...")
            try:
                results_ph = proportional_hazard_test(cph, analysis_data)
                p_values = results_ph.summary['p'].apply(lambda x: f"{x:.4f}")
                print(f"  Proportional hazards test p-values: {p_values}")
            except Exception as e:
                print(f"  Proportional hazards test failed: {str(e)}")

            if 'tst90' in model_info['covariates'] and 'tst90' in cph.summary.index:
                tst90_hr = cph.summary.loc['tst90', 'exp(coef)']
                tst90_hr_10 = tst90_hr ** 0.1
                print(f"  HR for TST90 per 10% increase: {tst90_hr_10:.3f}")

            if model_name == 'Model 4':
                print("\nPlotting adjusted survival curves...")
                try:
                    # ---- helper formatters ----
                    def _format_p(p):
                        if p is None or pd.isna(p):
                            return "P=NA"
                        return "P<0.001" if p < 0.001 else f"P={p:.3f}"

                    def _format_hrci(summary_df, g):
                        # HR and 95%CI for C(target_quartile)[T.g] vs reference (group 1)
                        if g == 1:
                            return "HR=1.00 (Ref)"
                        row_lab = f"C(target_quartile)[T.{g}]"
                        if summary_df is None or row_lab not in summary_df.index:
                            return "HR=NA"
                        hr = summary_df.loc[row_lab, 'exp(coef)']
                        lo = summary_df.loc[row_lab, 'exp(coef) lower 95%']
                        hi = summary_df.loc[row_lab, 'exp(coef) upper 95%']
                        return f"HR={hr:.2f} ({lo:.2f}-{hi:.2f})"

                    # ---- Overall log-rank p-value (across all groups) ----
                    overall_lr_p = np.nan
                    try:
                        overall_lr_p = multivariate_logrank_test(
                            analysis_data['time'],
                            analysis_data['target_quartile'],
                            analysis_data['event']
                        ).p_value
                    except Exception:
                        overall_lr_p = np.nan

                    plt.figure(figsize=(12, 8))
                    ax = plt.subplot(111)

                    group_prefix = "L" if target_feature in ['AHI', 'ODI'] else "Q"
                    group_type = "Tertile" if target_feature in ['AHI', 'ODI'] else "Quintile"

                    for quartile in all_groups:
                        pred_data = analysis_data.copy()
                        pred_data['target_quartile'] = quartile

                        surv_func = cph.predict_survival_function(pred_data)
                        mean_surv = surv_func.mean(axis=1)

                        hr_txt = _format_hrci(cph.summary, quartile)
                        cox_p_txt = ""
                        if quartile == max(all_groups):
                            # Keep Cox p-value only for the highest group (e.g., Q5)
                            row_lab = f"C(target_quartile)[T.{quartile}]"
                            pval = np.nan
                            try:
                                if hasattr(cph, 'summary') and (cph.summary is not None) and (row_lab in cph.summary.index) and ('p' in cph.summary.columns):
                                    pval = float(cph.summary.loc[row_lab, 'p'])
                            except Exception:
                                pval = np.nan
                            cox_p_txt = f", {_format_p(pval)}"
                        label = f"{group_prefix}{quartile}: {hr_txt}{cox_p_txt}"
                        ax.plot(mean_surv.index, mean_surv.values, label=label, lw=2)


                    # Overall log-rank P (one value for all curves)
                    ax.text(
                        0.98, 0.98,
                        f"Overall log-rank {_format_p(overall_lr_p)}",
                        transform=ax.transAxes,
                        ha='right',
                        va='top',
                        fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.6)
                    )
                    plt.title(f'Adjusted Survival Curves - {target_feature} {group_type} Groups (Model 4)')
                    plt.ylabel('Survival Probability')
                    plt.xlabel('Time (years)')
                    ax.set_xlim(0, 12)
                    # Put legend outside to avoid overlap with curves when labels are long
                    plt.legend(loc='lower left', ncol=1, fontsize=9, frameon=True)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'adjusted_survival_curves_{target_feature}.png'),
                                dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                    print(f"  Adjusted survival curves saved")
                except Exception as e:
                    print(f"  Failed to plot adjusted survival curves: {str(e)}")

            all_results.extend(results_for_model)

        except Exception as e:
            print(f"Error fitting {model_name}: {str(e)}")
            continue

    if all_results:
        results_df = pd.DataFrame(all_results)
        # 创建 'HR (95% CI)' 列，格式为 "HR (95% CI) ***"
        results_df['HR (95% CI)'] = results_df.apply(
            lambda row: f"{row['HR']} ({row['HR_lower_95']}, {row['HR_upper_95']}) {row['Significance']}", axis=1
        )

        summary_table = results_df.copy()
        summary_table['HR (95% CI)'] = summary_table.apply(
            lambda x: f"{x['HR']:.2f} ({x['HR_lower_95']:.2f}-{x['HR_upper_95']:.2f})", axis=1
        )
        summary_table['p-value'] = summary_table['p_value'].apply(
            lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001"
        )
        summary_table['Significance'] = summary_table['p_value'].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'NS'
        )

        print("\n" + "=" * 80)
        print("SUMMARY OF MULTIVARIABLE QUINTILE COX REGRESSION ANALYSIS")
        print("=" * 80)
        display_cols = ['model', 'quartile', 'HR (95% CI)', 'p-value',
                        'Significance', 'n_samples', 'concordance']
        print(summary_table[display_cols].to_string(index=False))

        plt.figure(figsize=(12, 8))

        model_labels = []
        hr_values = []
        lower_ci_values = []
        upper_ci_values = []

        for _, row in results_df.iterrows():
            model_labels.append(f"{row['model']} - Q{int(row['quartile'])}")
            hr_values.append(row['HR'])
            lower_ci_values.append(row['HR_lower_95'])
            upper_ci_values.append(row['HR_upper_95'])

        y_pos = np.arange(len(model_labels))

        plt.errorbar(
            x=hr_values,
            y=y_pos,
            xerr=[np.array(hr_values) - np.array(lower_ci_values),
                  np.array(upper_ci_values) - np.array(hr_values)],
            fmt='o',
            color='black',
            capsize=5,
            markersize=8
        )

        plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Reference (HR=1)')
        plt.yticks(y_pos, model_labels)
        plt.xlabel('Hazard Ratio (HR)')
        plt.ylabel('Model and Quintile')
        plt.title(f'Multivariable Quintile Cox Analysis: {target_feature}')
        plt.grid(True, axis='x', alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'multivariable_quartile_forest_{target_feature}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        results_df.to_csv(os.path.join(output_dir,
                                       f'multivariable_quartile_results_{target_feature}.csv'),
                          index=False)

        return results_df
    else:
        print("No valid results obtained from any model")
        return None
def create_hb_hr_groups(data,
                        hb_col='hypoxic_burden',
                        hr_col='HR surge',
                        group_col_name='HB_HR_group'):
    """Create four joint hypoxic burden/HR surge groups by median split."""

    df = data.copy()

    # Medians
    hb_median = df[hb_col].median()
    hr_median = df[hr_col].median()
    print(f"{hb_col} median = {hb_median:.4f}")
    print(f"{hr_col} median = {hr_median:.4f}")

    # Binary indicators
    df['HB_high'] = (df[hb_col] > hb_median).astype(int)       # 0 = low, 1 = high
    df['HR_high'] = (df[hr_col] > hr_median).astype(int)     # 0 = weak, 1 = strong

    # 4 groups
    def _assign_group(row):
        if row['HB_high'] == 0 and row['HR_high'] == 0:
            return 1  # Low HB & weak HR (reference)
        elif row['HB_high'] == 0 and row['HR_high'] == 1:
            return 2  # Low HB & strong HR
        elif row['HB_high'] == 1 and row['HR_high'] == 0:
            return 3  # High HB & weak HR
        else:
            return 4  # High HB & strong HR

    df[group_col_name] = df.apply(_assign_group, axis=1).astype(int)

    group_label_map = {
        1: "Low HB & Low HR surge (Ref)",
        2: "Low HB & High HR surge",
        3: "High HB & Low HR surge",
        4: "High HB & High HR surge"
    }

    print("\nGroup sizes:")
    for g in range(1, 5):
        n = (df[group_col_name] == g).sum()
        print(f"  Group {g}: {group_label_map[g]}  (n = {n})")

    return df, group_label_map, hb_median, hr_median


def plot_hb_hr_survival_curves(
        data,
        hb_col='hypoxic_burden',
        hr_col='HR surge',
        output_path="outputs/HB_HR_survival_curves.png",
        adjusted=False,
        covariates=None):
    """Plot survival curves for four joint hypoxic burden/HR surge groups."""
    df, group_label_map, hb_median, hr_median = create_hb_hr_groups(
        data, hb_col=hb_col, hr_col=hr_col, group_col_name='HB_HR_group'
    )

    # Keep essential columns
    km_data = df[['time', 'event', 'HB_HR_group']].copy()
    km_data = km_data.dropna(subset=['time', 'event', 'HB_HR_group'])
    km_data['event'] = km_data['event'].astype(int)
    km_data['HB_HR_group'] = km_data['HB_HR_group'].astype(int)

    def _format_hrci(summary, g: int) -> str:
        """Format per-group HR and 95% CI from Cox summary (vs Group 1)."""
        if g == 1:
            return "Ref"
        lab = f"C(HB_HR_group)[T.{g}]"
        if summary is None or lab not in summary.index:
            return "HR=NA"
        hr = summary.loc[lab, 'exp(coef)']
        lo = summary.loc[lab, 'exp(coef) lower 95%']
        hi = summary.loc[lab, 'exp(coef) upper 95%']
        return f"HR={hr:.2f} ({lo:.2f}-{hi:.2f})"


    # ---------- helpers ----------
    def _format_p(p: float) -> str:
        if p is None or pd.isna(p):
            return "P=NA"
        return "P<0.001" if p < 0.001 else f"P={p:.3f}"

    # Pairwise log-rank p-values vs Group 1 (for curve-end annotation)
    pairwise_lr_p = {}
    ref = km_data[km_data['HB_HR_group'] == 1]
    for g in [2, 3, 4]:
        sub = km_data[km_data['HB_HR_group'] == g]
        if len(ref) == 0 or len(sub) == 0:
            pairwise_lr_p[g] = np.nan
            continue
        lr_res = logrank_test(
            ref['time'], sub['time'],
            event_observed_A=ref['event'],
            event_observed_B=sub['event']
        )
        pairwise_lr_p[g] = lr_res.p_value

    # Cox Model 0 p-values (legend; vs Group 1)
    cox_data = km_data.copy()
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='time', event_col='event', formula="C(HB_HR_group)")
    cox_summary = cph.summary
    cox_p = {1: np.nan}
    for g in [2, 3, 4]:
        row_label = f"C(HB_HR_group)[T.{g}]"
        cox_p[g] = cox_summary.loc[row_label, 'p'] if row_label in cox_summary.index else np.nan
    def _format_hrci_from_summary(summary, g: int) -> str:
        """Format per-group HR and 95% CI from Cox summary (vs Group 1)."""
        if g == 1:
            return "Ref"
        row_label = f"C(HB_HR_group)[T.{g}]"
        if summary is None or row_label not in summary.index:
            return "HR=NA"
        hr = summary.loc[row_label, 'exp(coef)']
        lo = summary.loc[row_label, 'exp(coef) lower 95%']
        hi = summary.loc[row_label, 'exp(coef) upper 95%']
        return f"HR={hr:.2f} ({lo:.2f}-{hi:.2f})"


    # ============ 1. Kaplan–Meier curves ============
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    colors = ['C0', 'C1', 'C2', 'C3']  # matplotlib default colors

    for i, g in enumerate([1, 2, 3, 4]):
        sub = km_data[km_data['HB_HR_group'] == g]
        if len(sub) == 0:
            continue

        # Legend label with per-group Cox HR (95%CI) + p-value (vs Group 1)
        hr_txt = _format_hrci_from_summary(cox_summary, g)
        if g == 1:
            label = f"{group_label_map[g]} (n={len(sub)})"
        else:
            label = f"{group_label_map[g]} (n={len(sub)}), {hr_txt}, {_format_p(cox_p.get(g, np.nan))}"


        kmf.fit(durations=sub['time'], event_observed=sub['event'], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i], lw=2)

        # Curve-end annotation with pairwise Log-rank p (vs Group 1)
        if g != 1:
            sf = kmf.survival_function_.iloc[:, 0]
            x_end = float(sf.index.max())
            y_end = float(sf.iloc[-1])
            ax.annotate(
                f"Log-rank {_format_p(pairwise_lr_p.get(g, np.nan))}",
                xy=(x_end, y_end),
                xytext=(6, 0),
                textcoords="offset points",
                ha='left',
                va='center',
                fontsize=10,
                color=colors[i]
            )

    ax.set_xlabel("Follow-up time (years)", fontsize=12)
    ax.set_ylabel("CVD survival probability", fontsize=12)
    ax.set_title("CVD survival by hypoxic burden and HR surge", fontsize=14)
    ax.grid(True, alpha=0.3)

    legend_title = (
        f"Groups (cut at medians: HB = {hb_median:.2f}, HR surge = {hr_median:.2f})"
    )
    leg = ax.legend(loc='best', fontsize=9, title=legend_title, frameon=True, borderpad=0.4, labelspacing=0.3, handlelength=1.8, handletextpad=0.5)
    if leg is not None:
        leg.get_title().set_fontsize(10)

    # ============ 2. Overall multivariate log-rank test (4-group) ============
    logrank_results = multivariate_logrank_test(
        km_data['time'],
        km_data['HB_HR_group'],
        event_observed=km_data['event']
    )
    logrank_p = logrank_results.p_value

    # ============ 3. Optional: keep a caption with HR vs. Group 1 (as before) ============
    def _star(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'NS'

    hr_lines = []
    hr_lines.append("Reference: Group 1 (Low HB & Low HR surge)")
    for g in [2, 3, 4]:
        row_label = f"C(HB_HR_group)[T.{g}]"
        if row_label in cox_summary.index:
            hr = cox_summary.loc[row_label, 'exp(coef)']
            lower = cox_summary.loc[row_label, 'exp(coef) lower 95%']
            upper = cox_summary.loc[row_label, 'exp(coef) upper 95%']
            pval = cox_summary.loc[row_label, 'p']
            stars = _star(pval)
            p_txt = "P<0.001" if pval < 0.001 else f"P={pval:.4f}"
            hr_lines.append(
                f"Group {g} ({group_label_map[g]}): HR = {hr:.2f} "
                f"(95% CI {lower:.2f}–{upper:.2f}), {p_txt} {stars}"
            )
        else:
            hr_lines.append(f"Group {g}: HR = NA")

    hr_text = "\n".join(hr_lines)

    plt.tight_layout(rect=[0, 0.25, 1, 1])  # leave space for caption
    plt.gcf().text(
        0.01, 0.02,
        hr_text,
        ha='left',
        va='bottom',
        fontsize=10,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )

    # Save and show
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"\nSurvival curves figure saved to: {output_path}")
    print("\nCox Model 0 results (HB_HR_group):")
    print(cox_summary)

    return {
        "hb_median": hb_median,
        "hr_median": hr_median,
        "cox_summary": cox_summary,
        "logrank_p": logrank_p
    }


# 新增函数：输出 Model0 & Model4 的 HR、10年绝对风险，并绘图
def _interp_survival_at(surv_series, t):
    """Interpolate the survival function and return S(t)."""
    times = np.array(surv_series.index, dtype=float)
    surv_vals = np.array(surv_series.values, dtype=float)
    if t <= times.min():
        return float(surv_vals[0])
    if t >= times.max():
        return float(surv_vals[-1])
    return float(np.interp(t, times, surv_vals))

def plot_and_compare_model0_model4(
        data,
        hb_col='hypoxic_burden',
        hr_col='HR surge',
        out_dir="outputs",
        absolute_risk_time=10.0):
    """
    输出：
      - 未调整的 KM 曲线（Model 0）保存为: out_dir/HB_HR_KM_model0.png
      - 完全调整的（Model 4）调整后平均生存曲线保存为: out_dir/HB_HR_adjusted_model4.png
      - 汇总表格 CSV: out_dir/HB_HR_group_summary_model0_model4.csv

    额外增强（按需求）：
      - Model 0 与 Model 4 的图例中，为每个组（vs Group 1）标注 p 值；若 p<0.001 显示为 P<0.001。
        * Model 0：p 值来自 Cox（仅分组）模型
        * Model 4：p 值来自 Cox（分组+协变量）模型
      - 两张图每条曲线末端标注“Log-rank p”（与 Group 1 的两两 log-rank 检验）；若 p<0.001 显示为 P<0.001。
    """

    os.makedirs(out_dir, exist_ok=True)

    def _format_p(p: float) -> str:
        if p is None or pd.isna(p):
            return "P=NA"
        return "P<0.001" if p < 0.001 else f"P={p:.3f}"

    def _format_hrci(summary, g: int) -> str:
        """返回 'Ref' 或 'HR=1.23 (0.95–1.60)'（vs 组1）。"""
        if g == 1:
            return "Ref"
        lab = f"C(HB_HR_group)[T.{g}]"
        if summary is None or lab not in summary.index:
            return "HR=NA"
        hr = summary.loc[lab, 'exp(coef)']
        lo = summary.loc[lab, 'exp(coef) lower 95%']
        hi = summary.loc[lab, 'exp(coef) upper 95%']
        return f"HR={hr:.2f} ({lo:.2f}–{hi:.2f})"

    # 1) 根据中位数建组（和你原来实现一致）
    df_groups, group_label_map, hb_med, hr_med = create_hb_hr_groups(
        data, hb_col=hb_col, hr_col=hr_col, group_col_name='HB_HR_group'
    )

    km_data = df_groups[['time', 'event', 'HB_HR_group']].dropna(subset=['time', 'event', 'HB_HR_group']).copy()
    km_data['event'] = km_data['event'].astype(int)
    km_data['HB_HR_group'] = km_data['HB_HR_group'].astype(int)

    # Pairwise log-rank p-values vs Group 1 (for curve-end annotation; used in both plots)
    pairwise_lr_p0 = {}
    ref0 = km_data[km_data['HB_HR_group'] == 1]
    for g in [2, 3, 4]:
        sub = km_data[km_data['HB_HR_group'] == g]
        if len(ref0) == 0 or len(sub) == 0:
            pairwise_lr_p0[g] = np.nan
            continue
        pairwise_lr_p0[g] = logrank_test(
            ref0['time'], sub['time'],
            event_observed_A=ref0['event'],
            event_observed_B=sub['event']
        ).p_value

    # =========== Model 0: Unadjusted ============
    # Fit Cox Model 0 to get p-values in legend (vs group1)
    cph0 = CoxPHFitter()
    cph0.fit(km_data, duration_col='time', event_col='event', formula="C(HB_HR_group)")
    summary0 = cph0.summary
    cox_p0 = {1: np.nan}
    for g in [2, 3, 4]:
        lab = f"C(HB_HR_group)[T.{g}]"
        cox_p0[g] = summary0.loc[lab, 'p'] if lab in summary0.index else np.nan

    # Kaplan-Meier plotting
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    colors = ['C0', 'C1', 'C2', 'C3']
    for i, g in enumerate([1, 2, 3, 4]):
        sub = km_data[km_data['HB_HR_group'] == g]
        if len(sub) == 0:
            continue

        hr_txt = _format_hrci(summary0, g)
        if g == 1:
            label = f"{group_label_map[g]} (n={len(sub)})"
        else:
            label = f"{group_label_map[g]} (n={len(sub)}), {hr_txt}, {_format_p(cox_p0.get(g, np.nan))}"

        kmf.fit(durations=sub['time'], event_observed=sub['event'], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i])


    ax.set_xlabel("Follow-up time (years)")
    ax.set_xlim(0, 12)
    ax.set_ylabel("CVD survival probability")
    ax.set_title("Unadjusted KM: CVD survival by HB & HR surge groups")

    legend_title = f"Groups (medians: HB={hb_med:.2f}, HR={hr_med:.2f})"
    leg = ax.legend(loc='lower left', ncol=1, fontsize=9, title=legend_title,
                    frameon=True, borderpad=0.4, labelspacing=0.3,
                    handlelength=1.8, handletextpad=0.5)
    if leg is not None:
        leg.get_title().set_fontsize(10)

    # overall log-rank p-value（多组）
    lr0 = multivariate_logrank_test(km_data['time'], km_data['HB_HR_group'], event_observed=km_data['event'])
    lr_p0 = lr0.p_value

    # 仅标注一个总体 Log-rank 检验 P（四组整体比较），放在右上角
    ax.text(
        0.98, 0.98,
        f"Overall log-rank {_format_p(lr_p0)}",
        transform=ax.transAxes,
        ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.6)
    )
    plt.show()
    plt.tight_layout()
    model0_png = os.path.join(out_dir, "HB_HR_KM_model0.png")
    plt.savefig(model0_png, dpi=400, bbox_inches='tight')
    plt.close()

    # 10-year absolute risk (unadjusted KM) for each group: use Kaplan-Meier survival at t=absolute_risk_time
    km_unadj_surv_at_t = {}
    for g in [1, 2, 3, 4]:
        sub = km_data[km_data['HB_HR_group'] == g]
        if len(sub) == 0:
            km_unadj_surv_at_t[g] = np.nan
            continue
        kf = KaplanMeierFitter()
        kf.fit(sub['time'], event_observed=sub['event'])
        s_at_t = _interp_survival_at(kf.survival_function_.iloc[:, 0], absolute_risk_time)
        km_unadj_surv_at_t[g] = s_at_t

    # =========== Model 4: Fully adjusted ============
    model4_covs = [
        'age', 'gender', 'bmi', 'race', 'smoking', 'AHI',
        'arousal_index', 'TST', 'tst90', 'REM', 'minsat',
        'hypertension', 'diabetes', 'hyperlipidemia', 'asthma'
    ]
    model4_covs = [c for c in model4_covs if c in df_groups.columns]

    categorical_covs = [c for c in ['gender', 'race', 'smoking', 'hypertension', 'diabetes', 'hyperlipidemia', 'asthma']
                        if c in df_groups.columns]

    formula_terms = ["C(HB_HR_group)"]
    for cov in model4_covs:
        if cov in categorical_covs:
            formula_terms.append(f"C({cov})")
        else:
            formula_terms.append(cov)
    formula4 = " + ".join(formula_terms)

    model4_df = df_groups[['time', 'event', 'HB_HR_group'] + model4_covs].dropna().copy()
    model4_df['event'] = model4_df['event'].astype(int)
    model4_df['HB_HR_group'] = model4_df['HB_HR_group'].astype(int)

    summary4 = None
    cph4 = None
    model4_png = os.path.join(out_dir, "HB_HR_adjusted_model4.png")
    adjusted_surv_at_t = {g: np.nan for g in [1, 2, 3, 4]}

    # pairwise log-rank p-values on the Model4 subset (still unadjusted log-rank; used for curve-end annotation)
    pairwise_lr_p4 = {g: np.nan for g in [2, 3, 4]}
    if len(model4_df) > 0:
        ref4 = model4_df[model4_df['HB_HR_group'] == 1]
        for g in [2, 3, 4]:
            sub = model4_df[model4_df['HB_HR_group'] == g]
            if len(ref4) == 0 or len(sub) == 0:
                continue
            pairwise_lr_p4[g] = logrank_test(
                ref4['time'], sub['time'],
                event_observed_A=ref4['event'],
                event_observed_B=sub['event']
            ).p_value

    if len(model4_df) > 0:
        cph4 = CoxPHFitter()
        try:
            cph4.fit(model4_df, duration_col='time', event_col='event', formula=formula4)
            summary4 = cph4.summary
        except Exception as e:
            print("Warning: fitting Model 4 failed:", e)
            summary4 = None

        # Adjusted average survival curves by group + legend p-values (from Model 4 Cox) + curve-end log-rank p
        if cph4 is not None and summary4 is not None:
            # p-values for legend (vs group1)
            cox_p4 = {1: np.nan}
            for g in [2, 3, 4]:
                lab = f"C(HB_HR_group)[T.{g}]"
                cox_p4[g] = summary4.loc[lab, 'p'] if lab in summary4.index else np.nan

            plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)

            for i, g in enumerate([1, 2, 3, 4]):
                pred_df = model4_df.copy()
                pred_df['HB_HR_group'] = g

                surv_df = cph4.predict_survival_function(pred_df)
                mean_surv = surv_df.mean(axis=1)

                # legend label
                n_g = int((df_groups['HB_HR_group'] == g).sum())
                hr_txt = _format_hrci(summary4, g)
                if g == 1:
                    label = f"{group_label_map[g]} (n={n_g})"
                else:
                    label = f"{group_label_map[g]} (n={n_g}), {hr_txt}, {_format_p(cox_p4.get(g, np.nan))}"

                ax.plot(mean_surv.index, mean_surv.values, label=label)

                # survival at t for absolute risk
                s_at_t = _interp_survival_at(mean_surv, absolute_risk_time)
                adjusted_surv_at_t[g] = s_at_t


            ax.set_xlabel("Follow-up time (years)")
            ax.set_xlim(0, 12)
            ax.set_ylabel("Adjusted survival probability")
            ax.set_title("Adjusted survival curves (Model 4) - average predicted survival by group")
            leg = ax.legend(loc='lower left', ncol=1, fontsize=9, frameon=True, borderpad=0.4, labelspacing=0.3, handlelength=1.8, handletextpad=0.5)
            if leg is not None:
                leg.get_title().set_fontsize(10)

            # Optional: overall log-rank on the Model4 subset
            try:
                lr4 = multivariate_logrank_test(model4_df['time'], model4_df['HB_HR_group'], event_observed=model4_df['event'])
            except Exception:
                pass

            plt.tight_layout()
            plt.show()
            plt.savefig(model4_png, dpi=400, bbox_inches='tight')
            plt.close()

    # =========== 整理结果表格 ============
    rows = []
    for g in [1, 2, 3, 4]:
        n = int((df_groups['HB_HR_group'] == g).sum())

        # Model0 HR (95%CI)
        if g == 1:
            model0_hr_str = "Ref"
        else:
            label0 = f"C(HB_HR_group)[T.{g}]"
            if label0 in summary0.index:
                hr0 = summary0.loc[label0, 'exp(coef)']
                lo0 = summary0.loc[label0, 'exp(coef) lower 95%']
                hi0 = summary0.loc[label0, 'exp(coef) upper 95%']
                model0_hr_str = f"{hr0:.2f} ({lo0:.2f}-{hi0:.2f})"
            else:
                model0_hr_str = "NA"

        # Model4 HR (95%CI)
        if g == 1:
            model4_hr_str = "Ref"
        else:
            label4 = f"C(HB_HR_group)[T.{g}]"
            if summary4 is not None and label4 in summary4.index:
                hr4 = summary4.loc[label4, 'exp(coef)']
                lo4 = summary4.loc[label4, 'exp(coef) lower 95%']
                hi4 = summary4.loc[label4, 'exp(coef) upper 95%']
                model4_hr_str = f"{hr4:.2f} ({lo4:.2f}-{hi4:.2f})"
            else:
                model4_hr_str = "NA"

        # 绝对风险（absolute_risk_time 年）: 优先 Model4 调整后估计；否则未调整 KM
        if not np.isnan(adjusted_surv_at_t[g]):
            abs_risk = 1.0 - adjusted_surv_at_t[g]
        else:
            s_unadj = km_unadj_surv_at_t.get(g, np.nan)
            abs_risk = (1.0 - s_unadj) if not np.isnan(s_unadj) else np.nan

        rows.append({
            'Group': g,
            'Label': group_label_map[g],
            'n': n,
            'Model0_HR_95CI': model0_hr_str,
            'Model4_HR_95CI': model4_hr_str,
            f'AbsoluteRisk_{absolute_risk_time:g}yr': abs_risk
        })

    results_df = pd.DataFrame(rows)
    results_df[f'AbsoluteRisk_{absolute_risk_time:g}yr_pct'] = results_df[f'AbsoluteRisk_{absolute_risk_time:g}yr'].apply(
        lambda x: f"{100 * x:.1f}%" if (pd.notnull(x)) else "NA"
    )

    csv_path = os.path.join(out_dir, "HB_HR_group_summary_model0_model4.csv")
    results_df.to_csv(csv_path, index=False)

    print("\nGroup summary (saved to):", csv_path)
    print(results_df[['Group', 'Label', 'n', 'Model0_HR_95CI', 'Model4_HR_95CI',
                      f'AbsoluteRisk_{absolute_risk_time:g}yr_pct']].to_string(index=False))

    return {
        'results_df': results_df,
        'model0_png': model0_png,
        'model4_png': model4_png if os.path.exists(model4_png) else None,
        'csv': csv_path,
        'cox_model0_summary': summary0,
        'cox_model4_summary': summary4,
        'hb_median': hb_med,
        'hr_median': hr_med,
        'logrank_p': lr_p0
    }



# ==================== Main execution workflow ====================
def main():
    print("Starting CVD Survival Analysis...")

    # 1. Load and preprocess analysis data
    print("\nStep 1: Loading and preparing data...")
    data, feature_columns, covariate_cols, categorical_covariates = load_and_prepare_data()
    results = plot_and_compare_model0_model4(
        data,
        hb_col='hypoxic_burden',
        hr_col='HR surge',
        out_dir="outputs",
        absolute_risk_time=10.0
    )

    print("\nAvailable features:")
    for i, feature in enumerate(feature_columns):
        print(f"{i + 1:2d}. {feature}")

    print("\nAvailable covariates:")
    for i, cov in enumerate(covariate_cols):
        print(f"{i + 1:2d}. {cov}")

    # 2. Configure the output directory
    output_dir = os.environ.get("OUTPUT_DIR", "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")

    # 3. Interactive command-line analysis loop
    while True:
        print("\n" + "=" * 60)
        print("Please select analysis mode:")
        print("1. Univariate quintile analysis for all features")
        print("2. Multivariable quintile analysis for specific feature")
        print("3. Exit program")

        mode_input = input("Please enter option number (1/2/3): ").strip()

        if mode_input == '3':
            print("Program ended, thank you for using!")
            break
        elif mode_input == '1':
            print(f"\nStarting analysis of all {len(feature_columns)} features...")

            all_results = []

            for i, feature in enumerate(feature_columns):
                print(f"\nProgress: {i + 1}/{len(feature_columns)} - Analyzing: {feature}")
                try:
                    results_df = univariate_quartile_cox_analysis(data, feature, output_dir)
                    if results_df is not None and not results_df.empty:
                        all_results.append(results_df)

                        significant = any(results_df['p_value'] < 0.05)
                        status = "✓ Significant" if significant else "✗ Not significant"
                        print(f"{feature} analysis completed: {status}")
                    else:
                        print(f"{feature} analysis failed or no results")

                except Exception as e:
                    print(f"Error analyzing feature {feature}: {str(e)}")
                    import traceback
                    traceback.print_exc()

            if all_results:
                all_results_df = pd.concat(all_results, ignore_index=True)
                all_results_path = os.path.join(output_dir, "all_features_univariate_quartile_results.csv")
                all_results_df.to_csv(all_results_path, index=False)
                print(f"\nAll features analysis completed! Results saved to: {all_results_path}")

                significant_results = all_results_df[all_results_df['p_value'] < 0.05]
                if not significant_results.empty:
                    print("\nSignificant results summary (p < 0.05):")
                    for _, row in significant_results.iterrows():
                        print(f"{row['feature']} Q{int(row['quartile'])}: "
                              f"HR = {row['HR']:.4f}, p = {row['p_value']:.4f}")
                else:
                    print("\nNo significant results found (p < 0.05)")
            else:
                print("\nAll features analysis completed, but no valid results obtained")

        elif mode_input == '2':
            print("\nPlease enter the target feature name:")
            feature_input = input("Feature name: ").strip()

            if feature_input not in feature_columns:
                print(f"Error: Feature '{feature_input}' does not exist!")
                print("Please select a valid feature name from the list above")
                continue

            print("\nUse default covariate settings?")
            print("Default covariates: age, gender, bmi, race, smoking, TST, ahi, odi, minsat, tst90")
            use_default = input("Use default settings? (y/n): ").strip().lower()

            if use_default == 'y':
                covariates = None
            else:
                print("\nPlease enter covariate names (separated by commas):")
                print("Available covariates:", ", ".join(covariate_cols))
                cov_input = input("Covariates: ").strip()
                covariates = [cov.strip() for cov in cov_input.split(',')]

            print("\nPlease select variable transformation type:")
            print("1. Original")
            print("2. Log Transform")
            print("3. Square Root")
            print("4. Square")

            trans_choice = input("Please enter option number (1-4): ").strip()

            trans_map = {
                '1': 'Original',
                '2': 'Log_Transform',
                '3': 'Square_Root',
                '4': 'Square'
            }

            transformation_type = trans_map.get(trans_choice, 'Original')

            specific_output_dir = os.path.join(output_dir, f"{feature_input}_{transformation_type}")
            os.makedirs(specific_output_dir, exist_ok=True)

            print(f"\nPerforming multivariable quintile Cox analysis: "
                  f"{feature_input} ({transformation_type})")
            try:
                results_df = multivariable_quartile_cox_analysis(
                    data,
                    feature_input,
                    covariates,
                    specific_output_dir,
                    transformation_type=transformation_type
                )
                if results_df is not None and not results_df.empty:
                    print(f"\n{feature_input} ({transformation_type}) multivariable quintile analysis completed!")

                    best_model_concordance = results_df.groupby('model')['concordance'].mean()
                    best_model = best_model_concordance.idxmax()
                    print(
                        f"Best model discrimination: {best_model} "
                        f"(Average Concordance = {best_model_concordance[best_model]:.3f})"
                    )

                    significant = any(results_df['p_value'] < 0.05)
                    if significant:
                        print("✓ Statistically significant association found (p < 0.05)")
                    else:
                        print("✗ No statistically significant association found")

            except Exception as e:
                print(f"Error in multivariable quintile analysis for feature {feature_input}: {str(e)}")
                import traceback
                traceback.print_exc()

        else:
            print("Invalid option, please select again")

        print("\nAnalysis completed! You can continue with other analysis modes")


if __name__ == "__main__":
    main()