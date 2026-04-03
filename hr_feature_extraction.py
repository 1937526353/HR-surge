"""
Heart-rate feature extraction from EDF recordings.

This script reads EDF files, extracts the H.R channel, filters respiratory
events using an annotation workbook (NH.xlsx), and computes patient-level
heart-rate response features for downstream survival analysis.

Open-source notes:
- Local absolute paths were replaced with environment variables or relative paths.
- Only de-identified inputs should be used in public releases.
"""

# Open-source sanitized version
# Local absolute paths have been replaced with environment variables or relative paths.

import os
import re
import csv
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm


# ============================================================
#                   日志工具（可断点续跑）
# ============================================================

def setup_logger(log_path: str) -> logging.Logger:
    """Configure a logger that writes both to console and file.

    The log file is stored alongside the output feature table so that
    interrupted runs can be resumed and audited later.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("feature_extract")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 避免重复 handler（交互环境/重复运行）

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_log_path_from_output(output_path: str) -> str:
    """Return the log path corresponding to the output CSV file.

    Example
    -------
    hypoxia_features_all_patients.csv -> hypoxia_features_all_patients.log
    """
    base, _ = os.path.splitext(output_path)
    return base + ".log"


def load_processed_patient_ids(output_path: str) -> set[str]:
    """Load patient IDs that have already been written to the output table.

    This is the key mechanism enabling resumable processing after interruption.
    """
    if not os.path.exists(output_path):
        return set()

    try:
        # 只读 patient_id 列，快一些
        df = pd.read_csv(output_path, usecols=["patient_id"], dtype=str, encoding="utf-8-sig")
        return set(df["patient_id"].dropna().astype(str).tolist())
    except Exception:
        # 文件可能为空或损坏，尽量不阻塞运行
        return set()


def safe_append_row_csv(output_path: str, row: dict, fieldnames: list[str], fsync_every_row: bool = True):
    """Append a single row to a CSV file safely.

    When ``fsync_every_row`` is enabled, each completed patient record is flushed
    to disk to reduce the risk of data loss during long batch jobs.
    """
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        if fsync_every_row:
            os.fsync(f.fileno())


# ============================================================
#                 EDF / NPY 读写工具（仅 HR）
# ============================================================

def extract_patient_id(filename: str) -> str:
    """Extract the patient ID from the EDF filename.

    Example
    -------
    ``shhs1-200091_hr_sao2.edf`` -> ``200091``
    """
    match = re.search(r'shhs1-(\d+)_hr_sao2\.edf', filename)
    if match:
        return match.group(1)
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('-')
    if len(parts) > 1:
        id_part = parts[1].split('_')[0]
        return id_part
    return ''.join(filter(str.isdigit, filename))


def get_hr_npy_path(edf_path: str, cache_dir: str) -> str:
    """根据 EDF 文件名，返回 HR 的 npy 缓存路径"""
    base = os.path.splitext(os.path.basename(edf_path))[0]
    return os.path.join(cache_dir, base + "_hr.npy")


def load_edf_hr_signal(edf_path: str) -> np.ndarray:
    """Read an EDF file and extract the heart-rate channel (``H.R``)."""
    try:
        with pyedflib.EdfReader(edf_path) as f:
            hr_index = None
            for i, label in enumerate(f.getSignalLabels()):
                if label.strip() == 'H.R':
                    hr_index = i
                    break

            if hr_index is None:
                raise ValueError("HR 信号通道未找到 (label != 'H.R')")

            hr_signal = f.readSignal(hr_index)

            # 采样率不是 1Hz 也能跑，只做提示
            if f.getSampleFrequency(hr_index) != 1:
                # 不要 print，避免大量 IO 变慢；由上层 logger 统一记录
                pass

            return np.asarray(hr_signal, dtype=np.float32)

    except Exception as e:
        raise RuntimeError(f"EDF文件读取错误: {str(e)}") from e


def load_hr_from_cache(edf_path: str, cache_dir: str) -> np.ndarray:
    """从 npy 缓存中加载 HR"""
    hr_npy = get_hr_npy_path(edf_path, cache_dir)
    if not os.path.exists(hr_npy):
        raise FileNotFoundError(f"缓存不存在：{hr_npy}，请先运行 EDF->NPY 预转换")
    # mmap_mode='r'：节省内存（特别是并行时）
    return np.load(hr_npy, mmap_mode='r')


def convert_single_edf_to_npy(edf_path: str, cache_dir: str):
    """
    单个 EDF 转成 hr.npy
    如果已有缓存则跳过
    """
    hr_npy = get_hr_npy_path(edf_path, cache_dir)
    if os.path.exists(hr_npy):
        return

    os.makedirs(cache_dir, exist_ok=True)
    try:
        hr_signal = load_edf_hr_signal(edf_path)
        np.save(hr_npy, hr_signal.astype(np.float32, copy=False))
    except Exception:
        # 子进程里不建议依赖全局 logger，避免 handler 复制
        filename = os.path.basename(edf_path)
        print(f"[ERROR] 转换 {filename} 失败")


def convert_all_edf_to_npy(folder_path: str, cache_dir: str, max_workers=None):
    """
    预先将文件夹中所有 EDF 转成 HR 的 NumPy 缓存（多进程 + 进度条）
    """
    edf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.edf')
    ]

    print(f"\n========== EDF -> NPY(HR) 预转换：共 {len(edf_files)} 个文件 ==========\n")

    if not edf_files:
        print("没有找到 EDF 文件，预转换跳过")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(convert_single_edf_to_npy, path, cache_dir): path for path in edf_files}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="预转换进度"):
            pass

    print("\n[OK] EDF -> NPY(HR) 预转换完成\n")


# ============================================================
#                  HR 预处理（保留插值）
# ============================================================

def preprocess_hr_signal(hr_signal: np.ndarray) -> np.ndarray:
    """Preprocess the heart-rate signal.

    Steps include removing physiologically implausible values and linearly
    interpolating short missing segments.
    """
    # 1) Remove negative values
    hr_signal = np.where(hr_signal < 0, np.nan, hr_signal).astype(float)

    # 2) Remove physiologically implausible HR values
    hr_low_threshold = 40
    hr_high_threshold = 180
    hr_abnormal_mask = (hr_signal < hr_low_threshold) | (hr_signal > hr_high_threshold)
    hr_signal[hr_abnormal_mask] = np.nan

    # 3) Linearly interpolate short missing gaps
    max_gap_length = 100
    hr_series = pd.Series(hr_signal)
    hr_signal = hr_series.interpolate(
        method='linear',
        limit=max_gap_length,
        limit_direction='both'
    ).values

    return hr_signal


# ============================================================
#          事件读取（NH.xlsx：呼吸暂停/低通气 + 睡眠期过滤）
# ============================================================

def load_sleep_intervals(all_sheets: dict, pid_str: str):
    """Load valid sleep-stage intervals for one subject.

    Only N1, N2, N3, N4, and REM epochs are retained for event filtering.
    """
    sleep_intervals = []
    sleep_df = all_sheets.get('睡眠阶段', None)
    if sleep_df is None:
        return sleep_intervals

    sleep_stage_names = ['N1期', 'N2期', 'N3期', 'N4期', 'REM期']

    for _, row in sleep_df.iterrows():
        filename = row.get('文件名称')
        match = re.search(r'(\d{6})', str(filename))
        if not match or match.group(1) != pid_str:
            continue

        if row.get('睡眠阶段') not in sleep_stage_names:
            continue

        try:
            s = int(float(row['开始时间_秒']))
            e = int(float(row['结束时间_秒']))
        except Exception:
            continue

        if e > s:
            sleep_intervals.append((s, e))

    return sleep_intervals


def in_sleep(start: int, end: int, sleep_intervals) -> bool:
    """Return whether an event overlaps with any retained sleep interval."""
    if not sleep_intervals:
        return True
    for ss, ee in sleep_intervals:
        if not (end <= ss or start >= ee):
            return True
    return False


def load_events_from_annotation(annotation_file: str,
                                patient_id: str,
                                ignore_first_seconds: int = 100,
                                only_sleep: bool = True):
    """Load respiratory events for a single subject from the annotation workbook.

    Events are read from apnea and hypopnea sheets. By default, only events
    overlapping valid sleep stages are retained, and very early events are
    excluded to avoid unstable recording onset periods.
    """
    if not os.path.exists(annotation_file):
        return []

    try:
        all_sheets = pd.read_excel(annotation_file, sheet_name=None)
    except Exception:
        return []

    pid_str = str(patient_id).zfill(6)
    sleep_intervals = load_sleep_intervals(all_sheets, pid_str) if only_sleep else []

    event_sheet_names = ['呼吸暂停事件', '低通气事件_1', '低通气事件_2']
    events = []

    for sheet_name in event_sheet_names:
        df = all_sheets.get(sheet_name, None)
        if df is None:
            continue

        for _, row in df.iterrows():
            filename = row.get('文件名称')
            match = re.search(r'(\d{6})', str(filename))
            if not match or match.group(1) != pid_str:
                continue

            try:
                start = int(float(row['开始时间_秒']))
                end = int(float(row['结束时间_秒']))
            except Exception:
                continue

            if start < ignore_first_seconds:
                continue
            if end <= start:
                continue
            if only_sleep and not in_sleep(start, end, sleep_intervals):
                continue

            events.append((start, end))

    return events


# ============================================================
#            三个目标特征：HR surge / HR surge 30 / HR surge 60
# ============================================================

def format_feature_value(value):
    """Round floating-point feature values to two decimals for output."""
    if isinstance(value, float):
        return round(value, 2)
    return value


def extract_response_features_for_event(hr_signal: np.ndarray, event):
    """Compute event-level HR response features relative to a pre-event baseline.

    Three features are derived:
    - HR surge
    - HR surge 30
    - HR surge 60
    """
    start, end = event
    if start <= 0:
        return None

    pre_window = 60
    pre_start = max(0, start - pre_window)
    pre_hr = hr_signal[pre_start:start]
    if pre_hr.size == 0 or np.all(np.isnan(pre_hr)):
        return None
    pre_hr_mean = float(np.nanmean(pre_hr))
    if np.isnan(pre_hr_mean):
        return None

    # 事件窗口裁剪到信号长度内
    n = len(hr_signal)
    end_clip = min(end, n)
    end30 = min(end + 30, n)
    end60 = min(end + 60, n)

    event_hr = hr_signal[start:end_clip]
    event_hr30 = hr_signal[start:end30]
    event_hr60 = hr_signal[start:end60]

    # 任一窗口全 NaN 则跳过（避免 np.nanmax 抛错）
    if (event_hr.size == 0 or np.all(np.isnan(event_hr)) or
            event_hr30.size == 0 or np.all(np.isnan(event_hr30)) or
            event_hr60.size == 0 or np.all(np.isnan(event_hr60))):
        return None

    event_hr_max = float(np.nanmax(event_hr))
    event_hr_max30 = float(np.nanmax(event_hr30))
    event_hr_max60 = float(np.nanmax(event_hr60))

    response_amp = event_hr_max - pre_hr_mean
    response_amp30 = event_hr_max30 - pre_hr_mean
    response_amp60 = event_hr_max60 - pre_hr_mean

    return response_amp, response_amp30, response_amp60


def extract_all_features(edf_path: str,
                         annotation_file: str,
                         patient_id: str,
                         use_cache: bool = False,
                         cache_dir: str | None = None):
    """Extract patient-level HR response features from one EDF file.

    Event-level features are computed first and then averaged across all valid
    respiratory events for the subject.
    """
    # 载入 HR
    if use_cache and cache_dir is not None:
        hr_signal = load_hr_from_cache(edf_path, cache_dir)
    else:
        hr_signal = load_edf_hr_signal(edf_path)

    hr_signal = preprocess_hr_signal(hr_signal)

    # Load events (default: retain only events occurring during sleep)
    events = load_events_from_annotation(
        annotation_file=annotation_file,
        patient_id=patient_id,
        ignore_first_seconds=100,
        only_sleep=True
    )
    if not events:
        return None

    s1 = s2 = s3 = 0.0
    cnt = 0
    for ev in events:
        feats = extract_response_features_for_event(hr_signal, ev)
        if feats is None:
            continue
        a, a30, a60 = feats
        s1 += a
        s2 += a30
        s3 += a60
        cnt += 1

    if cnt == 0:
        return None

    avg_features = {
        'HR surge': format_feature_value(float(s1 / cnt)),
        'HR surge 30': format_feature_value(float(s2 / cnt)),
        'HR surge 60': format_feature_value(float(s3 / cnt)),
    }
    return avg_features


# ============================================================
#              多进程包装 & EDF 文件夹处理（可断点续跑）
# ============================================================

def process_single_edf(file_path: str,
                       annotation_file: str,
                       use_cache: bool,
                       cache_dir: str | None):
    """Worker function for multiprocessing-based EDF processing."""
    patient_id = extract_patient_id(os.path.basename(file_path))
    try:
        feats = extract_all_features(
            file_path,
            annotation_file=annotation_file,
            patient_id=patient_id,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        if feats is None:
            return {'patient_id': patient_id}  # 无事件/无有效窗口
        feats['patient_id'] = patient_id
        return feats
    except Exception as e:
        return {'patient_id': patient_id, 'error': str(e)}


def process_edf_folder(folder_path: str,
                       output_path: str,
                       annotation_file: str,
                       use_cache: bool = True,
                       cache_dir: str | None = None,
                       max_workers=None,
                       fsync_every_row: bool = True):
    """Process all EDF files in a folder and write results incrementally.

    This function supports resumable execution, multiprocessing, per-subject
    logging, and immediate CSV persistence after each completed subject.
    """
    log_path = get_log_path_from_output(output_path)
    logger = setup_logger(log_path)

    logger.info("========== 任务开始 ==========")
    logger.info(f"EDF文件夹: {folder_path}")
    logger.info(f"输出特征表: {output_path}")
    logger.info(f"日志文件: {log_path}")
    logger.info(f"标注文件: {annotation_file}")
    logger.info(f"use_cache={use_cache}, cache_dir={cache_dir}, max_workers={max_workers}")

    edf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.edf')
    ]
    logger.info(f"在文件夹中找到 {len(edf_files)} 个 EDF 文件")

    if not edf_files:
        logger.info("没有 EDF 文件，直接返回")
        return None

    # 读取已处理 patient_id，用于断点续跑
    processed_ids = load_processed_patient_ids(output_path)
    if processed_ids:
        logger.info(f"检测到已存在输出文件，已处理患者数: {len(processed_ids)}（将自动跳过）")

    # 过滤待处理 EDF（同一 patient_id 只保留一个文件，避免重复）
    to_process = []
    seen_run = set()
    for p in edf_files:
        pid = extract_patient_id(os.path.basename(p))
        if pid in processed_ids:
            continue
        if pid in seen_run:
            continue
        seen_run.add(pid)
        to_process.append(p)

    logger.info(f"本次需要处理 EDF 数: {len(to_process)}")

    if not to_process:
        logger.info("没有需要处理的文件（全部已完成），结束")
        return None

    # Fixed CSV schema to keep row structure consistent across runs
    fieldnames = ['patient_id', 'HR surge', 'HR surge 30', 'HR surge 60']

    # 先写一条 RUN START，方便断电后追踪
    logger.info(f"RUN_START | total={len(to_process)}")

    # Parallel computation; the main process writes CSV rows and logs incrementally
    ok_cnt = 0
    skip_cnt = 0
    err_cnt = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_single_edf, file_path, annotation_file, use_cache, cache_dir): file_path
            for file_path in to_process
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="处理EDF文件(并行)"):
            file_path = futures[fut]
            filename = os.path.basename(file_path)
            pid = extract_patient_id(filename)

            try:
                res = fut.result()
            except Exception as e:
                # 极少数情况下 fut.result() 本身失败
                res = {'patient_id': pid, 'error': str(e)}

            # 统一成固定列
            row = {k: 0 for k in fieldnames}
            row['patient_id'] = str(pid)

            if res is None:
                # 理论不会发生
                err_cnt += 1
                logger.error(f"FAILED | patient_id={pid} | file={filename} | reason=res is None")
                safe_append_row_csv(output_path, row, fieldnames, fsync_every_row=fsync_every_row)
                continue

            if 'error' in res:
                err_cnt += 1
                logger.error(f"ERROR | patient_id={pid} | file={filename} | error={res.get('error')}")
                # 仍然写入一行（特征为0），确保断点续跑不会反复卡在同一文件
                safe_append_row_csv(output_path, row, fieldnames, fsync_every_row=fsync_every_row)
                continue

            # 无有效特征（无事件/全NaN等）
            if (len(res.keys()) == 1 and 'patient_id' in res):
                skip_cnt += 1
                logger.info(f"NO_FEATURE | patient_id={pid} | file={filename}")
                safe_append_row_csv(output_path, row, fieldnames, fsync_every_row=fsync_every_row)
                continue

            # 正常结果
            ok_cnt += 1
            row['HR surge'] = res.get('HR surge', 0)
            row['HR surge 30'] = res.get('HR surge 30', 0)
            row['HR surge 60'] = res.get('HR surge 60', 0)

            safe_append_row_csv(output_path, row, fieldnames, fsync_every_row=fsync_every_row)
            logger.info(f"OK | patient_id={pid} | file={filename} | "
                        f"HR surge={row['HR surge']} "
                        f"HR surge 30={row['HR surge 30']} "
                        f"HR surge 60={row['HR surge 60']}")

    logger.info(f"RUN_END | ok={ok_cnt} | no_feature={skip_cnt} | error={err_cnt}")
    logger.info("========== 任务结束 ==========")

    # 返回 DataFrame（可选；大数据时不建议）
    try:
        df_out = pd.read_csv(output_path, encoding="utf-8-sig")
        return df_out
    except Exception:
        return None


# ============================================================
#                        主入口（路径不变）
# ============================================================

if __name__ == "__main__":
    # 路径（保持不变）
    folder_path = os.environ.get("EDF_FOLDER", "data/edf")         # EDF folder
    output_path = os.environ.get("FEATURE_OUTPUT_PATH", "outputs/hypoxia_features_all_patients.csv")
    annotation_file = os.environ.get("ANNOTATION_FILE", "data/NH.xlsx")                 # annotation workbook
    cache_dir = os.environ.get("CACHE_DIR", "data/npy_cache")          # EDF->NPY cache directory

    # Step 1: Optional EDF -> NPY(HR) pre-conversion for faster repeated runs
    convert_all_edf_to_npy(folder_path, cache_dir, max_workers=None)

    # Step 2: Extract features using cached HR signals, multiprocessing, and resumable output writing
    process_edf_folder(
        folder_path,
        output_path,
        annotation_file=annotation_file,
        use_cache=True,
        cache_dir=cache_dir,
        max_workers=None,
        fsync_every_row=True  # 更抗断电（更慢）。若更追求速度可改 False
    )
