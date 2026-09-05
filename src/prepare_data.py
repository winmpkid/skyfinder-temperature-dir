# 本项目新增的数据准备流程，不属于 DIR 官方代码。
# 将 SkyFinder 图片与气象表匹配、清洗后按日期划分，生成 data/manifest.csv。
# DIR 官方年龄实验读取已有 split 的 CSV；本项目在这里构建温度任务及时间划分。

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_DIR / "datasets"
IMAGES_DIR = DATASETS_DIR / "images"
METADATA_PATH = DATASETS_DIR / "complete_table_with_mcr.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MISSING_TEMPERATURE_VALUES = {-9999.0, -999.0, 999.0, 9999.0}

# 本项目数据修复：排除训练中发现的两个损坏 JPEG，避免 truncated image 错误。
CORRUPTED_FILENAMES = {
    "20130617_101231.jpg",
    "20131225_171222.jpg",
}


def find_images(images_dir: Path) -> pd.DataFrame:
    # 本项目新增：扫描下载的图片，用 camera_id 和 filename 确定样本身份。
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    records = []
    for image_path in images_dir.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # SkyFinder 本地目录约定：images/<CamId>/<Filename>。
        camera_id = image_path.parent.name.strip()
        if not camera_id.isdigit():
            continue

        records.append(
            {
                "filename": image_path.name.strip(),
                "camera_id": camera_id,
                # Store a project-relative path so the generated manifest remains
                # usable after the repository is moved or cloned elsewhere.
                "image_path": str(image_path.relative_to(PROJECT_DIR)),
            }
        )

    image_df = pd.DataFrame.from_records(records)
    if image_df.empty:
        raise RuntimeError(f"No image files were found under: {images_dir}")

    duplicate_mask = image_df.duplicated(
        subset=["camera_id", "filename"], keep=False
    )
    if duplicate_mask.any():
        print(
            "Warning: found "
            f"{int(duplicate_mask.sum()):,} duplicate image records; "
            "keeping the first copy."
        )
        image_df = image_df.drop_duplicates(
            subset=["camera_id", "filename"], keep="first"
        )

    return image_df.reset_index(drop=True)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    # 本项目新增：读取气象表 TempM 作为摄氏温度标签，过滤无效标签和时间戳。
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")

    required_columns = [
        "Filename",
        "CamId",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Min",
        "TempM",
    ]

    metadata = pd.read_csv(
        metadata_path,
        usecols=required_columns,
        low_memory=False,
    ).rename(
        columns={
            "Filename": "filename",
            "CamId": "camera_id",
            "TempM": "temperature",
        }
    )

    metadata["filename"] = metadata["filename"].astype(str).str.strip()
    metadata["camera_id"] = (
        metadata["camera_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    metadata["temperature"] = pd.to_numeric(
        metadata["temperature"], errors="coerce"
    )

    original_count = len(metadata)
    invalid_temperature = (
        metadata["temperature"].isna()
        | metadata["temperature"].isin(MISSING_TEMPERATURE_VALUES)
        | (metadata["temperature"] <= -100.0)
        | (metadata["temperature"] >= 100.0)
    )
    metadata = metadata.loc[~invalid_temperature].copy()

    print(f"Metadata rows: {original_count:,}")
    print(f"Rows with valid temperature: {len(metadata):,}")
    print(f"Removed invalid temperatures: {int(invalid_temperature.sum()):,}")

    for column in ["Year", "Month", "Day", "Hour", "Min"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")

    metadata["timestamp"] = pd.to_datetime(
        {
            "year": metadata["Year"],
            "month": metadata["Month"],
            "day": metadata["Day"],
            "hour": metadata["Hour"],
            "minute": metadata["Min"],
        },
        errors="coerce",
    )

    invalid_timestamp_count = int(metadata["timestamp"].isna().sum())
    if invalid_timestamp_count:
        print(f"Removed invalid timestamps: {invalid_timestamp_count:,}")
        metadata = metadata.dropna(subset=["timestamp"])

    metadata = metadata[
        ["filename", "camera_id", "temperature", "timestamp"]
    ].drop_duplicates(subset=["camera_id", "filename"], keep="first")

    return metadata.reset_index(drop=True)


def assign_temporal_split(
    camera_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    # 本项目划分方案：每个摄像头按日期先后分 train/val/test，同一天不跨 split。
    camera_df = camera_df.sort_values("timestamp").copy()
    camera_df["capture_date"] = camera_df["timestamp"].dt.strftime("%Y-%m-%d")

    unique_dates = np.array(sorted(camera_df["capture_date"].unique()))
    number_of_dates = len(unique_dates)
    if number_of_dates < 3:
        camera_id = camera_df["camera_id"].iloc[0]
        raise ValueError(
            f"Camera {camera_id} has only {number_of_dates} unique dates; "
            "at least three are required for train/val/test."
        )

    number_train = max(1, int(number_of_dates * train_ratio))
    number_val = max(1, int(number_of_dates * val_ratio))

    # 本项目边界处理：日期较少时仍为验证集和测试集各保留至少一天。
    if number_train + number_val >= number_of_dates:
        number_train = number_of_dates - 2
        number_val = 1

    split_by_date = {}
    for capture_date in unique_dates[:number_train]:
        split_by_date[capture_date] = "train"
    for capture_date in unique_dates[
        number_train : number_train + number_val
    ]:
        split_by_date[capture_date] = "val"
    for capture_date in unique_dates[number_train + number_val :]:
        split_by_date[capture_date] = "test"

    camera_df["split"] = camera_df["capture_date"].map(split_by_date)
    if camera_df["split"].isna().any():
        raise RuntimeError("At least one capture date was not assigned to a split.")

    return camera_df.drop(columns="capture_date")


def create_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    # 本项目新增：匹配图片与标签、排除损坏文件，再应用时间划分。
    print(f"Project directory: {PROJECT_DIR}")
    print("\nFinding downloaded images...")
    image_df = find_images(IMAGES_DIR)
    print(f"Images found: {len(image_df):,}")

    print("\nLoading metadata...")
    metadata_df = load_metadata(METADATA_PATH)

    manifest = image_df.merge(
        metadata_df,
        on=["camera_id", "filename"],
        how="inner",
        validate="one_to_one",
    )

    matched_keys = manifest[["camera_id", "filename"]].drop_duplicates()
    unmatched_images = image_df.merge(
        matched_keys,
        on=["camera_id", "filename"],
        how="left",
        indicator=True,
    )
    unmatched_images = unmatched_images.loc[
        unmatched_images["_merge"] == "left_only",
        ["image_path", "filename", "camera_id"],
    ]

    corrupted_mask = manifest["filename"].isin(CORRUPTED_FILENAMES)
    print(f"Removed corrupted images: {int(corrupted_mask.sum()):,}")
    manifest = manifest.loc[~corrupted_mask].copy()

    print(f"Matched images with valid temperature: {len(manifest):,}")
    print(f"Images without a valid metadata match: {len(unmatched_images):,}")
    if manifest.empty:
        raise RuntimeError("No images could be matched to valid metadata rows.")

    split_frames = [
        assign_temporal_split(camera_df)
        for _, camera_df in manifest.groupby("camera_id", sort=True)
    ]
    manifest = pd.concat(split_frames, ignore_index=True)

    manifest = manifest.sort_values(
        ["camera_id", "timestamp", "filename"]
    ).reset_index(drop=True)
    manifest = manifest[
        [
            "image_path",
            "filename",
            "camera_id",
            "temperature",
            "timestamp",
            "split",
        ]
    ]

    return manifest, unmatched_images.reset_index(drop=True)


def print_summary(manifest: pd.DataFrame) -> None:
    # 本项目检查：输出各摄像头及 split 的数量、温度范围，辅助发现分布差异。
    print("\nCounts by camera and split:")
    print(
        manifest.groupby(["camera_id", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )

    print("\nTemperature summary by split (degrees Celsius):")
    print(
        manifest.groupby("split")["temperature"]
        .agg(["count", "min", "mean", "median", "max"])
        .reindex(["train", "val", "test"])
        .round(2)
    )


def main() -> None:
    manifest, _ = create_manifest()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)

    print_summary(manifest)
    print(f"\nManifest saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
