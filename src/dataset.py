# DIR 对应：imdb-wiki-dir/datasets.py::IMDBWIKI 的样本权重准备与返回接口。
# https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/datasets.py
# 本项目适配：从 SkyFinder manifest 读取图片和温度，以 temperature 替代 age。
# 权重算法在 utils.py 中重写；本文件负责训练集筛选及图片、标签、权重对齐。
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from .config import BATCH_SIZE, IMAGE_SIZE, MANIFEST_PATH, RANDOM_SEED
    from .utils import compute_lds_weights
except ImportError:
    from config import BATCH_SIZE, IMAGE_SIZE, MANIFEST_PATH, RANDOM_SEED
    from utils import compute_lds_weights


PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size=IMAGE_SIZE):
    # 本项目改动：在 Dataset 外构建变换，使用 ImageNet 预训练模型的归一化参数。
    # 官方 IMDBWIKI 使用均值/标准差 0.5 和训练随机裁剪；这里不做随机裁剪。
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform



class SkyFinderDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        transform=None,
        use_lds=False,
        lds_bin_width=1.0,
        lds_kernel_size=5,
        lds_sigma=2.0,
    ):
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_DIR / manifest_path

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest CSV was not found: {manifest_path}. "
                "Run prepare_data.py first."
            )

        split = split.strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        df = pd.read_csv(manifest_path, low_memory=False)
        required_columns = {"image_path", "temperature", "split"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Manifest CSV is missing required columns: {missing_columns}"
            )

        df["split"] = df["split"].astype(str).str.strip().str.lower()
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

        self.transform = transform
        self.split = split
        self.df = df[df["split"] == split].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No samples were found for split: {split}")

        if self.df["temperature"].isna().any():
            raise ValueError(f"Split {split} contains invalid temperatures")

        self.use_lds = use_lds
        self.weights = None
        self.lds_info = None

        if self.use_lds:
            if self.split != "train":
                raise ValueError(
                    "LDS weights should only be enabled for the training split."
                )

            # 沿用官方训练数据加权的流程：只统计训练集，验证/测试不参与密度估计。
            # 本项目改动：调用独立权重函数，缓存 tensor 和分析信息，保持行顺序一致。
            temperatures = self.df["temperature"].to_numpy()
            weights, info = compute_lds_weights(
                temperatures,
                bin_width=lds_bin_width,
                kernel_size=lds_kernel_size,
                sigma=lds_sigma,
            )
            self.weights = torch.from_numpy(weights)
            self.lds_info = info

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path

        if not image_path.is_file():
            raise FileNotFoundError(f"Image was not found: {image_path}")

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        temperature = torch.tensor(
            float(row["temperature"]),
            dtype=torch.float32,
        )

        if self.use_lds:
            # DIR 对应：IMDBWIKI.__getitem__ 返回图片、标签和对应样本权重。
            # 本项目改动：温度与权重为标量 tensor，组 batch 后为 [batch_size]。
            weight = self.weights[index]
            return image, temperature, weight

        # 本项目接口：仅 LDS 训练返回三项；baseline、验证和测试仍返回两项。
        # 官方 IMDBWIKI 在未加权时也返回第三项，值为 1。
        return image, temperature


def create_dataloaders(
    manifest_path=MANIFEST_PATH,
    batch_size=BATCH_SIZE,
    num_workers=0,
    use_lds=False,
    lds_bin_width=1.0,
    lds_kernel_size=5,
    lds_sigma=2.0,
):
    # 本项目封装：集中建立三个 DataLoader，只向 train_dataset 传入 LDS 参数。
    train_transform, eval_transform = build_transforms()

    train_dataset = SkyFinderDataset(
        manifest_path,
        split="train",
        transform=train_transform,
        use_lds=use_lds,
        lds_bin_width=lds_bin_width,
        lds_kernel_size=lds_kernel_size,
        lds_sigma=lds_sigma,
    )
    val_dataset = SkyFinderDataset(
        manifest_path,
        split="val",
        transform=eval_transform,
    )
    test_dataset = SkyFinderDataset(
        manifest_path,
        split="test",
        transform=eval_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def run_smoke_test():
    # 本项目检查：读取每个 split 的一个 batch，确认图片和温度接口可用。
    loaders = create_dataloaders(num_workers=0)
    split_names = ["train", "val", "test"]

    for split_name, loader in zip(split_names, loaders):
        images, temperatures = next(iter(loader))
        print(
            f"{split_name}: samples={len(loader.dataset)}, "
            f"images={tuple(images.shape)}, "
            f"temperatures={tuple(temperatures.shape)}"
        )


if __name__ == "__main__":
    run_smoke_test()
