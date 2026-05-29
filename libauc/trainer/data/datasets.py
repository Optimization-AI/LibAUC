import logging
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from libauc.utils import ImbalancedDataGenerator
from ogb.graphproppred import PygGraphPropPredDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    """Wraps an existing dataset to return ``(image, target, index)`` tuples.

    Optionally selects a single column from multi-label targets when
    ``class_id`` is given.
    """

    def __init__(self, dataset, class_id=None):
        self.dataset = dataset
        self.targets = self._load_targets()
        if len(self.targets.shape) == 2 and class_id is not None:
            self.targets = self.targets[:, class_id: class_id + 1]

    def _load_targets(self):
        targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        return np.array(targets).astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Unpack task_id if TriSampler is being used
        task_id = None
        if isinstance(idx, (tuple, list)):
            idx, task_id = idx

        image, _ = self.dataset[idx]
        target = self.targets[idx]
        return image, target, (idx, task_id) if task_id is not None else idx


class ImageDataset(Dataset):
    """In-memory image dataset with train/test augmentation presets."""

    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
        self.images  = images.astype(np.uint8)
        self.targets = targets
        self.mode    = mode
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((crop_size, crop_size), padding=None),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image  = Image.fromarray(self.images[idx].astype('uint8'))
        target = self.targets[idx]
        image  = self.transform_train(image) if self.mode == 'train' else self.transform_test(image)
        return image, target, idx


class ChemicalDataset(Dataset):
    """OGB molecular graph dataset filtered to a single task column."""

    def __init__(self, dataset, class_id):
        indices  = dataset.indices()
        assert len(dataset.data.y.shape) == 2
        y        = dataset.data.y[indices, class_id]
        not_nan  = ~np.isnan(y.numpy())
        self.targets = y[not_nan].float()
        self.dataset = dataset[not_nan]

        pos   = int((self.targets == 1).sum())
        total = len(self.targets)
        logger.info(f"[ChemicalDataset] positive: {pos} | rate: {pos / total:.4f}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx], int(idx)


class GraphDataset(PygGraphPropPredDataset):
    """PygGraphPropPredDataset with integer-index support."""

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int64)):
            item     = self.get(self.indices()[idx])
            item.idx = torch.LongTensor([idx])
            return item
        else:
            return self.index_select(idx)


class MedicalImageCSVDataset(Dataset):
    """
    General-purpose CSV-backed medical image dataset.

    Expects a CSV (or DataFrame) with at least an image path column and a
    binary label column.  Image paths may be relative (resolved against
    ``image_root``) or absolute.

    Args:
        source:     Path to a metadata CSV **or** a ``pandas.DataFrame`` with
                    the required columns already loaded.  Passing a DataFrame
                    avoids writing temporary files.
        image_root: Directory that relative image paths are resolved against.
                    Ignored for absolute paths.
        image_col:  Column name containing the image filename / path.
        label_col:  Column name containing the binary label (0 / 1).
        transform:  torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        source,
        image_root: str,
        image_col: str,
        label_col: str,
        transform,
    ):
        if isinstance(source, pd.DataFrame):
            df = source.dropna(subset=[label_col]).reset_index(drop=True)
        else:
            df = pd.read_csv(source).dropna(subset=[label_col]).reset_index(drop=True)

        self.image_root  = image_root
        self.image_col   = image_col
        self.transform   = transform
        self.targets     = df[label_col].to_numpy().astype(np.float32)
        self.image_paths = df[image_col].tolist()

        pos   = int((self.targets == 1).sum())
        total = len(self.targets)
        logger.info(f"[MedicalImageCSVDataset] positive: {pos} | rate: {pos / total:.4f}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        rel  = self.image_paths[idx]
        path = rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx].reshape(-1), idx


# ---------------------------------------------------------------------------
# Shared transform factories
# ---------------------------------------------------------------------------

def _medical_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _medical_test_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# OGB graph dataset helpers
# ---------------------------------------------------------------------------

def _safe_import_pyg_globals():
    """Register PyG safe globals for torch.serialization when available."""
    import torch.serialization
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals(
            [DataEdgeAttr, DataTensorAttr, GlobalStorage]
        )
    except ImportError:
        pass


# OGB dataset name → (class_id to use for binary classification)
_OGB_CLASS_IDS: dict[str, int] = {
    "ogbg-molhiv":  0,
    "ogbg-moltox21": 0,
    "ogbg-molmuv":  1,
    "ogbg-molpcba": 0,
}


def _load_ogbg(name: str, class_id: int, root_path: str, splits: list):
    """Load any OGB graph-property-prediction dataset as binary classification.

    Args:
        name:      OGB dataset name (e.g. ``"ogbg-molhiv"``).
        class_id:  Column index in ``y`` to use as the binary label.
        root_path: Root directory for dataset caching.
        splits:    Evaluation splits to return (e.g. ``["val", "test"]``).

    Returns:
        (train_dataset, eval_datasets)
    """
    _safe_import_pyg_globals()
    dataset   = GraphDataset(name=name, root=root_path)
    split_idx = dataset.get_idx_split()
    train_dataset = ChemicalDataset(dataset[split_idx["train"]], class_id=class_id)

    eval_datasets = []
    for split in splits:
        if split == "val":
            eval_datasets.append(ChemicalDataset(dataset[split_idx["valid"]], class_id=class_id))
        elif split == "test":
            eval_datasets.append(ChemicalDataset(dataset[split_idx["test"]], class_id=class_id))
        else:
            raise NotImplementedError(
                f"Split '{split}' is not implemented for dataset '{name}'."
            )
    return train_dataset, eval_datasets


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, splits: list, **kwargs):
    """
    Load a dataset by name and return train + eval splits.

    Args:
        name:     Dataset identifier (case-insensitive).
        splits:   Evaluation splits to return, e.g. ``["val", "test"]``.
        **kwargs: Extra dataset-specific keyword arguments from the config.

    Returns:
        ``(train_dataset, eval_datasets)`` — both are
        :class:`torch.utils.data.Dataset` instances whose ``__getitem__``
        yields ``(data, label, index)`` tuples, as expected by the Trainer.
    """
    name      = name.lower()
    root_path = kwargs.get("root_path", "./data")

    # ── OGB graph datasets ───────────────────────────────────────────────────
    if name in _OGB_CLASS_IDS:
        return _load_ogbg(name, _OGB_CLASS_IDS[name], root_path, splits)

    # ── Image datasets ───────────────────────────────────────────────────────
    if name == "catvsdog":
        raise NotImplementedError(f"Dataset '{name}' is not yet implemented.")

    elif name == "chexpert":
        from libauc.datasets import CheXpert
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Subset

        root     = os.path.join(root_path, "CheXpert-v1.0-small")
        val_size = kwargs.get("val_size", 0.05)

        full_train = CheXpert(
            csv_path=os.path.join(root, 'train.csv'),
            image_root_path=root,
            use_upsampling=False,
            use_frontal=True,
            image_size=224,
            mode='train',
            class_index=-1,
            verbose=False,
        )

        all_targets  = np.array([full_train[i][1] for i in range(len(full_train))])
        strat_labels = all_targets[:, -1] if all_targets.ndim == 2 else all_targets
        train_indices, val_indices = train_test_split(
            np.arange(len(full_train)),
            test_size=val_size,
            stratify=strat_labels,
            random_state=42,
        )
        train_dataset = IndexedDataset(Subset(full_train, train_indices))
        val_dataset   = IndexedDataset(Subset(full_train, val_indices))

        test_dataset = IndexedDataset(CheXpert(
            csv_path=os.path.join(root, 'valid.csv'),
            image_root_path=root,
            use_upsampling=False,
            use_frontal=True,
            image_size=224,
            mode='valid',
            class_index=-1,
            verbose=False,
        ))

        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(val_dataset)
            elif split == 'test':
                eval_datasets.append(test_dataset)
            else:
                raise NotImplementedError(
                    f"Split '{split}' is not implemented for dataset '{name}'."
                )
        return train_dataset, eval_datasets

    elif name == "cifar10":
        from libauc.datasets import CIFAR10

        train_data, train_targets = CIFAR10(root=root_path, train=True).as_array()
        test_data,  test_targets  = CIFAR10(root=root_path, train=False).as_array()

        imratio   = kwargs.get("imratio", 0.1)
        generator = ImbalancedDataGenerator(verbose=True, random_seed=0)
        (train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
        (test_images,  test_labels)  = generator.transform(test_data,  test_targets,  imratio=0.5)

        train_dataset = ImageDataset(train_images, train_labels)
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(ImageDataset(train_images, train_labels, mode='test'))
            elif split == 'test':
                eval_datasets.append(ImageDataset(test_images, test_labels, mode='test'))
            else:
                raise NotImplementedError(
                    f"Split '{split}' is not implemented for dataset '{name}'."
                )
        return train_dataset, eval_datasets

    elif name == "pneumoniamnist":
        from medmnist import PneumoniaMNIST

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])
        train_dataset = IndexedDataset(
            PneumoniaMNIST(split='train', transform=train_transform, download=True, root=root_path)
        )
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(
                    PneumoniaMNIST(split='val', transform=test_transform, download=True, root=root_path)
                ))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(
                    PneumoniaMNIST(split='test', transform=test_transform, download=True, root=root_path)
                ))
            else:
                raise NotImplementedError(
                    f"Split '{split}' is not implemented for dataset '{name}'."
                )
        return train_dataset, eval_datasets

    elif name == "breastmnist":
        from medmnist import BreastMNIST

        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = IndexedDataset(
            BreastMNIST(split='train', transform=transform, download=True, root=root_path)
        )
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(
                    BreastMNIST(split='val', transform=transform, download=True, root=root_path)
                ))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(
                    BreastMNIST(split='test', transform=transform, download=True, root=root_path)
                ))
            else:
                raise NotImplementedError(
                    f"Split '{split}' is not implemented for dataset '{name}'."
                )
        return train_dataset, eval_datasets

    elif name == "chestmnist":
        from medmnist import ChestMNIST

        transform = transforms.Compose([transforms.ToTensor()])
        task      = kwargs.get("task", None)
        train_dataset = IndexedDataset(
            ChestMNIST(split='train', transform=transform, download=True, root=root_path), task
        )
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(
                    ChestMNIST(split='val', transform=transform, download=True, root=root_path), task
                ))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(
                    ChestMNIST(split='test', transform=transform, download=True, root=root_path), task
                ))
            else:
                raise NotImplementedError(
                    f"Split '{split}' is not implemented for dataset '{name}'."
                )
        return train_dataset, eval_datasets

    elif name == "melanoma":
        # Download dataset from Kaggle before use:
        #   https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256
        # Place the extracted contents under <root_path>/melanoma/
        from libauc.datasets import Melanoma

        root          = os.path.join(root_path, "melanoma")
        train_dataset = IndexedDataset(Melanoma(root=root, is_test=False, test_size=0.2))
        eval_datasets = []
        for split in splits:
            if split == 'val':
                eval_datasets.append(IndexedDataset(Melanoma(root=root, is_test=False, test_size=0.2)))
            elif split == 'test':
                eval_datasets.append(IndexedDataset(Melanoma(root=root, is_test=True, test_size=0.2)))
            else:
                raise NotImplementedError(
                    f"Split '{split}' is not implemented for dataset '{name}'."
                )
        return train_dataset, eval_datasets

    elif name == "ddsm":
        # Download dataset from Kaggle before use:
        #   https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
        # Place the extracted contents under <root_path>/ddsm/ with subdirs csv/ and jpeg/
        from sklearn.model_selection import train_test_split

        root     = os.path.join(root_path, "ddsm")
        csv_dir  = os.path.join(root, "csv")
        jpeg_dir = os.path.join(root, "jpeg")

        # ── 1. Build UID → absolute jpeg path lookup from dicom_info.csv ────
        dicom_df = pd.read_csv(os.path.join(csv_dir, "dicom_info.csv"))

        # Fix known labelling bug: 8-bit "Unknown" entries are ROI masks.
        dicom_df.loc[
            (dicom_df["SeriesDescription"] == "Unknown") & (dicom_df["BitsAllocated"] == 8),
            "SeriesDescription",
        ] = "ROI mask images"

        full_mammo = dicom_df[dicom_df["SeriesDescription"] == "full mammogram images"].copy()
        full_mammo["uid"]      = full_mammo["image_path"].str.extract(r"jpeg/([^/]+)/")
        full_mammo["abs_path"] = full_mammo["image_path"].apply(
            lambda p: p.replace("CBIS-DDSM/jpeg", jpeg_dir)
        )
        uid_to_path = full_mammo.set_index("uid")["abs_path"].to_dict()

        # ── 2. Load case-description CSVs ───────────────────────────────────
        def load_case_csv(csv_name):
            path = os.path.join(csv_dir, csv_name)
            if not os.path.exists(path):
                return None
            df  = pd.read_csv(path)
            col = next(c for c in df.columns if "image file path" in c.lower())
            df  = df.rename(columns={col: "image_file_path"})
            df["uid"]       = df["image_file_path"].str.extract(r"/([^/]+)/[^/]+$")
            df["pathology"] = df["pathology"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
            df["label"]     = (df["pathology"] == "MALIGNANT").astype(np.float32)
            return df[["patient_id", "uid", "label", "pathology"]]

        train_case_dfs, test_case_dfs = [], []
        for csv_name in ("calc_case_description_train_set.csv",
                         "mass_case_description_train_set.csv"):
            df = load_case_csv(csv_name)
            if df is not None:
                train_case_dfs.append(df)

        for csv_name in ("calc_case_description_test_set.csv",
                         "mass_case_description_test_set.csv"):
            df = load_case_csv(csv_name)
            if df is not None:
                test_case_dfs.append(df)

        train_cases = pd.concat(train_case_dfs, ignore_index=True).drop_duplicates(subset=["uid"])
        test_cases  = pd.concat(test_case_dfs,  ignore_index=True).drop_duplicates(subset=["uid"])

        # ── 3. Map UID → absolute image path ────────────────────────────────
        def attach_paths(df):
            df = df.copy()
            df["image_path"] = df["uid"].map(uid_to_path)
            return df.dropna(subset=["image_path"]).reset_index(drop=True)

        train_pool = attach_paths(train_cases)
        test_df    = attach_paths(test_cases)

        # ── 4. Stratified val split from training pool ───────────────────────
        val_size = kwargs.get("val_size", 0.1)
        train_idx, val_idx = train_test_split(
            np.arange(len(train_pool)),
            test_size=val_size,
            stratify=train_pool["label"].values,
            random_state=42,
        )
        train_df = train_pool.iloc[train_idx].reset_index(drop=True)
        val_df   = train_pool.iloc[val_idx].reset_index(drop=True)

        # ── 5. Build datasets — pass DataFrame directly to avoid temp files ──
        image_size      = kwargs.get("image_size", 224)
        train_transform = _medical_train_transform(image_size)
        test_transform  = _medical_test_transform(image_size)

        def _df_to_dataset(df, transform):
            return MedicalImageCSVDataset(
                source=df[["image_path", "label"]],
                image_root="",
                image_col="image_path",
                label_col="label",
                transform=transform,
            )

        train_dataset = _df_to_dataset(train_df, train_transform)
        eval_datasets = []
        for split in splits:
            if split == "val":
                eval_datasets.append(_df_to_dataset(val_df, test_transform))
            elif split == "test":
                eval_datasets.append(_df_to_dataset(test_df, test_transform))
            else:
                raise NotImplementedError(
                    f"Split '{split}' is not implemented for dataset '{name}'."
                )
        return train_dataset, eval_datasets

    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            "Please add a branch for it inside load_dataset()."
        )
