import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from torchvision.transforms.functional import resize as resize_mask
from torchvision.transforms.functional import InterpolationMode

# Mapping of raw Cityscapes IDs to 19-class training IDs (rest → 255)
CITYSCAPES_LABEL_MAPPING = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

class CityscapesDataset(datasets.Cityscapes):
    def __init__(self, root, split='train', transform=None):
        super().__init__(
            root,
            split=split,
            mode='fine',
            target_type='semantic',
            transform=None,
            target_transform=None
        )
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if self.transform:
            image, target = self.transform(image, target)
        return image, target


def get_transform(model_path):
    feature_extractor = SegformerImageProcessor.from_pretrained(model_path, local_files_only=True)

    def transform(image, target):
        # Resize and normalize the input image
        image = feature_extractor(image, return_tensors='pt')['pixel_values'].squeeze()

        # Resize the mask to 1024x1024 using nearest-neighbor to preserve class IDs
        target = resize_mask(target, size=[1024, 1024], interpolation=InterpolationMode.NEAREST)

        # Convert to numpy array for remapping
        target_np = np.array(target)
        label_map = np.full_like(target_np, fill_value=255)

        # Remap raw Cityscapes labels to 19-class IDs
        for raw_id, train_id in CITYSCAPES_LABEL_MAPPING.items():
            label_map[target_np == raw_id] = train_id

        # Convert final remapped mask to tensor
        target_tensor = torch.as_tensor(label_map, dtype=torch.long)

        return image, target_tensor

    return transform


def create_dataloaders(root_dir, model_path, batch_size=2, debug=False):
    transform_fn = get_transform(model_path)

    train_dataset = CityscapesDataset(
        root=root_dir,
        split='train',
        transform=transform_fn
    )

    test_dataset = CityscapesDataset(
        root=root_dir,
        split='val',
        transform=transform_fn
    )

    # ✅ Limit dataset size in debug mode
    if debug:
        print("🚧 DEBUG MODE: Using 100 training samples and 50 test samples")
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(100)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(50)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
