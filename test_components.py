import torch
from dataset import create_dataloaders
from model import GDFSegFormer

# Test model loading and forward pass
def test_model(model_path, device):
    try:
        print("Testing Model...")
        model = GDFSegFormer(num_classes=19, model_path=model_path).to(device)
        dummy_input = torch.randn(2, 3, 1024, 1024).to(device)
        output = model(dummy_input)
        print("Model test successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        return True
    except Exception as e:
        print("❌ Model test failed:")
        print(f"{type(e).__name__}: {e}")
        return False

# Test data loading and shape
def test_dataloader(root_dir, model_path, debug=False):
    try:
        print("Testing Dataloaders...")
        train_loader, test_loader = create_dataloaders(root_dir, model_path, batch_size=2, debug=debug)

        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        train_imgs, train_targets = train_batch
        test_imgs, test_targets = test_batch

        print("Train loader test successful!")
        print(f"Batch shapes - Images: {train_imgs.shape}, Targets: {train_targets.shape}")

        print("Test loader test successful!")
        print(f"Batch shapes - Images: {test_imgs.shape}, Targets: {test_targets.shape}")
        return True
    except Exception as e:
        print("❌ Dataloader test failed:")
        print(f"{type(e).__name__}: {e}")
        return False
