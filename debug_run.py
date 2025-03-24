import os
import torch
import time
from dataset import create_dataloaders
from model import GDFSegFormer
from trainer import Trainer
from test_components import test_model, test_dataloader

def debug_main():
    # Debug configuration
    model_path = '/Users/bhavyabhatia/Documents/Thesis/Project_Thesis/segformer_finetuned_model'
    root_dir = '/Users/bhavyabhatia/Documents/Thesis/Project_Thesis/cityscape'
    save_dir = '/Users/bhavyabhatia/Documents/Thesis/Project_Thesis/saved_models/debug'
    os.makedirs(save_dir, exist_ok=True)

    print("\n🚧 DEBUG MODE ENABLED 🚧")
    print(f"Using 100 training images and 50 validation images for fast testing")
    print(f"Model path: {model_path}")
    print(f"Save directory: {save_dir}")

    # Training parameters
    num_classes = 19
    num_epochs = 1
    batch_size = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Test components
    print("\nTesting model and dataloader...")
    if not test_model(model_path, device):
        print("Model test failed. Exiting.")
        return
    if not test_dataloader(root_dir, model_path, debug=True):  # Modified to accept debug flag
        print("Dataloader test failed. Exiting.")
        return

    print("\nAll components validated. Starting debug training...")

    try:
        train_loader, test_loader = create_dataloaders(root_dir, model_path, batch_size=1, debug=True)
        model = GDFSegFormer(num_classes=num_classes, model_path=model_path).to(device)
        trainer = Trainer(model, train_loader, test_loader, device, num_classes)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} [DEBUG MODE]")
            print("-" * 30)

            avg_train_loss = trainer.train_epoch(epoch, num_epochs)
            print(f'Average Training Loss: {avg_train_loss:.4f}')

            metrics = trainer.validate(epoch)

            print(f'\nValidation Metrics [DEBUG]:')
            print(f'Loss: {metrics["loss"]:.4f}, mIoU: {metrics["miou"]:.4f}, '
                  f'Acc: {metrics["accuracy"]:.4f}, FPS: {1.0 / metrics["inference_time"]:.2f}')

            print("⏩ Skipping checkpoint save (debug mode)")
            
            trainer.scheduler.step()

        print("\n✅ DEBUG TRAINING COMPLETE ✅")

    except Exception as e:
        print("\n❌ DEBUG RUN FAILED:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Error in: {e.__traceback__.tb_frame.f_code.co_name}, Line: {e.__traceback__.tb_lineno}")
        raise

if __name__ == "__main__":
    debug_main()
