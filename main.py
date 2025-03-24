import os
import torch
from dataset import create_dataloaders
from model import GDFSegFormer
from trainer import Trainer
from test_components import test_model, test_dataloader
import time

def main():
    # Configuration with your specific paths
    model_path = '/Users/bhavyabhatia/Documents/Thesis/Project_Thesis/segformer_finetuned_model'
    root_dir = '/Users/bhavyabhatia/Documents/Thesis/Project_Thesis/cityscape'
    save_dir = '/Users/bhavyabhatia/Documents/Thesis/Project_Thesis/saved_models'
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Print paths for verification
    print("\nPaths:")
    print(f"Model path: {model_path}")
    print(f"Data root: {root_dir}")
    print(f"Save directory: {save_dir}")
    
    # Verify paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Data root not found: {root_dir}")
    
    # Training parameters
    num_classes = 19
    num_epochs = 1
    batch_size = 2
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test components first
    print("\nTesting components before training...")
    if not test_model(model_path, device):
        print("Model test failed. Exiting.")
        return
    if not test_dataloader(root_dir, model_path):
        print("Dataloader test failed. Exiting.")
        return
    
    print("\nAll components tested successfully. Starting training...")
    
    try:
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(root_dir, model_path, batch_size)
        
        # Initialize model
        model = GDFSegFormer(num_classes=num_classes, model_path=model_path).to(device)
        
        # Initialize trainer
        trainer = Trainer(model, train_loader, test_loader, device, num_classes)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 20)
            
            # Train
            avg_train_loss = trainer.train_epoch(epoch, num_epochs)
            print(f'Average Training Loss: {avg_train_loss:.4f}')
            
            # Validate
            metrics = trainer.validate()
            
            print(f'\nValidation Metrics:')
            print(f'Average Loss: {metrics["loss"]:.4f}')
            print(f'mIoU: {metrics["miou"]:.4f}')
            print(f'Accuracy: {metrics["accuracy"]:.4f}')
            print(f'Precision: {metrics["precision"]:.4f}')
            print(f'Recall: {metrics["recall"]:.4f}')
            print(f'Average Inference Time: {metrics["inference_time"]*1000:.2f}ms')
            print(f'FPS: {1.0/metrics["inference_time"]:.2f}')

            # Save checkpoint every epoch
            epoch_ckpt_path = f"epoch_{epoch+1}_gdf_segformer.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'miou': epoch_miou,
            }, epoch_ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}_gdf_segformer.pth"))
            print(f"Saved model for epoch {epoch+1} to {epoch_ckpt_path}")


        

            
            # Save best model with timestamp
            if metrics["miou"] > trainer.best_miou:
                trainer.best_miou = metrics["miou"]

                # Fixed filename path for best model
                best_ckpt_path = os.path.join(SAVE_DIR, "best_gdf_segformer.pth")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'miou': metrics["miou"],
                    'metrics': metrics,
                    'config': {
                        'num_classes': num_classes,
                        'batch_size': batch_size,
                        'device': str(device)
                    }
                }, best_ckpt_path)

                print(f"✅ Best model updated with mIoU: {metrics['miou']:.4f} → saved to {best_ckpt_path}")

            
            trainer.scheduler.step()

        print("\nTraining completed!")
        
    except Exception as e:
        print(f"\nAn error occurred during training:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
        print(f"Line number: {e.__traceback__.tb_lineno}")
        raise

if __name__ == "__main__":
    main() 