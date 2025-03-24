# Ensure PyTorch is installed before running this script:
# pip install torch torchvision torchaudio
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("PyTorch is not installed. Please install it using 'pip install torch torchvision torchaudio'") from e

from transformers import SegformerForSemanticSegmentation

class GDFSegFormer(nn.Module):
    def __init__(self, num_classes, model_path):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            local_files_only=True,
            num_labels=num_classes
        )

        # Save config info
        self.hidden_sizes = self.segformer.config.hidden_sizes
        self.num_classes = num_classes

        # Create enhancement conv layer only for the last feature map
        hidden_size = self.hidden_sizes[-1]
        self.detail_conv = nn.Conv2d(hidden_size, hidden_size, 1)
        self.fusion_layer = nn.Conv2d(hidden_size * 2, hidden_size, 1)

    def forward(self, x, *args, **kwargs):
        target_shape = kwargs.get('target_shape', None)
        try:
            # Forward pass through encoder backbone
            encoder_outputs = self.segformer.segformer(
                x,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = encoder_outputs.hidden_states  # Tuple of 4 tensors (low to high resolution)

            # Get last (most semantic) feature map
            global_features = hidden_states[-1]  # [B, C, H, W]

            # Apply detail enhancement
            detail_features = self.detail_conv(global_features)
            fused_features = self.fusion_layer(torch.cat([global_features, detail_features], dim=1))

            # Replace the last hidden state with fused version
            modified_hidden_states = list(hidden_states)
            modified_hidden_states[-1] = fused_features

            # Decode expects list of features from encoder (usually from multiple stages)
            logits = self.segformer.decode_head(modified_hidden_states)

            # Upsample to input image resolution
            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            return logits

        except Exception as e:
            print("\nError in forward pass:")
            print(f"Input shape: {x.shape}")
            if 'hidden_states' in locals():
                print(f"Hidden states count: {len(hidden_states)}")
                print(f"Last hidden state shape: {hidden_states[-1].shape}")
            print("Model configuration:")
            print(f"- Hidden sizes: {self.hidden_sizes}")
            print(f"- Num classes: {self.num_classes}")
            print("Error details:")
            print(f"- Type: {type(e).__name__}")
            print(f"- Message: {str(e)}")
            raise

    @torch.no_grad()
    def check_shapes(self, x):
        print("\nShape check:")
        print(f"Input: {x.shape}")

        encoder_outputs = self.segformer.segformer(
            x,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = encoder_outputs.hidden_states
        print("\nEncoder hidden states shapes:")
        for idx, hs in enumerate(hidden_states):
            print(f"Stage {idx}: {hs.shape}")

        return True
