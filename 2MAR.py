import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# ============= Utility Functions =============
def check_tensor_shape(tensor, expected_dims, name="tensor"):
    """Validate tensor shape and format"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if len(tensor.shape) != expected_dims:
        raise ValueError(
            f"{name} must have {expected_dims} dimensions, "
            f"got shape {tensor.shape}"
        )
    return tensor.shape

def validate_spatial_tensor(tensor, name="tensor"):
    """Validate that tensor is in spatial format [B, C, H, W]"""
    shape = check_tensor_shape(tensor, expected_dims=4, name=name)
    return shape

# ============= Model Components =============
class DetailBranch(nn.Module):
    """Detail processing branch of GDFSegFormer"""
    def __init__(self, hidden_size):
        super().__init__()
        self.detail_conv = nn.Conv2d(hidden_size, hidden_size, 1)
    
    def forward(self, x):
        shape = validate_spatial_tensor(x, "detail_branch_input")
        return self.detail_conv(x)

class FusionLayer(nn.Module):
    """Fusion layer for combining global and detail features"""
    def __init__(self, hidden_size):
        super().__init__()
        self.fusion = nn.Conv2d(hidden_size * 2, hidden_size, 1)
    
    def forward(self, global_feat, detail_feat):
        validate_spatial_tensor(global_feat, "global_features")
        validate_spatial_tensor(detail_feat, "detail_features")
        
        if global_feat.shape != detail_feat.shape:
            raise ValueError(
                f"Shape mismatch: global {global_feat.shape} != "
                f"detail {detail_feat.shape}"
            )
        
        combined = torch.cat([global_feat, detail_feat], dim=1)
        return self.fusion(combined)

# ============= Main Model =============
class GDFSegFormer(nn.Module):
    """Main GDFSegFormer model"""
    def __init__(self, num_classes):
        super().__init__()
        # Initialize backbone
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            local_files_only=True,
            num_labels=num_classes
        )
        hidden_size = self.segformer.config.hidden_sizes[-1]
        
        # Initialize components
        self.detail_branch = DetailBranch(hidden_size)
        self.fusion_layer = FusionLayer(hidden_size)
        self.num_classes = num_classes
        
        print(f"Initialized GDFSegFormer with:")
        print(f"- Hidden size: {hidden_size}")
        print(f"- Num classes: {num_classes}")

    def _validate_input(self, x):
        """Validate input tensor"""
        shape = validate_spatial_tensor(x, "model_input")
        if shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {shape[1]}")
        return shape

    def forward(self, x):
        try:
            # Validate input
            input_shape = self._validate_input(x)
            print(f"\nProcessing batch:")
            print(f"Input shape: {input_shape}")
            
            # Encoder forward pass
            encoder_outputs = self.segformer.segformer(x, output_hidden_states=True)
            hidden_states = encoder_outputs.hidden_states
            global_features = hidden_states[-1]  # [B, C, H, W]
            
            # Validate encoder output
            encoder_shape = validate_spatial_tensor(global_features, "encoder_output")
            print(f"Encoder output shape: {encoder_shape}")
            
            # Detail branch
            detail_features = self.detail_branch(global_features)
            print(f"Detail features shape: {detail_features.shape}")
            
            # Feature fusion
            fused_features = self.fusion_layer(global_features, detail_features)
            print(f"Fused features shape: {fused_features.shape}")
            
            # Decoder head
            logits = self.segformer.decode_head(fused_features)
            print(f"Initial logits shape: {logits.shape}")
            
            # Upsample to input resolution
            logits = F.interpolate(
                logits,
                size=input_shape[2:],
                mode='bilinear',
                align_corners=False
            )
            print(f"Final logits shape: {logits.shape}")
            
            return logits
            
        except Exception as e:
            print("\nError in forward pass:")
            print(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
            print(f"Line number: {e.__traceback__.tb_lineno}")
            print(f"Error message: {str(e)}")
            raise

# ============= Testing Functions =============
def test_model(model, device):
    """Test model with dummy input"""
    print("\nRunning model test...")
    model.eval()
    with torch.no_grad():
        try:
            dummy_input = torch.randn(2, 3, 1024, 1024).to(device)
            output = model(dummy_input)
            print(f"\nTest successful!")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            return True
        except Exception as e:
            print(f"\nTest failed:")
            print(str(e))
            return False

# ============= Main Execution =============
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GDFSegFormer(num_classes=19).to(device)
    test_model(model, device)
