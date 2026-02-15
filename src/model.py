"""
Model architecture with DINOv3 backbone and detection head
"""
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.ops import nms
from safetensors.torch import load_file

# --- Direct import from the local dinov3 library ---
from dinov3.hub import backbones as dinov3_backbones

BACKBONE_BUILDERS = {
    'dinov3_vits16': dinov3_backbones.dinov3_vits16,
    'dinov3_vitb16': dinov3_backbones.dinov3_vitb16,
    'dinov3_vitl16': dinov3_backbones.dinov3_vitl16,
}
# --- End of direct import ---

def load_safetensors_weights(model, safetensors_path):
    """Loads weights from a .safetensors file directly into the backbone."""
    from safetensors.torch import load_file
    
    # Load the state dict from safetensors
    state_dict = load_file(safetensors_path, device="cpu")
    
    # Try to load directly into the backbone
    # The keys should match the backbone structure directly
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    
    print(f"✓ Weights loaded from {Path(safetensors_path).name}")
    if unexpected: 
        print(f"  - Unexpected keys not loaded: {len(unexpected)}")
    if missing: 
        print(f"  - Missing keys in model: {len(missing)}")
        # If we have many missing keys, the weights might not match
        if len(missing) > 10:
            print(f"  ⚠️ WARNING: Many keys missing ({len(missing)}). Weights may not have loaded correctly!")
            print(f"  First few missing keys: {missing[:5]}")
            print(f"  First few unexpected keys: {unexpected[:5] if unexpected else 'None'}")

class SpatialTuningAdapter(nn.Module):
    def __init__(self, in_dim, out_dim=256, num_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()])
        self.adapter = nn.Sequential(*layers)
    def forward(self, x): return self.adapter(x)

class SimpleDetectionHead(nn.Module):
    def __init__(self, in_dim, num_classes, num_queries=100):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, in_dim)
        self.cross_attn = nn.MultiheadAttention(in_dim, 8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(in_dim, in_dim*4), nn.GELU(), nn.Linear(in_dim*4, in_dim))
        self.norm1, self.norm2 = nn.LayerNorm(in_dim), nn.LayerNorm(in_dim)
        self.bbox_head = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, 4))
        self.class_head = nn.Linear(in_dim, num_classes + 1)
        self.objectness_head = nn.Linear(in_dim, 1)
    
    def forward(self, features):
        B = features.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, features, features)
        queries = self.norm1(queries + attn_out)
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        return self.bbox_head(queries).sigmoid(), self.class_head(queries), self.objectness_head(queries)
    
class DETRDecoderHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_queries=100, num_layers=6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Positional encoding for memory (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.class_head = nn.Linear(hidden_dim, num_classes + 1)

        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, memory):

        B, N, C = memory.shape

        # Add positional encoding
        memory = memory + self.pos_embed[:, :N]

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        hs = self.decoder(queries, memory)

        class_logits = self.class_head(hs)
        bbox = self.bbox_head(hs).sigmoid()

        return bbox, class_logits


class DINOv3Detector(nn.Module):
    def __init__(self, backbone_name='dinov3_vitl16', backbone_weights_path=None, num_classes=80, adapter_dim=256, num_queries=100, freeze_backbone=True):
        super().__init__()
        
        if backbone_name not in BACKBONE_BUILDERS:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Available: {list(BACKBONE_BUILDERS.keys())}")
        
        print(f"Building '{backbone_name}' architecture...")
        
        # If weights path is provided, try to load using it
        if backbone_weights_path and Path(backbone_weights_path).exists():
            try:
                # Determine file type and load accordingly
                weights_path = Path(backbone_weights_path)
                
                if weights_path.suffix == '.pth':
                    # Load .pth file
                    pretrained_weights = torch.load(weights_path, map_location='cpu')
                    # Handle potential nested structure
                    if 'model' in pretrained_weights:
                        pretrained_weights = pretrained_weights['model']
                    elif 'state_dict' in pretrained_weights:
                        pretrained_weights = pretrained_weights['state_dict']
                elif weights_path.suffix == '.safetensors':
                    # Load .safetensors file
                    from safetensors.torch import load_file
                    pretrained_weights = load_file(str(weights_path), device="cpu")
                else:
                    raise ValueError(f"Unsupported file format: {weights_path.suffix}")
                
                # Build the backbone
                self.backbone = BACKBONE_BUILDERS[backbone_name](pretrained=False)
                
                # Load the weights
                missing, unexpected = self.backbone.load_state_dict(pretrained_weights, strict=False)
                
                print(f"✓ Weights loaded from {weights_path.name}")
                
                # Calculate how many parameters were actually loaded
                total_params = len(list(self.backbone.state_dict().keys()))
                loaded_params = total_params - len(missing)
                load_percentage = (loaded_params / total_params) * 100
                
                if len(missing) > 0:
                    print(f"  - {len(missing)} missing keys ({100-load_percentage:.1f}% of parameters)")
                if len(unexpected) > 0:
                    print(f"  - {len(unexpected)} unexpected keys in weights file")
                    
                # Warn if less than 50% loaded
                if load_percentage < 50:
                    print(f"  ⚠️  WARNING: Only {load_percentage:.1f}% of parameters loaded!")
                    print(f"     The weights file may not match this architecture.")
                    print(f"     Training will proceed but results may be poor.")
                    print(f"     Consider using a smaller model or training from scratch.")
                else:
                    print(f"  ✓ {load_percentage:.1f}% of backbone parameters loaded successfully")
                    
            except Exception as e:
                print(f"  ⚠️  Failed to load weights: {e}")
                print(f"     Building backbone with random initialization...")
                self.backbone = BACKBONE_BUILDERS[backbone_name](pretrained=False)
        else:
            if backbone_weights_path:
                print(f"Warning: Weights file not found at {backbone_weights_path}")
            print("Building backbone with random initialization...")
            self.backbone = BACKBONE_BUILDERS[backbone_name](pretrained=False)
            print("Warning: No local weights path provided, using randomly initialized backbone.")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("✓ Backbone fully frozen.")

        else:
            print("✓ Fine-tuning last ViT blocks only.")

            # Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze last transformer blocks
            for name, param in self.backbone.named_parameters():
                if "blocks.10" in name or "blocks.11" in name:
                    param.requires_grad = True

            self.backbone.train()

        
        self.adapter = SpatialTuningAdapter(self.backbone.embed_dim, adapter_dim)
        # self.detection_head = SimpleDetectionHead(adapter_dim, num_classes, num_queries)
        self.detection_head = DETRDecoderHead(
            hidden_dim=adapter_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_layers=6)
        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n--- Model Params ---\nTotal: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)\n--------------------\n")

    def forward(self, images):
        # with torch.no_grad():
        #     patch_features = self.backbone.forward_features(images)['x_norm_patchtokens']
        if any(p.requires_grad for p in self.backbone.parameters()):
            patch_features = self.backbone.forward_features(images)['x_norm_patchtokens']
        else:
            with torch.no_grad():
                patch_features = self.backbone.forward_features(images)['x_norm_patchtokens']

        
        adapted = self.adapter(patch_features)
        # boxes_cxcywh, classes, objectness = self.detection_head(adapted)
        boxes_cxcywh, classes = self.detection_head(adapted)

        
        x_c, y_c, w, h = boxes_cxcywh.unbind(-1)
        boxes_xyxy = torch.stack([(x_c-w/2), (y_c-h/2), (x_c+w/2), (y_c+h/2)], dim=-1)
        # return boxes_xyxy, classes, objectness
        return boxes_xyxy, classes

    @torch.no_grad()
    def get_predictions(self, images, conf_threshold=0.1, nms_threshold=0.5):
        self.eval()
        # boxes_norm, class_logits, objectness_logits = self.forward(images)
        boxes_norm, class_logits = self.forward(images)
        results = []
        for i in range(images.shape[0]):
            # scores = torch.sigmoid(objectness_logits[i].squeeze(-1)) * torch.softmax(class_logits[i], -1)[:, :-1].max(-1).values
            prob = torch.nn.functional.softmax(class_logits[i], dim=-1)
            scores, labels = prob[..., :-1].max(-1)
            keep = scores > conf_threshold
            boxes, labels, scores = boxes_norm[i, keep], labels[keep], scores[keep]
            
            if boxes.shape[0] > 0:
                h, w = images.shape[-2:]
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h
                keep_nms = nms(boxes, scores, nms_threshold)
                boxes, labels, scores = boxes[keep_nms], labels[keep_nms], scores[keep_nms]
            results.append({'boxes': boxes, 'labels': labels, 'scores': scores})
        return results

def create_model(num_classes, backbone='dinov3_vitl16', adapter_dim=256, num_queries=100, backbone_weights_path=None, freeze_backbone=True):
    return DINOv3Detector(backbone, backbone_weights_path, num_classes, adapter_dim, num_queries, freeze_backbone)