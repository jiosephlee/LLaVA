import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

import torch.nn as nn
import torch
from .molformer import MolFormerVisionTower   # NEW import


from transformers import AutoTokenizer, AutoModel

# class MolFormerVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()
#         self.vision_tower_name = vision_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = args.mm_vision_select_feature

#         if not delay_load:
#             self.load_model()
#         else:
#             self.cfg_only = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True, dtype=torch.float32).config

#     def load_model(self, device_map=None):
#         self.image_processor = None
#         self.mol_processor = AutoTokenizer.from_pretrained(self.vision_tower_name, trust_remote_code=True)
#         self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
#         self.vision_tower.requires_grad_(False)
#         self.is_loaded = True

#     @torch.no_grad()
#     def forward(self, smiles):
        
#         # Manually move to device
#         smiles_input_ids = smiles['input_ids'].to(self.vision_tower.device)
#         smiles_attention_mask = smiles['attention_mask'].to(self.vision_tower.device)
        
#         outputs = self.vision_tower(
#             input_ids=smiles_input_ids,
#             attention_mask=smiles_attention_mask,
#             output_hidden_states=True
#         )
        
#         if self.select_feature == 'pooler':
#             feature_output = outputs.pooler_output.unsqueeze(1)
#         else:
#             feature_output = outputs.last_hidden_state

#         return feature_output

#     @property
#     def dummy_feature(self):
#         return torch.randn(1, 3, 224, 224) # Placeholder for dummy feature

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         return 768


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if 'molformer' in vision_tower.lower():
        return MolFormerVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if use_s2:
        return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
