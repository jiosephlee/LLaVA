# llava/model/multimodal_encoder/molformer.py

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoConfig, AutoTokenizer, AutoModel


class MolFormerVisionTower(nn.Module):
    """
    Drop-in vision tower replacement that encodes SMILES with a MolFormer-like HF model.
    - Safe for delay-load construction (no weights until load_model()).
    - Keeps dtype/device handling explicit (MolFormer stays float32; we only move device).
    - Respects mm_vision_select_feature: 'last_hidden_state' (default) or 'pooler'.
    """

    def __init__(self, vision_tower: str, args, delay_load: bool = False):
        super().__init__()
        self.vision_tower_name = vision_tower

        # Selection options (kept for parity with CLIP path and your unused MolFormerTower)
        self.select_layer = getattr(args, "mm_vision_select_layer", -1)
        self.select_feature = getattr(args, "mm_vision_select_feature", "last_hidden_state")

        # Set up placeholders
        self.image_processor = None         # there is no image processor for SMILES
        self.mol_processor: Optional[AutoTokenizer] = None
        self.vision_tower: Optional[nn.Module] = None

        # Crucial: define early so inference path can check it safely
        self.is_loaded: bool = False

        # Config-only when delay loading to avoid pulling full weights
        self._cfg_only = None
        if delay_load:
            self._cfg_only = AutoConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        else:
            self.load_model()

    # Accept dtype and device_map like the inference helper expects
    def load_model(self, device_map: Optional[Any] = None, dtype: Optional[torch.dtype] = None, **kwargs):
        if self.is_loaded:
            # Idempotent
            return

        # Tokenizer
        self.mol_processor = AutoTokenizer.from_pretrained(self.vision_tower_name, trust_remote_code=True)

        # If caller passed dtype under `dtype`, map it to HF's `torch_dtype` (but default to fp32 for MolFormer)
        torch_dtype = kwargs.pop("torch_dtype", dtype) or torch.float32

        # Model
        self.vision_tower = AutoModel.from_pretrained(
            self.vision_tower_name,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs
        )
        # No training on the tower
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, smiles: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Expects a dict with 'input_ids' and 'attention_mask' (as produced by self.mol_processor).
        Returns [B, T, H] (last_hidden_state) or [B, 1, H] (pooler).
        """
        if not self.is_loaded or self.vision_tower is None:
            raise RuntimeError("MolFormerVisionTower must be loaded via load_model() before forward().")

        dev = next(self.vision_tower.parameters()).device
        inputs = {
            "input_ids": smiles["input_ids"].to(dev),
            "attention_mask": smiles["attention_mask"].to(dev),
            "output_hidden_states": True
        }

        outs = self.vision_tower(**inputs)

        if self.select_feature == "pooler" and hasattr(outs, "pooler_output") and outs.pooler_output is not None:
            feats = outs.pooler_output.unsqueeze(1)   # [B, 1, H]
        else:
            feats = outs.last_hidden_state             # [B, T, H]

        return feats


    @property
    def dtype(self) -> torch.dtype:
        if self.is_loaded and self.vision_tower is not None:
            return next(self.vision_tower.parameters()).dtype
        # default expectation for MolFormer
        return torch.float32

    @property
    def device(self) -> torch.device:
        if self.is_loaded and self.vision_tower is not None:
            return next(self.vision_tower.parameters()).device
        return torch.device("cpu")

    @property
    def config(self):
        if self.is_loaded and self.vision_tower is not None:
            return self.vision_tower.config
        return self._cfg_only

    @property
    def hidden_size(self) -> int:
        cfg = self.config
        return getattr(cfg, "hidden_size", 768)

    @property
    def pad_token_id(self) -> Optional[int]:
        # Useful for debug assertions elsewhere
        return getattr(self.mol_processor, "pad_token_id", None) if self.mol_processor is not None else None

    # Allow inference code to call .to(device=..., dtype=...) safely
    def to(self, *args, **kwargs):
        if self.is_loaded and self.vision_tower is not None:
            self.vision_tower.to(*args, **kwargs)
        # Still move the wrapper module so attributes are consistent
        return super().to(*args, **kwargs)

    @property
    def dummy_feature(self):
        # Provide a small correctly-shaped dummy tensor for projector sanity checks
        return torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)