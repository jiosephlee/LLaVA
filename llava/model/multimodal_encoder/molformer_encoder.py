import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

class MolFormerTower(nn.Module):
    def __init__(self, model_name_or_path, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.model_name_or_path = model_name_or_path
        
        # NOTE: MolFormer doesn't have a direct equivalent to 'select_layer' like CLIP.
        # We will likely use the output of the last hidden state or a pooled output.
        # For now, this is a placeholder.
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'pooler') 

        if not delay_load:
            self.load_model()
        else:
            # NOTE: We can't get a config only for MolFormer in the same way as CLIP.
            # We'll load the full model when needed.
            pass

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.model_name_or_path} is already loaded, `load_model` called again, skipping.')
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True, device_map=device_map)
        self.molformer.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, forward_outs):
        if self.select_feature == 'pooler':
            # Use the pooled output which is suitable for sentence-level classification tasks
            return forward_outs.pooler_output
        elif self.select_feature == 'last_hidden_state':
            # Use the last hidden state
            return forward_outs.last_hidden_state
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

    @torch.no_grad()
    def forward(self, smiles_strings):
        inputs = self.tokenizer(smiles_strings, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device=self.device) for key, val in inputs.items()}

        forward_outs = self.molformer(**inputs)
        features = self.feature_select(forward_outs)
        return features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.molformer.dtype

    @property
    def device(self):
        return self.molformer.device

    @property
    def config(self):
        if self.is_loaded:
            return self.molformer.config
        else:
            # As noted, MolFormer doesn't have a config-only option like CLIP.
            # This will require the model to be loaded to access the config.
            raise RuntimeError("MolFormer model must be loaded to access config.")

    @property
    def hidden_size(self):
        # Assuming the model is loaded to access its config
        return self.config.hidden_size
