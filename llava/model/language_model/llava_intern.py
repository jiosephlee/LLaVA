#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen3ForCausalLM, Qwen3Model, Qwen3Config

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import torch.nn.functional as F
from llava.constants import IGNORE_INDEX


class LlavaInternConfig(Qwen3Config):
    model_type = "llava_intern"


class LlavaInternModel(LlavaMetaModel, Qwen3Model):
    config_class = LlavaInternConfig

    def __init__(self, config: Qwen3Config):
        super(LlavaInternModel, self).__init__(config)


class LlavaInternForCausalLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaInternConfig

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = LlavaInternModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Manually implements the from_pretrained logic to make the process explicit.
        """
        # 1. Load the pretrained Qwen3 model from Hugging Face.
        #    This gives us a standard Qwen3 model with all the language model weights.
        qwen_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )

        # 2. Create an instance of our custom LlavaInternForCausalLM class.
        #    We use the configuration from the loaded Qwen model to ensure consistency.
        config = qwen_model.config
        model = cls(config)

        # 3. Copy the weights from the loaded Qwen model to our new Llava model.
        #    `strict=False` is important here. It allows us to load the state dict
        #    even if our Llava model has extra parameters (like vision tower components)
        #    that are not present in the original Qwen model.
        model.load_state_dict(qwen_model.state_dict(), strict=False)

        return model

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        smiles: Optional[dict] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        """ This is where the image embeddings are injected into the input_ids. """
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                smiles=smiles
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        smiles: Optional[dict] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or smiles is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                smiles=smiles
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        smiles = kwargs.pop("smiles", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if smiles is not None:
            inputs['smiles'] = smiles
        return inputs

class DebugLlavaInternForCausalLM(LlavaInternForCausalLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        smiles=None,
        return_dict=None,
    ):
        # Prepare multimodal inputs like the parent
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                smiles=smiles
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if getattr(self, 'training_args', None) is not None and self.training_args.debug_mode and labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            with torch.no_grad():
                loss_val = loss_fct(shift_logits, shift_labels).item()
                print("\n--- Debugging Model Forward ---")
                print("Logits shape:", logits.shape)
                print("Labels shape:", labels.shape)
                predictions = torch.argmax(shift_logits, dim=-1)
                unmasked_indices = shift_labels != IGNORE_INDEX
                unmasked_labels = shift_labels[unmasked_indices]
                unmasked_predictions = predictions[unmasked_indices]
                print("Unmasked Labels sample:", unmasked_labels[:20])
                print("Predictions sample:", unmasked_predictions[:20])
                tok = getattr(self.training_args, 'tokenizer', None)
                if tok is not None and len(unmasked_labels) > 0:
                    print("Decoded Unmasked Labels sample:", tok.decode(unmasked_labels[:20]))
                    print("Decoded Predictions sample:", tok.decode(unmasked_predictions[:20]))
                print(f"Calculated Loss: {loss_val:.6f}")
                print("--- End Debugging ---\n")

        return outputs


AutoConfig.register("llava_intern", LlavaInternConfig)
AutoModelForCausalLM.register(LlavaInternConfig, LlavaInternForCausalLM)
