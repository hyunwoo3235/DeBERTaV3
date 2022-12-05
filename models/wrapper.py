from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Config, DebertaV2ForMaskedLM
from transformers.modeling_outputs import ModelOutput

from .deberta import DebertaV2ForPreTraining


@dataclass
class DebertaV2WrapperOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    dis_loss: Optional[torch.FloatTensor] = None
    gen_logits: torch.FloatTensor = None
    dis_logits: torch.FloatTensor = None
    replaced_ids: Optional[torch.BoolTensor] = None


class DebertaV2Wrapper(nn.Module):
    def __init__(
            self, generator_config: DebertaV2Config, discriminator_config: DebertaV2Config
    ):
        super().__init__()

        self.gen_config = generator_config
        self.dis_config = discriminator_config

        self.generator = DebertaV2ForMaskedLM(generator_config)
        self.discriminator = DebertaV2ForPreTraining(discriminator_config)

        self.gen_loss_fn = nn.CrossEntropyLoss()
        self.dis_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            is_mlm_applied=None,
            labels=None,
    ):
        gen_logits = self.generator(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
        ).logits

        with torch.no_grad():
            pred_toks = F.gumbel_softmax(gen_logits, tau=1).argmax(-1)
            generated = input_ids.clone()

            generated[is_mlm_applied] = pred_toks[is_mlm_applied]

            is_replaced = None
            if labels is not None:
                is_replaced = is_mlm_applied.clone()
                is_replaced[is_mlm_applied] = (
                        pred_toks[is_mlm_applied] != labels[is_mlm_applied]
                )

            gen_embeddings = self.generator.deberta.embeddings.word_embeddings(
                generated
            )
        disc_embeddings = self.discriminator.deberta.embeddings.word_embeddings(
            generated
        )
        inputs_embeds = (gen_embeddings + disc_embeddings) / 2

        dis_logits = self.discriminator(
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
        ).logits

        loss = None
        gen_loss = None
        dis_loss = None
        if labels is not None:
            gen_loss = self.gen_loss_fn(
                gen_logits.view(-1, self.gen_config.vocab_size), labels.view(-1)
            )
            dis_loss = self.dis_loss_fn(dis_logits, is_replaced.float())

            loss = gen_loss + dis_loss * 50

        return DebertaV2WrapperOutput(
            loss=loss,
            gen_loss=gen_loss,
            dis_loss=dis_loss,
            gen_logits=gen_logits,
            dis_logits=dis_logits,
            replaced_ids=is_replaced,
        )
