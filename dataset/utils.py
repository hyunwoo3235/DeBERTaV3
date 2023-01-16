from typing import Dict, Optional, Tuple

import flax
import numpy as np
from transformers import PreTrainedTokenizerBase


# from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_mlm_flax.py
@flax.struct.dataclass
class FlaxDataCollatorForMaskedLM:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    replace_prob = 0.1
    orginal_prob = 0.1

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, input_ids: np.ndarray) -> Dict[str, np.ndarray]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {"input_ids": input_ids}

        special_tokens_mask = self.get_special_tokens_mask(batch)

        batch["input_ids"], batch["labels"], batch["masked_indices"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    # get special tokens mask
    def get_special_tokens_mask(self, input_ids: np.ndarray) -> np.ndarray:
        special_tokens_mask = np.zeros_like(input_ids, dtype=np.bool)

        for special_token in self.tokenizer.all_special_ids:
            special_tokens_mask |= input_ids == special_token

        return special_tokens_mask

    def mask_tokens(
        self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool")
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype(
            "bool"
        )
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(
            self.tokenizer.vocab_size, size=labels.shape, dtype="i4"
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices
