import tensorflow as tf
import torch
from torch.utils.data import IterableDataset

from .utils import mask_tokens


class TFRecordDataset(IterableDataset):
    def __init__(
            self,
            filenames,
            compression_type="GZIP",
    ):
        self.filenames = filenames

        self.dataset = tf.data.TFRecordDataset(filenames, compression_type=compression_type)
        self.dataset = self.dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    @staticmethod
    def _parse_function(example_proto):
        features = {
            'text': tf.io.FixedLenFeature([512], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return parsed_features['text']

    def __iter__(self):
        for x in self.dataset:
            yield torch.tensor(x.numpy(), dtype=torch.long)


class TFRecordCollator:
    def __init__(
            self,
            tokenizer,
            mlm_probability=0.15,
            replace_prob=0.1,
            orginal_prob=0.1,
            ignore_index=-100,
    ):
        self.tokenizer = tokenizer
        self.special_token_indices = tokenizer.all_special_ids
        self.mlm_probability = mlm_probability
        self.replace_prob = replace_prob
        self.orginal_prob = orginal_prob
        self.ignore_index = ignore_index

    def __call__(self, batch):
        batch = torch.stack(batch, dim=0)

        masked_batch, labels, mlm_mask = mask_tokens(
            batch,
            self.tokenizer.mask_token_id,
            len(self.tokenizer),
            self.special_token_indices,
            self.mlm_probability,
            self.replace_prob,
            self.orginal_prob,
            self.ignore_index,
        )

        return {
            'input_ids': masked_batch,
            'labels': labels,
            'mlm_mask': mlm_mask,
        }
