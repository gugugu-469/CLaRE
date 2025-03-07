import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from constant import text_start, left_bracket, right_bracket
from utils import sequence_padding

  
class CLaREDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train'
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.schema = data_processor.schema #spo
        self.args = args
        self.len_to_label = args.len_to_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_token_type_ids, batch_mask_ids = [], [], []
        batch_entity_labels, batch_rel_labels = [], []
        for item in examples:
            entity_labels,  rel_labels, input_ids, token_type_ids, attention_mask = item
            batch_entity_labels.append(entity_labels)
            batch_rel_labels.append(rel_labels)
            batch_token_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_mask_ids.append(attention_mask)
            

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()

        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_rel_labels = torch.tensor(sequence_padding(batch_rel_labels, seq_dims=2)).long()

        return batch_token_ids, batch_token_type_ids, batch_mask_ids, batch_entity_labels, batch_rel_labels
    
    
    @staticmethod
    def collate_fn_test(examples):
        return [item['text'] for item in examples]
    
    