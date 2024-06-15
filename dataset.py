import random
import torch
from torch.utils.data import Dataset

from utils import split_text

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0)."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TextDataset(Dataset):
    def __init__(self, text, vocab_list, max_length=1024, mask_prob=0.15):
        self.text = text
        self.vocab_list = vocab_list
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.pad_idx = vocab_list.index("<PAD>")
        self.word_to_index = {word: word_index for word_index, word in enumerate(self.vocab_list)}

    def __getitem__(self, index):
        text = self.text[index]
        # print(len(text))
        text_splited = split_text(text)

        text_indices = [self.word_to_index.get(word, 1) for word in text_splited]
        masked_indices = []
        for i in text_indices:
            if random.random() < self.mask_prob:
                masked_indices.append(self.word_to_index["<MASK>"]) 
            else:
                masked_indices.append(i)

        # Pad sequences to max_length
        if len(masked_indices) < self.max_length:
            padding_length = self.max_length - len(masked_indices)
            masked_indices += [self.pad_idx] * padding_length
            text_indices += [self.pad_idx] * padding_length

        else:
            masked_indices = masked_indices[:self.max_length]
            text_indices = text_indices[:self.max_length]

        attention_mask = generate_square_subsequent_mask(len(masked_indices))

        return torch.tensor(masked_indices), torch.tensor(attention_mask), torch.tensor(text_indices)

    def __len__(self):
        return len(self.text)
