from torch.utils.data import Dataset


def split_text(text):
    words = text.split(' ')
    words = [word for word in words if word != '']
    return words


class TextDataset(Dataset):
    def __init__(self,
                 text,
                 vocab_list):
        self.text = text
        self.vocab_list = vocab_list

    def __getitem__(self, index):
        text = self.text[index]
        text_splited = split_text(text)

        word_to_index = {word: word_index for word_index, word in enumerate(self.vocab_list)}
        text_clean = [word_to_index.get(word, 1) for word in text_splited]

        return text_clean

    
    def __len__(self):
        return len(self.text)
