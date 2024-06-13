import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_vocab_list(text):
    vocab_set = set()

    for line in text:
        for word in line:
            vocab_set.add(word)

    vocab_list = list(vocab_set)

    return vocab_list