import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def split_text(text):
    words = text.split(' ')
    words = [word for word in words if word != '']
    return words


def get_vocab_list(text, existing_vocab=None):
    if existing_vocab:
        vocab_set = set(existing_vocab)
    else:
        vocab_set = set()

    for line in text:
        line = clean_text(line)
        words = split_text(line)
        for word in words:
            vocab_set.add(word)

    vocab_list = list(vocab_set)

    return vocab_list