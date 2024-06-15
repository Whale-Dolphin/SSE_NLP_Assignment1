import torch
import torch.nn as nn
import pickle
import math
import re
from model import MaskedLanguageModel, PositionalEncoding
from utils import split_text, clean_text

def load_vocab_list(vocab_list_path):
    with open(vocab_list_path, 'rb') as f:
        vocab_list = pickle.load(f)
    return vocab_list

def load_model(checkpoint_path, vocab_size, embedding_size, nhead, nhid, nlayers, device):
    model = MaskedLanguageModel(vocab_size, embedding_size, nhead, nhid, nlayers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def preprocess_text(text, vocab_list, max_length=1024):
    text = clean_text(text)
    words = split_text(text)
    pad_idx = vocab_list.index("<PAD>")
    word_to_index = {word: idx for idx, word in enumerate(vocab_list)}

    indices = [word_to_index.get(word, 1) for word in words]
    if len(indices) < max_length:
        padding_length = max_length - len(indices)
        indices += [pad_idx] * padding_length
        attention_mask = [1] * len(words) + [0] * padding_length
    else:
        indices = indices[:max_length]
        attention_mask = [1] * max_length

    return torch.tensor(indices).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)

def predict(model, text, vocab_list, device):
    text_indices, attention_mask = preprocess_text(text, vocab_list)

    text_indices[0][0] = 0
    print(text_indices)

    input_tensor = text_indices.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_indices = torch.argmax(output, dim=2).squeeze(0).cpu().numpy()

    print(predicted_indices.shape)
    print(predicted_indices)

    word_to_index = {word: idx for idx, word in enumerate(vocab_list)}

    index_to_word = {idx: word for word, idx in word_to_index.items()}
    predicted_words = [index_to_word[idx] for idx in predicted_indices if idx in index_to_word][:]

    return predicted_words


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spell Checking using Masked Language Model")
    parser.add_argument("--checkpoint", "-ckpt", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab file")
    parser.add_argument("--embedding_size", type=int, default=512, help="Size of the embedding layer")
    parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models")
    parser.add_argument("--nhid", type=int, default=2048, help="Dimension of the feedforward network model in nn.TransformerEncoder")
    parser.add_argument("--nlayers", type=int, default=6, help="Number of nn.TransformerEncoderLayer in nn.TransformerEncoder")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of input sequences")
    parser.add_argument("--text", type=str, required=True, help="Text to perform spell checking on")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.vocab_path, 'rb') as f:
        vocab_list = pickle.load(f)
    model = load_model(args.checkpoint, len(vocab_list), args.embedding_size, args.nhead, args.nhid, args.nlayers, device)
    corrected_text = predict(model, args.text, vocab_list, device)

    print("Original text:", args.text)
    print("Corrected text:", " ".join(corrected_text))
