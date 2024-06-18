import math
import re

from loguru import logger
import torch
import torch.nn as nn
import pickle

from model import MaskedLanguageModel, PositionalEncoding
from utils import split_text, clean_text


log_file_path = ".log/inference.log"
logger.add(log_file_path, format="{time} {level} {message}", level="INFO", rotation='50MB')


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

    # text_indices[0][0] = 0
    # print(text_indices)

    input_tensor = text_indices.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_indices = torch.argmax(output, dim=2).squeeze(0).cpu().numpy()

    # print(predicted_indices.shape)
    # print(predicted_indices)

    word_to_index = {word: idx for idx, word in enumerate(vocab_list)}

    logger.info(f"{predicted_indices}")

    index_to_word = {idx: word for word, idx in word_to_index.items()}
    predicted_words = [index_to_word[idx] for idx in predicted_indices if idx in index_to_word][:]

    return predicted_words


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Spell Checking using Masked Language Model")
#     parser.add_argument("--checkpoint", "-ckpt", type=str, required=True, help="Path to the trained model checkpoint")
#     parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab file")
#     parser.add_argument("--embedding_size", type=int, default=512, help="Size of the embedding layer")
#     parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models")
#     parser.add_argument("--nhid", type=int, default=2048, help="Dimension of the feedforward network model in nn.TransformerEncoder")
#     parser.add_argument("--nlayers", type=int, default=6, help="Number of nn.TransformerEncoderLayer in nn.TransformerEncoder")
#     parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of input sequences")
#     parser.add_argument("--text", type=str, required=True, help="Text to perform spell checking on")

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     with open(args.vocab_path, 'rb') as f:
#         vocab_list = pickle.load(f)
#     model = load_model(args.checkpoint, len(vocab_list), args.embedding_size, args.nhead, args.nhid, args.nlayers, device)
#     corrected_text = predict(model, args.text, vocab_list, device)

#     print("Original text:", args.text)
#     print("Corrected text:", " ".join(corrected_text))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spell Checking using Masked Language Model")
    parser.add_argument("--checkpoint", "-ckpt", type=str, default='/root/NLP_Assignment/Assignment1/Assignment1/transformer/checkpoints/spell_correct_base.pth', help="Path to the trained model checkpoint")
    parser.add_argument("--vocab_path", type=str, default='/root/NLP_Assignment/Assignment1/Assignment1/transformer/checkpoints/vocab_list.pkl', help="Path to the vocab file")
    parser.add_argument("--embedding_size", type=int, default=512, help="Size of the embedding layer")
    parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models")
    parser.add_argument("--nhid", type=int, default=2048, help="Dimension of the feedforward network model in nn.TransformerEncoder")
    parser.add_argument("--nlayers", type=int, default=6, help="Number of nn.TransformerEncoderLayer in nn.TransformerEncoder")
    parser.add_argument("--max_length", '-ml', type=int, default=1024, help="Maximum length of input sequences")
    parser.add_argument("--text_path", type=str, required=True, help="Text to perform spell checking on")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.vocab_path, 'rb') as f:
        vocab_list = pickle.load(f)

    lines = []
    with open(args.text_path, 'r', encoding='utf-8') as f:
        text = f.readlines()
    for line in text:
        lines.append(line.strip().split('\t')[2])
    # print(text[0])
    # print(lines[0])

    # text_len = []
    text_split = []
    for line in lines:
        line = split_text(line)
        text_split.append(line)
        # text_len.append(len(line))

    corrected_list = []

    model = load_model(args.checkpoint, len(vocab_list), args.embedding_size, args.nhead, args.nhid, args.nlayers, device)
    with open("result.txt", "w", encoding="utf-8") as out_file:
        for i, line in enumerate(text_split):
            count = 0
            new_line = []
            for j, word in enumerate(line):
                masked_line = line.copy()
                # 将当前词替换为<MASK>
                masked_line[j] = '<MASK>'
                masked_text = ' '.join(masked_line)
                logger.debug(f"{masked_text}")
                
                corrected_text = predict(model, masked_text, vocab_list, device)
                logger.debug(f"{i}: {corrected_text}")
                
                predicted_word = corrected_text[j]
                is_correct = (predicted_word == word)
                logger.info(f"{predicted_word}")

                new_line.append(predicted_word)

                if not is_correct:
                    count += 1
                    logger.info(f"text {i} has {count} spell error")
                    # line[j] = predicted_word
                    line_copy = line.copy()
                    line_copy[j] = predicted_word
                    logger.info(f"{line_copy}")
                
            # sentence = ' '.join(line[:text_len[i]])
            sentence = ' '.join(new_line[:])
            corrected_line = f"{i+1}\t{sentence}"
            out_file.write(corrected_line + "\n")

    print("Inference completed.")
