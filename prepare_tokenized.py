#imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download,HfApi
from collections import Counter, defaultdict
import os
import zipfile
from tqdm import tqdm
import unicodedata
import pickle
import random
random.seed(42)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

# Reserved tokens
RESERVED_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
WHITESPACE_MARKER = "▁"
seq_length = 150
batch_size = 64
base = "/kaggle/working/" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else ""

#helper functions
def fix_quotes(text):
    """Standardizes various quote characters to a single format."""
    text = re.sub(r"(``|“|”|‘‘|’’|(?:' ?'))", ' " ', text)
    text = re.sub(r"(`|'|‘|’)", " ' ", text)
    return text

def clean_text(text):
    """Applies Unicode normalization, quote fixing, and ASCII conversion."""
    text = unicodedata.normalize('NFKD', text)
    text = fix_quotes(text)
    # Decode after encoding to handle potential errors gracefully
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    """Applies Unicode NFKC normalization and lowercasing."""
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.casefold()

def _decode_token_text(token_bytes):
    """Helper to decode bytes to a string, with a fallback."""
    if isinstance(token_bytes, bytes):
        try:
            return token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback for bytes that aren't valid UTF-8
            return str(token_bytes)[2:-1]
    return str(token_bytes)

def load_training_data(train_path):
    """Loads raw text from a file for tokenizer training."""
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def train_sp_tokenizer(text, vocab_size):
    """Trains a SentencePiece-style tokenizer using a BPE approach."""
    # Initialize vocabulary with reserved tokens and all single bytes
    vocab = [token.encode("utf-8") for token in RESERVED_TOKENS]
    vocab.extend([bytes([i]) for i in range(256)])

    words = text.split()
    words = [WHITESPACE_MARKER + word for word in words if word]
    word_freqs = Counter(words)
    word_to_tokens = {
        w: [byte + len(RESERVED_TOKENS) for byte in w.encode("utf-8")]
        for w in word_freqs
    }

    stats = Counter()
    pair_to_words = defaultdict(set)
    for w, tokens in word_to_tokens.items():
        freq = word_freqs[w]
        for i in range(len(tokens) - 1):
            p = (tokens[i], tokens[i + 1])
            stats[p] += freq
            pair_to_words[p].add(w)

    merges = {}
    num_merges = max(0, vocab_size - len(vocab))

    for _ in tqdm(range(num_merges),"Training SP BPE"):
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        a, b = best_pair
        words_affected = list(pair_to_words.get(best_pair, []))

        merged_bytes = vocab[a] + vocab[b]
        vocab.append(merged_bytes)
        new_id = len(vocab) - 1
        merges[new_id] = best_pair

        for w in words_affected:
            freq = word_freqs[w]
            old_tokens = word_to_tokens[w]
            # Update stats by removing old pairs
            for i in range(len(old_tokens) - 1):
                p = (old_tokens[i], old_tokens[i + 1])
                if p in stats:
                    stats[p] -= freq
                    if stats[p] <= 0:
                        stats.pop(p)
                if p in pair_to_words:
                    pair_to_words[p].discard(w)
                    if not pair_to_words[p]:
                        pair_to_words.pop(p)

            # Create new token list for the word
            new_tokens = []
            i = 0
            while i < len(old_tokens):
                if i < len(old_tokens) - 1 and (old_tokens[i], old_tokens[i+1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            word_to_tokens[w] = new_tokens

            # Add stats for new pairs
            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i + 1])
                stats[p] += freq
                pair_to_words[p].add(w)

        stats.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)

    return vocab, merges

def tokenize_sentence(text, tokenizer_data):
    """Tokenizes a single sentence using the trained tokenizer."""
    vocab, _ = tokenizer_data
    vocab_map = { _decode_token_text(tok): i for i, tok in enumerate(vocab) }

    text = normalize_text(text)
    processed_text = WHITESPACE_MARKER + re.sub(r'\s+', WHITESPACE_MARKER, text.strip())

    ids = []
    i = 0
    while i < len(processed_text):
        # Find the longest matching token in the vocabulary
        longest_match = ""
        for j in range(len(processed_text), i, -1):
            sub = processed_text[i:j]
            if sub in vocab_map:
                longest_match = sub
                break

        if longest_match:
            ids.append(vocab_map[longest_match])
            i += len(longest_match)
        else:
            # Fallback to unknown token for the character
            ids.append(vocab_map["<unk>"])
            i += 1

    return ids

def detokenize_sentence(tokens, tokenizer_data):
    """Converts a sequence of token IDs back to a text sentence."""
    vocab, _ = tokenizer_data
    # Use a simple list lookup for id -> text
    id_to_text = [_decode_token_text(tok) for tok in vocab]

    byte_array = bytearray()
    for token_id in tokens:
        if 0 <= token_id < len(id_to_text):
            token_text = id_to_text[token_id]
            # Skip reserved tokens in the final output
            if token_text not in ("<s>", "</s>", "<pad>", "<unk>"):
                # The tokenizer vocabulary is stored as bytes, so we append the bytes directly
                # Find the original bytes from the vocab list
                byte_array.extend(vocab[token_id])

    # Decode the full byte array and clean up whitespace
    detokenized_text = byte_array.decode('utf-8', errors='replace')
    detokenized_text = detokenized_text.replace(WHITESPACE_MARKER, " ").strip()
    return re.sub(r'\s+', ' ', detokenized_text)

def encode_and_pad(vocab, sent_ids, max_length):
    """Encode and pad sequences for batch processing"""
    sos = [vocab["<s>"]]
    eos = [vocab["</s>"]]
    pad = [vocab["<pad>"]]

    if len(sent_ids) < max_length - 2: # -2 for SOS and EOS
        n_pads = max_length - 2 - len(sent_ids)
        return sos + sent_ids + eos + pad * n_pads
    else: # sent is longer than max_length; truncating
        truncated = sent_ids[:max_length - 2]
        return sos + truncated + eos
    
def save_corpus(data,dataVal):
    source_sentences_train=[]
    source_sentences_val=[]
    hindi_sentences_train=[]
    bengali_sentences_train=[]
    for language_pair, language_data in data.items():
        for entry_id, entry_data in language_data.get("Train", {}).items():
            source_sentences_train.append(clean_text(entry_data["source"].lower()))
            if language_pair == f"English-Bengali":
                bengali_sentences_train.append(fix_quotes(entry_data["target"]))
            elif language_pair == f"English-Hindi":
                hindi_sentences_train.append(fix_quotes(entry_data["target"]))

    for language_pair, language_data in dataVal.items():
        for entry_id, entry_data in language_data.get("Validation", {}).items():
            source_sentences_val.append(clean_text(entry_data["source"].lower()))

    with open(os.path.join(base, "Englishcorpus.txt"), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(source_sentences_train + source_sentences_val))

    with open(os.path.join(base, "Bengalicorpus.txt"), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(bengali_sentences_train))

    with open(os.path.join(base, "Hindicorpus.txt"), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(hindi_sentences_train))

    print("Corpus files created for all language.")

def prepare(lang,data,dataTest,vocab_size=15000, en_vocab_size=15000):
    """
    Prepares data for a given language:
    1. Cleans and extracts sentences.
    2. Writes corpus files for tokenizer training.
    3. Trains SentencePiece tokenizers.
    4. Tokenizes and encodes all data.
    5. Creates PyTorch DataLoaders.
    """
    # --- Step 1: Extract and clean sentences ---
    source_sentences_train, target_sentences_train = [], []
    source_sentences_test = []
    id_test = []

    for language_pair, language_data in data.items():
        if language_pair == f"English-{lang}":
            for entry_id, entry_data in language_data.get("Train", {}).items():
                source_sentences_train.append(clean_text(entry_data["source"].lower()))
                target_sentences_train.append(fix_quotes(entry_data["target"]))

    for language_pair, language_data in dataTest.items():
        if language_pair == f"English-{lang}":
            for entry_id, entry_data in language_data.get("Test", {}).items():
                source_sentences_test.append(clean_text(entry_data["source"].lower()))
                id_test.append(entry_id)

    # --- Step 2: Write corpus files for training tokenizers ---
    en_corpus_path = os.path.join(base, "Englishcorpus.txt")
    target_corpus_path = os.path.join(base, f"{lang}corpus.txt")

    # --- Step 3: Train SentencePiece tokenizers ---
    en_text = load_training_data(en_corpus_path)
    print(f"Training English-{lang}")
    en_vocab, en_merges = train_sp_tokenizer(normalize_text(en_text), en_vocab_size)

    en_tokenizer_data = (en_vocab, en_merges)
    en_word2index = { _decode_token_text(tok): i for i, tok in enumerate(en_vocab) }

    target_text = load_training_data(target_corpus_path)
    target_vocab, target_merges = train_sp_tokenizer(normalize_text(target_text), vocab_size)

    target_tokenizer_data = (target_vocab, target_merges)
    de_word2index = { _decode_token_text(tok): i for i, tok in enumerate(target_vocab) }

    print("SentencePiece tokenizers trained.")

    # --- Step 4: Tokenize and encode datasets ---
    en_train_tokenized = [tokenize_sentence(s, en_tokenizer_data) for s in source_sentences_train]
    de_train_tokenized = [tokenize_sentence(s, target_tokenizer_data) for s in target_sentences_train]
    en_test_tokenized = [tokenize_sentence(s, en_tokenizer_data) for s in source_sentences_test]

    en_train_encoded = [encode_and_pad(en_word2index, s, seq_length) for s in en_train_tokenized]
    de_train_encoded = [encode_and_pad(de_word2index, s, seq_length) for s in de_train_tokenized]
    en_test_encoded = [encode_and_pad(en_word2index, s, seq_length) for s in en_test_tokenized]

    # --- Step 5: Create DataLoaders ---
    train_x = np.array(en_train_encoded)
    train_y = np.array(de_train_encoded)
    test_x = np.array(en_test_encoded)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_ds = TensorDataset(torch.from_numpy(test_x))
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    with open(os.path.join(base,f"{lang}_prepared_data.pkl"), "wb") as f:
        pickle.dump({
            "id_val": id_test,
            "en_tokenizer_data": en_tokenizer_data,
            "target_tokenizer_data": target_tokenizer_data,
            "train_dl": train_dl,
            "test_ds": test_ds
        }, f)
    api = HfApi()
    user_secrets = UserSecretsClient()
    api.upload_file(
        path_or_fileobj=os.path.join(base,f"{lang}_prepared_data.pkl"),
        path_in_repo=f"{lang}_prepared_data.pkl",
        repo_id="Arnab-Datta-240185/CS779-Capstone-Project",
        repo_type="dataset",
        commit_message=f"Add preprocessed {lang} dataset",
        token=user_secrets.get_secret("hf_token")
    )
    print("Uploaded to hugging face successfully.")
    return id_test, en_tokenizer_data, target_tokenizer_data, train_dl, test_ds

if __name__=="__main__":
    # file_path = <Train dataset path> eg "train_data1.json"
    user_secrets = UserSecretsClient()
    file_path = hf_hub_download(
        repo_id="Arnab-Datta-240185/CS779-Capstone-Project",
        filename="train_data1.json",
        repo_type="dataset",
        token=user_secrets.get_secret("hf_token")
    )

    # Open and load the JSON content
    with open(file_path, "r") as file:
        data = json.load(file)

    # file_path_val = <Validation dataset path> eg "val_data1.json"
    file_path_val = hf_hub_download(
        repo_id="Arnab-Datta-240185/CS779-Capstone-Project",
        filename="val_data1.json",
        repo_type="dataset",
        token=user_secrets.get_secret("hf_token")
    )

    # Open and load the JSON content
    with open(file_path_val, "r") as file:
        dataVal = json.load(file)

    # file_path_test = <Test dataset path> eg "test_data1.json"
    file_path_test = hf_hub_download(
        repo_id="Arnab-Datta-240185/CS779-Capstone-Project",
        filename="test_data1.json",
        repo_type="dataset",
        token=user_secrets.get_secret("hf_token")
    )

    # Open and load the JSON content
    with open(file_path_test, "r") as file:
        dataTest = json.load(file)
    
    print("Dataset loaded")

    save_corpus(data,dataVal)
    langs = ["Bengali","Hindi"]
    for lang in langs:
        prepare(lang,data,dataTest,vocab_size=15000, en_vocab_size=15000)

