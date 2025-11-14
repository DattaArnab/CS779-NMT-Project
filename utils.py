import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import re
import unicodedata
import json
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pickle
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

WHITESPACE_MARKER = "▁"

def _decode_token_text(token_bytes):
    if isinstance(token_bytes, bytes):
        try:
            return token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return str(token_bytes)[2:-1]
    return str(token_bytes)

def detokenize_sentence(tokens, tokenizer_data):
    vocab, _ = tokenizer_data
    id_to_text = [_decode_token_text(tok) for tok in vocab]
    byte_array = bytearray()
    for token_id in tokens:
        if 0 <= token_id < len(id_to_text):
            token_text = id_to_text[token_id]
            if token_text not in ("<s>", "</s>", "<pad>", "<unk>"):
                byte_array.extend(vocab[token_id])
    detok = byte_array.decode('utf-8', errors='replace')
    detok = detok.replace(WHITESPACE_MARKER, " ").strip()
    return re.sub(r'\s+', ' ', detok)

def encode_and_pad(vocab_map, sent_ids, max_length):
    sos = [vocab_map["<s>"]]
    eos = [vocab_map["</s>"]]
    pad = [vocab_map["<pad>"]]
    if len(sent_ids) < max_length - 2:
        n_pads = max_length - 2 - len(sent_ids)
        return sos + sent_ids + eos + pad * n_pads
    else:
        truncated = sent_ids[:max_length - 2]
        return sos + truncated + eos

def prepare(lang, vocab_size=15000):
    user_secrets = UserSecretsClient()
    lng = hf_hub_download(
        repo_id="Arnab-Datta-240185/CS779-Capstone-Project",
        filename=f"{lang}_prepared_data.pkl",
        repo_type="dataset",
        token=user_secrets.get_secret("hf_token")
    )
    with open(lng, "rb") as f:
        dat = pickle.load(f)
    id_val = dat["id_val"]
    en_tokenizer_data = dat["en_tokenizer_data"]
    target_tokenizer_data = dat["target_tokenizer_data"]
    train_dl = dat["train_dl"]
    test_ds = dat["test_ds"]
    return id_val, en_tokenizer_data, target_tokenizer_data, train_dl, test_ds

def save_model(encoder, decoder, lang, epoch=None):
    base = "/kaggle/working/" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else ""
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    torch.save(encoder.state_dict(), f"{base}encoder_{lang}{suffix}.pt")
    torch.save(decoder.state_dict(), f"{base}decoder_{lang}{suffix}.pt")
    print(f"Model saved for {lang}{suffix}")

def _subsequent_mask(sz, device):
    # Additive mask for causal self-attention: -inf above diagonal
    mask = torch.full((sz, sz), float('-inf'), device=device)
    mask = torch.triu(mask, 1)
    return mask

def _makes_repeat_ngram(seq, n, new_tok):
    if n <= 0:
        return False
    tmp = seq + [new_tok]
    if len(tmp) < n:
        return False
    last = tuple(tmp[-n:])
    for i in range(len(tmp) - n):
        if tuple(tmp[i:i+n]) == last:
            return True
    return False

@torch.no_grad()
def beam_search_translate_transformer(
    encoder, decoder, src_tensor,
    SOS_ID, EOS_ID,
    src_pad_id=None,
    beam_width=5, max_length=150,
    length_penalty=0.7,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    min_length=5,
    temperature=1.0,
):
    device = src_tensor.device
    src_key_padding_mask = (src_tensor == src_pad_id) if (src_pad_id is not None) else None
    memory = encoder(src_tensor, src_key_padding_mask=src_key_padding_mask)  # [1, S, d]

    beams = [([SOS_ID], 0.0)]
    completed = []

    for step in range(1, max_length + 1):
        candidates = []
        B = len(beams)
        mem = memory.repeat(B, 1, 1)
        if src_key_padding_mask is not None:
            mem_kpm = src_key_padding_mask.repeat(B, 1)
        else:
            mem_kpm = None

        max_len = max(len(b[0]) for b in beams)
        T = max_len
        tgt_tensor = torch.full((B, T), SOS_ID, device=device, dtype=torch.long)
        for i, (tokens, _) in enumerate(beams):
            tgt_tensor[i, :len(tokens)] = torch.tensor(tokens, device=device, dtype=torch.long)

        tgt_mask = _subsequent_mask(T, device)

        logits = decoder(
            tgt_tensor,
            mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=mem_kpm
        )  # [B, T, V]
        step_logits = logits[:, -1, :] / max(temperature, 1e-6)
        log_probs = torch.log_softmax(step_logits, dim=-1)

        for b_idx, (tokens, score) in enumerate(beams):
            if repetition_penalty != 1.0 and tokens:
                uniq = list(set(tokens))
                log_probs[b_idx, uniq] = log_probs[b_idx, uniq] - math.log(repetition_penalty)

            if step < min_length:
                log_probs[b_idx, EOS_ID] = -1e9

            top_log_probs, top_idx = log_probs[b_idx].topk(beam_width)
            for i in range(beam_width):
                tok = top_idx[i].item()
                if no_repeat_ngram_size and _makes_repeat_ngram(tokens, no_repeat_ngram_size, tok):
                    continue
                new_tokens = tokens + [tok]
                new_score = score + top_log_probs[i].item()
                if tok == EOS_ID:
                    completed.append((new_tokens, new_score))
                else:
                    candidates.append((new_tokens, new_score))

        if not candidates and completed:
            break

        def norm_score(s, l):
            return s / (max(1, l) ** length_penalty)

        if candidates:
            candidates.sort(key=lambda x: norm_score(x[1], len(x[0])), reverse=True)
            beams = candidates[:beam_width]
        else:
            break

        if len(completed) >= beam_width:
            break

    final = completed if completed else beams
    final.sort(key=lambda x: (x[0][-1] == EOS_ID, x[1] / (max(1, len(x[0])) ** length_penalty)), reverse=True)
    best_tokens = final[0][0]
    trimmed = []
    for t in best_tokens:
        if t == SOS_ID:
            continue
        if t == EOS_ID:
            break
        trimmed.append(t)
    return trimmed

@torch.no_grad()
def translate(encoder, decoder, test_ds, SOS_ID, EOS_ID, target_tokenizer_data, src_tokenizer_data, use_beam_search=True):
    encoder.eval()
    decoder.eval()
    val_outs = []
    device = next(encoder.parameters()).device

    src_vocab_map = {_decode_token_text(tok): i for i, tok in enumerate(src_tokenizer_data[0])}
    SRC_PAD_ID = src_vocab_map.get("<pad>", None)

    for i in tqdm(range(len(test_ds)), desc=f"Translating"):
        input_tensor = test_ds[i][0].unsqueeze(0).to(device)  # [1, S]
        if use_beam_search:
            try:
                result_tokens = beam_search_translate_transformer(
                    encoder, decoder, input_tensor,
                    SOS_ID, EOS_ID,
                    src_pad_id=SRC_PAD_ID,
                    beam_width=5, max_length=150,
                    length_penalty=0.7,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    min_length=5,
                    temperature=1.0
                )
            except Exception:
                mem = encoder(input_tensor, src_key_padding_mask=(input_tensor == SRC_PAD_ID) if SRC_PAD_ID is not None else None)
                result_tokens = []
                cur = torch.tensor([[SOS_ID]], device=device, dtype=torch.long)
                for _ in range(150):
                    T = cur.size(1)
                    tgt_mask = _subsequent_mask(T, device)
                    logits = decoder(cur, mem, tgt_mask=tgt_mask)
                    next_id = logits[:, -1, :].argmax(-1)
                    nid = next_id.item()
                    if nid == EOS_ID:
                        break
                    result_tokens.append(nid)
                    cur = torch.cat([cur, next_id.view(1,1)], dim=1)
        else:
            mem = encoder(input_tensor, src_key_padding_mask=(input_tensor == SRC_PAD_ID) if SRC_PAD_ID is not None else None)
            result_tokens = []
            cur = torch.tensor([[SOS_ID]], device=device, dtype=torch.long)
            for _ in range(150):
                T = cur.size(1)
                tgt_mask = _subsequent_mask(T, device)
                logits = decoder(cur, mem, tgt_mask=tgt_mask)
                next_id = logits[:, -1, :].argmax(-1)
                nid = next_id.item()
                if nid == EOS_ID:
                    break
                result_tokens.append(nid)
                cur = torch.cat([cur, next_id.view(1,1)], dim=1)

        translated_sentence = detokenize_sentence(result_tokens, target_tokenizer_data)
        val_outs.append(translated_sentence)

    return val_outs
