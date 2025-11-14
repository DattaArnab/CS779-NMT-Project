import torch
import torch.nn as nn
import math
import pandas as pd
import os
from tqdm import tqdm
import traceback
from utils import prepare, save_model, translate, _decode_token_text

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, src, src_key_padding_mask=None):
        x = self.embed(src)  # [B, S, d]
        x = self.pos(x)
        memory = self.enc(x, src_key_padding_mask=src_key_padding_mask)  # [B, S, d]
        return memory

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, tgt_vocab_size, nhead, num_layers, dim_feedforward, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        y = self.embed(tgt)  # [B, T, d]
        y = self.pos(y)
        out = self.dec(
            y, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # [B, T, d]
        logits = self.generator(out)  # [B, T, V]
        return logits

def _subsequent_mask(sz, device):
    mask = torch.full((sz, sz), float('-inf'), device=device)
    mask = torch.triu(mask, 1)
    return mask

def train_language(lang, device_id, hp):
    try:
        if device_id >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device_id)
        else:
            device = torch.device('cpu')

        base = "/kaggle/working/" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else ""

        print(f"[{lang}] Preparing data on {device}...")
        id_val, en_tok_data, target_tok_data, train_dl, test_ds = prepare(lang, vocab_size=hp.get('vocab_size', 15000))

        # Vocab sizes and special IDs
        en_vocab_size = len(en_tok_data[0])
        de_vocab_size = len(target_tok_data[0])
        de_vocab_map = {_decode_token_text(tok): i for i, tok in enumerate(target_tok_data[0])}
        en_vocab_map = {_decode_token_text(tok): i for i, tok in enumerate(en_tok_data[0])}

        SOS_ID = de_vocab_map["<s>"]
        EOS_ID = de_vocab_map["</s>"]
        PAD_ID_TGT = de_vocab_map["<pad>"]
        PAD_ID_SRC = en_vocab_map["<pad>"]

        # Transformer hyperparameters
        d_model = hp.get('hidden_size', 300)
        nhead = hp.get('nheads', 8)
        num_layers_enc = hp.get('num_layers_enc', 4)
        num_layers_dec = hp.get('num_layers_dec', 4)
        dim_ff = hp.get('dim_feedforward', 4 * d_model)
        dropout = hp.get('dropout_rate', 0.1)

        encoder = TransformerEncoder(en_vocab_size, d_model, nhead, num_layers_enc, dim_ff, dropout, PAD_ID_SRC).to(device)
        decoder = TransformerDecoder(d_model, de_vocab_size, nhead, num_layers_dec, dim_ff, dropout, PAD_ID_TGT).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID_TGT, reduction='sum')
        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=hp.get('learning_rate', 1e-4), weight_decay=hp.get('weight_decay', 5e-6))
        dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=hp.get('learning_rate', 1e-4), weight_decay=hp.get('weight_decay', 5e-6))

        epochs = hp.get('epochs', 60)
        clip_value = hp.get('clip_value', 1.0)

        for epoch in range(epochs):
            encoder.train()
            decoder.train()
            epoch_losses = []

            for batch in tqdm(train_dl, desc=f"Epoch {epoch+1} {lang}"):
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                src = batch[0].to(device)  # [B, S]
                tgt = batch[1].to(device)  # [B, S]

                tgt_in = tgt[:, :-1]      # [B, S-1], starts with <s>
                tgt_out = tgt[:, 1:]      # [B, S-1], next tokens including </s>

                src_key_padding_mask = (src == PAD_ID_SRC)            # [B, S]
                tgt_key_padding_mask = (tgt_in == PAD_ID_TGT)         # [B, S-1]
                T = tgt_in.size(1)
                tgt_mask = _subsequent_mask(T, device)                # [T, T]

                memory = encoder(src, src_key_padding_mask=src_key_padding_mask)  # [B, S, d]
                logits = decoder(
                    tgt_in, memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )  # [B, T, V]

                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                nonpad = (tgt_out != PAD_ID_TGT).sum().clamp_min(1)
                loss = loss / nonpad

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_value)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_value)
                enc_optimizer.step()
                dec_optimizer.step()

                if not math.isnan(loss.item()):
                    epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            print(f"Epoch {epoch+1} [{lang}] Avg loss: {avg_loss:.4f}")

            if avg_loss < float('inf'):
                save_model(encoder, decoder, lang, epoch+1)

        save_model(encoder, decoder, lang)
        print(f"\n[{lang}] Translating with beam search...")
        # Pass both tokenizers so translate can build src masks
        val_outs = translate(encoder, decoder, test_ds, SOS_ID, EOS_ID, target_tok_data, en_tok_data, use_beam_search=True)

        df = pd.DataFrame({"ID": id_val, "Translation": val_outs})
        df.to_csv(f"{base}answers{lang[0]}.csv", index=False)
        print(f"[{lang}] Saved translations.")

    except Exception:
        print(f"--- FATAL ERROR IN PROCESS FOR {lang} ---")
        traceback.print_exc()
