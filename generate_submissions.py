import os
import math
import zipfile
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi 
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
import warnings
warnings.filterwarnings("ignore")
from utils import (
    prepare,
    detokenize_sentence,
    _decode_token_text,
    beam_search_translate_transformer,  # uses PyTorch only
)

# -------------------- CONFIG: edit these as needed --------------------

# Target languages to run
LANGS = ["Bengali","Hindi"]

# Model hyperparameters (must match CS779-Capstone-10)
HIDDEN_SIZE = 333
NHEADS = 9
NUM_LAYERS_ENC = 4
NUM_LAYERS_DEC = 4
DROPOUT = 0.15
DIM_FEEDFORWARD = 4 * HIDDEN_SIZE
MAX_LEN = 150

# Decoding parameters (you can change these)
DECODER_ARGS = {
    "beam_width": 5,
    "max_length": MAX_LEN,
    "length_penalty": 0.7,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2,
    "min_length": 5,
    "temperature": 1.0,
}

# Directory to look for weights. Defaults to /kaggle/working/ on Kaggle, else current dir.
def is_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

WEIGHTS_DIR = "/kaggle/working/" if is_kaggle() else "."

# Output location (same base as weights by default)
BASE_OUT = WEIGHTS_DIR

# ---------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, src, src_key_padding_mask=None):
        x = self.embed(src)
        x = self.pos(x)
        return self.enc(x, src_key_padding_mask=src_key_padding_mask)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, tgt_vocab_size, nhead, num_layers, dim_feedforward, dropout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        y = self.embed(tgt)
        y = self.pos(y)
        out = self.dec(
            y, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(out)

def _find_weights(weights_dir, lang, kind):
    """
    kind: 'encoder' or 'decoder'
    Prefer exact filename encoder_{lang}.pt / decoder_{lang}.pt
    If not found, try to pick the highest epoch file encoder_{lang}_epoch*.pt
    """
    base_exact = os.path.join(weights_dir, f"{kind}_{lang}.pt")
    if os.path.exists(base_exact):
        return base_exact

    # fallback: pick highest epoch if present
    best_path = None
    best_epoch = -1
    prefix = f"{kind}_{lang}_epoch"
    for fname in os.listdir(weights_dir):
        if fname.startswith(prefix) and fname.endswith(".pt"):
            try:
                num = fname[len(prefix):-3]
                num = int(num)
            except Exception:
                continue
            if num > best_epoch:
                best_epoch = num
                best_path = os.path.join(weights_dir, fname)
    if best_path is not None:
        return best_path
    else:
        return f"{kind}_{lang}"

def load_models_and_tokenizers(lang):
    # Load tokenizers and datasets using your pipeline
    id_val, en_tok_data, tgt_tok_data, _, test_ds = prepare(lang, vocab_size=15000)
    en_vocab_size = len(en_tok_data[0])
    tgt_vocab_size = len(tgt_tok_data[0])

    de_map = {_decode_token_text(tok): i for i, tok in enumerate(tgt_tok_data[0])}
    en_map = {_decode_token_text(tok): i for i, tok in enumerate(en_tok_data[0])}

    SOS_ID = de_map["<s>"]
    EOS_ID = de_map["</s>"]
    PAD_ID_TGT = de_map["<pad>"]
    PAD_ID_SRC = en_map["<pad>"]

    # Build models
    encoder = TransformerEncoder(
        en_vocab_size, HIDDEN_SIZE, NHEADS, NUM_LAYERS_ENC, DIM_FEEDFORWARD, DROPOUT, PAD_ID_SRC
    ).to(device)
    decoder = TransformerDecoder(
        HIDDEN_SIZE, tgt_vocab_size, NHEADS, NUM_LAYERS_DEC, DIM_FEEDFORWARD, DROPOUT, PAD_ID_TGT
    ).to(device)
    user_secrets = UserSecretsClient()
    # Load weights
    enc_path = _find_weights(WEIGHTS_DIR, lang, "encoder")
    dec_path = _find_weights(WEIGHTS_DIR, lang, "decoder")
    if not os.path.exists(enc_path) or not os.path.exists(dec_path):
        enc_path = hf_hub_download(
            repo_id="Arnab-Datta-240185/CS779-Capstone-Project",
            filename=f"encoder_{lang}.pt",
            repo_type="dataset",
            token=user_secrets.get_secret("hf_token"),
        )
        dec_path = hf_hub_download(
            repo_id="Arnab-Datta-240185/CS779-Capstone-Project",
            filename=f"decoder_{lang}.pt",
            repo_type="dataset",
            token=user_secrets.get_secret("hf_token"),
        )
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))
    encoder.eval()
    decoder.eval()

    return id_val, test_ds, encoder, decoder, en_tok_data, tgt_tok_data, PAD_ID_SRC, SOS_ID, EOS_ID

@torch.no_grad()
def translate_all(lang):
    id_val, test_ds, encoder, decoder, en_tok, tgt_tok, PAD_ID_SRC, SOS_ID, EOS_ID = load_models_and_tokenizers(lang)

    outputs = []
    for i in tqdm(range(len(test_ds)), desc=f"Translating [{lang}]"):
        src = test_ds[i][0].unsqueeze(0).to(device)  # [1, S]
        out_ids = beam_search_translate_transformer(
            encoder, decoder, src,
            SOS_ID, EOS_ID,
            src_pad_id=PAD_ID_SRC,
            beam_width=DECODER_ARGS["beam_width"],
            max_length=DECODER_ARGS["max_length"],
            length_penalty=DECODER_ARGS["length_penalty"],
            no_repeat_ngram_size=DECODER_ARGS["no_repeat_ngram_size"],
            repetition_penalty=DECODER_ARGS["repetition_penalty"],
            min_length=DECODER_ARGS["min_length"],
            temperature=DECODER_ARGS["temperature"],
        )
        outputs.append(detokenize_sentence(out_ids, tgt_tok))

    # Save per-language CSV
    lang_initial = "B" if lang.lower().startswith("beng") else "H"
    csv_path = os.path.join(BASE_OUT, f"answers{lang_initial}.csv")
    df = pd.DataFrame({"ID": id_val, "Translation": outputs})
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[{lang}] Saved: {csv_path}")
    return csv_path

def build_submission_zip():
    # Combine Bengali and Hindi CSVs
    path_b = os.path.join(BASE_OUT, "answersB.csv")
    path_h = os.path.join(BASE_OUT, "answersH.csv")
    if not (os.path.exists(path_b) and os.path.exists(path_h)):
        raise FileNotFoundError("answersB.csv or answersH.csv missing. Run both languages first.")

    df_b = pd.read_csv(path_b)
    df_h = pd.read_csv(path_h)
    df = pd.concat([df_b, df_h], ignore_index=True)

    combined_csv = os.path.join(BASE_OUT, "answersBH.csv")
    df.to_csv(combined_csv, index=False, encoding="utf-8")
    print(f"Combined results saved to {combined_csv}")

    # Write answer.csv in TSV format with header "ID\tTranslation"
    answer_path = os.path.join(BASE_OUT, "answer.csv")
    with open(answer_path, "w", encoding="utf-8") as f:
        f.write("ID\tTranslation\n")
        for _, row in df.iterrows():
            # Quote translation as in your notebooks
            f.write(f'{row["ID"]}\t"{row["Translation"]}"\n')

    # Zip it
    zip_path = os.path.join(BASE_OUT, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(answer_path, arcname=os.path.basename(answer_path))
    print(f"Created final submission file: {zip_path}")

if __name__ == "__main__":
    print(f"Device: {device}")
    for lang in LANGS:
        translate_all(lang)
    build_submission_zip()