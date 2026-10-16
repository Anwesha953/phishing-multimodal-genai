import os, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NAME = 'facebook/nllb-200-distilled-600M'
LANGS = {'de': 'deu_Latn', 'es': 'spa_Latn', 'fr': 'fra_Latn', 'pt': 'por_Latn',
         'nl': 'nld_Latn', 'id': 'ind_Latn', 'ja': 'jpn_Jpan'}
OUT = 'data/synthetic/translated_pages.csv'
CHUNK_CHARS = 450
BATCH = 16

if not torch.cuda.is_available():
    raise SystemExit('No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then rerun the setup cells.')
device = 'cuda'

idx = pd.read_csv('data/processed/image_features_index.csv')[['filename', 'label']]
lang = pd.read_csv('data/processed/pages_lang.csv')[['filename', 'lang', 'text']]
src = idx.merge(lang, on='filename', how='left')
src = src[src['lang'] == 'en'].copy()
src['text'] = src['text'].fillna('').astype(str)

def chunk(t):
    t = re.sub(r'\s+', ' ', t).strip()
    if len(t) <= CHUNK_CHARS:
        return t
    cut = t.rfind(' ', 0, CHUNK_CHARS)
    return t[:cut if cut > 200 else CHUNK_CHARS]

src['chunk'] = src['text'].apply(chunk)
src = src[src['chunk'].str.len() >= 40].reset_index(drop=True)
print('English source pages:', len(src), src['label'].value_counts().to_dict())

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['source_filename', 'label', 'lang', 'text'])
tok = AutoTokenizer.from_pretrained(NAME, src_lang='eng_Latn')
model = AutoModelForSeq2SeqLM.from_pretrained(NAME).to(device).half().eval()

for code, nllb in LANGS.items():
    if code in set(done['lang']):
        print('skip', code, '(already done)')
        continue
    outs = []
    bos = tok.convert_tokens_to_ids(nllb)
    for i in range(0, len(src), BATCH):
        texts = src['chunk'].iloc[i:i + BATCH].tolist()
        enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=200).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=220, num_beams=2)
        outs += tok.batch_decode(gen, skip_special_tokens=True)
    new = pd.DataFrame({'source_filename': src['filename'], 'label': src['label'], 'lang': code, 'text': outs})
    done = pd.concat([done, new], ignore_index=True)
    done.to_csv(OUT, index=False)
    print('done', code, '| total rows so far:', len(done))
print('Finished. Saved', OUT)
