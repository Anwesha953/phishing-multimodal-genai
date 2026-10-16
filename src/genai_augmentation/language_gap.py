import pandas as pd

idx = pd.read_csv('data/processed/image_features_index.csv')
lang = pd.read_csv('data/processed/pages_lang.csv')[['filename', 'lang', 'is_english']]
oof = pd.read_csv('results/fusion_oof_predictions.csv')
expl = pd.read_csv('results/fusion_explanations.csv')[['filename', 'predicted']]
assert len(oof) == len(idx) and (oof['url'].values == idx['url'].values).all(), 'files are out of step, rerun earlier cells'

d = idx[['filename', 'label']].copy()
d['y'] = (d['label'] == 'legit').astype(int)
d['Text only'] = (oof['Text only'].values >= 0.5).astype(int)
d['Image only (CLIP)'] = (oof['Image only'].values >= 0.5).astype(int)
d = d.merge(expl, on='filename', how='left')
d['Fused (stacked, host-only URL)'] = (d['predicted'] == 'legit').astype(int)
d = d.merge(lang, on='filename', how='left')
d['group'] = d['is_english'].map({1: 'English', 0: 'Non-English'})

models = ['Text only', 'Image only (CLIP)', 'Fused (stacked, host-only URL)']
rows = []
for m in models:
    row = {'model': m}
    for g in ['English', 'Non-English']:
        s = d[d['group'] == g]
        row[g + ' accuracy'] = round(float((s[m] == s['y']).mean()), 3)
        row[g + ' pages'] = len(s)
    rows.append(row)
res = pd.DataFrame(rows)
print(res.to_string(index=False))

print()
print('Phishing recall (share of phishing pages caught) by language group:')
for m in models:
    out = []
    for g in ['English', 'Non-English']:
        s = d[(d['group'] == g) & (d['y'] == 0)]
        out.append(f"{g} {float((s[m] == 0).mean()):.2f} (n={len(s)})")
    print(' ', m, '|', ' | '.join(out))
res.to_csv('results/language_gap.csv', index=False)
print('Saved results/language_gap.csv')
