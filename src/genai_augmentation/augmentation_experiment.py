import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold

idx = pd.read_csv('data/processed/image_features_index.csv')
lang = pd.read_csv('data/processed/pages_lang.csv')[['filename', 'is_english', 'text']]
df = idx.merge(lang, on='filename', how='left')
df['text'] = df['text'].fillna('').astype(str)
syn = pd.read_csv('data/synthetic/translated_pages.csv')
syn['text'] = syn['text'].fillna('').astype(str)
y = (df['label'] == 'legit').astype(int).values
groups = df['group'].values
texts = df['text'].values
files = df['filename'].values
non_en = (df['is_english'].values == 0)
print('Real pages:', len(df), '| non-English:', int(non_en.sum()), '| synthetic rows:', len(syn), '| languages:', sorted(syn['lang'].unique()))

def fit_predict(train_texts, train_y, test_texts):
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = tf.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=64, random_state=42).fit(Xtr)
    sc = StandardScaler().fit(svd.transform(Xtr))
    clf = LogisticRegressionCV(Cs=[0.0005, 0.002, 0.01, 0.05, 0.2], cv=3, max_iter=5000, scoring='accuracy')
    clf.fit(sc.transform(svd.transform(Xtr)), train_y)
    return clf.predict_proba(sc.transform(svd.transform(tf.transform(test_texts))))[:, 1]

def acc(mask, pred):
    return float((pred[mask] == y[mask]).mean())

def recall_phish(mask, pred):
    m = mask & (y == 0)
    return float((pred[m] == 0).mean())

rows, lost_gained = [], []
for seed in [42, 1, 2]:
    pa, pb = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups):
        pa[te] = fit_predict(texts[tr], y[tr], texts[te])
        s = syn[syn['source_filename'].isin(set(files[tr]))]
        pb[te] = fit_predict(np.concatenate([texts[tr], s['text'].values]),
                             np.concatenate([y[tr], (s['label'] == 'legit').astype(int).values]), texts[te])
    ha, hb = (pa >= 0.5).astype(int), (pb >= 0.5).astype(int)
    for name, mask in [('English', ~non_en), ('Non-English', non_en), ('All', np.ones(len(y), bool))]:
        rows.append({'seed': seed, 'group': name, 'baseline_acc': acc(mask, ha), 'augmented_acc': acc(mask, hb),
                     'baseline_phish_recall': recall_phish(mask, ha), 'augmented_phish_recall': recall_phish(mask, hb)})
    ne = non_en
    lost_gained.append((int(((ha == y) & (hb != y) & ne).sum()), int(((ha != y) & (hb == y) & ne).sum())))

res = pd.DataFrame(rows)
summary = res.groupby('group')[['baseline_acc', 'augmented_acc', 'baseline_phish_recall', 'augmented_phish_recall']].mean().round(3)
print('Per seed:')
print(res.round(3).to_string(index=False))
print()
print('Mean over 3 different fold splits (text branch only, real test pages only):')
print(summary.to_string())
print()
print('Non-English pages per seed: (made wrong by augmentation, made right by augmentation) =', lost_gained)
res.to_csv('results/augmentation_experiment.csv', index=False)
print('Saved results/augmentation_experiment.csv')
