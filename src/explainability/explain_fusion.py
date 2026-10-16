import os, re, textwrap
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

KEYS = ['login', 'verify', 'secure', 'update', 'account', 'bank', 'confirm', 'password']

def host_features(u):
    u = str(u)
    p = urlparse(u if '://' in u else 'http://' + u)
    host = p.netloc.lower().split(':')[0]
    hn = host[4:] if host.startswith('www.') else host
    parts = hn.split('.')
    n = max(len(hn), 1)
    digits = sum(c.isdigit() for c in hn)
    return [len(hn), hn.count('.'), hn.count('-'), digits, digits / n,
            int(bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', hn))), max(len(parts) - 2, 0),
            max(len(x) for x in parts), sum(k in hn for k in KEYS)]

def new_lr():
    return LogisticRegressionCV(Cs=[0.0005, 0.002, 0.01, 0.05, 0.2], cv=3, max_iter=5000, scoring='accuracy')

def fit_pred(Xtr, ytr, Xte):
    clf = new_lr()
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

idx = pd.read_csv('data/processed/image_features_index.csv')
pages = pd.read_csv('data/processed/pages_clean.csv')[['filename', 'text']]
df = idx.merge(pages, on='filename', how='left')
df['text'] = df['text'].fillna('').astype(str)
X_img = np.load('data/processed/image_features.npy')
assert len(df) == len(X_img)
X_host = np.array([host_features(u) for u in df['url']], dtype=float)
y = (df['label'] == 'legit').astype(int).values      # 0 = phishing, 1 = legitimate
groups = df['group'].values
texts = df['text'].values

MODS = ['url_host', 'text', 'image']
NICE = {'url_host': 'URL structure', 'text': 'page text', 'image': 'page screenshot'}
contrib = np.zeros((len(y), 3))
bias = np.zeros(len(y))
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X_img, y, groups))

for k, (tr, te) in enumerate(folds):
    sc_u = StandardScaler().fit(X_host[tr])
    sc_i = StandardScaler().fit(X_img[tr])
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english')
    Ttr = tf.fit_transform(texts[tr])
    svd = TruncatedSVD(n_components=64, random_state=42).fit(Ttr)
    sc_t = StandardScaler().fit(svd.transform(Ttr))
    train = {'url_host': sc_u.transform(X_host[tr]), 'text': sc_t.transform(svd.transform(Ttr)), 'image': sc_i.transform(X_img[tr])}
    test = {'url_host': sc_u.transform(X_host[te]), 'text': sc_t.transform(svd.transform(tf.transform(texts[te]))), 'image': sc_i.transform(X_img[te])}
    ytr, gtr = y[tr], groups[tr]
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros(len(ytr)), ytr, gtr))
    meta_tr = np.zeros((len(ytr), 3))
    meta_te = np.zeros((len(te), 3))
    for j, m in enumerate(MODS):
        for itr, ite in inner:
            meta_tr[ite, j] = fit_pred(train[m][itr], ytr[itr], train[m][ite])
        meta_te[:, j] = fit_pred(train[m], ytr, test[m])
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(meta_tr), ytr)
    contrib[te] = logit(meta_te) * meta.coef_[0]
    bias[te] = meta.intercept_[0]
    print(f'Fold {k + 1} done')

total = bias + contrib.sum(axis=1)
pred = (total >= 0).astype(int)
conf = 1 / (1 + np.exp(-np.abs(total)))
share = np.abs(contrib) / np.abs(contrib).sum(axis=1, keepdims=True)
print('Held-out accuracy (all pages pooled):', round(float((pred == y).mean()), 3))

def explain(i):
    verdict = 'legitimate' if pred[i] == 1 else 'PHISHING'
    parts = []
    for j in np.argsort(-share[i]):
        way = 'legitimate' if contrib[i, j] >= 0 else 'phishing'
        parts.append(f'{NICE[MODS[j]]} {share[i, j] * 100:.0f}% (points to {way})')
    return f'Predicted {verdict}, {conf[i] * 100:.0f}% confident. Evidence: ' + '; '.join(parts) + '.'

out = df[['filename', 'url', 'label']].copy()
out['predicted'] = np.where(pred == 1, 'legit', 'phishing')
out['confidence'] = conf.round(3)
for j, m in enumerate(MODS):
    out['share_' + m] = share[:, j].round(3)
out['explanation'] = [explain(i) for i in range(len(df))]
os.makedirs('results', exist_ok=True)
out.to_csv('results/fusion_explanations.csv', index=False)

print(pd.crosstab(out['label'], out['predicted']))
ok = out[out['label'] == out['predicted']]
summary = ok.groupby('label')[['share_' + m for m in MODS]].mean().round(2)
summary.columns = [NICE[m] for m in MODS]
print('Average share of evidence per signal (correct predictions only):')
print(summary)
summary.plot(kind='bar', rot=0, figsize=(7, 4))
plt.ylabel('Share of evidence')
plt.title('Which signal drove the decision')
plt.tight_layout()
plt.savefig('results/modality_evidence_share.png', dpi=150)
plt.show()

ex = ok[ok['label'] == 'phishing'].sample(min(6, int((ok['label'] == 'phishing').sum())), random_state=3)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (_, r) in zip(axes.flat, ex.iterrows()):
    ax.imshow(Image.open(r['filename']))
    ax.set_title(str(r['url'])[:40] + '\n' + textwrap.fill(r['explanation'], 50), fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/explanation_examples.png', dpi=130)
plt.show()
print('Saved results/fusion_explanations.csv and two figures in results/')
