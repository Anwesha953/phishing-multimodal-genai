import os, importlib.util, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

spec = importlib.util.spec_from_file_location('train_url_rf', 'src/url_branch/train_url_rf.py')
url_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(url_mod)

def new_lr():
    return LogisticRegressionCV(Cs=[0.0005, 0.002, 0.01, 0.05, 0.2], cv=3, max_iter=5000, scoring='accuracy')

idx = pd.read_csv('data/processed/image_features_index.csv')
pages = pd.read_csv('data/processed/pages_clean.csv')[['filename', 'text']]
df = idx.merge(pages, on='filename', how='left')
df['text'] = df['text'].fillna('').astype(str)
X_img = np.load('data/processed/image_features.npy')
assert len(df) == len(X_img)
X_url = np.array([url_mod.url_features(u) for u in df['url']], dtype=float)
y = (df['label'] == 'legit').astype(int).values      # 0 = phishing, 1 = legitimate
groups = df['group'].values
texts = df['text'].values
print('Pages:', len(df), '| phishing:', int((y == 0).sum()), '| legit:', int((y == 1).sum()))
print('Image feature size:', X_img.shape[1])

CONFIGS = {
    'URL only': ['url'], 'Text only': ['text'], 'Image only': ['image'],
    'URL + Text': ['url', 'text'], 'URL + Image': ['url', 'image'], 'Text + Image': ['text', 'image'],
    'URL + Text + Image (fusion)': ['url', 'text', 'image'],
}
LATE = 'Late fusion (average of 3 branches)'
MLP = 'Fusion with small neural network (MLP)'

def make_blocks(tr, te):
    sc_u = StandardScaler().fit(X_url[tr])
    sc_i = StandardScaler().fit(X_img[tr])
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english')
    Ttr = tf.fit_transform(texts[tr])
    svd = TruncatedSVD(n_components=64, random_state=42).fit(Ttr)
    sc_t = StandardScaler().fit(svd.transform(Ttr))
    train = {'url': sc_u.transform(X_url[tr]), 'text': sc_t.transform(svd.transform(Ttr)), 'image': sc_i.transform(X_img[tr])}
    test = {'url': sc_u.transform(X_url[te]), 'text': sc_t.transform(svd.transform(tf.transform(texts[te]))), 'image': sc_i.transform(X_img[te])}
    return train, test, (sc_u, tf, svd, sc_t, sc_i)

names = list(CONFIGS) + [LATE, MLP]
oof = {n: np.zeros(len(y)) for n in names}
fold_acc = {n: [] for n in names}
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X_img, y, groups))

for k, (tr, te) in enumerate(folds):
    train, test, _ = make_blocks(tr, te)
    for name, mods in CONFIGS.items():
        clf = new_lr()
        clf.fit(np.hstack([train[m] for m in mods]), y[tr])
        oof[name][te] = clf.predict_proba(np.hstack([test[m] for m in mods]))[:, 1]
    oof[LATE][te] = (oof['URL only'][te] + oof['Text only'][te] + oof['Image only'][te]) / 3
    allm = ['url', 'text', 'image']
    mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=0.01, max_iter=1000, random_state=42)
    mlp.fit(np.hstack([train[m] for m in allm]), y[tr])
    oof[MLP][te] = mlp.predict_proba(np.hstack([test[m] for m in allm]))[:, 1]
    for n in names:
        fold_acc[n].append(accuracy_score(y[te], (oof[n][te] >= 0.5).astype(int)))
    print(f'Fold {k + 1} done')

rows = []
for n in names:
    pred = (oof[n] >= 0.5).astype(int)
    rows.append({'model': n, 'accuracy': np.mean(fold_acc[n]), 'accuracy_std': np.std(fold_acc[n]),
                 'f1': f1_score(y, pred, average='weighted')})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
res.to_csv('results/fusion_results.csv', index=False)
pred_df = df[['url', 'label']].copy()
for n in names:
    pred_df[n] = oof[n]
pred_df.to_csv('results/fusion_oof_predictions.csv', index=False)

all_idx = np.arange(len(y))
train, _, prep = make_blocks(all_idx, all_idx)
final = new_lr().fit(np.hstack([train['url'], train['text'], train['image']]), y)
sc_u, tf, svd, sc_t, sc_i = prep
joblib.dump({'url_scaler': sc_u, 'tfidf': tf, 'svd': svd, 'text_scaler': sc_t, 'image_scaler': sc_i,
             'classifier': final,
             'block_sizes': {'url': train['url'].shape[1], 'text': train['text'].shape[1], 'image': train['image'].shape[1]}},
            'models/fusion_lr.joblib', compress=3)
print('Saved fusion model. Size (MB):', round(os.path.getsize('models/fusion_lr.joblib') / (1024 * 1024), 1))
