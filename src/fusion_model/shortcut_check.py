import os, re, importlib.util, numpy as np, pandas as pd
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

spec = importlib.util.spec_from_file_location('train_url_rf', 'src/url_branch/train_url_rf.py')
url_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(url_mod)

def new_lr():
    return LogisticRegressionCV(Cs=[0.0005, 0.002, 0.01, 0.05, 0.2], cv=3, max_iter=5000, scoring='accuracy')

def fit_pred(Xtr, ytr, Xte):
    clf = new_lr()
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

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

idx = pd.read_csv('data/processed/image_features_index.csv')
pages = pd.read_csv('data/processed/pages_clean.csv')[['filename', 'text']]
df = idx.merge(pages, on='filename', how='left')
df['text'] = df['text'].fillna('').astype(str)
X_img = np.load('data/processed/image_features.npy')
assert len(df) == len(X_img)
y = (df['label'] == 'legit').astype(int).values      # 0 = phishing, 1 = legitimate
groups = df['group'].values
texts = df['text'].values

parsed = df['url'].apply(lambda u: urlparse(str(u)))
df['has_path_or_query'] = parsed.apply(lambda p: int(len(p.path.strip('/')) > 0 or bool(p.query)))
df['uses_https'] = df['url'].astype(str).str.lower().str.startswith('https').astype(int)
df['starts_with_www'] = parsed.apply(lambda p: int(p.netloc.lower().startswith('www.')))
print('Share of URLs with each property, by class (1.00 = all of them):')
print(df.groupby('label')[['has_path_or_query', 'uses_https', 'starts_with_www']].mean().round(2))

X_full = np.array([url_mod.url_features(u) for u in df['url']], dtype=float)
X_host = np.array([host_features(u) for u in df['url']], dtype=float)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_full, y)
imp = pd.Series(rf.feature_importances_, index=url_mod.FEATURE_NAMES).sort_values(ascending=False)
print('Most important full-URL features:', imp.head(5).round(2).to_dict())

BLOCKS = (('url_full', X_full), ('url_host', X_host), ('image', X_img))

def prep(tr, te):
    sc = {k: StandardScaler().fit(X[tr]) for k, X in BLOCKS}
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english')
    Ttr = tf.fit_transform(texts[tr])
    svd = TruncatedSVD(n_components=64, random_state=42).fit(Ttr)
    sct = StandardScaler().fit(svd.transform(Ttr))
    train = {k: sc[k].transform(X[tr]) for k, X in BLOCKS}
    test = {k: sc[k].transform(X[te]) for k, X in BLOCKS}
    train['text'] = sct.transform(svd.transform(Ttr))
    test['text'] = sct.transform(svd.transform(tf.transform(texts[te])))
    return train, test

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

def stacked(btr, bte, ytr, gtr, mods):
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros(len(ytr)), ytr, gtr))
    meta_tr = np.zeros((len(ytr), len(mods)))
    meta_te = np.zeros((len(bte[mods[0]]), len(mods)))
    for j, m in enumerate(mods):
        for itr, ite in inner:
            meta_tr[ite, j] = fit_pred(btr[m][itr], ytr[itr], btr[m][ite])
        meta_te[:, j] = fit_pred(btr[m], ytr, bte[m])
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(meta_tr), ytr)
    return meta.predict_proba(logit(meta_te))[:, 1], meta.coef_[0]

CONC = {
    'URL only (full URL features)': ['url_full'],
    'URL only (host-only features)': ['url_host'],
    'Text only': ['text'],
    'Image only': ['image'],
    'URL (host) + Text': ['url_host', 'text'],
    'URL (host) + Image': ['url_host', 'image'],
    'Text + Image': ['text', 'image'],
    'Concatenation fusion (URL host + Text + Image)': ['url_host', 'text', 'image'],
    'Concatenation fusion (URL full + Text + Image)': ['url_full', 'text', 'image'],
}
STACK = {
    'Stacked fusion (URL host + Text + Image)': ['url_host', 'text', 'image'],
    'Stacked fusion (URL full + Text + Image)': ['url_full', 'text', 'image'],
}
names = list(CONC) + list(STACK)
oof = {n: np.zeros(len(y)) for n in names}
accs = {n: [] for n in names}
weights = {n: [] for n in STACK}
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X_img, y, groups))

for k, (tr, te) in enumerate(folds):
    train, test = prep(tr, te)
    for n, mods in CONC.items():
        oof[n][te] = fit_pred(np.hstack([train[m] for m in mods]), y[tr], np.hstack([test[m] for m in mods]))
    for n, mods in STACK.items():
        oof[n][te], w = stacked(train, test, y[tr], groups[tr], mods)
        weights[n].append(w)
    for n in names:
        accs[n].append(accuracy_score(y[te], (oof[n][te] >= 0.5).astype(int)))
    print(f'Fold {k + 1} done')

res = pd.DataFrame([{'model': n, 'accuracy': np.mean(accs[n]), 'accuracy_std': np.std(accs[n]),
                     'f1': f1_score(y, (oof[n] >= 0.5).astype(int), average='weighted')} for n in names])
print(res.round(3).to_string(index=False))
for n in STACK:
    print(n, '| average weight on', STACK[n], ':', np.round(np.mean(weights[n], axis=0), 2))
os.makedirs('results', exist_ok=True)
res.to_csv('results/shortcut_check_results.csv', index=False)
print('Saved results/shortcut_check_results.csv')
