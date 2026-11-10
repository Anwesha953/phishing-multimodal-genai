import ast, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Re-use the exact URL-feature function from the earlier explanation script (everything above 'def new_lr').
_tree = ast.parse(open('src/explainability/explain_fusion.py').read())
_stop = [n.lineno for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == 'new_lr'][0]
exec(compile(ast.Module(body=[n for n in _tree.body if n.lineno < _stop], type_ignores=[]), 'explain_fusion_head', 'exec'))
matplotlib.use('Agg')


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
y = (df['label'] == 'legit').astype(int).values          # 0 = phishing, 1 = legitimate
groups = df['group'].values
texts = df['text'].values

MODS = ['url_host', 'text', 'image']
NICE = {'url_host': 'URL structure', 'text': 'page text', 'image': 'page screenshot'}
n = len(y)
branch_p = np.zeros((n, 3))        # probability of "legit" from each signal alone (held-out pages)
shap_vals = np.zeros((n, 3))       # SHAP value of each signal for the fused "legit" score
base = np.zeros(n)                 # SHAP base value
total_direct = np.zeros(n)         # fused score computed the old way (to check SHAP adds up)
agree = []
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
    Ltr, Lte = logit(meta_tr), logit(meta_te)
    manual = meta.coef_[0] * (Lte - Ltr.mean(axis=0))
    try:
        sv = np.asarray(shap.LinearExplainer(meta, Ltr).shap_values(Lte))
        agree.append(bool(sv.shape == manual.shape and np.allclose(sv, manual, atol=1e-6)))
        if not agree[-1]:
            sv = manual
    except Exception:
        sv = manual
        agree.append(None)
    shap_vals[te] = sv
    branch_p[te] = meta_te
    base[te] = meta.intercept_[0] + (meta.coef_[0] * Ltr.mean(axis=0)).sum()
    total_direct[te] = meta.intercept_[0] + (Lte * meta.coef_[0]).sum(axis=1)
    print(f'Fold {k + 1} done', flush=True)

total = base + shap_vals.sum(axis=1)
print('\nSHAP library agrees with the hand formula in each fold:', agree)
print('SHAP values add up to the fused score (max difference):', float(np.abs(total - total_direct).max()))

# ---------------- metrics table (phishing = positive class) ----------------
def rep(name, p_legit):
    p_ph = 1 - p_legit
    pred = (p_ph >= 0.5).astype(int)
    yp = 1 - y
    return {'model': name, 'accuracy': accuracy_score(yp, pred), 'precision': precision_score(yp, pred),
            'recall': recall_score(yp, pred), 'f1': f1_score(yp, pred), 'auc': roc_auc_score(yp, p_ph)}

p_fused_legit = 1 / (1 + np.exp(-total))
tab = pd.DataFrame([rep('URL structure only (host)', branch_p[:, 0]), rep('Page text only', branch_p[:, 1]),
                    rep('Screenshot only (CLIP)', branch_p[:, 2]), rep('Fused (all three, ours)', p_fused_legit)]).round(3)
os.makedirs('results', exist_ok=True)
tab.to_csv('results/metrics_table.csv', index=False)
print('\nTable 1: held-out results, phishing = positive class (pooled over 5 domain-grouped folds)')
print(tab.to_string(index=False))

# ---------------- SHAP-based natural-language explanations ----------------
c = -shap_vals                         # evidence pointing toward PHISHING
pred_ph = total < 0
conf = 1 / (1 + np.exp(-np.abs(total)))
share = np.abs(c) / np.abs(c).sum(axis=1, keepdims=True)


def sentence(i):
    direction = 1 if pred_ph[i] else -1
    support = direction * c[i]                       # > 0 means this signal backs the final verdict
    order = np.argsort(-support)
    verdict = 'Flagged as phishing' if pred_ph[i] else 'Judged legitimate'
    backers = [j for j in order if support[j] > 0]
    doubters = [j for j in order[::-1] if support[j] < 0]
    if backers:
        top = backers[0]
        s = f'{verdict} ({conf[i]:.0%} confident), mainly because of the {NICE[MODS[top]]} ({share[i, top]:.0%} of the evidence)'
        if len(backers) > 1:
            s += f' and the {NICE[MODS[backers[1]]]} ({share[i, backers[1]]:.0%})'
    else:
        s = f'{verdict} ({conf[i]:.0%} confident), but no single signal strongly backed it'
    if doubters:
        s += '; the ' + ' and the '.join(NICE[MODS[j]] for j in doubters) + ' pointed the other way'
    return s + '.'


out = pd.DataFrame({'filename': df['filename'], 'url': df['url'], 'label': df['label'],
                    'predicted': np.where(pred_ph, 'phishing', 'legit'), 'confidence': conf.round(3),
                    'shap_url': (-shap_vals[:, 0]).round(3), 'shap_text': (-shap_vals[:, 1]).round(3),
                    'shap_image': (-shap_vals[:, 2]).round(3)})
out['explanation'] = [sentence(i) for i in range(n)]
out.to_csv('results/shap_explanations.csv', index=False)
print('\nFused accuracy:', round(float((np.where(pred_ph, 0, 1) == y).mean()), 3))
print('\nExample explanations:')
for i in list(np.where(pred_ph & (y == 0))[0][:3]) + list(np.where(~pred_ph & (y == 1))[0][:2]) + list(np.where(pred_ph != (y == 0))[0][:2]):
    print('-', df['url'][i][:60], '| true:', df['label'][i], '|', sentence(i))

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(6.5, 4))
w = 0.38
xs = np.arange(3)
for off, (cls, name) in zip([-w / 2, w / 2], [(0, 'phishing pages'), (1, 'legitimate pages')]):
    ax.bar(xs + off, np.abs(shap_vals[y == cls]).mean(axis=0), w, label=name)
ax.set_xticks(xs)
ax.set_xticklabels([NICE[m] for m in MODS])
ax.set_ylabel('mean |SHAP value| (log-odds)')
ax.set_title('How much each signal drives the verdict (SHAP)')
ax.legend()
plt.tight_layout()
plt.savefig('results/shap_modality_importance.png', dpi=160)
plt.close()

ok_ph = list(np.where(pred_ph & (y == 0))[0])
ok_lg = list(np.where(~pred_ph & (y == 1))[0])
bad = list(np.where(pred_ph != (y == 0))[0])
picks = []
for pool, kk in [(ok_ph, 2), (ok_lg, 1), (bad, 1)]:
    pool = sorted(pool, key=lambda i: -conf[i])[:kk]
    picks += pool
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
for ax in axes.ravel()[len(picks):]:
    ax.axis('off')
for ax, i in zip(axes.ravel(), picks):
    colors = ['#c0392b' if v > 0 else '#2e86c1' for v in c[i]]
    ax.barh([NICE[m] for m in MODS], c[i], color=colors)
    ax.axvline(0, color='black', lw=0.8)
    tag = 'CORRECT' if (pred_ph[i] == (y[i] == 0)) else 'WRONG'
    ax.set_title(f"{df['url'][i][:38]}\ntrue: {df['label'][i]} | predicted: {out['predicted'][i]} ({tag})", fontsize=9)
    ax.set_xlabel('SHAP: red = toward phishing, blue = toward legit', fontsize=8)
plt.tight_layout()
plt.savefig('results/shap_examples.png', dpi=160)
plt.close()
print('\nSaved: results/metrics_table.csv, results/shap_explanations.csv, results/shap_modality_importance.png, results/shap_examples.png')
