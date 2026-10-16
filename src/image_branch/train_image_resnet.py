import os, joblib, numpy as np, pandas as pd, torch
from urllib.parse import urlparse
from PIL import Image
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score

def reg_domain(u):
    host = urlparse(str(u)).netloc.lower().split(':')[0]
    parts = host.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else host

device = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_csv('data/processed/pages_clean.csv')
df = df[df['filename'].apply(os.path.exists)].reset_index(drop=True)
df['y'] = (df['label'] == 'legit').astype(int)   # 0 = phishing, 1 = legitimate
df['group'] = df['url'].apply(reg_domain)

n = df['y'].value_counts().min()
df = pd.concat([g.sample(n, random_state=42) for _, g in df.groupby('y')]).reset_index(drop=True)
print('Balanced set:', df['y'].value_counts().to_dict())

net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
net.fc = torch.nn.Identity()
net = net.to(device).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
feats = []
with torch.no_grad():
    for i in range(0, len(df), 32):
        batch = torch.stack([tf(Image.open(f).convert('RGB')) for f in df['filename'][i:i + 32]]).to(device)
        feats.append(net(batch).cpu().numpy())
X = np.vstack(feats)
y = df['y'].values

oof_pred = np.zeros(len(df), dtype=int)
oof_prob = np.zeros(len(df))
accs = []
for k, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups=df['group'])):
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X[tr], y[tr])
    oof_pred[te] = clf.predict(X[te])
    oof_prob[te] = clf.predict_proba(X[te])[:, 1]
    accs.append(accuracy_score(y[te], oof_pred[te]))
    print(f'Fold {k + 1}: accuracy {accs[-1]:.3f} ({len(te)} test images)')

print(f'Mean accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}')
print(classification_report(y, oof_pred, target_names=['phishing', 'legitimate']))

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
np.save('data/processed/image_features.npy', X)
df['image_oof_prob_legit'] = oof_prob
df[['filename', 'url', 'label', 'group', 'image_oof_prob_legit']].to_csv('data/processed/image_features_index.csv', index=False)
joblib.dump(LogisticRegression(max_iter=3000).fit(X, y), 'models/image_resnet18_lr.joblib')
pd.DataFrame([{'model': 'Image-only ResNet18 features + Logistic Regression (5-fold grouped CV)',
               'accuracy': np.mean(accs), 'accuracy_std': np.std(accs),
               'f1': f1_score(y, oof_pred, average='weighted'), 'images': len(df)}]).to_csv('results/image_branch_metrics.csv', index=False)
print('Saved.')
