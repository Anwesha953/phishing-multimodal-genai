import os, numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, AutoModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

os.makedirs('results', exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
idx = pd.read_csv('data/processed/image_features_index.csv')
files = idx['filename'].tolist()
y = (idx['label'] == 'legit').astype(int).values   # 0 = phishing, 1 = legitimate
groups = idx['group'].values
print('Pages:', len(idx), '| phishing:', int((y == 0).sum()), '| legit:', int((y == 1).sum()))

def as_2d(out):
    if torch.is_tensor(out):
        return out
    for name in ('image_embeds', 'pooler_output', 'last_hidden_state'):
        v = getattr(out, name, None)
        if v is not None:
            return v if v.dim() == 2 else v[:, 0]
    return out[0]

def extract(fn, mean, std):
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    outs = []
    with torch.no_grad():
        for i in range(0, len(files), 16):
            batch = torch.stack([tf(Image.open(f).convert('RGB')) for f in files[i:i + 16]]).to(device)
            outs.append(fn(batch).float().cpu().numpy())
    return np.vstack(outs)

r18_path = 'data/processed/image_features_resnet18.npy'
if not os.path.exists(r18_path):
    np.save(r18_path, np.load('data/processed/image_features.npy'))
feats = {'ResNet18 (previous)': np.load(r18_path)}

clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
feats['CLIP ViT-B/32'] = extract(lambda b: as_2d(clip.get_image_features(pixel_values=b)),
                                 [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
del clip
print('CLIP features done')

dino = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
def dino_fn(b):
    h = dino(pixel_values=b).last_hidden_state
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
feats['DINOv2-small'] = extract(dino_fn, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
del dino
print('DINOv2 features done')
feats['CLIP + DINOv2'] = np.hstack([feats['CLIP ViT-B/32'], feats['DINOv2-small']])

folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups))

def evaluate(X):
    prob, accs = np.zeros(len(y)), []
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=[0.0005, 0.002, 0.01, 0.05, 0.2], cv=3, max_iter=5000, scoring='accuracy'))
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
        accs.append(accuracy_score(y[te], (prob[te] >= 0.5).astype(int)))
    return prob, accs

rows, probs = [], {}
for name, X in feats.items():
    prob, accs = evaluate(X)
    probs[name] = prob
    rows.append({'backbone': name, 'accuracy': np.mean(accs), 'accuracy_std': np.std(accs),
                 'f1': f1_score(y, (prob >= 0.5).astype(int), average='weighted')})
    print(f'{name}: accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f}')

res = pd.DataFrame(rows)
res.to_csv('results/image_backbone_comparison.csv', index=False)
print(res.round(3).to_string(index=False))

best = res.sort_values('accuracy', ascending=False).iloc[0]['backbone']
print('Best image feature extractor:', best)
np.save('data/processed/image_features.npy', feats[best])
idx['image_oof_prob_legit'] = probs[best]
idx.to_csv('data/processed/image_features_index.csv', index=False)
b = res.set_index('backbone').loc[best]
pd.DataFrame([{'model': f'Image-only {best} + Logistic Regression (5-fold grouped CV)',
               'accuracy': b['accuracy'], 'accuracy_std': b['accuracy_std'], 'f1': b['f1'], 'images': len(y)}]
             ).to_csv('results/image_branch_metrics.csv', index=False)
print('Saved. The best features are now in data/processed/image_features.npy for the fusion step.')
