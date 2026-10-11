import re, os, joblib
from urllib.parse import urlparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

FEATURE_NAMES = ['url_length', 'host_length', 'path_length', 'dots', 'hyphens', 'underscores',
                 'slashes', 'question_marks', 'equals', 'at_symbols', 'ampersands', 'percents',
                 'digit_count', 'letter_count', 'is_https', 'host_is_ip', 'host_dots',
                 'suspicious_keywords']

def url_features(u):
    u = str(u)
    p = urlparse(u if '://' in u else 'http://' + u)
    host = p.netloc
    return [
        len(u), len(host), len(p.path), u.count('.'), u.count('-'), u.count('_'),
        u.count('/'), u.count('?'), u.count('='), u.count('@'), u.count('&'), u.count('%'),
        sum(c.isdigit() for c in u), sum(c.isalpha() for c in u),
        int(u.lower().startswith('https')),
        int(bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', host.split(':')[0]))),
        host.count('.'),
        sum(k in u.lower() for k in ['login', 'verify', 'secure', 'update', 'account', 'bank', 'confirm', 'password']),
    ]

if __name__ == '__main__':
    df = pd.read_csv('data/raw/uci_phishing.csv', usecols=['URL', 'label']).dropna()
    df = df.sample(n=50000, random_state=42)

    X = pd.DataFrame([url_features(u) for u in df['URL']], columns=FEATURE_NAMES)
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print(classification_report(y_test, pred, target_names=['phishing', 'legitimate']))

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    joblib.dump(clf, 'models/url_rf.joblib', compress=3)
    pd.DataFrame([{'model': 'URL-only Random Forest',
                   'accuracy': accuracy_score(y_test, pred),
                   'f1': f1_score(y_test, pred)}]).to_csv('results/url_branch_metrics.csv', index=False)
    print('Model size (MB):', round(os.path.getsize('models/url_rf.joblib') / (1024 * 1024), 1))
