import os, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_emails():
    parts = [pd.read_csv('data/raw/phishing_email_part1.csv'),
             pd.read_csv('data/raw/phishing_email_part2.csv')]
    return pd.concat(parts, ignore_index=True)

def detect_columns(df):
    label_col = None
    for c in df.columns:
        if str(c).lower() in ('label', 'class', 'target', 'email type', 'type'):
            label_col = c
            break
    if label_col is None:
        label_col = min(df.columns, key=lambda c: df[c].nunique())
    text_cols = [c for c in df.columns if c != label_col and df[c].dtype == object]
    text_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return text_col, label_col

if __name__ == '__main__':
    df = load_emails()
    text_col, label_col = detect_columns(df)
    print('Columns in file:', list(df.columns))
    print('Rows:', len(df), '| text column:', text_col, '| label column:', label_col)

    df = df[[text_col, label_col]].dropna().drop_duplicates()
    if df[label_col].dtype == object:
        codes, uniques = pd.factorize(df[label_col])
        print('Label mapping:', dict(enumerate(uniques)))
        y = codes
    else:
        y = df[label_col].values
        print('Label counts:', dict(pd.Series(y).value_counts()))
    X = df[text_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5, stop_words='english')
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    print(classification_report(y_test, pred))

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    joblib.dump({'vectorizer': vec, 'model': clf}, 'models/text_tfidf_lr.joblib', compress=3)
    pd.DataFrame([{'model': 'Text-only TF-IDF + Logistic Regression',
                   'accuracy': accuracy_score(y_test, pred),
                   'f1': f1_score(y_test, pred, average='weighted')}]).to_csv('results/text_branch_metrics.csv', index=False)
    print('Model size (MB):', round(os.path.getsize('models/text_tfidf_lr.joblib') / (1024 * 1024), 1))
