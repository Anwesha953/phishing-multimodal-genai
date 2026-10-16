import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0

df = pd.read_csv('data/processed/pages_clean.csv')
df['text'] = df['text'].fillna('').astype(str)

def lang(t):
    try:
        return detect(t[:1000])
    except LangDetectException:
        return 'unknown'

df['lang'] = df['text'].apply(lang)
df['is_english'] = (df['lang'] == 'en').astype(int)
df.to_csv('data/processed/pages_lang.csv', index=False)

top = df['lang'].value_counts().head(8).index
print('Pages by language and class:')
print(pd.crosstab(df['lang'].where(df['lang'].isin(top), 'other'), df['label']))
print()
print('English vs non-English:')
print(pd.crosstab(df['is_english'].map({1: 'English', 0: 'Non-English'}), df['label']))
