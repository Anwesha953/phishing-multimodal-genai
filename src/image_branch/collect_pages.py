import sys, os, io, time, random
from urllib.parse import urlparse
import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

OUT_CSV = 'data/processed/pages_dataset.csv'
IMG_DIR = 'data/processed/pages_img'
ATTEMPTED = 'data/processed/attempted_urls.txt'
ERROR_MARKERS = ["can't be reached", "can’t be reached", 'err_', '404', 'not found', 'server is down',
                 'suspended', 'domain for sale', 'access denied', '403', 'forbidden', 'bad gateway',
                 'under construction', 'parked', 'suspected phishing', 'deceptive', 'just a moment',
                 'checking your browser', 'attention required', 'expired', 'unavailable']

def phishing_candidates():
    urls = []
    if os.path.exists('data/raw/phishtank_fresh.csv'):
        urls += pd.read_csv('data/raw/phishtank_fresh.csv')['url'].dropna().astype(str).tolist()
    if os.path.exists('data/raw/openphish_fresh.txt'):
        urls += [l.strip() for l in open('data/raw/openphish_fresh.txt') if l.strip().startswith('http')]
    return urls

def legit_candidates():
    d = pd.read_csv('data/raw/uci_phishing.csv', usecols=['URL', 'label'])
    return d[d['label'] == 1]['URL'].astype(str).tolist()

def pick(urls, seed=42):
    random.Random(seed).shuffle(urls)
    seen, out = {}, []
    for u in urls:
        host = urlparse(u).netloc.lower()
        if seen.get(host, 0) >= 2:
            continue
        seen[host] = seen.get(host, 0) + 1
        out.append(u)
    return out

def grab(driver, u, label, idx):
    driver.get(u)
    time.sleep(1.5)
    text = driver.find_element(By.TAG_NAME, 'body').text.replace('\n', ' ').strip()
    low = text.lower()
    if len(text) < 80 or (len(text) < 800 and any(m in low for m in ERROR_MARKERS)):
        return None
    img = Image.open(io.BytesIO(driver.get_screenshot_as_png())).convert('RGB').resize((512, 400))
    fn = f'{IMG_DIR}/{label}_{idx:05d}.jpg'
    img.save(fn, quality=70)
    return {'filename': fn, 'url': u, 'final_url': driver.current_url, 'label': label,
            'title': driver.title, 'text': text[:5000]}

if __name__ == '__main__':
    label, attempts = sys.argv[1], int(sys.argv[2])
    os.makedirs(IMG_DIR, exist_ok=True)
    cands = pick(phishing_candidates() if label == 'phishing' else legit_candidates())
    rows = pd.read_csv(OUT_CSV).to_dict('records') if os.path.exists(OUT_CSV) else []
    attempted = set(open(ATTEMPTED).read().split('\n')) if os.path.exists(ATTEMPTED) else set()
    todo = [u for u in cands if u not in attempted][:attempts]
    print('Candidates:', len(cands), '| trying now:', len(todo))

    opts = Options()
    for a in ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--window-size=1280,1024']:
        opts.add_argument(a)
    opts.binary_location = '/usr/bin/google-chrome'
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(12)

    kept = 0
    for i, u in enumerate(todo):
        with open(ATTEMPTED, 'a') as f:
            f.write(u + '\n')
        try:
            row = grab(driver, u, label, len(rows))
        except Exception:
            row = None
            try:
                driver.switch_to.alert.accept()
            except Exception:
                pass
        if row:
            rows.append(row)
            kept += 1
        if (i + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f'{i + 1}/{len(todo)} tried, kept {kept}')
    driver.quit()
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print('Done. Kept', kept, 'of', len(todo), '| totals:', pd.DataFrame(rows)['label'].value_counts().to_dict())
