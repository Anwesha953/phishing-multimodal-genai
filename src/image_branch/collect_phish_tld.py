import sys, os, random, urllib.request
import pandas as pd
from urllib.parse import urlparse
sys.path.insert(0, 'src/image_branch')
import collect_pages as cp

mode, attempts = sys.argv[1], int(sys.argv[2])
TLD = {'european': ('.de', '.fr', '.es', '.it', '.pt', '.nl', '.pl', '.ro', '.se', '.dk', '.fi', '.cz', '.hu', '.gr',
                    '.bg', '.hr', '.sk', '.si', '.at', '.be', '.no', '.lt', '.lv', '.ee'),
       'indian': ('.in',)}[mode]
SRC = ['https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt',
       'https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE-today.txt',
       'https://openphish.com/feed.txt']
links = []
for s in SRC:
    try:
        raw = urllib.request.urlopen(urllib.request.Request(s, headers={'User-Agent': 'Mozilla/5.0'}), timeout=180).read().decode('utf-8', 'ignore')
        links += [l.strip() for l in raw.split('\n') if l.strip().startswith('http')]
    except Exception as e:
        print('could not download', s, e)
rows = pd.read_csv(cp.OUT_CSV).to_dict('records') if os.path.exists(cp.OUT_CSV) else []
attempted = set(open(cp.ATTEMPTED).read().split('\n')) if os.path.exists(cp.ATTEMPTED) else set()
seen, urls = set(), []
for u in links:
    host = urlparse(u).netloc.lower().split(':')[0]
    if not host.endswith(TLD) or u in attempted or host in seen:
        continue
    seen.add(host)
    urls.append(u)
random.Random(11).shuffle(urls)
todo = urls[:attempts]
print(mode, '| phishing links on these domains:', len(urls), '| trying now:', len(todo))

os.makedirs(cp.IMG_DIR, exist_ok=True)
driver = cp.make_driver()
kept, restarts = 0, 0
for i, u in enumerate(todo):
    with open(cp.ATTEMPTED, 'a') as f:
        f.write(u + '\n')
    try:
        row = cp.grab(driver, u, 'phishing', len(rows))
    except Exception:
        row = None
        try:
            driver.switch_to.alert.accept()
        except Exception:
            pass
    if row:
        rows.append(row)
        kept += 1
    if (not row and not cp.alive(driver)) or (i + 1) % 60 == 0:
        try:
            driver.quit()
        except Exception:
            pass
        driver = cp.make_driver()
        restarts += 1
    if (i + 1) % 20 == 0:
        pd.DataFrame(rows).to_csv(cp.OUT_CSV, index=False)
        print(f'{i + 1}/{len(todo)} tried, kept {kept}, browser restarts {restarts}', flush=True)
try:
    driver.quit()
except Exception:
    pass
pd.DataFrame(rows).to_csv(cp.OUT_CSV, index=False)
print('Done.', mode, 'kept', kept, 'of', len(todo), '| totals:', pd.DataFrame(rows)['label'].value_counts().to_dict())
