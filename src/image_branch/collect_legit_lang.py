
import sys, os, io, random, zipfile, urllib.request
import pandas as pd
from urllib.parse import urlparse
sys.path.insert(0, 'src/image_branch')
import collect_pages as cp

mode, attempts = sys.argv[1], int(sys.argv[2])
SEED_INDIAN = ['bhaskar.com', 'jagran.com', 'amarujala.com', 'livehindi.com', 'patrika.com', 'dinamalar.com',
               'dailythanthi.com', 'vikatan.com', 'eenadu.net', 'sakshi.com', 'mathrubhumi.com', 'manoramaonline.com',
               'anandabazar.com', 'lokmat.com', 'loksatta.com', 'sandesh.com', 'prajavani.net', 'andhrajyothy.com',
               'divyabhaskar.co.in', 'hindi.news18.com', 'navbharattimes.indiatimes.com', 'tamil.oneindia.com',
               'hindi.oneindia.com', 'telugu.oneindia.com', 'bengali.oneindia.com', 'marathi.abplive.com',
               'abplive.com', 'ndtv.in', 'aajtak.in', 'jansatta.com', 'prabhatkhabar.com', 'dainik-b.in']
TLD = {'european': ('.de', '.fr', '.es', '.it', '.pt', '.nl', '.pl', '.ro', '.se', '.dk', '.fi', '.cz', '.hu', '.gr',
                    '.bg', '.hr', '.sk', '.si', '.at', '.be', '.no', '.lt', '.lv', '.ee'),
       'indian': ('.in',),
       'english': ('.com', '.org', '.net', '.uk', '.au', '.ca', '.nz', '.ie', '.edu', '.gov')}[mode]

LIST = 'data/raw/legitimate_urls.csv'
if os.path.exists(LIST):
    doms = pd.read_csv(LIST, header=None, names=['rank', 'domain'])['domain'].astype(str).tolist()
else:
    raw = urllib.request.urlopen('https://tranco-list.eu/top-1m.csv.zip', timeout=90).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    doms = pd.read_csv(z.open(z.namelist()[0]), header=None, names=['rank', 'domain'])['domain'].astype(str).tolist()
doms = [d.lower() for d in doms[:150000]]
cand = [d for d in doms if d.endswith(TLD)]
if mode == 'indian':
    cand = SEED_INDIAN + cand

rows = pd.read_csv(cp.OUT_CSV).to_dict('records') if os.path.exists(cp.OUT_CSV) else []
hosts = set(urlparse(str(r['url'])).netloc.lower().replace('www.', '') for r in rows)
attempted = set(open(cp.ATTEMPTED).read().split('\n')) if os.path.exists(cp.ATTEMPTED) else set()
urls = []
for d in cand:
    u = 'https://' + d
    if d.replace('www.', '') in hosts or u in attempted:
        continue
    urls.append(u)
urls = list(dict.fromkeys(urls))
if mode != 'indian':
    random.Random(5).shuffle(urls)
todo = urls[:attempts]
print(mode, '| candidates:', len(urls), '| trying now:', len(todo))

os.makedirs(cp.IMG_DIR, exist_ok=True)
driver = cp.make_driver()
kept, restarts = 0, 0
for i, u in enumerate(todo):
    with open(cp.ATTEMPTED, 'a') as f:
        f.write(u + '\n')
    try:
        row = cp.grab(driver, u, 'legit', len(rows))
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
