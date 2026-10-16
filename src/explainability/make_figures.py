import textwrap
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

e = pd.read_csv('results/fusion_explanations.csv')

def grid(df, path, title):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=15)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, r) in zip(axes.flat, df.iterrows()):
        ax.imshow(Image.open(r['filename']))
        ax.set_title(f"{str(r['label']).upper()}: " + str(r['url'])[:42], fontsize=10, pad=8)
        caption = textwrap.fill(str(r['explanation']).replace('Evidence: ', 'Evidence:\n'), 54)
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha='center', va='top', fontsize=9)
    plt.subplots_adjust(hspace=0.65, wspace=0.12, top=0.92, bottom=0.08)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)

ok = e[e['label'] == e['predicted']]
good = pd.concat([ok[ok['label'] == l].sample(min(3, int((ok['label'] == l).sum())), random_state=7) for l in ['phishing', 'legit']])
grid(good, 'results/explanation_examples.png', 'Correct predictions with explanations')

bad = e[e['label'] != e['predicted']]
missed = bad[bad['label'] == 'phishing']
false_alarm = bad[bad['label'] == 'legit']
mistakes = pd.concat([missed.sample(min(3, len(missed)), random_state=7), false_alarm.sample(min(3, len(false_alarm)), random_state=7)])
grid(mistakes, 'results/explanation_errors.png', 'Mistakes: missed phishing (first row) and false alarms (second row)')
print('Saved results/explanation_examples.png and results/explanation_errors.png')
