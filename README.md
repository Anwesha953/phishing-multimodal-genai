# Multimodal Generative AI for Phishing and Social-Engineering Attack Detection

Authors: Anwesha Das (2305604), Aakash Pandey (23053641). Guide: Sujata Swain.

## What it does
Fuses three signals of the same website: URL (host-level features), page text (TF-IDF + SVD + logistic regression) and page screenshot (CLIP ViT-B/32 + logistic regression). A logistic-regression meta-learner combines them (stacked fusion). Each decision is explained with exact SHAP values and turned into a plain-language sentence.

## Data
2,144 collected pages (1,299 legitimate, 845 phishing): data/processed/pages_dataset_lang.csv.
Evaluation set: 952 pages (final_index.csv, final_features.npy), balanced inside each language group, max 2 pages per registered domain per label.
Indian-language phishing pages are almost absent from public feeds (a finding in itself).

## Results (952 pages, nested grouped CV by registered domain)
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| URL structure only (host) | 0.679 | 0.744 | 0.544 | 0.629 | 0.709 |
| Page text only | 0.804 | 0.829 | 0.765 | 0.796 | 0.875 |
| Screenshot only (CLIP) | 0.769 | 0.769 | 0.769 | 0.769 | 0.829 |
| Fused (all three) | 0.824 | 0.836 | 0.805 | 0.820 | 0.895 |

Fused accuracy by language group: English 0.839 (708 pages), European 0.795 (200), Other 0.705 (44).
Files: results_final/metrics_table.csv, language_group_accuracy.csv, shap_explanations.csv, shap_modality_importance.png, shap_examples.png. Earlier results on the first 396-page set are in results/.

## Other findings
- Full-URL features gave 97.8% because of a dataset shortcut; only host-level URL features are used.
- GenAI augmentation (NLLB machine-translated samples, 1,960 rows, 7 languages) did not help a TF-IDF text model or multilingual MiniLM.
- Human rating of explanation clarity (1-5, 10 items): mean 3.0 so far (3 responses).

## Run
python src/image_branch/collect_pages.py LABEL ATTEMPTS
python src/explainability/metrics_and_shap_final.py
