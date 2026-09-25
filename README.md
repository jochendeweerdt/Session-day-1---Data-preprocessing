# Session day 1 – Introduction and data preprocessing

Postgraduate Studies in Business Analytics & AI (KU Leuven) · Session day 1 · Prof. dr. Jochen De Weerdt

This repository contains all material of the first session day:

| Session | Topic | Material |
|---|---|---|
| 1.1 | Introduction to the programme and to business analytics and AI | [Slides (PDF)](slides/Session_1.1_Introduction.pdf) |
| 1.2 – 1.3 | Data preprocessing, exploratory data analysis and anomaly detection | [Slides (PDF)](slides/Session_1.2-1.3_Data_preprocessing.pdf) · four hands-on notebooks (below) |

We use the notebooks during sessions 1.2 and 1.3: the lecturer runs a short demo, then you do a small exercise
yourself. Afterwards you can keep using them as a reference.

You do not need to install anything. Everything runs in your browser with **Google Colab**.

---

## The slides

The slides are in the [`slides`](slides) folder as PDF files. Click a file to view it on GitHub, or use the
download button to save it.

- **Session 1.1 – Introduction**: the programme, practical details and the dissertation, what business analytics
  and AI are about, the data analytics process, AI projects and their challenges.
- **Sessions 1.2 – 1.3 – Data preprocessing**: data selection and leakage, exploratory data analysis, data cleaning
  and transformation, anomaly detection. The hands-on slides link to the notebooks, and section numbers in the
  notebooks (for example **§3.4**) match the slides.

---

## Before the session

1. Make sure you have a **Google account** (a personal Gmail account works best).
2. Open [colab.research.google.com](https://colab.research.google.com) once and sign in, so you know it works.
3. Bring a laptop. A tablet works for reading, but typing code is much easier on a laptop.

No prior Python experience is required for Exercise A. Exercise B asks you to write a few lines of code.

---

## The notebooks

| # | Notebook | Topic | Data | Open |
|---|---|---|---|---|
| 1 | `NB1_selection_and_leakage` | Data selection, the prediction point and data leakage | Telecom churn | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jochendeweerdt/Session-day-1---Data-preprocessing/blob/main/notebooks/NB1_selection_and_leakage.ipynb) |
| 2 | `NB2_exploratory_data_analysis` | Exploratory data analysis (EDA) | House sales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jochendeweerdt/Session-day-1---Data-preprocessing/blob/main/notebooks/NB2_exploratory_data_analysis.ipynb) |
| 3 | `NB3_cleaning_transformation_pipelines` | Cleaning, transformation and pipelines | Telecom churn | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jochendeweerdt/Session-day-1---Data-preprocessing/blob/main/notebooks/NB3_cleaning_transformation_pipelines.ipynb) |
| 4 | `NB4_anomaly_detection` | Anomaly detection | Credit card transactions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jochendeweerdt/Session-day-1---Data-preprocessing/blob/main/notebooks/NB4_anomaly_detection.ipynb) |

The slides of sessions 1.2 – 1.3 contain the same links and a QR code for each notebook.

---

## How to work with a notebook

1. **Open** the notebook with its *Open in Colab* button (above, on the slides, or at the top of the notebook).
2. Colab may warn that the notebook was *not authored by Google*. That is expected: click **Run anyway**.
3. **Save your own copy**: `File → Save a copy in Drive`. Work in that copy, otherwise your changes are lost
   when you close the tab.
4. **Run the cells from top to bottom**: click a cell and press `Shift + Enter`, or use `Runtime → Run all`.
   The first code cell installs a few extra libraries and loads the data; this takes up to a minute.
5. Follow the demo. The section numbers in the notebook (for example **§3.4**) match the numbers on the slides.

### The exercises

Every notebook has two kinds of exercises:

- **Exercise A – basic.** Change a value in a line marked `# <- change`, run the cell again, and answer
  the questions in the text. No programming needed.
- **Exercise B – more challenging.** Write a few lines of code yourself.

**Solutions** are in cells titled *Solution*. Their code is hidden: click *Show code* (or double-click the
cell) to see it. Try the exercise first.

### If something goes wrong

| Problem | What to do |
|---|---|
| `NameError: name '...' is not defined` | A cell above was not run. Use `Runtime → Run before` or `Runtime → Run all`. |
| The data does not load | Check your internet connection and run the first code cell again. |
| Colab asks to restart the session after installing libraries | Click *Restart session*, then run all cells again. |
| The session disconnected (after a break) | `Runtime → Run all`. Your edits in your saved copy are kept. |
| Something is completely broken | Open the original notebook again via the link and start from a fresh copy. |

---

## The datasets

All data files are in the `data` folder and are loaded automatically by the notebooks.

| File | Content |
|---|---|
| `churn_sample.xlsx` | A 36-row excerpt of a telecom churn dataset, as a data owner might send it to you. |
| `churn.csv` | 5,040 customers with the columns of the excerpt plus a few extra columns from the data owner. The data is **synthetic** and contains typical real-world problems on purpose: leaky columns, missing values, inconsistent spellings and formats, impossible values, extreme but valid values, duplicates. |
| `housing.csv` | 3,000 (artificial) house sales with the sale price as target. Note: the columns are separated by `;`. |
| `creditcard_sample.csv` | 30,492 card transactions of European cardholders, of which 492 are fraudulent. The variables `V1`–`V28` are principal components of confidential features. |

The credit card data is a subsample of the dataset by Dal Pozzolo, Caelen, Johnson and Bontempi,
*Calibrating probability with undersampling for unbalanced classification* (IEEE Symposium on Computational
Intelligence and Data Mining, 2015), Machine Learning Group, Université Libre de Bruxelles, available on
Kaggle under the Open Database License (ODbL).

---

## Running the notebooks on your own computer (optional)

```bash
git clone https://github.com/jochendeweerdt/Session-day-1---Data-preprocessing.git
cd Session-day-1---Data-preprocessing
pip install -r requirements.txt
jupyter lab
```

When run locally, the notebooks read the data from the `data` folder instead of from GitHub.

---

## For the lecturer: maintaining this repository

- Slides: export both decks from PowerPoint as PDF (`File → Export → PDF`) and upload them to the `slides` folder
  under exactly these names, so the links above keep working: `Session_1.1_Introduction.pdf` and
  `Session_1.2-1.3_Data_preprocessing.pdf`. Hide the Wi-Fi slide of session 1.1 before exporting.

- The notebooks are generated by `tools/build_nb1.py` … `tools/build_nb4.py` (run them from the `tools`
  folder). Edit the builder, not the `.ipynb`, so changes are not lost.
- `tools/make_churn.py` regenerates `churn.csv`; `tools/make_creditcard_sample.py` the credit card subsample.
- `python tools/run_nb.py notebooks/*.ipynb` executes all notebooks as a test.
- `tools/set_repo_links.py <account>/<repository> <deck.pptx>` updates the repository name in the notebooks
  and the Colab links and QR codes on the slides (needed only if the repository moves).
- Tested with Python 3.11, scikit-learn 1.8, skrub 0.10 and PyOD 3.6.
