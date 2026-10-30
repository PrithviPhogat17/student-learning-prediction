# Data

## Primary dataset: UCI "Student Performance"
* Creator: Paulo Cortez. Donated 2014. DOI: https://doi.org/10.24432/C5TG7T
* Page: https://archive.ics.uci.edu/dataset/320/student+performance
* Licence: **Creative Commons Attribution 4.0 (CC BY 4.0)** - sharing and adaptation allowed with credit.
* Content: two Portuguese secondary schools; Mathematics (`student-mat.csv`, 395 rows) and Portuguese
  (`student-por.csv`, 649 rows); 33 columns; no missing values reported by the source.
* Introductory paper: P. Cortez and A. Silva, "Using data mining to predict secondary school student performance",
  Proceedings of 5th Annual Future Business Technology Conference, 2008.

Citation: *Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T*

The data files are **not included** in this project. Get them with:

    python scripts/download_data.py

or manually: download the ZIP from the page above, unzip (a nested `student.zip` is inside; unzip it too) and copy
`student-mat.csv` into `data/raw/`.

## Folders
* `raw/` - original CSV (git-ignored).
* `processed/` - `train.csv`, `val.csv`, `test.csv` written by training (git-ignored). They hold the selected columns,
  the anonymous `record_id`, `G3` and the `at_risk` label (not scaled).
* `demo/` - optional **synthetic** file created by `scripts/make_demo_data.py`. It is random data and must never be
  presented as real student data.

## Privacy
Only anonymous row numbers are used. The loader keeps just the 14 model inputs plus G3; other personal columns
(sex, age, address, parents' jobs, alcohol use, ...) are dropped at load time.
