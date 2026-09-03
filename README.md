# Agricultural Production Prediction

This project trains and evaluates a regression model that predicts agricultural
production from district, area, weather, irrigation, seed, fertilizer,
pesticide, and year information.

## What was corrected

- The supplied `test_data.csv` is now used for an independent evaluation.
- `Total_Cost` is excluded from the predictors. In the supplied test data it is
  exactly ten times `Production`, so using it would leak the answer into the
  model and produce a misleadingly high score.
- `District` is one-hot encoded instead of being treated as a continuous value.
- Regression is assessed with R-squared, MAE, RMSE, and nonzero-target MAPE.
- Predictions are constrained to be nonnegative. A record with zero cultivated
  area returns zero production.
- Input datasets and command-line prediction values are validated.
- Model behavior is covered by automated tests.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`

## Setup

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python thesis_model.py
```

The command trains only on `training_data.csv`, evaluates only on
`test_data.csv`, and prints a five-row prediction preview.

## Make a prediction

Pass the ten values in this exact order:

1. District
2. Area
3. Rainfall
4. irrigation
5. total irregation
6. seed_required
7. seed_available
8. fertilizer
9. Pesticides
10. Year

Example:

```bash
python thesis_model.py --predict 1 900 2488 4 22 486 47.093 0.248 1122.322 2018
```

The unusual spelling `total irregation` is retained because it is the column
name in the original CSV files.

## Run the tests

```bash
python -m unittest -v
```

The corrected workflow is also available in `Untitled20.ipynb` for use in
Jupyter Notebook or Google Colab.
