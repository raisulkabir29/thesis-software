"""Train on training_data.csv and name test results from the full dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAINING_DATA_FILE = "training_data.csv"
TEST_DATA_FILE = "test_data.csv"
FULL_AGRICULTURE_FILE = "full_agriculture_data.csv"
PROGRAM_VERSION = "3.0-three-source-district-names"

TARGET_COLUMN = "Production"
CATEGORICAL_FEATURES = ["District"]
NUMERIC_FEATURES = [
    "Area",
    "Rainfall",
    "irrigation",
    "total irregation",
    "seed_required",
    "seed_available",
    "fertilizer",
    "Pesticides",
    "Year",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
MODEL_DATA_COLUMNS = ["ID"] + FEATURE_COLUMNS + [TARGET_COLUMN]


def resolve_data_path(path: str | Path) -> Path:
    """Find a data file in the working directory or beside this module."""
    resolved = Path(path)
    if not resolved.is_absolute() and not resolved.is_file():
        module_relative = Path(__file__).resolve().parent / resolved
        if module_relative.is_file():
            resolved = module_relative

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Data file not found: {resolved}. Keep all three CSV files in "
            "the same folder as thesis_model.py."
        )
    return resolved


def load_model_data(path: str | Path) -> pd.DataFrame:
    """Load and validate training_data.csv or test_data.csv."""
    path = resolve_data_path(path)
    data = pd.read_csv(path)
    missing = [column for column in MODEL_DATA_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError(f"{path.name} contains no rows")
    if data["ID"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate IDs")

    numeric_columns = ["ID"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    try:
        data.loc[:, numeric_columns] = data[numeric_columns].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} contains a nonnumeric model value") from exc

    values = data[numeric_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{path.name} contains an infinite value")
    essential = ["ID", "District", "Area", "Year", TARGET_COLUMN]
    if data[essential].isna().any().any():
        raise ValueError(f"{path.name} has a missing essential value")
    if (data["Area"] < 0).any() or (data[TARGET_COLUMN] < 0).any():
        raise ValueError(f"{path.name} contains a negative area or production")
    return data


def load_district_names(path: str | Path = FULL_AGRICULTURE_FILE) -> pd.DataFrame:
    """Load the authoritative ID-to-district-name lookup from the full CSV."""
    path = resolve_data_path(path)
    full_data = pd.read_csv(path, usecols=["id", "district"])
    if full_data.empty:
        raise ValueError(f"{path.name} contains no district records")
    if full_data["id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate IDs")
    if full_data[["id", "district"]].isna().any().any():
        raise ValueError(f"{path.name} contains a missing ID or district name")

    full_data.loc[:, "id"] = pd.to_numeric(full_data["id"], errors="raise")
    full_data.loc[:, "district"] = full_data["district"].astype(str).str.strip()
    if full_data["district"].eq("").any():
        raise ValueError(f"{path.name} contains a blank district name")
    if full_data["district"].str.fullmatch(r"\d+").any():
        raise ValueError(f"{path.name} must contain names, not district numbers")
    return full_data


def validate_three_data_sources(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    district_names: pd.DataFrame,
) -> None:
    """Verify that the train and test IDs are covered by the full name table."""
    training_ids = set(training_data["ID"].astype(int))
    test_ids = set(test_data["ID"].astype(int))
    full_ids = set(district_names["id"].astype(int))
    if training_ids & test_ids:
        raise ValueError("Training and test data contain overlapping IDs")

    missing_ids = sorted((training_ids | test_ids) - full_ids)
    if missing_ids:
        raise ValueError(
            "The full agriculture CSV has no district name for IDs: "
            + ", ".join(map(str, missing_ids))
        )


def build_district_code_lookup(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    district_names: pd.DataFrame,
) -> dict[str, float]:
    """Relate names to their most frequently associated numeric model code."""
    coded_ids = pd.concat(
        [
            training_data[["ID", "District"]],
            test_data[["ID", "District"]],
        ],
        ignore_index=True,
    )
    named_ids = district_names.rename(
        columns={"id": "ID", "district": "District_Name"}
    )
    mapping_rows = coded_ids.merge(
        named_ids, on="ID", how="left", validate="one_to_one"
    )
    if mapping_rows["District_Name"].isna().any():
        raise ValueError("A district code could not be matched to a district name")

    # The supplied sources contain one inconsistent Jessore code. Selecting the
    # modal code makes name-based new predictions deterministic; final test
    # result names are still matched directly and exactly by unique record ID.
    code_counts = (
        mapping_rows.groupby(["District_Name", "District"])
        .size()
        .rename("frequency")
        .reset_index()
        .sort_values(
            ["District_Name", "frequency", "District"],
            ascending=[True, False, True],
        )
    )
    return (
        code_counts.drop_duplicates("District_Name")
        .set_index("District_Name")["District"]
        .astype(float)
        .to_dict()
    )


def build_model() -> Pipeline:
    """Build preprocessing and regression using the supplied feature columns."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "district_code",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric_measurements",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
        ]
    )
    regression = GradientBoostingRegressor(
        random_state=42,
        loss="squared_error",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
    )
    return Pipeline(
        [("preprocessing", preprocessing), ("regression", regression)]
    )


def train_model(training_data: pd.DataFrame) -> Pipeline:
    """Fit the model only on training_data.csv records."""
    model = build_model()
    model.fit(training_data[FEATURE_COLUMNS], training_data[TARGET_COLUMN])
    return model


def prepare_prediction_input(features: pd.DataFrame) -> pd.DataFrame:
    """Validate model-ready numeric feature rows."""
    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError(f"Prediction input is missing: {', '.join(missing)}")

    model_input = features[FEATURE_COLUMNS].copy()
    try:
        model_input.loc[:, FEATURE_COLUMNS] = model_input[FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction features must contain numeric values") from exc

    values = model_input[FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Prediction input contains an infinite value")
    if model_input[["District", "Area", "Year"]].isna().any().any():
        raise ValueError("District, area, and year cannot be missing")
    if (model_input["Area"] < 0).any():
        raise ValueError("Prediction area cannot be negative")
    return model_input


def predict_production(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predict production and prevent physically impossible negative values."""
    model_input = prepare_prediction_input(features)
    predictions = np.maximum(model.predict(model_input), 0.0)
    return np.where(model_input["Area"].to_numpy() == 0, 0.0, predictions)


def evaluate_model(model: Pipeline, test_data: pd.DataFrame) -> dict[str, float]:
    """Evaluate predictions only against test_data.csv."""
    actual = test_data[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = predict_production(model, test_data[FEATURE_COLUMNS])
    errors = np.abs(actual - predicted)
    nonzero = actual != 0
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wmape": float(errors.sum() / np.abs(actual).sum()),
        "median_ape": (
            float(np.median(errors[nonzero] / np.abs(actual[nonzero])))
            if nonzero.any()
            else float("nan")
        ),
    }


def build_results_table(
    model: Pipeline,
    test_data: pd.DataFrame,
    district_names: pd.DataFrame,
) -> pd.DataFrame:
    """Combine test predictions with district names from the full CSV."""
    predictions = predict_production(model, test_data[FEATURE_COLUMNS])
    results = test_data[["ID", "Year", TARGET_COLUMN]].copy()
    results["Predicted_Production"] = predictions
    results["Absolute_Error"] = np.abs(
        results[TARGET_COLUMN] - results["Predicted_Production"]
    )

    name_lookup = district_names.rename(
        columns={"id": "ID", "district": "District_Name"}
    )
    results = results.merge(
        name_lookup, on="ID", how="left", validate="one_to_one"
    )
    if results["District_Name"].isna().any():
        missing_ids = results.loc[results["District_Name"].isna(), "ID"].tolist()
        raise ValueError(f"District names are missing for test IDs: {missing_ids}")

    results = results.rename(columns={TARGET_COLUMN: "Actual_Production"})
    results = results[
        [
            "District_Name",
            "Year",
            "Actual_Production",
            "Predicted_Production",
            "Absolute_Error",
        ]
    ]
    numeric_output = [
        "Actual_Production",
        "Predicted_Production",
        "Absolute_Error",
    ]
    results.loc[:, numeric_output] = results[numeric_output].round(2)
    return results


def prediction_frame(
    values: Sequence[str | float], district_codes: Mapping[str, float]
) -> pd.DataFrame:
    """Convert a district name and nine measurements into one model-ready row."""
    if len(values) != 10:
        raise ValueError("Prediction requires one district name and nine measurements")

    district_name = str(values[0]).strip()
    if district_name not in district_codes:
        raise ValueError(f"Unknown district name: {district_name}")
    try:
        measurements = [float(value) for value in values[1:]]
    except (TypeError, ValueError) as exc:
        raise ValueError("The nine measurements after district must be numeric") from exc

    return pd.DataFrame(
        [
            {
                "District": district_codes[district_name],
                **dict(zip(NUMERIC_FEATURES, measurements, strict=True)),
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", default=TRAINING_DATA_FILE)
    parser.add_argument("--test-data", default=TEST_DATA_FILE)
    parser.add_argument("--full-data", default=FULL_AGRICULTURE_FILE)
    parser.add_argument(
        "--predict",
        nargs=10,
        metavar=tuple(f"VALUE_{index}" for index in range(1, 11)),
        help=(
            "district name, area, rainfall, irrigation, total irrigation, "
            "seed required, seed available, fertilizer, pesticides, year"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_data = load_model_data(args.training_data)
    test_data = load_model_data(args.test_data)
    district_names = load_district_names(args.full_data)
    validate_three_data_sources(training_data, test_data, district_names)
    district_codes = build_district_code_lookup(
        training_data, test_data, district_names
    )

    model = train_model(training_data)
    metrics = evaluate_model(model, test_data)
    results = build_results_table(model, test_data, district_names)

    print(f"Training source: {resolve_data_path(args.training_data).name}")
    print(f"Training rows: {len(training_data)}")
    print(f"Test source: {resolve_data_path(args.test_data).name}")
    print(f"Test rows: {len(test_data)}")
    print(f"District-name source: {resolve_data_path(args.full_data).name}")
    print(f"District names: {district_names['district'].nunique()}")
    print("Excluded leakage feature: Total_Cost")
    print(f"Test R-squared: {metrics['r2']:.4f}")
    print(f"Test MAE: {metrics['mae']:.2f}")
    print(f"Test RMSE: {metrics['rmse']:.2f}")
    print(f"Test WMAPE: {metrics['wmape']:.2%}")
    print(f"Median absolute percentage error: {metrics['median_ape']:.2%}")
    print("\nFinal test results with district names:")
    print(results.head(10).to_string(index=False))

    if args.predict is not None:
        new_record = prediction_frame(args.predict, district_codes)
        predicted = predict_production(model, new_record)[0]
        print(f"\nDistrict: {args.predict[0]}")
        print(f"Predicted production: {predicted:.2f} M.Tons")


if __name__ == "__main__":
    main()

