"""Tests proving that training, test, and district-name CSVs are all used."""

import unittest
from pathlib import Path

from thesis_model_v3 import (
    FEATURE_COLUMNS,
    FULL_AGRICULTURE_FILE,
    TEST_DATA_FILE,
    TRAINING_DATA_FILE,
    build_district_code_lookup,
    build_results_table,
    evaluate_model,
    load_district_names,
    load_model_data,
    predict_production,
    prediction_frame,
    train_model,
    validate_three_data_sources,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent


class ThesisModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load all three required CSV files exactly as the application does.
        cls.training_data = load_model_data(
            PROJECT_DIRECTORY / TRAINING_DATA_FILE
        )
        cls.test_data = load_model_data(PROJECT_DIRECTORY / TEST_DATA_FILE)
        cls.district_names = load_district_names(
            PROJECT_DIRECTORY / FULL_AGRICULTURE_FILE
        )
        validate_three_data_sources(
            cls.training_data, cls.test_data, cls.district_names
        )
        cls.district_codes = build_district_code_lookup(
            cls.training_data, cls.test_data, cls.district_names
        )
        cls.model = train_model(cls.training_data)

    def test_fetches_training_data_csv(self) -> None:
        self.assertEqual(len(self.training_data), 304)
        self.assertEqual(self.training_data["ID"].nunique(), 304)

    def test_fetches_test_data_csv(self) -> None:
        self.assertEqual(len(self.test_data), 38)
        self.assertEqual(self.test_data["ID"].nunique(), 38)

    def test_fetches_district_names_from_full_csv(self) -> None:
        self.assertEqual(len(self.district_names), 342)
        self.assertEqual(self.district_names["district"].nunique(), 44)
        self.assertIn("Bandarban", set(self.district_names["district"]))
        self.assertFalse(
            self.district_names["district"].str.fullmatch(r"\d+").any()
        )

    def test_training_and_test_ids_cover_the_full_csv(self) -> None:
        combined_ids = set(self.training_data["ID"]) | set(self.test_data["ID"])
        self.assertEqual(combined_ids, set(self.district_names["id"]))
        self.assertFalse(set(self.training_data["ID"]) & set(self.test_data["ID"]))

    def test_numeric_district_codes_map_to_names(self) -> None:
        self.assertEqual(len(self.district_codes), 44)
        self.assertEqual(self.district_codes["Bandarban"], 1.0)
        self.assertEqual(self.district_codes["Chittagong"], 2.0)

    def test_id_target_and_total_cost_are_not_features(self) -> None:
        self.assertNotIn("ID", FEATURE_COLUMNS)
        self.assertNotIn("Production", FEATURE_COLUMNS)
        self.assertNotIn("Total_Cost", FEATURE_COLUMNS)

    def test_model_trains_on_training_and_evaluates_on_test(self) -> None:
        metrics = evaluate_model(self.model, self.test_data)
        self.assertGreater(metrics["r2"], 0.90)
        self.assertGreaterEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["rmse"], 0.0)

    def test_final_results_show_names_and_not_numeric_ids(self) -> None:
        results = build_results_table(
            self.model, self.test_data, self.district_names
        )
        self.assertEqual(
            results.columns.tolist(),
            [
                "District_Name",
                "Year",
                "Actual_Production",
                "Predicted_Production",
                "Absolute_Error",
            ],
        )
        self.assertNotIn("ID", results.columns)
        self.assertNotIn("District", results.columns)
        self.assertEqual(results.iloc[0]["District_Name"], "Bandarban")

    def test_every_test_result_has_a_district_name(self) -> None:
        results = build_results_table(
            self.model, self.test_data, self.district_names
        )
        self.assertEqual(len(results), len(self.test_data))
        self.assertFalse(results["District_Name"].isna().any())
        self.assertTrue(results["District_Name"].map(type).eq(str).all())

    def test_predictions_are_nonnegative(self) -> None:
        predictions = predict_production(self.model, self.test_data[FEATURE_COLUMNS])
        self.assertTrue((predictions >= 0).all())

    def test_zero_area_returns_zero_production(self) -> None:
        row = self.test_data[FEATURE_COLUMNS].iloc[[0]].copy()
        row.loc[:, "Area"] = 0
        self.assertEqual(predict_production(self.model, row)[0], 0.0)

    def test_new_prediction_accepts_a_district_name(self) -> None:
        row = prediction_frame(
            ["Bandarban", 900, 2488, 4, 22, 486, 47.093, 0.248, 1122.322, 2018],
            self.district_codes,
        )
        self.assertEqual(row.loc[0, "District"], 1.0)
        self.assertGreaterEqual(predict_production(self.model, row)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
