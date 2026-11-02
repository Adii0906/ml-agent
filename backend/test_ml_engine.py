import csv
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml_engine import MLEngine


class MLEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.temp_dir.name)
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            self.engine = MLEngine(str(self.root / "experiments.db"))

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        return path

    async def test_analyze_tabular_reports_statistics(self):
        dataset = self.write_csv(
            "dataset.csv",
            [
                ["value", "group", "target"],
                ["1", "A", "yes"],
                ["2", "A", "no"],
                ["2", "A", "no"],
                ["", "B", "yes"],
            ],
        )

        analysis = await self.engine.analyze_dataset(str(dataset), "tabular")

        self.assertEqual(analysis["rows"], 4)
        self.assertEqual(analysis["columns"], 3)
        self.assertEqual(analysis["missing_values"]["value"], 1)
        self.assertEqual(analysis["duplicate_rows"], 1)
        self.assertEqual(analysis["numeric_features"], ["value"])
        self.assertEqual(analysis["class_distribution"], {"A": 3, "B": 1})
        self.assertEqual(analysis["numeric_stats"]["value"]["count"], 3.0)

    async def test_analyze_text_reports_character_word_and_line_counts(self):
        content = "hello world\nML tests"
        dataset = self.root / "notes.txt"
        dataset.write_text(content, encoding="utf-8")

        analysis = await self.engine.analyze_dataset(str(dataset), "text")

        self.assertEqual(analysis["total_characters"], len(content))
        self.assertEqual(analysis["total_words"], 4)
        self.assertEqual(analysis["total_lines"], 1)
        self.assertEqual(analysis["avg_word_length"], len(content) / 4)

    async def test_analyze_unsupported_dataset_type_returns_error(self):
        analysis = await self.engine.analyze_dataset("unused.csv", "image")

        self.assertEqual(analysis, {"error": "Unsupported dataset type"})

    async def test_experiment_can_be_created_listed_and_deleted(self):
        experiment_id = await self.engine.create_experiment(
            name="test experiment",
            algorithm="Random Forest",
            dataset_path="dataset.csv",
            config={"target_column": "target"},
        )

        experiment = await self.engine.get_experiment(experiment_id)
        experiments = await self.engine.get_all_experiments()

        self.assertEqual(experiment["config"], {"target_column": "target"})
        self.assertEqual(experiment["status"], "initializing")
        self.assertEqual([item["id"] for item in experiments], [experiment_id])
        self.assertTrue(await self.engine.delete_experiment(experiment_id))
        self.assertIsNone(await self.engine.get_experiment(experiment_id))
        self.assertFalse(await self.engine.delete_experiment(experiment_id))

    async def test_training_completes_and_persists_metrics(self):
        dataset = self.write_csv(
            "training.csv",
            [
                ["score", "color", "target"],
                ["1", "red", "no"],
                ["2", "blue", "no"],
                ["3", "red", "no"],
                ["4", "blue", "no"],
                ["5", "red", "no"],
                ["6", "blue", "yes"],
                ["7", "red", "yes"],
                ["8", "blue", "yes"],
                ["9", "red", "yes"],
                ["10", "blue", "yes"],
            ],
        )
        experiment_id = await self.engine.create_experiment(
            name="training test",
            algorithm="Decision Tree",
            dataset_path=str(dataset),
            config={"target_column": "target", "task_type": "classification"},
        )

        with patch("ml_engine.joblib.dump") as dump_model:
            updates = [
                update
                async for update in self.engine.train_model_streaming(
                    experiment_id,
                    "Decision Tree",
                    str(dataset),
                    {"target_column": "target", "task_type": "classification"},
                )
            ]

        experiment = await self.engine.get_experiment(experiment_id)

        self.assertEqual(updates[-1]["stage"], "completed")
        self.assertEqual(updates[-1]["progress"], 100)
        self.assertEqual(experiment["status"], "completed")
        self.assertEqual(experiment["metrics"]["task_type"], "classification")
        self.assertEqual(experiment["metrics"]["n_samples"], 10)
        self.assertEqual(experiment["metrics"]["n_features"], 2)
        self.assertGreaterEqual(experiment["metrics"]["accuracy"], 0)
        self.assertLessEqual(experiment["metrics"]["accuracy"], 1)
        self.assertEqual(json.loads(json.dumps(experiment["metrics"])), experiment["metrics"])
        dump_model.assert_called_once()
        self.assertEqual(dump_model.call_args.args[1], f"model_{experiment_id}.joblib")


if __name__ == "__main__":
    unittest.main()
