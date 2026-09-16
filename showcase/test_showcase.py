"""Static export tests. Run with python -m unittest discover -s showcase -p 'test_*.py'."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("showcase_build", Path(__file__).with_name("build.py"))
build_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_module)


class ShowcaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.data = build_module.build(cls.output)
        cls.html = (cls.output / "index.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_actual_archive_decisions(self):
        by_id = {r["id"]: r for r in self.data["reports"]}
        self.assertEqual(sum(by_id["formal"]["checks"].values()), 18)
        self.assertEqual(len(by_id["formal"]["checks"]), 22)
        self.assertEqual(by_id["formal"]["seeds"], 50)
        self.assertEqual(sum(by_id["holdout"]["checks"].values()), 22)
        self.assertEqual(sum(by_id["development"]["checks"].values()), 22)

    def test_all_scenarios_and_distribution_metrics_present(self):
        for report in self.data["reports"]:
            self.assertEqual(len(report["summary"]), 16)
            for stats in report["summary"].values():
                self.assertEqual(set(stats), set(build_module.METRICS))

    def test_provenance_and_finite_json(self):
        exported = json.loads((self.output / "data.json").read_text(encoding="utf-8"))
        self.assertEqual(exported, self.data)
        for report in exported["reports"]:
            self.assertEqual(len(report["sha256"]), 64)
            self.assertTrue((build_module.ROOT / report["path"] / "mechanism_report.json").is_file())

    def test_offline_self_contained_html(self):
        self.assertNotIn("/* SHOWCASE_CSS */", self.html)
        self.assertNotIn("/* SHOWCASE_JS */", self.html)
        self.assertNotIn('{"reports":[]}', self.html)
        self.assertIn('type="application/json"', self.html)
        self.assertIn('id="failed-only"', self.html)
        self.assertIn('id="motion-toggle"', self.html)
        self.assertIn('prefers-reduced-motion', self.html)
        self.assertIn('Stage 1 未完成', self.html)
        self.assertNotIn("fetch(", self.html)


if __name__ == "__main__":
    unittest.main()
