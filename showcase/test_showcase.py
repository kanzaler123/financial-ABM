"""Static export tests. Run with python -m unittest discover -s showcase -p 'test_*.py'."""
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("showcase_build", HERE / "build.py")
build_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_module)


def text_keys(language: str, page: str = "sandbox") -> set[str]:
    source = (HERE / page / "i18n.js").read_text(encoding="utf-8")
    block = source.split(f"// {language}\n", 1)[1].split("\n    },", 1)[0]
    return set(re.findall(r"^\s{6}(\w+):", block, flags=re.MULTILINE))


CJK = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")


class UntranslatedText(HTMLParser):
    """Chinese text or aria-labels in a template that no i18n attribute (or script-set id) covers."""

    VOID = {"meta", "link", "br", "img", "input", "hr", "source", "stop", "circle", "ellipse", "path"}

    def __init__(self, dynamic_ids: set[str]):
        super().__init__()
        self.dynamic_ids, self.stack, self.problems = dynamic_ids, [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if CJK.search(attrs.get("aria-label") or "") and "data-i18n-aria" not in attrs:
            self.problems.append(f"<{tag}> aria-label {attrs['aria-label']}")
        if tag not in self.VOID:
            self.stack.append((tag, attrs))

    def handle_startendtag(self, tag, attrs):
        attrs = dict(attrs)
        if CJK.search(attrs.get("aria-label") or "") and "data-i18n-aria" not in attrs:
            self.problems.append(f"<{tag}> aria-label {attrs['aria-label']}")

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()[0] == tag:
                break

    def handle_data(self, data):
        if not CJK.search(data) or not self.stack:
            return
        tags = [tag for tag, _ in self.stack]
        if "noscript" in tags or tags[-1] == "title" and "svg" not in tags:  # no script / set via document.title
            return
        tag, attrs = self.stack[-1]
        if "data-i18n" in attrs or "data-i18n-text" in attrs or attrs.get("id") in self.dynamic_ids:
            return
        self.problems.append(f"<{tag}> {data.strip()[:40]}")


def untranslated(page: str, dynamic_ids: set[str]) -> list[str]:
    parser = UntranslatedText(dynamic_ids)
    parser.feed((HERE / page / "index.html").read_text(encoding="utf-8"))
    return parser.problems


SITE_URL = "https://kanzaler123.github.io/financial-ABM/"
PREVIEW_TAGS = ("og:title", "og:description", "og:type", "og:url", "og:image", "og:image:width", "og:image:height",
                "og:image:alt", "twitter:card", "twitter:title", "twitter:description", "twitter:image")


class MetaTags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and "content" in attrs and (attrs.get("property") or attrs.get("name")):
            self.tags.append((attrs.get("property") or attrs.get("name"), attrs["content"]))


def head_meta(html: str) -> list[tuple[str, str]]:
    parser = MetaTags()
    parser.feed(html.split("</head>", 1)[0])
    return parser.tags


class ShowcaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.data = build_module.build(cls.output)
        cls.html = (cls.output / "stage1" / "index.html").read_text(encoding="utf-8")
        cls.sandbox_html = (cls.output / "index.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    # -- stage 1 exhibit (historical archive) --------------------------------------
    def test_actual_archive_decisions(self):
        by_id = {r["id"]: r for r in self.data["stage1"]["reports"]}
        self.assertEqual(sum(by_id["formal"]["checks"].values()), 18)
        self.assertEqual(len(by_id["formal"]["checks"]), 22)
        self.assertEqual(by_id["formal"]["seeds"], 50)
        self.assertEqual(sum(by_id["holdout"]["checks"].values()), 22)
        self.assertEqual(sum(by_id["development"]["checks"].values()), 22)

    def test_all_scenarios_and_distribution_metrics_present(self):
        for report in self.data["stage1"]["reports"]:
            self.assertEqual(len(report["summary"]), 16)
            for stats in report["summary"].values():
                self.assertEqual(set(stats), set(build_module.METRICS))

    def test_provenance_and_finite_json(self):
        exported = json.loads((self.output / "stage1" / "data.json").read_text(encoding="utf-8"))
        self.assertEqual(exported, self.data["stage1"])
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
        self.assertIn('href="../"', self.html)
        self.assertNotIn("fetch(", self.html)

    # -- greenwashing policy sandbox (home page) -------------------------------------
    def test_sandbox_is_offline_and_self_contained(self):
        for marker in ("/* SANDBOX_CSS */", "/* SANDBOX_CORE */", "/* SANDBOX_DESK */", "/* SANDBOX_TEXT */", "/* SANDBOX_APP */", '{"sandbox":null}'):
            self.assertNotIn(marker, self.sandbox_html)
        self.assertIn("SandboxDesk", self.sandbox_html)
        for element in ('id="sphere"', 'id="spectro"', 'id="wire-feed"', 'id="trace-body"', 'id="ticket-select"'):
            self.assertIn(element, self.sandbox_html)
        self.assertNotIn("fetch(", self.sandbox_html)
        self.assertIsNone(re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://', self.sandbox_html))
        self.assertIn('href="stage1/"', self.sandbox_html)
        self.assertIn('data-domain="ai_washing"', self.sandbox_html)
        self.assertIn('prefers-reduced-motion', self.sandbox_html)
        exported = json.loads((self.output / "sandbox-data.json").read_text(encoding="utf-8"))
        self.assertEqual(exported, self.data["sandbox"])

    def test_sandbox_has_share_link(self):
        self.assertIn('id="share-link"', self.sandbox_html)

    # -- panel 1 CSV export ----------------------------------------------------------------
    def test_sandbox_has_csv_export_button_with_bilingual_text(self):
        self.assertIn('id="export-csv"', self.sandbox_html)
        template = (HERE / "sandbox" / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-i18n="exportCsv"', template)
        for key in ("zh", "en"):
            self.assertIn("exportCsv", text_keys(key))

    def test_csv_export_uses_a_blob_and_no_network_request(self):
        app = (HERE / "sandbox" / "app.js").read_text(encoding="utf-8")
        export = app.split("const exportCsv", 1)[1].split("const selectFirm", 1)[0]
        self.assertIn("new Blob", export)
        self.assertIn("text/csv", export)
        self.assertIn("URL.createObjectURL", export)
        self.assertNotIn("fetch(", export)
        self.assertNotIn("XMLHttpRequest", export)

    def test_csv_export_columns_and_policy_header(self):
        app = (HERE / "sandbox" / "app.js").read_text(encoding="utf-8")
        self.assertIn("csvRow(['id', 'size_class', 'market_cap', 'action', 'amount'])", app)
        self.assertIn("transparency tau", app)
        self.assertIn("fixed fine", app)
        self.assertIn("proportional penalty", app)
        self.assertIn("audit probability", app)

    def test_csv_cell_escaping_matches_rfc4180(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is not installed")
        source = (HERE / "sandbox" / "app.js").read_text(encoding="utf-8")
        match = re.search(r"const csvCell = value => \{\n(.*?)\n  \};", source, re.DOTALL)
        self.assertIsNotNone(match, "csvCell helper not found in app.js")
        cases = [
            ("plain", "plain"),
            ("a,b", '"a,b"'),
            ('say "hi"', '"say ""hi"""'),
            ("line\nbreak", '"line\nbreak"'),
            ("123", "123"),
            ("", ""),
        ]
        script = (
            "const csvCell = value => {\n" + match.group(1) + "\n};\n"
            "const cases = " + json.dumps(cases) + ";\n"
            "for (const [input, expected] of cases) {\n"
            "  const got = csvCell(input);\n"
            "  if (got !== expected) { console.error('FAIL ' + JSON.stringify(input) + ' => ' + JSON.stringify(got) + ', want ' + JSON.stringify(expected)); process.exit(1); }\n"
            "}\n"
            "console.log('csvCell ok');\n"
        )
        result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_pages_open_in_english_unless_chinese_was_chosen(self):
        self.assertIn('<html lang="en">', (HERE / "sandbox" / "index.html").read_text(encoding="utf-8"))
        for page in ("sandbox", "stage1"):
            app = (HERE / page / "app.js").read_text(encoding="utf-8")
            self.assertIn("storage.get() === 'zh' ? 'zh' : 'en'", app, page)

    def test_desk_replay_is_labelled_as_simplified_dynamics(self):
        template = (HERE / "sandbox" / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-i18n="wireTag"', template)
        self.assertIn('data-i18n="p2Tag"', template.split('id="decision"')[0])
        text = (HERE / "sandbox" / "i18n.js").read_text(encoding="utf-8")
        self.assertIn("回放 · 简化动态 · 非研究结果", text)
        self.assertIn("Replay · simplified dynamics · not a research result", text)

    def test_sandbox_labels_simplified_dynamics_and_ai_washing(self):
        zh = (HERE / "sandbox" / "i18n.js").read_text(encoding="utf-8")
        self.assertIn("无市场反馈 · 非研究结果", zh)
        self.assertIn("No market feedback · not a research result", zh)
        self.assertIn("AI-washing 领域没有完整模型结果", zh)
        self.assertIn("There are no full-model results for AI-washing", zh)

    def test_every_text_key_exists_in_both_languages(self):
        zh, en = text_keys("zh"), text_keys("en")
        self.assertGreater(len(zh), 150)
        self.assertEqual(zh, en)
        template = (HERE / "sandbox" / "index.html").read_text(encoding="utf-8")
        used = set(re.findall(r'data-i18n(?:-aria)?="(\w+)"', template))
        app = (HERE / "sandbox" / "app.js").read_text(encoding="utf-8")
        used |= set(re.findall(r"\bt\('(\w+)'", app))
        used |= set(re.findall(r"'((?:action|size|class|contrast|bracket)[A-Z]\w*)'", app))
        used |= {row["name"] for row in self.data["sandbox"]["stage_f"]["primary"]}
        self.assertEqual(used - zh, set())

    def test_stage1_text_keys_exist_in_both_languages(self):
        zh, en = text_keys("zh", "stage1"), text_keys("en", "stage1")
        self.assertGreater(len(zh), 200)
        self.assertEqual(zh, en)
        template = (HERE / "stage1" / "index.html").read_text(encoding="utf-8")
        used = set(re.findall(r'data-i18n(?:-aria|-text)?="(\w+)"', template))
        app = (HERE / "stage1" / "app.js").read_text(encoding="utf-8")
        used |= set(re.findall(r"\bt\('(\w+)'", app))
        used |= {f"step{i}{part}" for i in range(1, 7) for part in ("Title", "Text")}
        used |= {f"{prefix}_{metric}" for metric in build_module.METRICS for prefix in ("metric", "metricNote", "metricOption")}
        for report in self.data["stage1"]["reports"]:
            used |= {f"{prefix}_{name}" for name in report["summary"] for prefix in ("scenario", "scenarioNote")}
            used |= {f"{prefix}_{name}" for name in report["checks"] for prefix in ("check", "checkNote")}
        self.assertEqual(used - zh, set())
        self.assertIn('id="lang-toggle"', self.html)
        self.assertIn("ExhibitText", self.html)

    def test_no_chinese_text_escapes_the_language_switch(self):
        stage1_dynamic = {"step-title", "step-description", "kpi-decision", "kpi-burn", "kpi-runs", "gate-status",
                          "gate-summary", "gate-title", "scenario-detail", "motion-toggle"}
        self.assertEqual(untranslated("stage1", stage1_dynamic), [])
        self.assertEqual(untranslated("sandbox", set()), [])

    def test_sandbox_numbers_trace_to_committed_files(self):
        payload = self.data["sandbox"]
        source = payload["stage_f"]["source"]
        analysis = build_module.ROOT / source["path"]
        self.assertEqual(hashlib.sha256(analysis.read_bytes()).hexdigest(), source["sha256"])
        self.assertEqual(json.loads(analysis.read_text(encoding="utf-8"))["experiment_id"], source["experiment_id"])
        firms = payload["firms"]
        self.assertEqual(firms["seed"], 51001)
        self.assertEqual(set(firms["domains"]), {"esg", "ai_washing"})
        for domain in firms["domains"].values():
            self.assertEqual(set(domain["firms"]), build_module.SANDBOX_FIRM_FIELDS)
            self.assertEqual(len(domain["firms"]["benefit"]), 200)

    def test_readme_stage_f_numbers_match_the_exported_results(self):
        stage_f = self.data["sandbox"]["stage_f"]
        s = stage_f["secondary"]
        p2 = next(row for row in stage_f["primary"] if row["name"] == "P2_I_minus_F")
        p1 = next(row for row in stage_f["primary"] if row["name"] == "P1_market_full_period")
        ranges = [(s["decay0_I_minus_B"]["mean"], s["I_minus_B"]["mean"], 2, 2),
                  (s["decay0_F_minus_B"]["mean"], s["F_minus_B"]["mean"], 3, 2),
                  (s["decay0_I_minus_F"]["mean"], p2["mean"], 3, 2)]
        tau = next(row["tau_star"] for row in stage_f["tau_star_check"] if row["w"] == 0.2)
        for path, joiner in (("README.zh-CN.md", " 到 "), ("README.md", " to ")):
            text = (build_module.ROOT / path).read_text(encoding="utf-8")
            for low, high, low_digits, high_digits in ranges:
                self.assertIn(f"+{low:.{low_digits}f}{joiner}+{high:.{high_digits}f} SD", text, path)
            self.assertIn(f"+{p1['mean']:.2f} SD [+{p1['ci_low']:.2f}, +{p1['ci_high']:.2f}]", text, path)
            self.assertIn(f"τ* ≈ {tau:.2f}", text, path)

    # -- link previews (Open Graph / X cards) ----------------------------------------
    def test_pages_declare_english_link_previews_with_the_copied_card(self):
        card = self.output / "social-card.png"
        self.assertEqual(card.read_bytes(), (HERE / "social" / "social-card.png").read_bytes())
        self.assertEqual(build_module.png_size(card), (1200, 630))
        self.assertLess(card.stat().st_size, 5_000_000)  # X's limit for card images
        for page, html in (("", self.sandbox_html), ("stage1/", self.html)):
            with self.subTest(page=page or "home"):
                pairs = head_meta(html)
                keys = [key for key, _ in pairs]
                self.assertEqual([key for key in PREVIEW_TAGS if keys.count(key) != 1], [])
                tags = dict(pairs)
                self.assertEqual(tags["og:type"], "website")
                self.assertEqual(tags["og:url"], SITE_URL + page)
                self.assertEqual(tags["twitter:card"], "summary_large_image")
                # Crawlers need an absolute URL, and it must name the file the build copied.
                self.assertEqual(tags["og:image"], SITE_URL + card.name)
                self.assertEqual(tags["twitter:image"], tags["og:image"])
                self.assertEqual((int(tags["og:image:width"]), int(tags["og:image:height"])), build_module.png_size(card))
                self.assertLessEqual(len(tags["twitter:title"]), 70)
                self.assertLessEqual(len(tags["twitter:description"]), 200)
                self.assertLessEqual(len(tags["og:image:alt"]), 420)
                for key, value in pairs:
                    self.assertTrue(value.strip(), key)
                    self.assertIsNone(CJK.search(value), f"{key} should be English")

    def test_repository_social_preview_is_ready_to_upload(self):
        preview = build_module.ROOT / ".github" / "assets" / "social-preview.png"
        self.assertEqual(build_module.png_size(preview), (1280, 640))
        self.assertLess(preview.stat().st_size, 1_000_000)  # GitHub refuses social previews of 1 MB or more

    def test_build_rejects_stage_f_numbers_from_another_analysis(self):
        payload = self.data["sandbox"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("showcase/data/sandbox_firms.json", "showcase/data/simplification.json",
                             payload["firms"]["experiment_spec"], payload["firms"]["domains"]["ai_washing"]["domain_file"]):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((build_module.ROOT / relative).read_bytes())
            stage_f = json.loads((build_module.ROOT / "showcase/data/stage_f.json").read_text(encoding="utf-8"))
            (root / "showcase/data/stage_f.json").write_text(json.dumps(stage_f), encoding="utf-8")
            analysis = root / stage_f["source"]["path"]
            analysis.parent.mkdir(parents=True, exist_ok=True)
            analysis.write_text(json.dumps({"experiment_id": stage_f["source"]["experiment_id"], "edited": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer match"):
                build_module.sandbox_payload(root, "test")


if __name__ == "__main__":
    unittest.main()
