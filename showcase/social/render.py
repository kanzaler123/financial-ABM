"""Render the link preview card (card.html) to PNG with a headless Chromium browser.

  showcase/social/social-card.png    1200×630  og:image / twitter:image of both pages; build.py copies it into _site
  .github/assets/social-preview.png  1280×640  GitHub repository social preview, uploaded by hand
                                               (Settings → General → Social preview)

Run after editing card.html, then commit both PNGs: python showcase/social/render.py [--browser PATH]
The build never starts a browser. The committed images were rendered by Chrome on Windows
(Bahnschrift, Segoe UI, Cascadia Mono); elsewhere the font stacks fall back and the text reflows.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build import ROOT, SOCIAL_CARD_SIZE, png_size  # noqa: E402

TARGETS = (
    (HERE / "social-card.png", SOCIAL_CARD_SIZE),
    (ROOT / ".github" / "assets" / "social-preview.png", (1280, 640)),
)
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge")
INSTALLS = (
    "{ProgramFiles}/Google/Chrome/Application/chrome.exe",
    "{ProgramFiles(x86)}/Google/Chrome/Application/chrome.exe",
    "{LOCALAPPDATA}/Google/Chrome/Application/chrome.exe",
    "{ProgramFiles(x86)}/Microsoft/Edge/Application/msedge.exe",
    "{ProgramFiles}/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def find_browser() -> str:
    for name in BROWSERS:
        if found := shutil.which(name):
            return found
    for pattern in INSTALLS:
        try:
            path = Path(pattern.format_map(os.environ))
        except KeyError:
            continue
        if path.is_file():
            return str(path)
    raise SystemExit("No Chrome, Chromium or Edge found; pass --browser PATH")


def render(browser: str, source: Path, target: Path, size: tuple[int, int]) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        shot = Path(scratch) / "card.png"
        # A throwaway profile keeps the user's running browser out of it.
        command = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                   "--no-first-run", "--no-default-browser-check", f"--user-data-dir={Path(scratch) / 'profile'}",
                   f"--window-size={size[0]},{size[1]}", f"--screenshot={shot}", source.as_uri()]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0 or not shot.is_file():
            raise SystemExit(f"Browser failed ({result.returncode}):\n{result.stderr[-2000:]}")
        if png_size(shot) != size:
            raise SystemExit(f"Expected {size[0]}x{size[1]}, browser wrote {png_size(shot)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shot, target)
    print(f"{target.relative_to(ROOT).as_posix()}: {size[0]}x{size[1]}, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--browser", help="Chrome, Chromium or Edge executable (default: search PATH and usual installs)")
    args = parser.parse_args()
    browser = args.browser or find_browser()
    for target, size in TARGETS:
        render(browser, HERE / "card.html", target, size)
