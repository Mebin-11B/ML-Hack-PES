"""M4 - Render Folium HTML maps to PNG for the PDF report (headless Chromium via Playwright).

    pip install playwright && python -m playwright install chromium-headless-shell
    python report/screenshot_maps.py results/maps/eda_00000046.html report/figures/map_eda.png \
        results/maps/hmm_helps_00000077.html report/figures/map_hmm_helps.png \
        results/maps/hmm_fails_00000007.html report/figures/map_hmm_fails.png
Needs internet (OpenStreetMap tiles). Arguments are (html, png) pairs.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(args: list[str]) -> None:
    pairs = list(zip(args[0::2], args[1::2]))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        for src, out in pairs:
            page.goto(Path(src).resolve().as_uri(), wait_until="networkidle", timeout=90_000)
            page.wait_for_timeout(2500)  # let tiles finish painting
            page.screenshot(path=out)
            print("wrote", out)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
