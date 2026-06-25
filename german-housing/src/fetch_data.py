"""Download the Eurostat source files used by the project."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# name -> Eurostat query (everything after /data/)
EUROSTAT = {
    "hpi_official": "prc_hpi_q?geo=DE&unit=I15_Q&purchase=TOTAL&format=JSON",
    "rent_index": "prc_hicp_midx?geo=DE&coicop=CP041&unit=I15&format=JSON",
    "gdp_real": "namq_10_gdp?geo=DE&na_item=B1GQ&unit=CLV15_MEUR&s_adj=SCA&format=JSON",
    "unemployment": "une_rt_m?geo=DE&age=TOTAL&sex=T&unit=PC_ACT&s_adj=SA&format=JSON",
    "hicp_all": "prc_hicp_midx?geo=DE&coicop=CP00&unit=I15&format=JSON",
    "long_rate": "irt_lt_mcby_m?geo=DE&int_rt=MCBY&format=JSON",
    "population": "demo_gind?geo=DE&indic_de=AVG&format=JSON",
}


def _get(url: str, dest: Path, tries: int = 6) -> dict:
    for attempt in range(1, tries + 1):
        subprocess.run(
            [
                "curl",
                "-sS",
                "--max-time",
                "60",
                "-A",
                "Mozilla/5.0 (research)",
                url,
                "-o",
                str(dest),
            ],
            check=True,
        )

        try:
            payload = json.loads(dest.read_text(encoding="utf-8"))
            if payload.get("value"):
                return payload
        except Exception:
            pass

        time.sleep(2 * attempt)

    raise RuntimeError(f"failed to fetch valid data from {url}")


def main():
    for name, query in EUROSTAT.items():
        payload = _get(f"{EUROSTAT_BASE_URL}/{query}", RAW / f"{name}.json")
        time_index = payload["dimension"]["time"]["category"]["index"]
        periods = sorted(time_index.items(), key=lambda item: item[1])
        print(f"Eurostat {name:14s} n={len(periods):4d}  {periods[0][0]}..{periods[-1][0]}")

    print("\nAll raw sources saved to", RAW)


if __name__ == "__main__":
    main()
