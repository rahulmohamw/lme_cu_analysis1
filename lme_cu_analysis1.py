#!/usr/bin/env python3
"""
LME Copper Price Analysis
Runs in GitHub Actions every 6 h, generates docs/analysis.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from io import StringIO
from statistics import mean, stdev
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from scipy import stats

CSV_URL = (
    "https://infilearnai.com/LME_Cu_Dashboard/"
    "lme_copper_historical_data.csv"
)
MAX_RETRIES = 3
BACKOFF = [5, 15, 45]  # seconds
TIMEOUT = 30


class CopperPriceAnalyzer:
    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.analysis_results: dict[str, Any] = {}

    # ─────────────────────────── Networking ──────────────────────────── #

    def fetch_data_from_website(self) -> None:
        """Download CSV with retry + MIME-type guard."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"🔄 Fetching data (attempt {attempt}) …")
                r = requests.get(CSV_URL, timeout=TIMEOUT)
                r.raise_for_status()

                if "text/csv" not in r.headers.get("Content-Type", ""):
                    raise ValueError(
                        f"Unexpected MIME type: {r.headers.get('Content-Type')}"
                    )

                self.df = self._parse_csv(r.text)
                return  # success → exit method

            except (RequestException, ValueError, pd.errors.ParserError) as e:
                print(f"⚠️  Attempt {attempt} failed: {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(BACKOFF[attempt - 1])

    @staticmethod
    def _parse_csv(raw: str) -> pd.DataFrame:
        """Parse CSV, auto-detect columns, basic cleaning."""
        df = pd.read_csv(StringIO(raw))

        date_col = next(
            (c for c in df.columns if "date" in c.lower()), df.columns[0]
        )
        price_col = next(
            (
                c
                for c in df.columns
                if any(k in c.lower() for k in ("price", "cash", "settlement"))
            ),
            None,
        )
        if price_col is None:
            raise ValueError("Could not detect price column")

        df = df.rename(columns={date_col: "Date", price_col: "Price"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = (
            df.dropna(subset=["Date", "Price"])
            .query("Price > 0")
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if len(df) < 2000:
            raise ValueError(f"Dataset too small: {len(df)} rows")

        # enrich calendar fields
        df["Year"] = df["Date"].dt.year
        df["MonthName"] = df["Date"].dt.month_name()
        df["DayName"] = df["Date"].dt.day_name()

        print(
            f"🎯 Clean dataset: {len(df)} rows "
            f"({df['Date'].min().date()} → {df['Date'].max().date()})"
        )
        return df

    # ──────────────────────────── Analytics ──────────────────────────── #

    def run(self) -> None:
        """Fetch data + compute all analyses."""
        self.fetch_data_from_website()
        self.analysis_results = {
            "key_metrics": self._basic_stats(),
            "seasonality": self._seasonality(),
            "trend": self._trend(),
            "monthly_fluctuations": self._monthly_fluct(),
            "weekly_patterns": self._weekly_patterns(),
        }
        # inject summary fields
        km = self.analysis_results["key_metrics"]
        km["trend_direction"] = self.analysis_results["trend"][
            "trend_direction"
        ]
        km["best_month"] = self.analysis_results["seasonality"]["best_month"]
        km["best_day"] = self.analysis_results["seasonality"]["best_day"]

    # -------------------- individual analysis helpers ------------------ #
    def _basic_stats(self) -> dict[str, Any]:
        p = self.df["Price"]
        return {
            "average_price": round(p.mean(), 2),
            "min_price": round(p.min(), 2),
            "max_price": round(p.max(), 2),
            "volatility": round(p.std(ddof=1) / p.mean() * 100, 2),
            "total_records": len(p),
        }

    def _seasonality(self) -> dict[str, Any]:
        mo_mean = self.df.groupby("MonthName")["Price"].mean().round(2)
        dow_mean = self.df.groupby("DayName")["Price"].mean().round(2)
        return {
            "monthly": {"mean": mo_mean.to_dict()},
            "day_of_week": {"mean": dow_mean.to_dict()},
            "best_month": mo_mean.idxmax(),
            "best_day": dow_mean.idxmax(),
        }

    def _trend(self) -> dict[str, Any]:
        x = np.arange(len(self.df))
        slope, _, r, p, _ = stats.linregress(x, self.df["Price"].values)
        direction = "Upward" if p < 0.05 and slope > 0 else "Downward" if p < 0.05 else "Flat"
        return {
            "slope": round(slope, 6),
            "r_squared": round(r**2, 4),
            "p_value": round(p, 4),
            "trend_direction": direction,
        }

    def _monthly_fluctuations(self) -> dict[str, Any]:
        g = (
            self.df.assign(YearMonth=self.df["Date"].dt.to_period("M"))
            .groupby("YearMonth")["Price"]
            .mean()
        )
        pct = g.pct_change().dropna() * 100
        return {
            "mom_changes": pct.round(2).tolist(),
            "mom_dates": pct.index.astype(str).tolist(),
            "base_price": round(g.mean(), 2),
            "volatility": round(pct.std(), 2),
        }

    def _weekly_patterns(self) -> dict[str, Any]:
        m = self.df.groupby("DayName")["Price"].agg(["mean", "count"]).round(2)
        overall = self.df["Price"].mean()
        return {
            "best_day": m["mean"].idxmax(),
            "weekly_performance": {
                d: {
                    "average_price": m.loc[d, "mean"],
                    "vs_monthly_avg": round((m.loc[d, "mean"] - overall) / overall * 100, 2),
                    "is_better_than_monthly": m.loc[d, "mean"] > overall,
                }
                for d in m.index
            },
            "monthly_baseline": round(overall, 2),
        }

    # ──────────────────────────── Output ─────────────────────────────── #

    def save_results(self) -> None:
        """Write docs/analysis.json + helper files atomically."""
        os.makedirs("docs", exist_ok=True)

        payload = {
            "status": "success",
            "message": "Analysis completed successfully",
            "timestamp": datetime.utcnow().isoformat(),
            "data_source": CSV_URL,
            "results": self.analysis_results,
            "raw_data": {
                "dates": self.df["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "prices": self.df["Price"].round(2).tolist(),
            },
            "metadata": {
                "total_records": len(self.df),
                "date_range": {
                    "start": self.df["Date"].min().strftime("%Y-%m-%d"),
                    "end": self.df["Date"].max().strftime("%Y-%m-%d"),
                },
                "analysis_version": "1.1",
            },
        }

        self._atomic_write("docs/analysis.json", json.dumps(payload, indent=2))
        self._atomic_write(
            "docs/last_updated.json",
            json.dumps(
                {
                    "timestamp": payload["timestamp"],
                    "unix_timestamp": int(datetime.utcnow().timestamp()),
                }
            ),
        )
        self._atomic_write(
            "docs/index.html",
            _INDEX_HTML_TEMPLATE.replace("{JSON_URL}", CSV_URL),
        )
        print("💾 docs/analysis.json written atomically")

    @staticmethod
    def _atomic_write(path: str, content: str) -> None:
        """Write via temp + rename to avoid half-written files."""
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)


# ────────────────────────────── HTML ───────────────────────────────── #

_INDEX_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LME Copper Analysis API</title>
<style>
body{font-family:system-ui;padding:2rem 5%;max-width:900px;margin:auto;}
code{background:#f5f5f5;padding:2px 4px;border-radius:3px;}
</style>
</head>
<body>
<h1>📈 LME Copper Price Analysis API</h1>
<p>Automated every 6 h by GitHub Actions.</p>

<h2>Endpoints</h2>
<ul>
<li><code>GET /analysis.json</code> – full analysis</li>
<li><code>GET /last_updated.json</code> – last run timestamp</li>
</ul>

<p>Data source: <a href="{JSON_URL}">{JSON_URL}</a></p>
</body>
</html>"""


# ──────────────────────────── Entry-point ──────────────────────────── #

def main() -> None:
    print("🚀 Starting Copper price analysis")
    analyzer = CopperPriceAnalyzer()
    try:
        analyzer.run()
        analyzer.save_results()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Fatal error: {exc}")
        sys.exit(1)
    print("✅ Completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
