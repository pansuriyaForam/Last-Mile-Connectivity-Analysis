# ── stdlib ────────────────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import cKDTree


logging.basicConfig(
    level=logging.WARNING,          # keep Streamlit console clean
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

# ── Config ────────────────────────────────────────────────────────────────────
# (verbatim from corrected notebook Cell 3)

@dataclass
class Config:
    hmrl_dir:   Path = field(default_factory=lambda: Path("Data/hmrl"))
    tgsrtc_dir: Path = field(default_factory=lambda: Path("Data/tgsrtc"))
    visuals_dir: Path = field(default_factory=lambda: Path("visuals"))

    utm_crs:         int   = 32644
    wgs84_crs:       int   = 4326
    walk_radius_m:   float = 800.0
    feeder_radius_m: float = 3000.0
    detour_factor:   float = 1.35

    bbox_lat_min: float = 17.20
    bbox_lat_max: float = 17.65
    bbox_lon_min: float = 78.20
    bbox_lon_max: float = 78.75

    peak_start: str = "07:00:00"
    peak_end:   str = "10:00:00"
    peak_hours: float = 3.0

    w_density:   float = 0.50
    w_frequency: float = 0.40
    w_walkzone:  float = 0.10

    mclp_budget:            int   = 10
    mclp_candidate_grid_m:  float = 500.0
    mclp_desert_grid_m:     float = 250.0
    mclp_coverage_radius_m: float = 800.0

    equity_desert_threshold:  float = 4.0
    equity_desert_budget_pct: float = 0.5
    equity_lmci_weight_eps:   float = 0.5

    export_dpi: int = 300

    red_line_stop_ids: List[str] = field(default_factory=lambda: [
        "MYP", "JNT", "KPH", "KUK", "BLR", "MSP", "BTN", "ERA", "ESI", "SRN",
        "AME", "PUN", "IRM", "KHA", "LKP", "ASM", "NAM", "GAB", "OMC", "MGB",
        "MKL", "NEM", "MSB", "DSN", "CHP", "VOM", "LBN"
    ])

    red_line_name_map: dict = field(default_factory=lambda: {
        "Balanagar": "Balanagar (Dr.B.R. Ambedkar Balanagar)",
        "S. R. Nagar": "S.R. Nagar",
        "Panjagutta": "Punjagutta",
        "Erra Manzil": "Irrum Manzil",
        "Mahatma Gandhi Bus Station": "MG Bus Station",
        "Dilsukh Nagar": "Dilsukhnagar",
        "L. B. Nagar": "LB Nagar",
    })

    def __post_init__(self) -> None:
        self.visuals_dir.mkdir(parents=True, exist_ok=True)
        assert abs(self.w_density + self.w_frequency + self.w_walkzone - 1.0) < 1e-9


CFG = Config()

# ── GTFSLoader ────────────────────────────────────────────────────────────────
# (verbatim from corrected notebook Cell 4)

class GTFSLoader:
    HMRL_FILES   = ["stops.txt"]
    TGSRTC_FILES = ["stops.txt", "stop_times.txt", "trips.txt", "routes.txt"]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._log = logging.getLogger("GTFSLoader")

    def _validate_files(self, directory: Path, required: List[str]) -> None:
        missing = [f for f in required if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing GTFS files in '{directory}': {missing}\n"
                f"Download from: https://data.telangana.gov.in"
            )

    def _bbox_filter(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        c = self.cfg
        mask = (
            df["stop_lat"].between(c.bbox_lat_min, c.bbox_lat_max) &
            df["stop_lon"].between(c.bbox_lon_min, c.bbox_lon_max)
        )
        return df[mask].reset_index(drop=True)

    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False, **kwargs)

    def load_metro_stops(self) -> pd.DataFrame:
        self._validate_files(self.cfg.hmrl_dir, self.HMRL_FILES)
        df = self._load_csv(self.cfg.hmrl_dir / "stops.txt")
        df.columns = df.columns.str.strip().str.lower()
        df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
        df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
        df["location_type"] = pd.to_numeric(df["location_type"], errors="coerce")
        df = df.dropna(subset=["stop_lat", "stop_lon"])
        df = self._bbox_filter(df, "HMRL stops")

        df = df[df["location_type"] == 1].copy()
        _access_pattern = r"Arm|Lift|Escalator|Staircase|Combined Staircase"
        df = df[~df["stop_name"].str.contains(_access_pattern, case=False, na=False)].copy()

        red_line = df[df["stop_id"].isin(self.cfg.red_line_stop_ids)].copy()
        red_line["stop_name"] = red_line["stop_name"].replace(self.cfg.red_line_name_map)
        return red_line[["stop_id", "stop_name", "stop_lat", "stop_lon"]].reset_index(drop=True)

    def load_bus_stops(self) -> pd.DataFrame:
        self._validate_files(self.cfg.tgsrtc_dir, self.TGSRTC_FILES)
        df = self._load_csv(self.cfg.tgsrtc_dir / "stops.txt")
        df.columns = df.columns.str.strip().str.lower()
        df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
        df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
        df = df.dropna(subset=["stop_lat", "stop_lon"])
        df = self._bbox_filter(df, "TGSRTC stops")
        return df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].reset_index(drop=True)

    def load_stop_times(self) -> pd.DataFrame:
        df = self._load_csv(self.cfg.tgsrtc_dir / "stop_times.txt")
        df.columns = df.columns.str.strip().str.lower()
        return df


# ── Fallback data (from notebook Cell 5) ─────────────────────────────────────
# Paste _make_fallback_metro() and _make_fallback_bus() here verbatim.
# They are used when live GTFS files are absent so the app still runs.
# Below is the canonical fallback from the notebook:

def _make_fallback_metro() -> pd.DataFrame:
    STATIONS = [
        {"stop_id": "MYP", "stop_name": "Miyapur",                              "stop_lat": 17.4950, "stop_lon": 78.3490},
        {"stop_id": "JNT", "stop_name": "JNTU College",                         "stop_lat": 17.4930, "stop_lon": 78.3600},
        {"stop_id": "KPH", "stop_name": "KPHB Colony",                          "stop_lat": 17.4910, "stop_lon": 78.3730},
        {"stop_id": "KUK", "stop_name": "Kukatpally",                           "stop_lat": 17.4870, "stop_lon": 78.3910},
        {"stop_id": "BLR", "stop_name": "Balanagar (Dr.B.R. Ambedkar Balanagar)","stop_lat": 17.4780, "stop_lon": 78.4030},
        {"stop_id": "MSP", "stop_name": "Moosapet",                             "stop_lat": 17.4710, "stop_lon": 78.4160},
        {"stop_id": "BTN", "stop_name": "Bharat Nagar",                         "stop_lat": 17.4660, "stop_lon": 78.4280},
        {"stop_id": "ERA", "stop_name": "Erragadda",                            "stop_lat": 17.4610, "stop_lon": 78.4360},
        {"stop_id": "ESI", "stop_name": "ESI Hospital",                         "stop_lat": 17.4560, "stop_lon": 78.4430},
        {"stop_id": "SRN", "stop_name": "S.R. Nagar",                           "stop_lat": 17.4510, "stop_lon": 78.4500},
        {"stop_id": "AME", "stop_name": "Ameerpet",                             "stop_lat": 17.4378, "stop_lon": 78.4480},
        {"stop_id": "PUN", "stop_name": "Punjagutta",                           "stop_lat": 17.4284, "stop_lon": 78.4484},
        {"stop_id": "IRM", "stop_name": "Irrum Manzil",                         "stop_lat": 17.4210, "stop_lon": 78.4490},
        {"stop_id": "KHA", "stop_name": "Khairatabad",                          "stop_lat": 17.4155, "stop_lon": 78.4500},
        {"stop_id": "LKP", "stop_name": "Lakdi-ka-pul",                         "stop_lat": 17.4080, "stop_lon": 78.4530},
        {"stop_id": "ASM", "stop_name": "Assembly",                             "stop_lat": 17.4020, "stop_lon": 78.4560},
        {"stop_id": "NAM", "stop_name": "Nampally",                             "stop_lat": 17.3950, "stop_lon": 78.4630},
        {"stop_id": "GAB", "stop_name": "Gandhi Bhavan",                        "stop_lat": 17.3900, "stop_lon": 78.4680},
        {"stop_id": "OMC", "stop_name": "Osmania Medical College",              "stop_lat": 17.3840, "stop_lon": 78.4740},
        {"stop_id": "MGB", "stop_name": "MG Bus Station",                       "stop_lat": 17.3790, "stop_lon": 78.4800},
        {"stop_id": "MKL", "stop_name": "Malakpet",                             "stop_lat": 17.3740, "stop_lon": 78.4870},
        {"stop_id": "NEM", "stop_name": "New Market",                           "stop_lat": 17.3680, "stop_lon": 78.4920},
        {"stop_id": "MSB", "stop_name": "Musarambagh",                          "stop_lat": 17.3620, "stop_lon": 78.4960},
        {"stop_id": "DSN", "stop_name": "Dilsukhnagar",                         "stop_lat": 17.3680, "stop_lon": 78.5260},
        {"stop_id": "CHP", "stop_name": "Chaitanyapuri",                        "stop_lat": 17.3630, "stop_lon": 78.5360},
        {"stop_id": "VOM", "stop_name": "Victoria Memorial",                    "stop_lat": 17.3580, "stop_lon": 78.5450},
        {"stop_id": "LBN", "stop_name": "LB Nagar",                             "stop_lat": 17.3490, "stop_lon": 78.5520},
    ]
    return pd.DataFrame(STATIONS)


def _make_fallback_bus(rng: np.random.Generator) -> tuple:
    """Generate calibrated synthetic TGSRTC data when live GTFS is absent."""
    # ~3 000 bus stops distributed across the Red Line corridor
    n = 3000
    lats = rng.uniform(17.34, 17.50, n)
    lons = rng.uniform(78.34, 78.56, n)
    stop_ids = [f"BUS{i:05d}" for i in range(n)]
    bus_df = pd.DataFrame({
        "stop_id": stop_ids,
        "stop_name": [f"Bus Stop {i}" for i in range(n)],
        "stop_lat": lats,
        "stop_lon": lons,
    })
    # Synthetic stop_times: 20 trips per stop during 07–10
    trip_ids  = [f"T{i:04d}" for i in range(200)]
    stop_time_rows = []
    for sid in stop_ids[:500]:          # keep synthetic data small
        for t in rng.choice(trip_ids, size=5, replace=False):
            hr  = rng.integers(7, 10)
            mn  = rng.integers(0, 60)
            stop_time_rows.append({
                "trip_id": t,
                "stop_id": sid,
                "departure_time": f"{hr:02d}:{mn:02d}:00",
                "arrival_time":   f"{hr:02d}:{mn:02d}:00",
            })
    stop_times = pd.DataFrame(stop_time_rows)
    return bus_df, stop_times


# ── TransitOptimizer ──────────────────────────────────────────────────────────

class TransitOptimizer:
    def __init__(self, cfg: Config) -> None:
        self.cfg  = cfg
        self._log = logging.getLogger("TransitOptimizer")
        self.gdf_metro:       Optional[gpd.GeoDataFrame] = None
        self.gdf_bus:         Optional[gpd.GeoDataFrame] = None
        self.gdf_lmci:        Optional[gpd.GeoDataFrame] = None
        self.mclp_report:     Optional[pd.DataFrame]     = None
        self._stop_times_raw: Optional[pd.DataFrame]     = None
        self._fallback_mode:  bool = False
        self._selected_mclp_xy: list = []

    @staticmethod
    def _to_gdf(df, lat_col="stop_lat", lon_col="stop_lon", crs_out=32644):
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )
        return gdf.to_crs(epsg=crs_out)

    @staticmethod
    def _minmax_norm(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else pd.Series(
            np.zeros(len(series)), index=series.index
        )

    def build(self, metro_df, bus_df, stop_times, fallback=False):
        self._fallback_mode  = fallback
        self._stop_times_raw = stop_times.copy()
        self.gdf_metro = self._to_gdf(metro_df, crs_out=self.cfg.utm_crs)
        self.gdf_bus   = self._to_gdf(bus_df,   crs_out=self.cfg.utm_crs)

        metro_xy = np.column_stack([self.gdf_metro.geometry.x, self.gdf_metro.geometry.y])
        bus_xy   = np.column_stack([self.gdf_bus.geometry.x,   self.gdf_bus.geometry.y])

        tree = cKDTree(metro_xy)
        dist_euc, idx = tree.query(bus_xy, k=1, workers=-1)
        dist_net = dist_euc * self.cfg.detour_factor

        self.gdf_bus = self.gdf_bus.copy()
        self.gdf_bus["nearest_metro_idx"] = idx
        self.gdf_bus["nearest_metro_id"]  = self.gdf_metro.iloc[idx]["stop_id"].values
        self.gdf_bus["dist_to_metro_m"]   = dist_net
        self.gdf_bus["in_walk_zone"]       = dist_net <= self.cfg.walk_radius_m
        self.gdf_bus["in_feeder_zone"]     = dist_net <= self.cfg.feeder_radius_m
        return self

    def compute_peak_frequency(self):
        st = self._stop_times_raw.copy()
        if "departure_time" not in st.columns and "arrival_time" in st.columns:
            st["departure_time"] = st["arrival_time"]

        def _parse(t):
            try:
                p = str(t).split(":")
                return int(p[0]) + int(p[1]) / 60.0 + int(p[2]) / 3600.0
            except Exception:
                return None

        st["dep_hr"] = st["departure_time"].apply(_parse)
        st = st.dropna(subset=["dep_hr"])
        peak = st[st["dep_hr"].between(7.0, 10.0)]

        freq = (
            peak.groupby("stop_id")["trip_id"]
            .nunique()
            .div(self.cfg.peak_hours)
            .rename("peak_freq")
            .reset_index()
        )
        self.gdf_bus = self.gdf_bus.merge(freq, on="stop_id", how="left")
        self.gdf_bus["peak_freq"] = self.gdf_bus["peak_freq"].fillna(0.0)
        return self

    def compute_lmci(self):
        feeder = self.gdf_bus[self.gdf_bus["in_feeder_zone"]].copy()

        metrics = feeder.groupby("nearest_metro_id").agg(
            stop_count_3km  = ("stop_id",        "count"),
            avg_peak_freq   = ("peak_freq",       "mean"),
            stop_count_800m = ("in_walk_zone",    "sum"),
            avg_dist_m      = ("dist_to_metro_m", "mean"),
        ).reset_index().rename(columns={"nearest_metro_id": "stop_id"})

        gdf = self.gdf_metro.merge(metrics, on="stop_id", how="left").fillna(0)

        gdf["norm_density"]   = self._minmax_norm(gdf["stop_count_3km"].astype(float))
        gdf["norm_frequency"] = self._minmax_norm(gdf["avg_peak_freq"].astype(float))
        gdf["norm_walkzone"]  = self._minmax_norm(gdf["stop_count_800m"].astype(float))

        cfg = self.cfg
        gdf["LMCI"] = 10.0 * (
            cfg.w_density   * gdf["norm_density"]   +
            cfg.w_frequency * gdf["norm_frequency"]  +
            cfg.w_walkzone  * gdf["norm_walkzone"]
        )

        def _cat(v):
            if v >= 7:  return "Well-Connected"
            if v >= 4:  return "Moderate"
            return "Transit Desert"

        gdf["category"] = gdf["LMCI"].apply(_cat)
        self.gdf_lmci = gdf
        return self

    def run_greedy_mclp(self):
        """Equity-weighted MCLP — paste full method from notebook Cell 6."""
        # ── Minimal version (produces LMCI_after / LMCI_improvement) ─────
        feeder = self.gdf_bus[self.gdf_bus["in_feeder_zone"]].copy()
        demand_xy = np.column_stack([feeder.geometry.x, feeder.geometry.y])

        # Equity weight per demand point: inversely proportional to nearest
        # station's LMCI (poorer connectivity = higher weight)
        eps = self.cfg.equity_lmci_weight_eps
        lmci_map = self.gdf_lmci.set_index("stop_id")["LMCI"].to_dict()
        feeder["lmci_nearest"] = feeder["nearest_metro_id"].map(lmci_map).fillna(5.0)
        feeder["eq_weight"]    = 1.0 / (feeder["lmci_nearest"] + eps)

        # Build candidate grid
        x_min, y_min = demand_xy.min(axis=0)
        x_max, y_max = demand_xy.max(axis=0)
        s = self.cfg.mclp_candidate_grid_m
        xs = np.arange(x_min, x_max + s, s)
        ys = np.arange(y_min, y_max + s, s)
        gx, gy = np.meshgrid(xs, ys)
        candidates = np.column_stack([gx.ravel(), gy.ravel()])

        demand_tree = cKDTree(demand_xy)
        cov_r = self.cfg.mclp_coverage_radius_m
        cand_coverage = demand_tree.query_ball_point(candidates, r=cov_r)

        total_weight  = feeder["eq_weight"].sum()
        eq_weights    = feeder["eq_weight"].values
        covered       = np.zeros(len(feeder), dtype=bool)
        selected_xy   = []
        report_rows   = []

        desert_ids = set(
            self.gdf_lmci.loc[
                self.gdf_lmci["LMCI"] < self.cfg.equity_desert_threshold, "stop_id"
            ]
        )
        desert_budget = max(1, int(self.cfg.mclp_budget * self.cfg.equity_desert_budget_pct))

        for step in range(1, self.cfg.mclp_budget + 1):
            desert_remaining = max(0, desert_budget - len(selected_xy))
            best_gain, best_idx = -1.0, -1
            for j, pts in enumerate(cand_coverage):
                gain = sum(eq_weights[p] for p in pts if not covered[p])
                if gain > best_gain:
                    best_gain, best_idx = gain, j
            if best_gain <= 0:
                break

            # Mark covered
            for p in cand_coverage[best_idx]:
                covered[p] = True

            xy = candidates[best_idx]
            selected_xy.append(xy)

            pct_cov = covered.sum() / len(feeder) * 100
            pct_wcov = (eq_weights[covered].sum() / total_weight * 100)
            report_rows.append({
                "step":              step,
                "candidate_lat":     0.0,   # placeholder; compute if needed
                "candidate_lon":     0.0,
                "weighted_gain":     best_gain,
                "pct_weighted_cov":  pct_wcov,
                "pct_improvement":   pct_wcov - (report_rows[-1]["pct_weighted_cov"] if report_rows else 0),
            })

        self.mclp_report     = pd.DataFrame(report_rows)
        self._selected_mclp_xy = selected_xy

        # ── Recompute LMCI_after ──────────────────────────────────────────
        self.gdf_bus = self.gdf_bus.copy()
        self.gdf_bus["covered_by_mclp"] = False
        self.gdf_bus.loc[feeder.index, "covered_by_mclp"] = covered

        feeder_after = self.gdf_bus[self.gdf_bus["in_feeder_zone"]].copy()
        metrics_after = feeder_after.groupby("nearest_metro_id").agg(
            stop_count_3km_after  = ("stop_id",           "count"),
            avg_peak_freq_after   = ("peak_freq",          "mean"),
            stop_count_800m_after = ("in_walk_zone",       "sum"),
        ).reset_index().rename(columns={"nearest_metro_id": "stop_id"})

        # Bonus: MCLP-covered demand points virtually added to walk zone
        mclp_bonus = (
            feeder_after[feeder_after["covered_by_mclp"]]
            .groupby("nearest_metro_id")
            .size()
            .rename("mclp_bonus")
            .reset_index()
            .rename(columns={"nearest_metro_id": "stop_id"})
        )
        metrics_after = metrics_after.merge(mclp_bonus, on="stop_id", how="left")
        metrics_after["mclp_bonus"] = metrics_after["mclp_bonus"].fillna(0)
        metrics_after["stop_count_800m_after"] += metrics_after["mclp_bonus"] * 0.5

        gdf = self.gdf_lmci.merge(metrics_after, on="stop_id", how="left").fillna(0)

        for col in ["stop_count_3km_after", "avg_peak_freq_after", "stop_count_800m_after"]:
            if col not in gdf.columns:
                gdf[col] = gdf[col.replace("_after", "")]

        gdf["norm_density_after"]   = self._minmax_norm(gdf["stop_count_3km_after"].astype(float))
        gdf["norm_frequency_after"] = self._minmax_norm(gdf["avg_peak_freq_after"].astype(float))
        gdf["norm_walkzone_after"]  = self._minmax_norm(gdf["stop_count_800m_after"].astype(float))

        cfg = self.cfg
        gdf["LMCI_after"] = 10.0 * (
            cfg.w_density   * gdf["norm_density_after"]   +
            cfg.w_frequency * gdf["norm_frequency_after"]  +
            cfg.w_walkzone  * gdf["norm_walkzone_after"]
        )
        gdf["LMCI_improvement"] = gdf["LMCI_after"] - gdf["LMCI"]
        self.gdf_lmci = gdf
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINE — cached so Streamlit only runs it once per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Running pipeline…")
def run_pipeline() -> tuple[pd.DataFrame, bool]:
    """
    Load data → build → LMCI → MCLP.
    Returns (gdf_lmci as plain DataFrame, fallback_mode flag).
    """
    loader   = GTFSLoader(CFG)
    rng      = np.random.default_rng(42)
    fallback = False

    try:
        metro_df = loader.load_metro_stops()
    except FileNotFoundError:
        metro_df = _make_fallback_metro()
        fallback = True

    try:
        bus_df     = loader.load_bus_stops()
        stop_times = loader.load_stop_times()
    except FileNotFoundError:
        bus_df, stop_times = _make_fallback_bus(rng)
        fallback = True

    opt = (
        TransitOptimizer(CFG)
        .build(metro_df, bus_df, stop_times, fallback=fallback)
        .compute_peak_frequency()
        .compute_lmci()
        .run_greedy_mclp()
    )

    # Return plain DataFrame (GeoDataFrame not pickle-safe across Streamlit reruns)
    df = pd.DataFrame(opt.gdf_lmci.drop(columns=["geometry"], errors="ignore"))
    return df, fallback


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CAT_COLOURS = {
    "Well-Connected": "#2ecc71",
    "Moderate":       "#f39c12",
    "Transit Desert": "#e74c3c",
}

def category_badge(cat: str) -> str:
    colour = CAT_COLOURS.get(cat, "#888")
    return (
        f'<span style="background:{colour};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-size:0.82rem;font-weight:600">{cat}</span>'
    )

def lmci_gauge(value: float, label: str, colour: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 2),
        number={"suffix": " / 10", "font": {"size": 26, "color": colour}},
        title={"text": label, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1, "tickcolor": "#ccc"},
            "bar":  {"color": colour, "thickness": 0.28},
            "bgcolor": "#f0f2f6",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 4], "color": "#fdecea"},
                {"range": [4, 7], "color": "#fff8e1"},
                {"range": [7, 10],"color": "#e8f8f0"},
            ],
            "threshold": {
                "line": {"color": colour, "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#2c3e50",
    )
    return fig


def before_after_bar(row_a: pd.Series, row_b: Optional[pd.Series] = None):
    """
    Grouped bar chart: before / after LMCI for one or two stations.
    """
    stations = [row_a["stop_name"]]
    befores  = [row_a["LMCI"]]
    afters   = [row_a["LMCI_after"]]

    if row_b is not None:
        stations.append(row_b["stop_name"])
        befores.append(row_b["LMCI"])
        afters.append(row_b["LMCI_after"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before MCLP",
        x=stations,
        y=befores,
        marker_color="#e74c3c",
        marker_line_color="#c0392b",
        marker_line_width=1.2,
        text=[f"{v:.2f}" for v in befores],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="After MCLP",
        x=stations,
        y=afters,
        marker_color="#2ecc71",
        marker_line_color="#27ae60",
        marker_line_width=1.2,
        text=[f"{v:.2f}" for v in afters],
        textposition="outside",
    ))

    # Threshold lines
    fig.add_hline(y=4, line_dash="dot", line_color="#e74c3c",
                  annotation_text="Transit Desert threshold (4)",
                  annotation_position="top left", annotation_font_size=10)
    fig.add_hline(y=7, line_dash="dot", line_color="#2ecc71",
                  annotation_text="Well-Connected threshold (7)",
                  annotation_position="top left", annotation_font_size=10)

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 11], title="LMCI (0–10)"),
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#2c3e50",
        margin=dict(t=50, b=20, l=10, r=10),
        height=380,
    )
    return fig


def all_stations_chart(df: pd.DataFrame, highlight: list[str]):
    """Horizontal bar of all 27 stations, sorted by LMCI, highlighted."""
    df_s = df[["stop_name", "LMCI", "LMCI_after", "category"]].sort_values("LMCI").reset_index(drop=True)

    colours = [
        "#1a73e8" if name in highlight else CAT_COLOURS.get(cat, "#95a5a6")
        for name, cat in zip(df_s["stop_name"], df_s["category"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_s["stop_name"],
        x=df_s["LMCI"],
        orientation="h",
        name="Before",
        marker_color=colours,
        opacity=0.55,
        text=[f"{v:.2f}" for v in df_s["LMCI"]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=df_s["stop_name"],
        x=df_s["LMCI_after"],
        orientation="h",
        name="After",
        marker_color=colours,
        opacity=1.0,
        text=[f"{v:.2f}" for v in df_s["LMCI_after"]],
        textposition="outside",
    ))
    fig.add_vline(x=4, line_dash="dot", line_color="#e74c3c")
    fig.add_vline(x=7, line_dash="dot", line_color="#2ecc71")

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(range=[0, 12], title="LMCI (0–10)"),
        yaxis_title="",
        height=max(480, len(df_s) * 22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#2c3e50",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(t=30, b=10, l=180, r=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HYD Metro LMCI Explorer",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 18px 22px;
    border-left: 5px solid;
    margin-bottom: 8px;
  }
  .metric-card .value {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
    font-family: 'DM Mono', monospace;
  }
  .metric-card .label {
    font-size: 0.78rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
  }
  .delta-positive { color: #27ae60; }
  .delta-negative { color: #e74c3c; }
  .delta-zero     { color: #95a5a6; }
  .section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6c757d;
    margin: 20px 0 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🚇 Hyderabad Metro Red Line — Last-Mile Connectivity Explorer")
st.caption("LMCI Before & After Equity-Weighted MCLP Optimisation · Miyapur ↔ LB Nagar")

# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Loading pipeline…"):
    lmci_df, is_fallback = run_pipeline()

if is_fallback:
    st.warning(
        "⚠️  Running on **synthetic fallback data** — GTFS files not found at "
        "`Data/hmrl/` and `Data/tgsrtc/`. LMCI values are illustrative only.",
        icon="⚠️",
    )

station_names = sorted(lmci_df["stop_name"].tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR — station selection
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📍 Station Selection")
    st.markdown('<p class="section-header">Primary station</p>', unsafe_allow_html=True)
    primary = st.selectbox("Select a station", station_names, index=station_names.index("Ameerpet"))

    st.markdown('<p class="section-header">Compare with (optional)</p>', unsafe_allow_html=True)
    compare_on = st.checkbox("Enable side-by-side comparison", value=False)
    secondary  = None
    if compare_on:
        secondary = st.selectbox(
            "Select second station",
            [s for s in station_names if s != primary],
            index=0,
        )

    st.divider()
    st.markdown("### ⚙️ LMCI Weights")
    st.caption("Adjust weights to explore sensitivity (must sum to 1.0)")
    w_d = st.slider("Density weight",   0.0, 1.0, CFG.w_density,   0.05)
    w_f = st.slider("Frequency weight", 0.0, 1.0, CFG.w_frequency, 0.05)
    w_w = st.slider("Walk-zone weight", 0.0, 1.0, CFG.w_walkzone,  0.05)
    total_w = round(w_d + w_f + w_w, 4)
    if abs(total_w - 1.0) > 0.001:
        st.error(f"Weights sum to {total_w:.2f} — must equal 1.00")
        weights_ok = False
    else:
        st.success("Weights: ✓")
        weights_ok = True

    st.divider()
    st.markdown("### 📊 Data Mode")
    mode_label = "🔴 Synthetic fallback" if is_fallback else "✅ Live GTFS"
    st.markdown(f"**{mode_label}**")
    st.caption(f"Stations loaded: {len(lmci_df)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. RECALCULATE LMCI on custom weights if changed
# ─────────────────────────────────────────────────────────────────────────────

def recalc_lmci(df: pd.DataFrame, wd: float, wf: float, ww: float) -> pd.DataFrame:
    """Recompute LMCI / LMCI_after from raw normalised components with new weights."""
    df = df.copy()
    if all(c in df.columns for c in ["norm_density", "norm_frequency", "norm_walkzone"]):
        df["LMCI"] = 10.0 * (wd * df["norm_density"] + wf * df["norm_frequency"] + ww * df["norm_walkzone"])
    if all(c in df.columns for c in ["norm_density_after", "norm_frequency_after", "norm_walkzone_after"]):
        df["LMCI_after"] = 10.0 * (
            wd * df["norm_density_after"] + wf * df["norm_frequency_after"] + ww * df["norm_walkzone_after"]
        )
    if "LMCI" in df.columns and "LMCI_after" in df.columns:
        df["LMCI_improvement"] = df["LMCI_after"] - df["LMCI"]

    def _cat(v):
        if v >= 7:  return "Well-Connected"
        if v >= 4:  return "Moderate"
        return "Transit Desert"
    if "LMCI" in df.columns:
        df["category"] = df["LMCI"].apply(_cat)
    return df


display_df = recalc_lmci(lmci_df, w_d, w_f, w_w) if weights_ok else lmci_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

row_a = display_df[display_df["stop_name"] == primary].iloc[0]
row_b = display_df[display_df["stop_name"] == secondary].iloc[0] if secondary else None

# ── 7a. Station detail cards ──────────────────────────────────────────────────

def render_station_panel(row: pd.Series, col):
    lmci_b  = row["LMCI"]
    lmci_a  = row.get("LMCI_after", lmci_b)
    delta   = row.get("LMCI_improvement", 0.0)
    cat     = row.get("category", "Moderate")
    cat_col = CAT_COLOURS.get(cat, "#888")

    delta_cls   = "delta-positive" if delta > 0.01 else ("delta-negative" if delta < -0.01 else "delta-zero")
    delta_arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "●")

    with col:
        st.markdown(f"### {row['stop_name']}")
        st.markdown(category_badge(cat), unsafe_allow_html=True)
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="metric-card" style="border-color:{cat_col}">'
            f'<div class="label">LMCI Before</div>'
            f'<div class="value" style="color:{cat_col}">{lmci_b:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="metric-card" style="border-color:#2ecc71">'
            f'<div class="label">LMCI After</div>'
            f'<div class="value" style="color:#2ecc71">{lmci_a:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div class="metric-card" style="border-color:#3498db">'
            f'<div class="label">Delta (Δ)</div>'
            f'<div class="value {delta_cls}">{delta_arrow} {delta:+.2f}</div></div>',
            unsafe_allow_html=True,
        )

        g1, g2 = st.columns(2)
        g1.plotly_chart(lmci_gauge(lmci_b, "Before MCLP", cat_col),   use_container_width=True)
        g2.plotly_chart(lmci_gauge(lmci_a, "After MCLP",  "#2ecc71"), use_container_width=True)

        # Raw metrics table
        raw_cols = {
            "Bus stops (3 km)": "stop_count_3km",
            "Avg peak freq":    "avg_peak_freq",
            "Walk-zone stops":  "stop_count_800m",
            "Avg dist (m)":     "avg_dist_m",
        }
        meta = {k: row.get(v, "—") for k, v in raw_cols.items()}
        st.markdown('<p class="section-header">Raw metrics</p>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(meta, index=["Value"]).T.rename(columns={"Value": ""}),
            use_container_width=True,
            hide_index=False,
        )


if compare_on and row_b is not None:
    col_a, col_sep, col_b = st.columns([5, 0.15, 5])
    col_sep.markdown("<div style='border-left:1px solid #dee2e6;height:100%;margin:0 auto'></div>",
                     unsafe_allow_html=True)
    render_station_panel(row_a, col_a)
    render_station_panel(row_b, col_b)
else:
    render_station_panel(row_a, st.container())

st.divider()

# ── 7b. Before / After bar chart ─────────────────────────────────────────────

st.markdown('<p class="section-header">Before vs After LMCI — Bar Chart</p>', unsafe_allow_html=True)
st.plotly_chart(
    before_after_bar(row_a, row_b),
    use_container_width=True,
)

st.divider()

# ── 7c. All-stations overview ─────────────────────────────────────────────────

st.markdown('<p class="section-header">All 27 Stations — LMCI Ranking (Before → After)</p>',
            unsafe_allow_html=True)
st.caption(
    "Faded bar = Before · Solid bar = After · Blue highlight = selected station(s). "
    "Red dotted line = Transit Desert threshold (4) · Green dotted = Well-Connected (7)."
)
highlight = [primary] + ([secondary] if secondary else [])
st.plotly_chart(all_stations_chart(display_df, highlight), use_container_width=True)

st.divider()

# ── 7d. Full data table ───────────────────────────────────────────────────────
with st.expander("📋 Full LMCI data table"):
    show_cols = ["stop_name", "category", "LMCI", "LMCI_after", "LMCI_improvement",
                 "stop_count_3km", "avg_peak_freq", "stop_count_800m"]
    show_cols = [c for c in show_cols if c in display_df.columns]
    fmt = {c: "{:.2f}" for c in ["LMCI", "LMCI_after", "LMCI_improvement", "avg_peak_freq"]}
    st.dataframe(
        display_df[show_cols].sort_values("LMCI", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#adb5bd;font-size:0.78rem;margin-top:30px'>"
    "Hyderabad Metro Red Line · Last-Mile Connectivity Index · "
    "Equity-Weighted MCLP Optimisation</div>",
    unsafe_allow_html=True,
)
