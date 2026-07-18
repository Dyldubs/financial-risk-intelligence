"""
Logistic Regression Scorecard with Weight of Evidence (WoE) Binning.

Industry-standard credit scorecard methodology applied to fraud detection:

  1. Bin continuous variables into quantile intervals (fine classing)
  2. Calculate Weight of Evidence (WoE) and Information Value (IV) per bin
  3. Select features by IV strength (>0.02 = usable; focus on >0.1)
  4. Train Logistic Regression on WoE-transformed features
  5. Scale coefficients into additive integer points via the PDO formula

Scorecard interpretation
------------------------
  Each transaction receives an integer score:
    Score = Offset + Σ Points(feature_i, bin_j)
  Higher score → higher fraud risk. Fully additive: analysts can audit
  exactly which bins drive any individual score.

PDO Scaling (fraud-risk framing: higher score = higher risk)
------------------------------------------------------------
  B      = PDO / log(2)
  A      = base_score + B * log(base_odds)     # base_odds = good:bad (50:1)
  Offset = A + B * intercept
  Points_ij = B * β_i * WoE_{i,j}

  Derivation:
    Score = A + B * log_odds_fraud
    At log_odds = log(1/base_odds): Score = A - B*log(base_odds) = base_score
    → A = base_score + B*log(base_odds)
    When fraud odds double: Score increases by B*log(2) = PDO ✓

References
----------
  Siddiqi, N. (2006). Credit Risk Scorecards. Wiley.
  APRA CPG 220 Model Risk Management.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split

from src.config import ROOT_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# ── Scaling parameters ────────────────────────────────────────────────────────
BASE_SCORE = 600     # score at base_odds
BASE_ODDS  = 50.0    # good:bad ratio at base score (50:1 ≈ 2 % fraud rate)
PDO        = 20      # points that increase score when fraud odds double

# ── Binning defaults ──────────────────────────────────────────────────────────
N_BINS   = 8    # quantile bins for continuous features (fine classing)
MIN_BADS = 3    # minimum fraud events per bin before merging

# Features treated as categorical (no quantile split)
CATEGORICAL_FEATURES: set[str] = {
    "merchant_risk_tier",
    "high_risk_country",
    "is_weekend",
    "is_night",
}

# IV predictive power labels — Siddiqi (2006)
_IV_BANDS = [
    (0.5,  "Suspicious (check for leakage)"),
    (0.3,  "Strong"),
    (0.1,  "Medium"),
    (0.02, "Weak"),
    (0.0,  "Useless"),
]

# Save paths
_SCORECARD_DIR = ROOT_DIR / "models"
BINNER_PATH    = _SCORECARD_DIR / "woe_binner.joblib"
LR_PATH        = _SCORECARD_DIR / "lr_scorecard.joblib"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iv_label(iv: float) -> str:
    for threshold, label in _IV_BANDS:
        if iv >= threshold:
            return label
    return "Useless"


def _safe_woe(dist_bad: float, dist_good: float, eps: float = 1e-6) -> float:
    """WoE = ln(dist_bad / dist_good). Clipped to ±4 to avoid overflow."""
    return float(np.clip(np.log((dist_bad + eps) / (dist_good + eps)), -4, 4))


def gini_from_auc(auc: float) -> float:
    """Gini coefficient = 2*AUC - 1."""
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov stat: max|TPR - FPR| across all thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


# ── WoE Binner ────────────────────────────────────────────────────────────────

class WoEBinner:
    """
    sklearn-style transformer: raw feature values → WoE-encoded values.

    Methodology
    -----------
    * Continuous features: quantile binning (pd.qcut).
      Bins with fewer than min_bads fraud events are merged left→right until
      the threshold is met.  The woe_map stores each *original* interval
      string → its merged-bin WoE, so pd.cut at transform time works
      transparently.
    * Categorical/binary features: group by value → WoE per unique value.

    Parameters
    ----------
    n_bins   : Number of quantile bins for continuous features.
    min_bads : Minimum fraud events required per (merged) bin.
    """

    def __init__(self, n_bins: int = N_BINS, min_bads: int = MIN_BADS):
        self.n_bins   = n_bins
        self.min_bads = min_bads
        # Populated by fit()
        self.feature_info_: dict[str, dict] = {}
        self.features_: list[str] = []

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        y = pd.Series(y, name="target", dtype=int).reset_index(drop=True)
        X = X.reset_index(drop=True)

        total_bad  = int(y.sum())
        total_good = int((y == 0).sum())
        if total_bad == 0 or total_good == 0:
            raise ValueError("y must contain both fraud (1) and legitimate (0) samples.")

        self.feature_info_ = {}
        for col in X.columns:
            is_cat = col in CATEGORICAL_FEATURES or X[col].nunique() <= 6
            if is_cat:
                info = self._fit_categorical(X[col], y, total_bad, total_good)
            else:
                info = self._fit_continuous(X[col], y, total_bad, total_good)
            self.feature_info_[col] = info

        self.features_ = list(X.columns)
        return self

    # ── categorical ──────────────────────────────────────────────────────────

    def _fit_categorical(
        self, x: pd.Series, y: pd.Series, total_bad: int, total_good: int
    ) -> dict:
        df = pd.DataFrame({"x": x.astype(str), "y": y})
        counts = (
            df.groupby("x")["y"]
            .agg(n_bad="sum", n_total="count")
            .assign(n_good=lambda d: d["n_total"] - d["n_bad"])
        )
        table = self._compute_woe_iv(counts, total_bad, total_good)
        # woe_map: str(value) → WoE
        woe_map = table.set_index("bin")["woe"].to_dict()
        return {
            "type":    "categorical",
            "cuts":    None,
            "woe_map": woe_map,
            "iv":      float(table["iv_contrib"].sum()),
            "table":   table,
        }

    # ── continuous ───────────────────────────────────────────────────────────

    def _fit_continuous(
        self, x: pd.Series, y: pd.Series, total_bad: int, total_good: int
    ) -> dict:
        # Fine classing: quantile bins
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binned, cuts = pd.qcut(x, q=self.n_bins, retbins=True, duplicates="drop")

        # Build raw counts keyed by interval string
        df = pd.DataFrame({"bin": binned.astype(str), "y": y})
        raw = (
            df.groupby("bin")["y"]
            .agg(n_bad="sum", n_total="count")
            .assign(n_good=lambda d: d["n_total"] - d["n_bad"])
        )

        # Coarse classing: merge bins with too few bads
        # Key contract: woe_map maps *original* interval strings → merged WoE
        woe_map, coarse_rows = self._merge_and_compute_woe(
            raw, total_bad, total_good
        )

        coarse_table = pd.DataFrame(coarse_rows)
        return {
            "type":    "continuous",
            "cuts":    cuts,
            "woe_map": woe_map,
            "iv":      float(coarse_table["iv_contrib"].sum()) if len(coarse_table) else 0.0,
            "table":   coarse_table,
        }

    def _merge_and_compute_woe(
        self,
        raw: pd.DataFrame,      # index = original bin str; cols: n_bad, n_good, n_total
        total_bad:  int,
        total_good: int,
    ) -> tuple[dict, list]:
        """
        Merge fine bins so each coarse bin has ≥ min_bads fraud cases.

        Returns
        -------
        woe_map   : dict  original_bin_str → woe_of_its_merged_bin
        coarse_rows : list of dicts for the scorecard display table
        """
        # Walk through fine bins in order, accumulating until min_bads met
        orig_labels = list(raw.index)           # ordered interval strings
        groups: list[tuple[list, int, int]] = []  # (original_labels, n_bad, n_good)
        pending_labels: list[str] = []
        pending_bad   = 0
        pending_good  = 0

        for lbl in orig_labels:
            row = raw.loc[lbl]
            pending_labels.append(lbl)
            pending_bad  += int(row["n_bad"])
            pending_good += int(row["n_good"])
            if pending_bad >= self.min_bads:
                groups.append((list(pending_labels), pending_bad, pending_good))
                pending_labels, pending_bad, pending_good = [], 0, 0

        # Flush any remaining pending labels into the last group
        if pending_labels and groups:
            last_lbls, last_bad, last_good = groups[-1]
            groups[-1] = (
                last_lbls + pending_labels,
                last_bad  + pending_bad,
                last_good + pending_good,
            )
        elif pending_labels:
            groups = [(pending_labels, pending_bad, pending_good)]

        # Now compute WoE for each merged group and build woe_map
        woe_map: dict[str, float] = {}
        coarse_rows: list[dict]   = []

        for orig_lbls, n_bad, n_good in groups:
            n_total  = n_bad + n_good
            dist_bad  = n_bad  / max(total_bad,  1)
            dist_good = n_good / max(total_good, 1)
            woe       = _safe_woe(dist_bad, dist_good)
            iv_c      = (dist_bad - dist_good) * woe

            # Display label: first bin … last bin
            display = orig_lbls[0] if len(orig_lbls) == 1 else f"{orig_lbls[0]}…{orig_lbls[-1]}"
            coarse_rows.append({
                "bin":        display,
                "n_bad":      n_bad,
                "n_good":     n_good,
                "n_total":    n_total,
                "event_rate": round(n_bad / max(n_total, 1), 4),
                "dist_bad":   round(dist_bad,  4),
                "dist_good":  round(dist_good, 4),
                "woe":        round(woe, 4),
                "iv_contrib": round(iv_c, 4),
            })

            # Map every original label to this WoE
            for orig_lbl in orig_lbls:
                woe_map[orig_lbl] = woe

        return woe_map, coarse_rows

    def _compute_woe_iv(
        self, counts: pd.DataFrame, total_bad: int, total_good: int
    ) -> pd.DataFrame:
        """Compute WoE and IV for a categorical groupby result (index = bin label)."""
        rows = []
        for lbl, row in counts.iterrows():
            n_bad    = int(row["n_bad"])
            n_good   = int(row["n_good"])
            n_total  = int(row["n_total"])
            dist_bad  = n_bad  / max(total_bad,  1)
            dist_good = n_good / max(total_good, 1)
            woe       = _safe_woe(dist_bad, dist_good)
            iv_c      = (dist_bad - dist_good) * woe
            rows.append({
                "bin":        str(lbl),
                "n_bad":      n_bad,
                "n_good":     n_good,
                "n_total":    n_total,
                "event_rate": round(n_bad / max(n_total, 1), 4),
                "dist_bad":   round(dist_bad,  4),
                "dist_good":  round(dist_good, 4),
                "woe":        round(woe, 4),
                "iv_contrib": round(iv_c, 4),
            })
        return pd.DataFrame(rows)

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace each feature value with its fitted WoE. Unknown bins → 0."""
        X   = X.reset_index(drop=True).copy()
        out = pd.DataFrame(index=X.index)
        for col in self.features_:
            if col not in X.columns:
                out[col] = 0.0
                continue
            info = self.feature_info_[col]
            if info["type"] == "categorical":
                mapped = X[col].astype(str).map(info["woe_map"])
            else:
                # pd.cut with stored quantile edges → same interval strings
                binned = pd.cut(X[col], bins=info["cuts"], include_lowest=True)
                mapped = binned.astype(str).map(info["woe_map"])
            out[col] = mapped.fillna(0.0).astype(float)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ── reporting ─────────────────────────────────────────────────────────────

    def get_iv_summary(self) -> pd.DataFrame:
        """One row per feature: IV value and predictive strength label."""
        rows = [
            {
                "feature":  feat,
                "iv":       round(info["iv"], 4),
                "strength": _iv_label(info["iv"]),
            }
            for feat, info in self.feature_info_.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def get_woe_table(self) -> pd.DataFrame:
        """Full WoE table: one row per (feature, bin)."""
        frames = []
        for feat, info in self.feature_info_.items():
            tbl = info["table"].copy()
            tbl.insert(0, "feature", feat)
            tbl["iv_feature"] = round(info["iv"], 4)
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Logistic Regression Scorecard ─────────────────────────────────────────────

class LRScoreCard:
    """
    Logistic Regression on WoE features scaled to additive integer points.

    Scoring formula
    ---------------
      B      = PDO / log(2)
      A      = base_score + B * log(base_odds)
      Score  = A + B * decision_function(X_woe)
             = Offset + Σ Points(feature_i, bin_j)
      Offset = A + B * intercept
      Points_ij = round(B * β_i * WoE_{i,j})

    Parameters
    ----------
    base_score : Score at the base_odds point (default 600)
    base_odds  : Good:bad ratio at base score; 50 → 2 % fraud rate
    pdo        : Points to double the fraud odds (default 20)
    C          : LR regularisation strength (inverse)
    """

    def __init__(
        self,
        base_score: int   = BASE_SCORE,
        base_odds:  float = BASE_ODDS,
        pdo:        int   = PDO,
        C:          float = 1.0,
    ):
        self.base_score = base_score
        self.base_odds  = base_odds
        self.pdo        = pdo
        self.C          = C

        self.lr_:              LogisticRegression | None = None
        self.B_:               float | None = None
        self.A_:               float | None = None
        self.offset_:          float | None = None
        self.feature_coefs_:   dict  | None = None
        self.scorecard_table_: pd.DataFrame | None = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_woe:  pd.DataFrame,
        y:      pd.Series,
        binner: WoEBinner,
    ) -> "LRScoreCard":
        """
        Train LR on WoE features and compute the additive points table.

        Parameters
        ----------
        X_woe   : WoE-transformed feature matrix (from WoEBinner.transform)
        y       : binary fraud labels
        binner  : fitted WoEBinner (needed to map WoE values → points)
        """
        self.lr_ = LogisticRegression(
            C=self.C,
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
        )
        self.lr_.fit(X_woe, y)

        # PDO scaling
        self.B_ = self.pdo / np.log(2)
        # A = base_score + B * log(base_odds)  so that:
        # Score(log_odds = log(1/base_odds)) = A + B*log(1/base_odds)
        #   = base_score + B*log(base_odds) - B*log(base_odds) = base_score ✓
        self.A_ = self.base_score + self.B_ * np.log(self.base_odds)

        intercept     = float(self.lr_.intercept_[0])
        self.offset_  = round(self.A_ + self.B_ * intercept)

        self.feature_coefs_ = dict(zip(X_woe.columns, self.lr_.coef_[0]))
        self.scorecard_table_ = self._build_scorecard_table(binner)
        return self

    def _build_scorecard_table(self, binner: WoEBinner) -> pd.DataFrame:
        rows = []
        for feat, info in binner.feature_info_.items():
            coef = self.feature_coefs_.get(feat, 0.0)
            for _, row in info["table"].iterrows():
                woe    = float(row["woe"])
                points = round(self.B_ * coef * woe)
                rows.append({
                    "feature":    feat,
                    "bin":        str(row["bin"]),
                    "n_total":    int(row["n_total"]),
                    "n_bad":      int(row["n_bad"]),
                    "event_rate": float(row["event_rate"]),
                    "woe":        round(woe, 4),
                    "iv_contrib": float(row["iv_contrib"]),
                    "coef":       round(coef, 4),
                    "points":     int(points),
                })

        offset_row = pd.DataFrame([{
            "feature":    "__OFFSET__",
            "bin":        "(base intercept contribution)",
            "n_total":    None,
            "n_bad":      None,
            "event_rate": None,
            "woe":        None,
            "iv_contrib": None,
            "coef":       None,
            "points":     self.offset_,
        }])
        return pd.concat([offset_row, pd.DataFrame(rows)], ignore_index=True)

    # ── scoring ───────────────────────────────────────────────────────────────

    def predict_score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Return integer fraud risk scores (higher = higher risk)."""
        # decision_function = α + Σ β_i * WoE_i = log(p_fraud / p_legit)
        log_odds = self.lr_.decision_function(X_woe)
        return np.round(self.A_ + self.B_ * log_odds).astype(int)

    def predict_proba(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Return P(fraud) for each row."""
        return self.lr_.predict_proba(X_woe)[:, 1]

    def score_to_tier(self, y_prob: np.ndarray) -> np.ndarray:
        """Map fraud probabilities to APRA-style risk tiers."""
        tiers = np.where(
            y_prob >= 0.85, "CRITICAL",
            np.where(y_prob >= 0.60, "HIGH",
            np.where(y_prob >= 0.30, "MEDIUM", "LOW"))
        )
        return tiers

    def score_transactions(
        self, df: pd.DataFrame, binner: WoEBinner
    ) -> pd.DataFrame:
        """Full pipeline: raw DataFrame → scorecard_score, risk_tier, recommended_action."""
        feats  = [f for f in binner.features_ if f in df.columns]
        X_woe  = binner.transform(df[feats])
        scores = self.predict_score(X_woe)
        probs  = self.predict_proba(X_woe)
        tiers  = self.score_to_tier(probs)

        out = df.copy()
        out["scorecard_score"]  = scores
        out["fraud_probability"] = np.round(probs, 4)
        out["risk_tier"]         = tiers
        out["recommended_action"] = pd.Series(tiers).map({
            "CRITICAL": "Immediately escalate to Financial Crime. Freeze transaction.",
            "HIGH":     "Flag for same-day analyst review. Hold funds pending clearance.",
            "MEDIUM":   "Queue for daily review. Monitor account activity.",
            "LOW":      "Log for audit trail. No immediate action required.",
        }).values
        return out


# ── Training entrypoint ───────────────────────────────────────────────────────

def train_scorecard(
    df:         pd.DataFrame,
    n_bins:     int   = N_BINS,
    min_bads:   int   = MIN_BADS,
    lr_C:       float = 1.0,
    min_iv:     float = 0.02,
    test_size:  float = 0.20,
) -> dict:
    """
    End-to-end scorecard training pipeline.

    Steps
    -----
    1. Engineer is_night feature from hour
    2. Stratified train/test split
    3. Fit WoEBinner on training data (all features)
    4. Select features with IV ≥ min_iv
    5. Re-fit binner on selected features
    6. Transform to WoE space
    7. Fit LRScoreCard
    8. Evaluate: AUC-PR, AUC-ROC, Gini, KS
    9. Log to MLflow and save model files

    Returns
    -------
    dict with keys:
      binner, model, scorecard_table, iv_summary, all_iv_summary,
      metrics, usable_features, X_test, y_test, scores_test, y_prob_test
    """
    from src.data.features import SYNTHETIC_NUMERIC_FEATURES

    if "is_fraud" not in df.columns:
        raise ValueError("DataFrame must have an 'is_fraud' column.")

    # Engineer is_night
    df = df.copy()
    if "hour" in df.columns:
        df["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(int)

    feature_cols = [
        f for f in SYNTHETIC_NUMERIC_FEATURES + ["is_night"] if f in df.columns
    ]
    X = df[feature_cols].copy()
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # ── 1. IV screening pass (all features) ──────────────────────────────────
    binner_full = WoEBinner(n_bins=n_bins, min_bads=min_bads)
    binner_full.fit(X_train, y_train)
    all_iv = binner_full.get_iv_summary()

    usable = all_iv[all_iv["iv"] >= min_iv]["feature"].tolist()
    if not usable:
        raise ValueError(
            f"No features have IV ≥ {min_iv}. Lower min_iv or check data."
        )
    print(f"  Features with IV ≥ {min_iv}: {usable}")

    # ── 2. Re-fit binner on selected features ────────────────────────────────
    binner = WoEBinner(n_bins=n_bins, min_bads=min_bads)
    binner.fit(X_train[usable], y_train)

    X_train_woe = binner.transform(X_train[usable])
    X_test_woe  = binner.transform(X_test[usable])

    # ── 3. Fit scorecard ─────────────────────────────────────────────────────
    card = LRScoreCard(C=lr_C)
    card.fit(X_train_woe, y_train, binner)

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    y_prob_test  = card.predict_proba(X_test_woe)
    scores_test  = card.predict_score(X_test_woe)
    auc_roc      = roc_auc_score(y_test, y_prob_test)
    auc_pr       = average_precision_score(y_test, y_prob_test)
    gini         = gini_from_auc(auc_roc)
    ks           = ks_statistic(y_test.values, y_prob_test)

    metrics = {
        "auc_roc":             round(auc_roc, 4),
        "auc_pr":              round(auc_pr,  4),
        "gini":                round(gini,    4),
        "ks_stat":             round(ks,      4),
        "n_features_selected": len(usable),
    }

    # ── 5. MLflow logging ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="LR_Scorecard"):
        mlflow.log_params({
            "n_bins":     n_bins,
            "min_iv":     min_iv,
            "lr_C":       lr_C,
            "n_features": len(usable),
            "features":   ",".join(usable),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(card.lr_, artifact_path="lr_model")

    # ── 6. Save artefacts ────────────────────────────────────────────────────
    _SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(binner, BINNER_PATH)
    joblib.dump(card,   LR_PATH)
    print(f"  Saved binner → {BINNER_PATH}")
    print(f"  Saved model  → {LR_PATH}")

    return {
        "binner":          binner,
        "model":           card,
        "scorecard_table": card.scorecard_table_,
        "iv_summary":      binner.get_iv_summary(),
        "all_iv_summary":  all_iv,
        "metrics":         metrics,
        "usable_features": usable,
        "X_test":          X_test,
        "y_test":          y_test,
        "scores_test":     scores_test,
        "y_prob_test":     y_prob_test,
    }


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_scorecard() -> tuple[WoEBinner, LRScoreCard]:
    """Load fitted binner and scorecard from disk."""
    if not BINNER_PATH.exists() or not LR_PATH.exists():
        raise FileNotFoundError(
            "Scorecard model files not found.\n"
            "Run: python scripts/train.py --model scorecard"
        )
    return joblib.load(BINNER_PATH), joblib.load(LR_PATH)
