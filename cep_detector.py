#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   CEP DETECTOR  —  Compression → Expansion → Pullback Signal    ║
║   Single-file · Oanda API · Streamlit-ready · GitHub-safe       ║
╠══════════════════════════════════════════════════════════════════╣
║  Structure interne :                                             ║
║   § 1  Imports & Configuration                                   ║
║   § 2  Oanda Data Fetcher        (≡ data_fetcher.py)            ║
║   § 3  Indicators                (≡ indicators.py)              ║
║   § 4  CEP Engine                (≡ cep_engine.py)              ║
║   § 5  Scanner                   (≡ scanner.py)                 ║
║   § 6  Streamlit UI              (≡ main.py)                    ║
╚══════════════════════════════════════════════════════════════════╝

Déploiement Streamlit Cloud :
  → Ajouter dans Settings > Secrets :
      OANDA_ACCESS_TOKEN = "votre-token"
      OANDA_ACCOUNT_ID   = "votre-compte"
      OANDA_ENVIRONMENT  = "practice"   # ou "live"

Usage local :
  streamlit run cep_detector.py
"""

__version__ = "1.0.0"
__author__  = "CEP Detector"

# ═══════════════════════════════════════════════════════════════════
# § 1 — IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import oandapyV20
    import oandapyV20.endpoints.instruments as oanda_instruments
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

# ── Streamlit Secrets → GitHub-safe ─────────────────────────────
def get_config() -> dict:
    """
    Charge les credentials depuis st.secrets (Streamlit Cloud).
    Les tokens ne sont JAMAIS hardcodés ici.
    """
    try:
        return {
            "access_token": st.secrets["OANDA_ACCESS_TOKEN"],
            "account_id":   st.secrets.get("OANDA_ACCOUNT_ID", ""),
            "environment":  st.secrets.get("OANDA_ENVIRONMENT", "practice"),
        }
    except Exception:
        return {"access_token": "", "account_id": "", "environment": "practice"}

# ── Liste d'instruments par défaut (format Oanda) ───────────────
DEFAULT_INSTRUMENTS = [
    # ── Majeurs USD (7) ─────────────────────────────────────────
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "USD_CAD", "AUD_USD", "NZD_USD",
    # ── Croisés EUR (6) ─────────────────────────────────────────
    "EUR_GBP", "EUR_JPY", "EUR_CHF",
    "EUR_CAD", "EUR_AUD", "EUR_NZD",
    # ── Croisés GBP (5) ─────────────────────────────────────────
    "GBP_JPY", "GBP_CHF", "GBP_CAD",
    "GBP_AUD", "GBP_NZD",
    # ── Croisés AUD (4) ─────────────────────────────────────────
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    # ── Croisés NZD (3) ─────────────────────────────────────────
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    # ── Croisés CAD & CHF (3) ───────────────────────────────────
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
    # ── Métaux & Indices (4) ────────────────────────────────────
    "XAU_USD", "NAS100_USD", "SPX500_USD", "US30_USD",
]

# ── Paramètres CEP (calibrés — ajustables via sidebar) ──────────
DEFAULT_CEP_PARAMS = {
    # Seuil de compression : spread < ATR × ratio → marché en range
    "compression_atr_ratio":    0.50,
    # Seuil d'expansion : spread > ATR × ratio → momentum détecté
    "expansion_atr_ratio":      1.00,
    # Nombre minimum de bougies H4 en compression pour être valide
    "compression_min_candles":  5,
    # Nombre minimum de bougies H4 consécutives en expansion
    "expansion_min_candles":    2,
    # La tendance D1 doit être alignée depuis au moins N bougies
    "d1_alignment_min_candles": 5,
    # Score minimum pour émettre un signal (sur 8)
    "min_score":                5,
    # Nombre de bougies à télécharger
    "candles_d1":               120,
    "candles_h4":               200,
}


# ═══════════════════════════════════════════════════════════════════
# § 2 — OANDA DATA FETCHER
# ═══════════════════════════════════════════════════════════════════

def fetch_candles(
    access_token: str,
    instrument:   str,
    granularity:  str,   # "D", "H4", "H1", "M30" …
    count:        int,
    environment:  str = "practice",
) -> pd.DataFrame:
    """
    Télécharge les bougies OHLCV depuis l'API Oanda.

    Returns
    -------
    pd.DataFrame  colonnes : open, high, low, close, volume
                  index    : DatetimeIndex (UTC)
    """
    if not OANDA_AVAILABLE:
        raise ImportError("oandapyV20 n'est pas installé. Voir requirements.txt.")

    client = oandapyV20.API(access_token=access_token, environment=environment)

    params = {
        "count":       count,
        "granularity": granularity,
        "price":       "M",          # mid-price
    }

    r = oanda_instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(r)

    rows = []
    for c in r.response["candles"]:
        if c["complete"]:            # ignorer la bougie en cours (incomplète)
            rows.append({
                "time":   pd.to_datetime(c["time"]),
                "open":   float(c["mid"]["o"]),
                "high":   float(c["mid"]["h"]),
                "low":    float(c["mid"]["l"]),
                "close":  float(c["mid"]["c"]),
                "volume": int(c["volume"]),
            })

    if not rows:
        raise ValueError(f"Aucune donnée retournée pour {instrument} ({granularity})")

    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


def fetch_multi_timeframe(
    access_token: str,
    instrument:   str,
    environment:  str,
    params:       dict,
) -> tuple:
    """Retourne (df_D1, df_H4) avec indicateurs déjà calculés."""
    df_d1 = fetch_candles(access_token, instrument, "D",  params["candles_d1"],  environment)
    df_h4 = fetch_candles(access_token, instrument, "H4", params["candles_h4"], environment)
    return df_d1, df_h4


# ═══════════════════════════════════════════════════════════════════
# § 3 — INDICATORS
# ═══════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """EMA standard (méthode Wilder via ewm)."""
    return series.ewm(span=length, adjust=False).mean()


def calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (14 périodes)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute au DataFrame :
      ema5, ema9, ema13, ema20, ema50
      atr14
      spread      = |ema5 - ema50|  (en valeur brute)
      spread_norm = spread / atr14  (normalisé → comparable entre actifs)
    """
    df = df.copy()
    df["ema5"]        = calc_ema(df["close"], 5)
    df["ema9"]        = calc_ema(df["close"], 9)
    df["ema13"]       = calc_ema(df["close"], 13)
    df["ema20"]       = calc_ema(df["close"], 20)
    df["ema50"]       = calc_ema(df["close"], 50)
    df["atr14"]       = calc_atr(df, 14)
    df["spread"]      = (df["ema5"] - df["ema50"]).abs()
    df["spread_norm"] = df["spread"] / df["atr14"]
    return df.dropna()


# ═══════════════════════════════════════════════════════════════════
# § 4 — CEP ENGINE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CEPResult:
    """
    Résultat complet de l'analyse CEP pour un instrument.
    Le système NE génère JAMAIS d'entrée — il signale une opportunité.
    """
    instrument:           str
    direction:            str    # "BUY" | "SELL" | "NONE" | "ERROR"
    score:                int    # 0–8
    signal:               bool   # True si score >= min_score ET CEP complet

    # ── Flags composants ────────────────────────────────────────
    trend_d1:             bool  = False
    d1_alignment_bars:    int   = 0
    compression_detected: bool  = False
    compression_bars:     int   = 0
    expansion_detected:   bool  = False
    expansion_bars:       int   = 0
    ema_slope_ok:         bool  = False
    spread_high:          bool  = False

    # ── Niveaux clés ────────────────────────────────────────────
    ema20_h4:      float = 0.0
    ema50_h4:      float = 0.0
    current_price: float = 0.0
    spread_norm:   float = 0.0

    score_details: list  = field(default_factory=list)
    error_msg:     str   = ""

    def to_telegram(self) -> str:
        """Format message Telegram (pour usage futur)."""
        if not self.signal:
            return ""
        emoji = "📈" if self.direction == "BUY" else "📉"
        lines = [
            f"🔔 *SETUP CEP — {self.instrument}*",
            f"Direction : {emoji} *{self.direction}*",
            f"Score     : *{self.score}/8*",
            "",
            f"{'✅' if self.trend_d1 else '❌'} Tendance D1 alignée ({self.d1_alignment_bars} bougies)",
            f"{'✅' if self.compression_detected else '❌'} Compression H4 ({self.compression_bars} bougies)",
            f"{'✅' if self.expansion_detected else '❌'} Expansion H4 ({self.expansion_bars} bougies conséc.)",
            f"{'✅' if self.ema_slope_ok else '❌'} Pente EMA20 & EMA50",
            "",
            f"⏳ Attendre pullback vers :",
            f"   EMA20 → `{self.ema20_h4:.5f}`",
            f"   EMA50 → `{self.ema50_h4:.5f}`",
            "",
            f"_Pas d'entrée immédiate — signal de préparation uniquement._",
        ]
        return "\n".join(lines)


# ── Helpers internes ─────────────────────────────────────────────

def _count_d1_alignment(df_d1: pd.DataFrame, direction: str, min_bars: int) -> tuple:
    """
    Compte depuis combien de bougies D1 la tendance est alignée.
    Returns (is_valid: bool, count: int)
    """
    count = 0
    for i in range(len(df_d1) - 1, max(len(df_d1) - 60, -1), -1):
        row = df_d1.iloc[i]
        if direction == "BUY":
            ok = (row.ema5 > row.ema9 > row.ema13 > row.ema20 > row.ema50
                  and row.close > row.ema20)
        else:
            ok = (row.ema5 < row.ema9 < row.ema13 < row.ema20 < row.ema50
                  and row.close < row.ema20)
        if ok:
            count += 1
        else:
            break
    return count >= min_bars, count


def _detect_compression(df: pd.DataFrame, params: dict) -> tuple:
    """
    Détecte une phase de compression.

    Critères combinés (Option A + B du document) :
      1. spread_norm < compression_atr_ratio
      2. spread décroissant progressivement (≥ 60% des pas)
      3. Durée ≥ compression_min_candles

    Returns (detected: bool, bar_count: int)
    """
    threshold = params["compression_atr_ratio"]
    min_bars  = params["compression_min_candles"]

    # Compter les bougies consécutives en dessous du seuil (en remontant)
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if df["spread_norm"].iloc[i] < threshold:
            count += 1
        else:
            break

    if count < min_bars:
        return False, count

    # Vérifier que le spread est bien décroissant (Option B)
    window = df["spread"].iloc[-count:]
    diffs  = window.diff().dropna()
    pct_decreasing = (diffs < 0).sum() / len(diffs) if len(diffs) > 0 else 0

    return pct_decreasing >= 0.55, count   # 55% des pas en baisse = compression réelle


def _detect_expansion(df: pd.DataFrame, params: dict, direction: str) -> tuple:
    """
    Détecte une phase d'expansion APRÈS la compression.

    Critères :
      1. spread_norm > expansion_atr_ratio
      2. spread croissant sur N bougies consécutives
      3. Alignement EMA correspond à la direction

    Returns (detected: bool, bar_count: int)
    """
    threshold = params["expansion_atr_ratio"]
    min_bars  = params["expansion_min_candles"]

    count = 0
    for i in range(len(df) - 1, max(len(df) - 20, -1), -1):
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1] if i > 0 else row

        above_threshold = row.spread_norm > threshold
        growing         = row.spread > prev_row.spread

        if direction == "BUY":
            ema_ok = row.ema5 > row.ema9 > row.ema13 > row.ema20 > row.ema50
        else:
            ema_ok = row.ema5 < row.ema9 < row.ema13 < row.ema20 < row.ema50

        if above_threshold and growing and ema_ok:
            count += 1
        else:
            break

    return count >= min_bars, count


def _calc_score(result: "CEPResult", df_h4: pd.DataFrame, params: dict) -> tuple:
    """
    Calcule le score de qualité 0-8 et retourne (score, details).

    Grille :
      +2  Alignement EMA parfait H4
      +2  Compression nette (durée × qualité)
      +2  Expansion forte (spread_norm > 1.5× seuil)
      +1  Pente EMA20/50 confirmée
      +1  Distance EMA élevée (spread > 1.2× seuil d'expansion)
    """
    score   = 0
    details = []
    last    = df_h4.iloc[-1]

    # ── Alignement EMA H4 (+2) ──────────────────────────────────
    if result.direction == "BUY":
        h4_aligned = last.ema5 > last.ema9 > last.ema13 > last.ema20 > last.ema50
    else:
        h4_aligned = last.ema5 < last.ema9 < last.ema13 < last.ema20 < last.ema50

    if h4_aligned:
        score += 2
        details.append(("Alignement EMA parfait (H4)", 2))

    # ── Compression (+1 ou +2) ──────────────────────────────────
    if result.compression_detected:
        strong_comp = result.compression_bars >= params["compression_min_candles"] * 2
        pts = 2 if strong_comp else 1
        score += pts
        details.append((f"Compression {'nette' if strong_comp else 'détectée'} ({result.compression_bars}b)", pts))

    # ── Expansion (+1 ou +2) ────────────────────────────────────
    if result.expansion_detected:
        strong_exp = last.spread_norm > params["expansion_atr_ratio"] * 1.5
        pts = 2 if strong_exp else 1
        score += pts
        details.append((f"Expansion {'forte >1.5×ATR' if strong_exp else 'en cours'} ({result.expansion_bars}b)", pts))

    # ── Pente EMA20/50 (+1) ─────────────────────────────────────
    if result.direction == "BUY":
        slope_ok = (df_h4["ema20"].iloc[-1] > df_h4["ema20"].iloc[-2] and
                    df_h4["ema50"].iloc[-1] > df_h4["ema50"].iloc[-2])
    else:
        slope_ok = (df_h4["ema20"].iloc[-1] < df_h4["ema20"].iloc[-2] and
                    df_h4["ema50"].iloc[-1] < df_h4["ema50"].iloc[-2])

    result.ema_slope_ok = slope_ok
    if slope_ok:
        score += 1
        details.append(("Pente EMA20/50 confirmée", 1))

    # ── Distance EMA élevée (+1) ────────────────────────────────
    spread_high = last.spread_norm > params["expansion_atr_ratio"] * 1.2
    result.spread_high = spread_high
    if spread_high:
        score += 1
        details.append(("Distance EMA élevée (>1.2×seuil)", 1))

    return score, details


# ── Moteur principal ─────────────────────────────────────────────

def run_cep_engine(
    instrument: str,
    df_d1:      pd.DataFrame,
    df_h4:      pd.DataFrame,
    params:     dict,
) -> CEPResult:
    """
    Pipeline CEP complet pour un instrument.

    Étapes (conformes au document de spécification) :
      1. Filtre directionnel D1
      2. Détection compression H4
      3. Détection expansion H4
      4. Validation momentum
      5. Calcul score 0-8
      6. Émission signal si score ≥ min_score ET CEP complet
    """
    result = CEPResult(instrument=instrument, direction="NONE", score=0, signal=False)

    # ── Données suffisantes ? ────────────────────────────────────
    if len(df_d1) < 60 or len(df_h4) < 60:
        result.error_msg = "Données insuffisantes"
        return result

    last_d1 = df_d1.iloc[-1]
    last_h4 = df_h4.iloc[-1]

    result.ema20_h4      = float(last_h4.ema20)
    result.ema50_h4      = float(last_h4.ema50)
    result.current_price = float(last_h4.close)
    result.spread_norm   = float(last_h4.spread_norm)

    # ── Étape 1 : Filtre directionnel D1 ─────────────────────────
    bull = (last_d1.ema5 > last_d1.ema9 > last_d1.ema13 > last_d1.ema20 > last_d1.ema50
            and df_d1["ema20"].iloc[-1] > df_d1["ema20"].iloc[-2]
            and df_d1["ema50"].iloc[-1] > df_d1["ema50"].iloc[-2]
            and last_d1.close > last_d1.ema20)

    bear = (last_d1.ema5 < last_d1.ema9 < last_d1.ema13 < last_d1.ema20 < last_d1.ema50
            and df_d1["ema20"].iloc[-1] < df_d1["ema20"].iloc[-2]
            and df_d1["ema50"].iloc[-1] < df_d1["ema50"].iloc[-2]
            and last_d1.close < last_d1.ema20)

    if not bull and not bear:
        result.error_msg = "Pas de tendance D1 claire"
        return result   # ← REJET : aucune tendance confirmée

    result.direction = "BUY" if bull else "SELL"

    aligned, align_bars = _count_d1_alignment(
        df_d1, result.direction, params["d1_alignment_min_candles"]
    )
    result.trend_d1          = aligned
    result.d1_alignment_bars = align_bars

    if not aligned:
        result.error_msg = f"Tendance D1 trop récente ({align_bars}b < {params['d1_alignment_min_candles']}b)"
        return result

    # ── Étape 2 : Compression H4 ──────────────────────────────────
    # On exclut les dernières bougies (potentielle expansion en cours)
    offset = max(params["expansion_min_candles"] + 1, 3)
    df_h4_pre_expansion = df_h4.iloc[:-offset]

    comp_ok, comp_bars = _detect_compression(df_h4_pre_expansion, params)
    result.compression_detected = comp_ok
    result.compression_bars     = comp_bars

    # ── Étape 3 : Expansion H4 ────────────────────────────────────
    exp_ok, exp_bars = _detect_expansion(df_h4, params, result.direction)
    result.expansion_detected = exp_ok
    result.expansion_bars     = exp_bars

    # ── Condition CEP complète ────────────────────────────────────
    # Les 3 phases doivent être présentes pour calculer le score
    if not (result.trend_d1 and result.compression_detected and result.expansion_detected):
        phases = []
        if not result.compression_detected:
            phases.append("compression absente")
        if not result.expansion_detected:
            phases.append("expansion absente")
        result.error_msg = ", ".join(phases).capitalize()
        return result

    # ── Étape 4 : Score qualité ───────────────────────────────────
    result.score, result.score_details = _calc_score(result, df_h4, params)
    result.signal = result.score >= params["min_score"]

    return result


# ═══════════════════════════════════════════════════════════════════
# § 5 — SCANNER MULTI-ACTIFS
# ═══════════════════════════════════════════════════════════════════

def run_scanner(
    access_token:       str,
    environment:        str,
    instruments_list:   list,
    params:             dict,
    progress_callback=None,
) -> list:
    """
    Scanne tous les instruments et retourne une liste de CEPResult
    triée par : signal d'abord, puis score décroissant.

    progress_callback(pct: float, msg: str) → appelé à chaque étape
    """
    results = []
    total   = len(instruments_list)

    for i, instrument in enumerate(instruments_list):

        if progress_callback:
            progress_callback(i / total, f"Analyse de {instrument}…")

        try:
            df_d1, df_h4 = fetch_multi_timeframe(access_token, instrument, environment, params)
            df_d1 = add_indicators(df_d1)
            df_h4 = add_indicators(df_h4)
            result = run_cep_engine(instrument, df_d1, df_h4, params)

        except Exception as e:
            result = CEPResult(
                instrument=instrument,
                direction="ERROR",
                score=0,
                signal=False,
                error_msg=str(e)[:80],
            )

        results.append(result)
        time.sleep(0.35)    # respect rate-limit Oanda (~3 req/s)

    if progress_callback:
        progress_callback(1.0, "Scan terminé.")

    # Tri : signaux d'abord, puis score décroissant
    results.sort(key=lambda r: (r.signal, r.score), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# § 6 — STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

# ── Helpers visuels ──────────────────────────────────────────────

def _score_badge(score: int) -> str:
    if score >= 6: return f"🟢 {score}/8"
    if score >= 4: return f"🟡 {score}/8"
    return f"🔴 {score}/8"

def _dir_emoji(direction: str) -> str:
    return {"BUY": "📈", "SELL": "📉"}.get(direction, "⚠️")

def _score_color(score: int) -> str:
    if score >= 6: return "#00C853"
    if score >= 4: return "#FFD600"
    return "#FF5252"


# ── Page principale ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="CEP Detector",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Chargement credentials (secrets uniquement) ────────────
    cfg          = get_config()
    access_token = cfg["access_token"]
    environment  = cfg["environment"]

    # ── En-tête ────────────────────────────────────────────────
    st.title("📊 CEP Detector")
    st.markdown(
        "**Compression → Expansion → Pullback** — Détecteur de setup.  "
        "Aucune entrée automatique. Attendre le pullback sur EMA20/EMA50."
    )
    st.divider()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("📋 Instruments")
        instruments_input = st.text_area(
            "Un par ligne (format Oanda)",
            value="\n".join(DEFAULT_INSTRUMENTS),
            height=200,
            help="Ex: EUR_USD, GBP_JPY, XAU_USD…",
        )
        instruments_list = [
            x.strip().upper()
            for x in instruments_input.strip().split("\n")
            if x.strip()
        ]

        st.divider()
        st.subheader("🎛️ Paramètres CEP")

        min_score = st.slider(
            "Score minimum (sur 8)",
            min_value=3, max_value=8,
            value=DEFAULT_CEP_PARAMS["min_score"],
            help="En dessous : pas de signal émis"
        )
        comp_ratio = st.slider(
            "Seuil compression (× ATR)",
            min_value=0.2, max_value=1.0, step=0.05,
            value=DEFAULT_CEP_PARAMS["compression_atr_ratio"],
            help="spread < ATR × seuil → marché en range"
        )
        exp_ratio = st.slider(
            "Seuil expansion (× ATR)",
            min_value=0.5, max_value=2.5, step=0.1,
            value=DEFAULT_CEP_PARAMS["expansion_atr_ratio"],
            help="spread > ATR × seuil → momentum détecté"
        )
        comp_min = st.slider(
            "Min bougies compression",
            min_value=3, max_value=20,
            value=DEFAULT_CEP_PARAMS["compression_min_candles"],
        )
        d1_min = st.slider(
            "Min bougies tendance D1",
            min_value=3, max_value=20,
            value=DEFAULT_CEP_PARAMS["d1_alignment_min_candles"],
        )

        params = {
            **DEFAULT_CEP_PARAMS,
            "min_score":                min_score,
            "compression_atr_ratio":    comp_ratio,
            "expansion_atr_ratio":      exp_ratio,
            "compression_min_candles":  comp_min,
            "d1_alignment_min_candles": d1_min,
        }

        st.divider()
        scan_btn = st.button(
            "🔍 Lancer le scan",
            type="primary",
            use_container_width=True,
            disabled=not bool(access_token),
        )

        if not access_token:
            st.warning("Credentials manquants. Vérifiez la configuration des secrets.")

    # ── Secrets manquants → instructions déploiement ──────────
    if not access_token:
        st.error("⚠️ Credentials non configurés.")
        with st.expander("📖 Comment configurer les secrets ?"):
            st.code(
                "# .streamlit/secrets.toml\n"
                'OANDA_ACCESS_TOKEN = "votre-token-ici"\n'
                'OANDA_ACCOUNT_ID   = "votre-account-id"\n'
                'OANDA_ENVIRONMENT  = "practice"   # ou "live"',
                language="toml",
            )
            st.markdown(
                "Sur **Streamlit Cloud** → *Settings* → *Secrets* → coller le bloc ci-dessus.  \n"
                "Pour usage **local** : créer le fichier `.streamlit/secrets.toml`."
            )
        return

    # ── Lancement scan ─────────────────────────────────────────
    if scan_btn:
        progress_bar  = st.progress(0, text="Initialisation…")
        status_text   = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(min(pct, 1.0), text=msg)
            status_text.text(msg)

        with st.spinner("Scan en cours…"):
            results = run_scanner(
                access_token, environment, instruments_list, params, update_progress
            )

        progress_bar.empty()
        status_text.empty()

        st.session_state["results"]   = results
        st.session_state["scan_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.session_state["params"]    = params

    # ── Aucun résultat encore ──────────────────────────────────
    if "results" not in st.session_state:
        st.markdown("### ← Cliquez sur **Lancer le scan** pour démarrer.")
        return

    results   = st.session_state["results"]
    scan_time = st.session_state.get("scan_time", "")
    st.caption(f"Dernier scan : {scan_time}")

    # ── Métriques résumé ───────────────────────────────────────
    signals  = [r for r in results if r.signal]
    buy_sig  = [r for r in signals if r.direction == "BUY"]
    sell_sig = [r for r in signals if r.direction == "SELL"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instruments analysés", len(results))
    c2.metric("🔔 Signaux actifs",    len(signals))
    c3.metric("📈 Setups BUY",        len(buy_sig))
    c4.metric("📉 Setups SELL",       len(sell_sig))

    st.divider()

    # ── Cartes de signal ───────────────────────────────────────
    if signals:
        st.subheader(f"🔔 Setups CEP actifs — score ≥ {params['min_score']}/8")
        st.caption("Signal de préparation uniquement. Attendre le pullback avant toute entrée.")

        for r in signals:
            header = (
                f"{_dir_emoji(r.direction)} **{r.instrument}**  "
                f"·  {_score_badge(r.score)}  "
                f"·  {r.direction}"
            )
            with st.expander(header, expanded=True):

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Prix actuel",  f"{r.current_price:.5f}")
                col2.metric("EMA20 H4",     f"{r.ema20_h4:.5f}",
                            delta="← pullback" if r.direction == "BUY" else "← pullback",
                            delta_color="off")
                col3.metric("EMA50 H4",     f"{r.ema50_h4:.5f}")
                col4.metric("Spread/ATR",   f"{r.spread_norm:.2f}×")

                st.markdown("---")
                ca, cb = st.columns(2)

                with ca:
                    st.markdown("**Conditions :**")
                    st.write(f"{'✅' if r.trend_d1 else '❌'}  Tendance D1 ({r.d1_alignment_bars} bougies)")
                    st.write(f"{'✅' if r.compression_detected else '❌'}  Compression H4 ({r.compression_bars} bougies)")
                    st.write(f"{'✅' if r.expansion_detected else '❌'}  Expansion H4 ({r.expansion_bars} bougies)")
                    st.write(f"{'✅' if r.ema_slope_ok else '❌'}  Pente EMA20/50")

                with cb:
                    st.markdown("**Score détaillé :**")
                    for label, pts in r.score_details:
                        st.write(f"  `+{pts}`  {label}")

                st.info(
                    "⏳ **Action** : surveiller un pullback vers l'EMA20 ou l'EMA50.  \n"
                    "Pas d'entrée immédiate — le signal indique que le setup existe."
                )
    else:
        st.info(
            f"Aucun setup CEP avec un score ≥ {params['min_score']}/8 au moment du scan.  \n"
            "Abaissez le score minimum ou attendez un prochain cycle."
        )

    # ── Tableau complet ────────────────────────────────────────
    st.divider()
    with st.expander("📋 Tous les résultats du scan", expanded=False):

        rows = []
        for r in results:
            rows.append({
                "Instrument":    r.instrument,
                "Direction":     r.direction,
                "Score":         r.score,
                "Signal":        "🔔" if r.signal else "—",
                "D1 Tendance":   f"✅ {r.d1_alignment_bars}b" if r.trend_d1  else "❌",
                "Compression":   f"✅ {r.compression_bars}b"  if r.compression_detected else "❌",
                "Expansion":     f"✅ {r.expansion_bars}b"    if r.expansion_detected   else "❌",
                "Pente EMA":     "✅" if r.ema_slope_ok else "❌",
                "EMA20 H4":      f"{r.ema20_h4:.5f}"      if r.ema20_h4      else "—",
                "EMA50 H4":      f"{r.ema50_h4:.5f}"      if r.ema50_h4      else "—",
                "Spread/ATR":    f"{r.spread_norm:.2f}×"  if r.spread_norm   else "—",
                "Info":          r.error_msg or "OK",
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=8, format="%d/8"
                ),
            },
        )


# ── Point d'entrée ───────────────────────────────────────────────
if __name__ == "__main__":
    main()
