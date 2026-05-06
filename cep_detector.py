#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   CEP DETECTOR v3.1  —  Compression → Expansion → Pullback      ║
║   Alertes PRÉCOCES (Expansion confirmée) · Signaux propres      ║
╠══════════════════════════════════════════════════════════════════╣
║  CHANGEMENTS v3.1 :                                              ║
║   1. Signal Telegram à l'EXPANSION (pas au pullback)            ║
║   2. Message clair : "Prépare-toi au pullback"                  ║
║   3. Suppression 2ème alerte Telegram                           ║
║   4. Option H1 disponible (désactivée par défaut)               ║
║   5. Score minimum relevé pour signaux plus propres             ║
══════════════════════════════════════════════════════════════════╝

Configuration recommandée pour signaux PROPRES :
  → Timeframes : D1 + H4 (H1 désactivé)
  → Score minimum : 6/8 (au lieu de 5/8)
  → Volume ratio : 1.5× (au lieu de 1.2×)
  → Expansion max age : 3 bougies H4 (au lieu de 6)
"""

__version__ = "3.1.0"
__author__   = "CEP Detector"

# ═══════════════════════════════════════════════════════════════════
# § 1 — IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

import time
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

try:
    import oandapyV20
    import oandapyV20.endpoints.instruments as oanda_instruments
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False


# ── Credentials ──────────────────────────────────────────────────
def get_config() -> dict:
    try:
        return {
            "access_token":     st.secrets["OANDA_ACCESS_TOKEN"],
            "account_id":       st.secrets.get("OANDA_ACCOUNT_ID", ""),
            "environment":      st.secrets.get("OANDA_ENVIRONMENT", "practice"),
            "telegram_token":   st.secrets.get("TELEGRAM_TOKEN", ""),
            "telegram_chat_id": st.secrets.get("TELEGRAM_CHAT_ID", ""),
        }
    except Exception:
        return {
            "access_token": "", "account_id": "",
            "environment": "practice",
            "telegram_token": "", "telegram_chat_id": "",
        }


# ── Telegram ─────────────────────────────────────────────────────
def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    if not bot_token or not chat_id:
        return False
    try:
        url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }, timeout=10)
        return resp.ok
    except Exception:
        return False


# ── Instruments ──────────────────────────────────────────────────
DEFAULT_INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF",
    "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD",
    "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "XAU_USD", "NAS100_USD", "SPX500_USD", "US30_USD",
]

# ── Paramètres CEP v3.1 — OPTIMISÉS POUR SIGNAUX PROPRES ────────
DEFAULT_CEP_PARAMS = {
    # Compression
    "atr_contraction_ratio":    0.75,   # ATR actuel < ATR passé × ratio
    "compression_atr_ratio":    0.50,   # spread_norm < seuil
    "compression_min_candles":  8,      # ↑ 6→8 pour signaux plus propres
    "atr_lookback":             20,     # bougies H4 pour mesurer contraction ATR
    # Expansion
    "expansion_atr_ratio":      1.00,   # spread_norm > seuil
    "expansion_min_candles":    2,
    "expansion_max_age":        3,      # ↓ 6→3 pour signaux plus frais
    "volume_ratio_min":         1.5,    # ↑ 1.2→1.5 pour confirmation volume forte
    # Tendance D1
    "d1_alignment_min_candles": 5,
    # Pullback (pour info, pas pour signal Telegram)
    "pullback_distance_min":    0.15,
    "pullback_ema50_zone":      3.0,
    "pullback_direction_bars":  2,
    # Signal
    "min_score":    6,          # ↑ 5→6 pour signaux plus qualitatifs
    "candles_d1":   120,
    "candles_h4":   200,
    # Timeframes
    "use_h1":       False,      # False = signaux propres | True = plus de signaux
    "candles_h1":   200,
    "h1_min_score": 7,          # Score encore plus élevé pour H1
}


# ═══════════════════════════════════════════════════════════════════
# § 2 — OANDA DATA FETCHER
# ═══════════════════════════════════════════════════════════════════

def fetch_candles(
    client:      "oandapyV20.API",
    instrument:  str,
    granularity: str,
    count:       int,
) -> pd.DataFrame:
    if not OANDA_AVAILABLE:
        raise ImportError("oandapyV20 n'est pas installé.")

    params = {"count": count, "granularity": granularity, "price": "M"}
    r = oanda_instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(r)

    rows = []
    for c in r.response["candles"]:
        if c["complete"]:
            rows.append({
                "time":   pd.to_datetime(c["time"]),
                "open":   float(c["mid"]["o"]),
                "high":   float(c["mid"]["h"]),
                "low":    float(c["mid"]["l"]),
                "close":  float(c["mid"]["c"]),
                "volume": int(c["volume"]),
            })

    if not rows:
        raise ValueError(f"Aucune donnée pour {instrument} ({granularity})")

    return pd.DataFrame(rows).set_index("time").sort_index()


def fetch_multi_timeframe(
    client:     "oandapyV20.API",
    instrument: str,
    params:     dict,
) -> dict:
    """
    Retourne un dict avec D1, H4 et optionnellement H1
    """
    result = {
        "D1": fetch_candles(client, instrument, "D",  params["candles_d1"]),
        "H4": fetch_candles(client, instrument, "H4", params["candles_h4"]),
    }
    
    if params.get("use_h1", False):
        result["H1"] = fetch_candles(client, instrument, "H1", params["candles_h1"])
    
    return result


# ═══════════════════════════════════════════════════════════════════
# § 3 — INDICATORS
# ═══════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema5"]         = calc_ema(df["close"], 5)
    df["ema9"]         = calc_ema(df["close"], 9)
    df["ema13"]        = calc_ema(df["close"], 13)
    df["ema20"]        = calc_ema(df["close"], 20)
    df["ema50"]        = calc_ema(df["close"], 50)
    df["atr14"]        = calc_atr(df, 14)
    df["spread"]       = (df["ema5"] - df["ema50"]).abs()
    df["spread_norm"]  = df["spread"] / df["atr14"]
    df["volume_ma20"]  = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    return df.dropna()


# ═══════════════════════════════════════════════════════════════════
# § 4 — CEP ENGINE v3.1
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CEPResult:
    instrument:           str
    timeframe:            str    # "D1", "H4", ou "H1"
    direction:            str    # "BUY" | "SELL" | "NONE" | "ERROR"
    score:                int    # 0–8
    signal:               bool   # TRUE = Expansion confirmée, prépare-toi

    # ── Tendance D1 ─────────────────────────────────────────────
    trend_d1:             bool  = False
    d1_alignment_bars:    int   = 0
    d1_ema_slope_ok:      bool  = False

    # ── Compression ─────────────────────────────────────────────
    compression_detected: bool  = False
    compression_bars:     int   = 0
    atr_contracted:       bool  = False

    # ── Expansion (TRIGGER PRINCIPAL) ───────────────────────────
    expansion_detected:   bool  = False
    expansion_bars:       int   = 0
    expansion_age:        int   = 0
    volume_confirmed:     bool  = False

    # ── Zone de pullback (INFO pour l'utilisateur) ──────────────
    pullback_pending:     bool  = False
    in_pullback_zone:     bool  = False
    near_ema50_zone:      bool  = False

    # ── Pentes ──────────────────────────────────────────────────
    ema_slope_ok:         bool  = False

    # ── Niveaux ─────────────────────────────────────────────────
    ema20:         float = 0.0
    ema50:         float = 0.0
    current_price: float = 0.0
    spread_norm:   float = 0.0
    atr:           float = 0.0
    volume_ratio:  float = 0.0

    score_details: list  = field(default_factory=list)
    error_msg:     str   = ""

    # ── Telegram v3.1 — ALERTE PRÉCOCE ──────────────────────────
    def to_telegram_signal(self) -> str:
        """
        ALERTE PRÉCOCE : Expansion confirmée après compression.
        Message : "Les conditions sont réunies. Prépare-toi au pullback."
        """
        if not self.signal:
            return ""
        
        emoji   = "📈" if self.direction == "BUY" else "📉"
        tf_text = f"({self.timeframe})" if self.timeframe != "H4" else ""
        
        lines = [
            f"🔔 *SETUP CEP CONFIRMÉ — {self.instrument}* {tf_text} {emoji}",
            f"",
            f"Direction : *{self.direction}*  |  Score : *{self.score}/8*",
            f"",
            f"✅ *CONDITIONS RÉUNIES :* ",
            f"  • Tendance D1 : {'✅' if self.trend_d1 else '❌'} ({self.d1_alignment_bars}b)",
            f"  • Compression → Expansion : {'✅' if self.expansion_detected else '❌'}",
            f"  • Volume : {self.volume_ratio:.1f}× moyenne {'✅' if self.volume_confirmed else '⚠️'}",
            f"  • Âge expansion : {self.expansion_age} bougies",
            f"",
            f"📍 *NIVEAUX ACTUELS :* ",
            f"  • Prix : `{self.current_price:.5f}`",
            f"  • EMA20 : `{self.ema20:.5f}`",
            f"  • EMA50 : `{self.ema50:.5f}`",
            f"  • ATR : `{self.atr:.5f}`",
            f"",
            f"🎯 *PROCHAINE ÉTAPE :* ",
            f"  1. Ouvre le graphique {self.timeframe}",
            f"  2. Attends le retour du prix sur EMA20 ou EMA50",
            f"  3. Cherche une bougie de confirmation (pinbar, engulfing)",
            f"  4. Entre manuellement avec SL sous l'EMA50",
            f"",
            f"⚠️ *N'ENTRE PAS MAINTENANT !* ",
            f"_Attends le pullback + confirmation price action._",
        ]
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────

def _count_d1_alignment(df_d1: pd.DataFrame, direction: str, min_bars: int) -> tuple:
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
    threshold = params["compression_atr_ratio"]
    min_bars  = params["compression_min_candles"]
    lookback  = params["atr_lookback"]

    count = 0
    for i in range(len(df) - 1, -1, -1):
        if df["spread_norm"].iloc[i] < threshold:
            count += 1
        else:
            break

    if count < min_bars:
        return False, False, count

    # Décroissance progressive du spread
    window  = df["spread"].iloc[-count:]
    diffs   = window.diff().dropna()
    pct_dec = (diffs < 0).sum() / len(diffs) if len(diffs) > 0 else 0
    if pct_dec < 0.55:
        return False, False, count

    # Contraction ATR réelle
    atr_contracted = False
    if len(df) > lookback:
        atr_now  = df["atr14"].iloc[-1]
        atr_past = df["atr14"].iloc[-lookback]
        if atr_past > 0:
            atr_contracted = atr_now < atr_past * params["atr_contraction_ratio"]

    return True, atr_contracted, count


def _detect_expansion(df: pd.DataFrame, params: dict, direction: str) -> tuple:
    threshold = params["expansion_atr_ratio"]
    min_bars  = params["expansion_min_candles"]
    max_age   = params["expansion_max_age"]
    vol_min   = params["volume_ratio_min"]

    count = 0
    for i in range(len(df) - 1, max(len(df) - max_age - 5, 1), -1):
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1]

        spread_ok = row.spread_norm > threshold
        atr_ok    = row.atr14 >= prev_row.atr14

        if direction == "BUY":
            ema_ok = row.ema5 > row.ema9 > row.ema13 > row.ema20 > row.ema50
        else:
            ema_ok = row.ema5 < row.ema9 < row.ema13 < row.ema20 < row.ema50

        if spread_ok and atr_ok and ema_ok:
            count += 1
        else:
            break

    if count < min_bars:
        return False, count, 0, False

    age = count
    if age > max_age:
        return False, count, age, False

    vol_confirmed = False
    if "volume_ratio" in df.columns and count > 0:
        exp_vol       = df["volume_ratio"].iloc[-count:].mean()
        vol_confirmed = exp_vol >= vol_min

    return True, count, age, vol_confirmed


def _detect_pullback_zone(df: pd.DataFrame, direction: str, params: dict) -> tuple:
    """
    Détection de zone de pullback (INFO seulement, pas pour signal Telegram)
    """
    last      = df.iloc[-1]
    atr       = last.atr14
    price     = last.close
    ema20     = last.ema20
    ema50     = last.ema50
    dist_min  = params["pullback_distance_min"]
    n_bars    = params["pullback_direction_bars"]
    ema50_thr = params["pullback_ema50_zone"]

    dist_ema20 = abs(price - ema20)
    dist_ema50 = abs(price - ema50)

    moving_toward_ema = False
    if len(df) >= n_bars + 1:
        recent_closes = df["close"].iloc[-(n_bars + 1):]
        if direction == "BUY":
            moving_toward_ema = recent_closes.iloc[-1] < recent_closes.iloc[0]
        else:
            moving_toward_ema = recent_closes.iloc[-1] > recent_closes.iloc[0]

    if direction == "BUY":
        above_ema20 = price > ema20
        pullback_pending = above_ema20 and dist_ema20 > atr * dist_min

        in_pullback_zone = (
            above_ema20
            and dist_ema20 <= atr * 1.5
            and dist_ema20 >= atr * 0.05
            and moving_toward_ema
        )

        near_ema50_zone = (
            price <= ema20
            and dist_ema50 <= atr * ema50_thr
            and moving_toward_ema
        )

    else:  # SELL
        below_ema20 = price < ema20
        pullback_pending = below_ema20 and dist_ema20 > atr * dist_min

        in_pullback_zone = (
            below_ema20
            and dist_ema20 <= atr * 1.5
            and dist_ema20 >= atr * 0.05
            and moving_toward_ema
        )

        near_ema50_zone = (
            price >= ema20
            and dist_ema50 <= atr * ema50_thr
            and moving_toward_ema
        )

    return pullback_pending, in_pullback_zone, near_ema50_zone


def _calc_score(result: "CEPResult", df_tf: pd.DataFrame, params: dict) -> tuple:
    """
    Score 0–8 pour signaux PROPRES
    """
    score   = 0
    details = []

    # +1 Durée D1
    if result.d1_alignment_bars >= 10:
        score += 1
        details.append((f"Tendance D1 longue ({result.d1_alignment_bars}b ≥ 10)", 1))

    # +1 Pente D1
    if result.d1_ema_slope_ok:
        score += 1
        details.append(("Pente EMA20/50 D1 confirmée", 1))

    # +1 Compression + ATR
    if result.compression_detected and result.atr_contracted:
        score += 1
        details.append((f"Compression solide (spread + ATR, {result.compression_bars}b)", 1))

    # +1 Compression longue
    if result.compression_bars >= params["compression_min_candles"] * 2:
        score += 1
        details.append((f"Compression longue ({result.compression_bars}b)", 1))

    # +1 Expansion FRAÎCHE (critique pour signaux propres)
    if result.expansion_detected and result.expansion_age <= 2:
        score += 2  # ← Double points pour expansion très fraîche
        details.append((f"Expansion TRÈS fraîche ({result.expansion_age}b) ← CRITIQUE", 2))
    elif result.expansion_detected and result.expansion_age <= params["expansion_max_age"]:
        score += 1
        details.append((f"Expansion fraîche ({result.expansion_age}b)", 1))

    # +1 Volume ÉLEVÉ (critique pour signaux propres)
    if result.volume_ratio >= 2.0:
        score += 2  # ← Double points pour volume très fort
        details.append((f"Volume EXCEPTIONNEL ({result.volume_ratio:.1f}×)", 2))
    elif result.volume_confirmed:
        score += 1
        details.append((f"Volume élevé ({result.volume_ratio:.1f}× moyenne)", 1))

    # +1 Pente EMA timeframe
    if result.ema_slope_ok:
        score += 1
        details.append(("Pente EMA20/50 confirmée", 1))

    # Cap à 8
    score = min(score, 8)

    return score, details


# ── Moteur principal v3.1 ────────────────────────────────────────

def run_cep_engine(
    instrument: str,
    timeframe:  str,
    df_d1:      pd.DataFrame,
    df_tf:      pd.DataFrame,  # H4 ou H1 selon le cas
    params:     dict,
) -> CEPResult:
    """
    Pipeline CEP v3.1 :
      1. Filtre D1 (tendance de fond)
      2. Compression sur timeframe cible
      3. Expansion sur timeframe cible ← TRIGGER PRINCIPAL
      4. Signal IMMÉDIAT (pas d'attente pullback)
      5. Score renforcé pour signaux propres
    """
    result = CEPResult(
        instrument=instrument,
        timeframe=timeframe,
        direction="NONE",
        score=0,
        signal=False
    )

    if len(df_d1) < 60 or len(df_tf) < 60:
        result.error_msg = "Données insuffisantes"
        return result

    last_d1 = df_d1.iloc[-1]
    last_tf = df_tf.iloc[-1]

    result.ema20         = float(last_tf.ema20)
    result.ema50         = float(last_tf.ema50)
    result.current_price = float(last_tf.close)
    result.spread_norm   = float(last_tf.spread_norm)
    result.atr           = float(last_tf.atr14)
    result.volume_ratio  = float(last_tf.get("volume_ratio", 0.0))

    # ── 1. Filtre D1 ─────────────────────────────────────────────
    bull = (last_d1.ema5 > last_d1.ema9 > last_d1.ema13 > last_d1.ema20 > last_d1.ema50
            and last_d1.close > last_d1.ema20)
    bear = (last_d1.ema5 < last_d1.ema9 < last_d1.ema13 < last_d1.ema20 < last_d1.ema50
            and last_d1.close < last_d1.ema20)

    if not bull and not bear:
        result.error_msg = "Pas de tendance D1 claire"
        return result

    result.direction = "BUY" if bull else "SELL"

    # Pente EMA D1 — bloquante
    if result.direction == "BUY":
        d1_slope = (df_d1["ema20"].iloc[-1] > df_d1["ema20"].iloc[-2] and
                    df_d1["ema50"].iloc[-1] > df_d1["ema50"].iloc[-2])
    else:
        d1_slope = (df_d1["ema20"].iloc[-1] < df_d1["ema20"].iloc[-2] and
                    df_d1["ema50"].iloc[-1] < df_d1["ema50"].iloc[-2])
    result.d1_ema_slope_ok = d1_slope
    if not d1_slope:
        result.error_msg = "Pente EMA D1 non confirmée"
        return result

    aligned, align_bars = _count_d1_alignment(
        df_d1, result.direction, params["d1_alignment_min_candles"]
    )
    result.trend_d1          = aligned
    result.d1_alignment_bars = align_bars
    if not aligned:
        result.error_msg = f"Tendance D1 trop récente ({align_bars}b)"
        return result

    # ── 2. Compression ───────────────────────────────────────────
    offset     = max(params["expansion_min_candles"] + 1, 3)
    df_comp    = df_tf.iloc[:-offset]
    comp_ok, atr_contracted, comp_bars = _detect_compression(df_comp, params)
    result.compression_detected = comp_ok
    result.atr_contracted       = atr_contracted
    result.compression_bars     = comp_bars

    # ── 3. Expansion ← TRIGGER PRINCIPAL ────────────────────────
    exp_ok, exp_bars, exp_age, vol_ok = _detect_expansion(df_tf, params, result.direction)
    result.expansion_detected = exp_ok
    result.expansion_bars     = exp_bars
    result.expansion_age      = exp_age
    result.volume_confirmed   = vol_ok

    # ── 4. Pente EMA timeframe ──────────────────────────────────
    if result.direction == "BUY":
        slope_ok = (df_tf["ema20"].iloc[-1] > df_tf["ema20"].iloc[-2] and
                    df_tf["ema50"].iloc[-1] > df_tf["ema50"].iloc[-2])
    else:
        slope_ok = (df_tf["ema20"].iloc[-1] < df_tf["ema20"].iloc[-2] and
                    df_tf["ema50"].iloc[-1] < df_tf["ema50"].iloc[-2])
    result.ema_slope_ok = slope_ok

    # ── 5. Zone pullback (INFO seulement) ───────────────────────
    pb_pending, pb_active, pb_ema50 = _detect_pullback_zone(df_tf, result.direction, params)
    result.pullback_pending  = pb_pending
    result.in_pullback_zone  = pb_active
    result.near_ema50_zone   = pb_ema50

    # ── 6. Score ─────────────────────────────────────────────────
    result.score, result.score_details = _calc_score(result, df_tf, params)

    # ── 7. SIGNAL = Expansion confirmée (PAS pullback) ──────────
    min_score_tf = params["h1_min_score"] if timeframe == "H1" else params["min_score"]
    
    result.signal = (
        result.trend_d1 and
        result.d1_ema_slope_ok and
        result.compression_detected and
        result.expansion_detected and  # ← LE TRIGGER
        result.volume_confirmed and
        result.ema_slope_ok and
        result.score >= min_score_tf
    )

    return result


# ═══════════════════════════════════════════════════════════════════
# § 5 — SCANNER v3.1
# ═══════════════════════════════════════════════════════════════════

def run_scanner(
    access_token:     str,
    environment:      str,
    instruments_list: list,
    params:           dict,
    progress_callback=None,
) -> list:
    results = []
    total   = len(instruments_list)

    client = oandapyV20.API(access_token=access_token, environment=environment)

    for i, instrument in enumerate(instruments_list):
        if progress_callback:
            progress_callback(i / total, f"Analyse de {instrument}…")
        
        try:
            # Récupère D1 + H4 (+ H1 si activé)
            dfs = fetch_multi_timeframe(client, instrument, params)
            
            # Ajoute indicateurs
            for tf in dfs:
                dfs[tf] = add_indicators(dfs[tf])
            
            # Analyse H4 (toujours)
            result_h4 = run_cep_engine(
                instrument, "H4", dfs["D1"], dfs["H4"], params
            )
            results.append(result_h4)
            
            # Analyse H1 (si activé)
            if params.get("use_h1", False) and "H1" in dfs:
                result_h1 = run_cep_engine(
                    instrument, "H1", dfs["D1"], dfs["H1"], params
                )
                results.append(result_h1)
                
        except Exception as e:
            result = CEPResult(
                instrument=instrument,
                timeframe="H4",
                direction="ERROR",
                score=0,
                signal=False,
                error_msg=str(e)[:80],
            )
            results.append(result)
        
        time.sleep(0.35)

    if progress_callback:
        progress_callback(1.0, "Scan terminé.")

    # Tri : score + expansion fraîche + volume
    results.sort(
        key=lambda r: (
            r.signal,
            r.score,
            -r.expansion_age,  # Plus frais en premier
            -r.volume_ratio,   # Plus de volume en premier
        ),
        reverse=True,
    )
    return results


# ═══════════════════════════════════════════════════════════════════
# § 6 — STREAMLIT UI v3.1
# ═══════════════════════════════════════════════════════════════════

def _score_badge(score: int) -> str:
    if score >= 7: return f"🟢 {score}/8"
    if score >= 5: return f"🟡 {score}/8"
    return f"🔴 {score}/8"

def _dir_emoji(direction: str) -> str:
    return {"BUY": "📈", "SELL": "📉"}.get(direction, "⚠️")

def _phase_badge(r: CEPResult) -> str:
    if r.expansion_age == 0:
        return "⏳ Expansion non détectée"
    elif r.expansion_age <= 2:
        return "🔥 Expansion TRÈS fraîche"
    elif r.expansion_age <= 4:
        return "✅ Expansion fraîche"
    else:
        return "⚠️ Expansion âgée"


def main():
    st.set_page_config(
        page_title="CEP Detector v3.1 — Signaux Propres",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg          = get_config()
    access_token = cfg["access_token"]
    environment  = cfg["environment"]
    tg_token     = cfg["telegram_token"]
    tg_chat_id   = cfg["telegram_chat_id"]

    st.title("🎯 CEP Detector v3.1")
    st.markdown(
        "**Alertes PRÉCOCES** — Expansion confirmée après compression.  \n"
        "**TU exécutes** le pullback sur EMA20/50 manuellement."
    )
    st.info(
        "📌 **Mode SIGNAUX PROPRES activé** : Score ≥6/8, Volume ≥1.5×, Expansion ≤3 bougies"
    )
    st.divider()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("📋 Instruments")
        instruments_input = st.text_area(
            "Un par ligne",
            value="\n".join(DEFAULT_INSTRUMENTS),
            height=200,
        )
        instruments_list = [
            x.strip().upper()
            for x in instruments_input.strip().split("\n")
            if x.strip()
        ]

        st.divider()
        st.subheader("🎛️ Paramètres — MODE PROPRES")
        
        st.markdown("**Timeframes**")
        use_h1 = st.checkbox(
            "Activer H1 (plus de signaux, plus de bruit)",
            value=False,
            help="Décoché = signaux très propres (D1+H4 uniquement)"
        )
        
        st.markdown("**Qualité du signal**")
        min_score  = st.slider("Score minimum", 5, 8, value=6)
        vol_ratio  = st.slider("Volume minimum (× moyenne)", 1.0, 3.0, step=0.1, value=1.5)
        exp_age    = st.slider("Fraîcheur expansion max", 2, 6, value=3)
        comp_min   = st.slider("Min bougies compression", 6, 15, value=8)
        
        st.markdown("**Seuils techniques**")
        comp_ratio = st.slider("Seuil compression (× ATR)", 0.3, 1.0, step=0.05, value=0.50)
        atr_ratio  = st.slider("Ratio contraction ATR", 0.5, 0.95, step=0.05, value=0.75)

        params = {
            **DEFAULT_CEP_PARAMS,
            "use_h1":                 use_h1,
            "min_score":              min_score,
            "h1_min_score":           7 if use_h1 else 6,
            "volume_ratio_min":       vol_ratio,
            "expansion_max_age":      exp_age,
            "compression_min_candles": comp_min,
            "compression_atr_ratio":  comp_ratio,
            "atr_contraction_ratio":  atr_ratio,
        }

        st.divider()
        st.subheader("⏱️ Scan automatique")
        auto_scan        = st.toggle("Activer", value=False)
        refresh_interval = st.select_slider(
            "Intervalle", options=[5, 10, 15, 30, 60], value=15,
            format_func=lambda x: f"{x} min", disabled=not auto_scan,
        )
        
        if use_h1:
            st.warning("⚠️ H1 activé : plus de signaux mais qualité réduite")
        
        if tg_token and tg_chat_id:
            st.success("🔔 Telegram configuré ✅")
        else:
            st.caption("💬 Configure Telegram pour recevoir les alertes")

        st.divider()
        scan_btn = st.button(
            "🔍 Lancer le scan", type="primary",
            use_container_width=True, disabled=not bool(access_token),
        )

    # ── Auto-refresh ───────────────────────────────────────────
    if auto_scan and AUTOREFRESH_AVAILABLE and access_token:
        st_autorefresh(interval=refresh_interval * 60 * 1000, key="cep_autorefresh")
        scan_btn = True

    # ── Credentials ────────────────────────────────────────────
    if not access_token:
        st.error("⚠️ Credentials manquants")
        return

    # ── Scan ───────────────────────────────────────────────────
    if scan_btn:
        progress_bar = st.progress(0, text="Initialisation…")
        status_text  = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(min(pct, 1.0), text=msg)
            status_text.text(msg)

        with st.spinner("Scan en cours…"):
            results = run_scanner(
                access_token, environment, instruments_list, params, update_progress
            )

        progress_bar.empty()
        status_text.empty()

        # ── Telegram — UNE SEULE ALERTE (Expansion) ────────────
        new_signals = [r for r in results if r.signal]
        prev_keys   = st.session_state.get("prev_signal_keys", set())
        
        # Clé unique : instrument + timeframe + direction
        cur_keys = {f"{r.instrument}_{r.timeframe}_{r.direction}" for r in new_signals}
        
        # Nouveaux signaux uniquement
        fresh_signals = [
            r for r in new_signals
            if f"{r.instrument}_{r.timeframe}_{r.direction}" not in prev_keys
        ]

        # Envoie Telegram
        if tg_token and tg_chat_id:
            for r in fresh_signals:
                send_telegram(tg_token, tg_chat_id, r.to_telegram_signal())

        # Sauvegarde état
        st.session_state["prev_signal_keys"] = cur_keys
        st.session_state["results"]          = results
        st.session_state["scan_time"]        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.session_state["params"]           = params
        st.session_state["fresh_count"]      = len(fresh_signals)

    if "results" not in st.session_state:
        st.markdown("### ← Clique sur **Lancer le scan**")
        return

    results       = st.session_state["results"]
    scan_time     = st.session_state.get("scan_time", "")
    fresh_count   = st.session_state.get("fresh_count", 0)

    # ── Header ─────────────────────────────────────────────────
    col_time, col_sig = st.columns([3, 1])
    col_time.caption(f"Dernier scan : {scan_time}")
    if fresh_count:
        col_sig.success(f"🔔 {fresh_count} nouveau(x) signal(aux)")

    # ── Métriques ──────────────────────────────────────────────
    signals      = [r for r in results if r.signal]
    buy_sig      = [r for r in signals if r.direction == "BUY"]
    sell_sig     = [r for r in signals if r.direction == "SELL"]
    fresh_exp    = [r for r in signals if r.expansion_age <= 2]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Analysés", len(results))
    c2.metric("🔔 Signaux", len(signals))
    c3.metric("📈 BUY", len(buy_sig))
    c4.metric("📉 SELL", len(sell_sig))
    c5.metric("🔥 Très frais", len(fresh_exp))

    st.divider()

    # ── Cartes signaux ─────────────────────────────────────────
    if signals:
        st.subheader(f"🔔 Setups CEP — Expansion confirmée")

        for r in signals:
            phase  = _phase_badge(r)
            header = (
                f"{_dir_emoji(r.direction)} **{r.instrument}**  ·  "
                f"{r.timeframe}  ·  {_score_badge(r.score)}"
            )
            with st.expander(header, expanded=True):

                if r.expansion_age <= 2:
                    st.success("🔥 **EXPANSION TRÈS FRAÎCHE** — Signal de haute qualité")
                elif r.volume_ratio >= 2.0:
                    st.info("📊 **VOLUME EXCEPTIONNEL** — Confirmation forte")
                
                st.markdown(f"**Phase actuelle** : {phase}")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Prix", f"{r.current_price:.5f}")
                col2.metric("EMA20", f"{r.ema20:.5f}")
                col3.metric("EMA50", f"{r.ema50:.5f}")
                col4.metric("ATR", f"{r.atr:.5f}")
                col5.metric("Volume", f"{r.volume_ratio:.1f}×")

                st.markdown("---")
                st.markdown("**✅ Conditions validées :**")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"{'✅' if r.trend_d1 else '❌'} Tendance D1 ({r.d1_alignment_bars}b)")
                    st.write(f"{'✅' if r.d1_ema_slope_ok else '❌'} Pente D1")
                    st.write(f"{'✅' if r.compression_detected else '❌'} Compression ({r.compression_bars}b)")
                    st.write(f"{'✅' if r.atr_contracted else '❌'} ATR contracté")
                
                with c2:
                    st.write(f"{'✅' if r.expansion_detected else '❌'} Expansion ({r.expansion_bars}b)")
                    st.write(f"⏱️ Âge : {r.expansion_age} bougies")
                    st.write(f"{'✅' if r.volume_confirmed else '❌'} Volume ({r.volume_ratio:.1f}×)")
                    st.write(f"{'✅' if r.ema_slope_ok else '❌'} Pente {r.timeframe}")

                st.markdown("---")
                st.markdown("**🎯 Ton plan d'action :**")
                st.info(
                    f"1. Ouvre le graphique {r.instrument} en {r.timeframe}\n"
                    f"2. Attends le retour sur EMA20 (`{r.ema20:.5f}`) ou EMA50 (`{r.ema50:.5f}`)\n"
                    f"3. Cherche une bougie de confirmation (pinbar, engulfing)\n"
                    f"4. Entre avec SL sous/au-dessus de l'EMA50\n"
                    f"5. TP = 2-3× ATR (`{r.atr:.5f}`)"
                )

                with st.expander("📊 Score détaillé"):
                    for label, pts in r.score_details:
                        icon = f"`+{pts}`" if pts > 0 else "` 0`"
                        st.write(f"{icon} {label}")

    else:
        st.info(
            f"Aucun setup avec score ≥ {params['min_score']}/8.  \n"
            "C'est normal en mode SIGNAUX PROPRES — mieux vaut attendre la qualité."
        )

    # ── Tableau complet ────────────────────────────────────────
    st.divider()
    with st.expander("📋 Tous les résultats", expanded=False):
        
        rows = []
        for r in results:
            rows.append({
                "Instrument": r.instrument,
                "TF": r.timeframe,
                "Direction": r.direction,
                "Score": r.score,
                "Signal": "🔔" if r.signal else "—",
                "Phase": _phase_badge(r) if r.signal else "—",
                "Âge Exp": f"{r.expansion_age}b" if r.expansion_detected else "—",
                "Volume": f"{r.volume_ratio:.1f}×",
                "Prix": f"{r.current_price:.5f}",
                "EMA20": f"{r.ema20:.5f}",
                "EMA50": f"{r.ema50:.5f}",
            })

        if rows:
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


if __name__ == "__main__":
    main()
