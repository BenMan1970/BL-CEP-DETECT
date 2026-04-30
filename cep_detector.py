#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   CEP DETECTOR v2  —  Compression → Expansion → Pullback        ║
║   Single-file · Oanda API · Streamlit-ready · GitHub-safe       ║
╠══════════════════════════════════════════════════════════════════╣
║  Structure interne :                                             ║
║   § 1  Imports & Configuration                                   ║
║   § 2  Oanda Data Fetcher                                        ║
║   § 3  Indicators                                                ║
║   § 4  CEP Engine v2 (logique améliorée)                        ║
║   § 5  Scanner                                                   ║
║   § 6  Streamlit UI                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Améliorations v2 :                                              ║
║   • Compression = ATR contractant + EMA spread étroit           ║
║   • Expansion  = ATR + volume croissants + alignement EMA        ║
║   • Détection de zone de pullback (prix en retour EMA20/50)     ║
║   • Fraîcheur de l'expansion (setup le plus récent en tête)     ║
║   • Score 0-8 non redondant (chaque point mesure qqch distinct) ║
║   • Filtre D1 renforcé (pente ATR + qualité tendance)           ║
╚══════════════════════════════════════════════════════════════════╝

Déploiement Streamlit Cloud :
  → Ajouter dans Settings > Secrets :
      OANDA_ACCESS_TOKEN = "votre-token"
      OANDA_ACCOUNT_ID   = "votre-compte"
      OANDA_ENVIRONMENT  = "practice"
      TELEGRAM_TOKEN     = "123456:ABC..."
      TELEGRAM_CHAT_ID   = "123456789"

Usage local :
  streamlit run cep_detector_v2.py
"""

__version__ = "2.0.0"
__author__  = "CEP Detector"

# ═══════════════════════════════════════════════════════════════════
# § 1 — IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

import time
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

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

# ── Streamlit Secrets ────────────────────────────────────────────
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
            "access_token": "", "account_id": "", "environment": "practice",
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

# ── Instruments par défaut ───────────────────────────────────────
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

# ── Paramètres CEP v2 ────────────────────────────────────────────
DEFAULT_CEP_PARAMS = {
    # ── Compression ─────────────────────────────────────────────
    # ATR actuel doit être < ATR passé × ce ratio (contraction)
    "atr_contraction_ratio":    0.75,
    # spread EMA normalisé doit être < ce seuil
    "compression_atr_ratio":    0.50,
    # durée minimum de compression en bougies H4
    "compression_min_candles":  6,
    # lookback ATR pour mesurer la contraction (bougies H4)
    "atr_lookback":             20,

    # ── Expansion ───────────────────────────────────────────────
    # spread EMA normalisé doit être > ce seuil
    "expansion_atr_ratio":      1.00,
    # bougies consécutives minimum en expansion
    "expansion_min_candles":    2,
    # fraîcheur max : l'expansion doit avoir démarré il y a ≤ N bougies
    "expansion_max_age":        6,
    # ratio de croissance volume minimum vs moyenne (ex: 1.2 = +20%)
    "volume_ratio_min":         1.2,

    # ── Tendance D1 ─────────────────────────────────────────────
    "d1_alignment_min_candles": 5,

    # ── Pullback ────────────────────────────────────────────────
    # Distance minimum prix/EMA20 pour considérer que le pullback
    # n'a pas encore eu lieu (en multiples d'ATR H4)
    "pullback_distance_min":    0.15,

    # ── Signal ──────────────────────────────────────────────────
    "min_score":   5,
    "candles_d1":  120,
    "candles_h4":  200,
}


# ═══════════════════════════════════════════════════════════════════
# § 2 — OANDA DATA FETCHER
# ═══════════════════════════════════════════════════════════════════

def fetch_candles(
    access_token: str,
    instrument:   str,
    granularity:  str,
    count:        int,
    environment:  str = "practice",
) -> pd.DataFrame:
    if not OANDA_AVAILABLE:
        raise ImportError("oandapyV20 n'est pas installé.")

    client = oandapyV20.API(access_token=access_token, environment=environment)
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
    access_token: str,
    instrument:   str,
    environment:  str,
    params:       dict,
) -> tuple:
    df_d1 = fetch_candles(access_token, instrument, "D",  params["candles_d1"],  environment)
    df_h4 = fetch_candles(access_token, instrument, "H4", params["candles_h4"], environment)
    return df_d1, df_h4


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


def calc_volume_ma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Moyenne mobile simple du volume tick Oanda."""
    return df["volume"].rolling(window=length).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute :
      ema5/9/13/20/50
      atr14
      atr14_slow   : ATR sur une fenêtre plus longue (lookback) pour comparer
      spread       : |ema5 - ema50| brut
      spread_norm  : spread / atr14
      volume_ma20  : moyenne mobile volume 20 périodes
      volume_ratio : volume / volume_ma20
    """
    df = df.copy()
    df["ema5"]       = calc_ema(df["close"], 5)
    df["ema9"]       = calc_ema(df["close"], 9)
    df["ema13"]      = calc_ema(df["close"], 13)
    df["ema20"]      = calc_ema(df["close"], 20)
    df["ema50"]      = calc_ema(df["close"], 50)
    df["atr14"]      = calc_atr(df, 14)
    df["spread"]     = (df["ema5"] - df["ema50"]).abs()
    df["spread_norm"] = df["spread"] / df["atr14"]
    df["volume_ma20"] = calc_volume_ma(df, 20)
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    return df.dropna()


# ═══════════════════════════════════════════════════════════════════
# § 4 — CEP ENGINE v2
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CEPResult:
    instrument:           str
    direction:            str    # "BUY" | "SELL" | "NONE" | "ERROR"
    score:                int    # 0–8
    signal:               bool

    # ── Flags composants ────────────────────────────────────────
    trend_d1:             bool  = False
    d1_alignment_bars:    int   = 0
    d1_atr_trending:      bool  = False   # NOUVEAU : ATR D1 en expansion

    compression_detected: bool  = False
    compression_bars:     int   = 0
    atr_contracted:       bool  = False   # NOUVEAU : ATR H4 contracté confirmé

    expansion_detected:   bool  = False
    expansion_bars:       int   = 0
    expansion_age:        int   = 0       # NOUVEAU : fraîcheur (bougies depuis début)
    volume_confirmed:     bool  = False   # NOUVEAU : volume expansion > moyenne

    pullback_pending:     bool  = False   # NOUVEAU : prix pas encore revenu sur EMA
    in_pullback_zone:     bool  = False   # NOUVEAU : prix en retour actif vers EMA

    ema_slope_ok:         bool  = False
    d1_ema_slope_ok:      bool  = False   # NOUVEAU : pente EMA D1

    # ── Niveaux clés ────────────────────────────────────────────
    ema20_h4:      float = 0.0
    ema50_h4:      float = 0.0
    current_price: float = 0.0
    spread_norm:   float = 0.0
    atr_h4:        float = 0.0
    volume_ratio:  float = 0.0

    score_details: list  = field(default_factory=list)
    error_msg:     str   = ""

    def to_telegram(self) -> str:
        if not self.signal:
            return ""
        emoji   = "📈" if self.direction == "BUY" else "📉"
        phase   = "⏳ Pullback en attente" if self.pullback_pending else "🎯 Pullback en cours"
        age_txt = f"{self.expansion_age} bougie(s) H4" if self.expansion_age else "—"

        lines = [
            f"🔔 *SETUP CEP v2 — {self.instrument}*",
            f"Direction  : {emoji} *{self.direction}*",
            f"Score      : *{self.score}/8*",
            f"Phase      : {phase}",
            "",
            f"{'✅' if self.trend_d1 else '❌'} Tendance D1 ({self.d1_alignment_bars}b) {'+ pente ✅' if self.d1_ema_slope_ok else ''}",
            f"{'✅' if self.compression_detected else '❌'} Compression H4 ({self.compression_bars}b) {'+ ATR contracté ✅' if self.atr_contracted else ''}",
            f"{'✅' if self.expansion_detected else '❌'} Expansion H4 ({self.expansion_bars}b) — âge : {age_txt}",
            f"{'✅' if self.volume_confirmed else '❌'} Volume expansion : {self.volume_ratio:.1f}× moyenne",
            f"{'✅' if self.ema_slope_ok else '❌'} Pente EMA20/50 H4",
            "",
            f"📍 Prix actuel : `{self.current_price:.5f}`",
            f"🎯 Pullback vers EMA20 → `{self.ema20_h4:.5f}`",
            f"🎯 Pullback vers EMA50 → `{self.ema50_h4:.5f}`",
            f"📊 ATR H4 : `{self.atr_h4:.5f}`",
            "",
            f"_Attendre le retour du prix sur EMA20 ou EMA50 avant d'entrer._",
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
    """
    Compression v2 — double confirmation :
      1. spread_norm < seuil (EMAs serrées)
      2. ATR actuel < ATR de N bougies en arrière × ratio (volatilité contractante)
      3. Durée ≥ min_candles
      4. Spread progressivement décroissant (≥ 55% des pas)

    Returns (detected: bool, atr_contracted: bool, bar_count: int)
    """
    threshold  = params["compression_atr_ratio"]
    min_bars   = params["compression_min_candles"]
    lookback   = params["atr_lookback"]

    # Bougies consécutives sous le seuil spread (en remontant)
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if df["spread_norm"].iloc[i] < threshold:
            count += 1
        else:
            break

    if count < min_bars:
        return False, False, count

    # Confirmation décroissance spread
    window = df["spread"].iloc[-count:]
    diffs  = window.diff().dropna()
    pct_dec = (diffs < 0).sum() / len(diffs) if len(diffs) > 0 else 0
    if pct_dec < 0.55:
        return False, False, count

    # Confirmation ATR contractant
    atr_contracted = False
    if len(df) > lookback:
        atr_now  = df["atr14"].iloc[-1]
        atr_past = df["atr14"].iloc[-lookback]
        atr_contracted = atr_now < atr_past * params["atr_contraction_ratio"]

    return True, atr_contracted, count


def _detect_expansion(df: pd.DataFrame, params: dict, direction: str) -> tuple:
    """
    Expansion v2 :
      1. spread_norm > seuil
      2. ATR croissant
      3. Volume > moyenne (confirmation momentum)
      4. Alignement EMA dans la direction
      5. Fraîcheur : l'expansion a démarré depuis ≤ expansion_max_age bougies

    Returns (detected: bool, bar_count: int, age: int, volume_confirmed: bool)
    """
    threshold = params["expansion_atr_ratio"]
    min_bars  = params["expansion_min_candles"]
    max_age   = params["expansion_max_age"]
    vol_min   = params["volume_ratio_min"]

    # Compter bougies consécutives en expansion (en remontant)
    count = 0
    for i in range(len(df) - 1, max(len(df) - max_age - 5, -1), -1):
        if i <= 0:
            break
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1]

        spread_ok = row.spread_norm > threshold
        atr_ok    = row.atr14 >= prev_row.atr14   # ATR stable ou croissant

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

    # L'expansion doit être récente
    age = count   # nombre de bougies depuis le début de l'expansion
    if age > max_age:
        return False, count, age, False

    # Confirmation volume sur les bougies d'expansion
    vol_confirmed = False
    if "volume_ratio" in df.columns:
        exp_vol = df["volume_ratio"].iloc[-count:].mean()
        vol_confirmed = exp_vol >= vol_min

    return True, count, age, vol_confirmed


def _detect_pullback_zone(
    df: pd.DataFrame,
    direction: str,
    params: dict,
) -> tuple:
    """
    Détecte la situation par rapport au pullback :

    - pullback_pending  : expansion terminée, prix encore loin des EMAs
                          → setup prêt, attendre le retour
    - in_pullback_zone  : prix en train de revenir vers EMA20/50
                          → setup optimal, surveiller l'entrée de près

    Returns (pullback_pending: bool, in_pullback_zone: bool)
    """
    last     = df.iloc[-1]
    atr      = last.atr14
    price    = last.close
    ema20    = last.ema20
    ema50    = last.ema50
    dist_min = params["pullback_distance_min"]

    dist_ema20 = abs(price - ema20)
    dist_ema50 = abs(price - ema50)

    if direction == "BUY":
        # Prix au-dessus des EMAs après expansion haussière
        above_ema20 = price > ema20
        # Pullback pending : prix encore bien au-dessus de l'EMA20
        pullback_pending = above_ema20 and dist_ema20 > atr * dist_min
        # In pullback zone : prix en train de descendre vers EMA20/50
        # (spread se réduit, prix entre EMA20 et 1.5×ATR au-dessus)
        in_pullback_zone = above_ema20 and dist_ema20 <= atr * 1.5 and dist_ema20 >= atr * 0.05
    else:
        # Prix en dessous des EMAs après expansion baissière
        below_ema20 = price < ema20
        pullback_pending = below_ema20 and dist_ema20 > atr * dist_min
        in_pullback_zone = below_ema20 and dist_ema20 <= atr * 1.5 and dist_ema20 >= atr * 0.05

    return pullback_pending, in_pullback_zone


def _calc_score(result: "CEPResult", df_h4: pd.DataFrame, params: dict) -> tuple:
    """
    Score v2 — 8 points SANS redondance (chaque critère mesure qqch de distinct) :

      +1  Tendance D1 longue (>10 bougies alignées)
      +1  Pente EMA20/50 D1 confirmée
      +1  Compression longue (>2× min_candles)  ET  ATR contracté
      +1  ATR H4 contracté seul (si compression courte mais ATR confirme)
      +1  Expansion récente (âge ≤ 3 bougies = setup frais)
      +1  Volume expansion > moyenne × ratio
      +1  Pente EMA20/50 H4 confirmée
      +1  Prix en zone de pullback actif (in_pullback_zone)
    """
    score   = 0
    details = []

    # ── +1 Durée tendance D1 ────────────────────────────────────
    if result.d1_alignment_bars >= 10:
        score += 1
        details.append((f"Tendance D1 solide ({result.d1_alignment_bars}b ≥ 10)", 1))

    # ── +1 Pente EMA D1 ─────────────────────────────────────────
    if result.d1_ema_slope_ok:
        score += 1
        details.append(("Pente EMA20/50 D1 haussière/baissière", 1))

    # ── +1 Compression forte + ATR contracté ────────────────────
    if result.compression_detected and result.atr_contracted:
        score += 1
        details.append((f"Compression confirmée (spread + ATR, {result.compression_bars}b)", 1))
    elif result.compression_detected:
        # compression spread seule : pas de point mais pas bloquant
        details.append((f"Compression spread seule ({result.compression_bars}b) — ATR non contracté", 0))

    # ── +1 Compression très longue (coil puissant) ──────────────
    if result.compression_bars >= params["compression_min_candles"] * 2:
        score += 1
        details.append((f"Compression longue ({result.compression_bars}b ≥ {params['compression_min_candles']*2})", 1))

    # ── +1 Expansion fraîche ────────────────────────────────────
    if result.expansion_detected and result.expansion_age <= 3:
        score += 1
        details.append((f"Expansion fraîche ({result.expansion_age}b)", 1))
    elif result.expansion_detected:
        details.append((f"Expansion âgée ({result.expansion_age}b > 3) — setup moins frais", 0))

    # ── +1 Volume confirmé ──────────────────────────────────────
    if result.volume_confirmed:
        score += 1
        details.append((f"Volume expansion élevé ({result.volume_ratio:.1f}× moyenne)", 1))

    # ── +1 Pente EMA H4 ─────────────────────────────────────────
    last = df_h4.iloc[-1]
    if result.direction == "BUY":
        slope_ok = (df_h4["ema20"].iloc[-1] > df_h4["ema20"].iloc[-2] and
                    df_h4["ema50"].iloc[-1] > df_h4["ema50"].iloc[-2])
    else:
        slope_ok = (df_h4["ema20"].iloc[-1] < df_h4["ema20"].iloc[-2] and
                    df_h4["ema50"].iloc[-1] < df_h4["ema50"].iloc[-2])
    result.ema_slope_ok = slope_ok
    if slope_ok:
        score += 1
        details.append(("Pente EMA20/50 H4 confirmée", 1))

    # ── +1 Zone de pullback actif ───────────────────────────────
    if result.in_pullback_zone:
        score += 1
        details.append(("Prix en zone pullback actif ← ENTRER EN ALERTE", 1))
    elif result.pullback_pending:
        details.append(("Pullback pas encore amorcé — surveiller", 0))

    return score, details


# ── Moteur principal ─────────────────────────────────────────────

def run_cep_engine(
    instrument: str,
    df_d1:      pd.DataFrame,
    df_h4:      pd.DataFrame,
    params:     dict,
) -> CEPResult:
    """
    Pipeline CEP v2 :
      1. Filtre tendance D1 (alignement + pente + durée)
      2. Compression H4 (spread + ATR contractant)
      3. Expansion H4 (spread + ATR croissant + volume + fraîcheur)
      4. Zone de pullback (où en est le prix par rapport aux EMAs)
      5. Score 0-8 non redondant
      6. Signal si score ≥ min_score
    """
    result = CEPResult(instrument=instrument, direction="NONE", score=0, signal=False)

    if len(df_d1) < 60 or len(df_h4) < 60:
        result.error_msg = "Données insuffisantes"
        return result

    last_d1 = df_d1.iloc[-1]
    last_h4 = df_h4.iloc[-1]

    result.ema20_h4      = float(last_h4.ema20)
    result.ema50_h4      = float(last_h4.ema50)
    result.current_price = float(last_h4.close)
    result.spread_norm   = float(last_h4.spread_norm)
    result.atr_h4        = float(last_h4.atr14)
    result.volume_ratio  = float(last_h4.volume_ratio) if "volume_ratio" in df_h4.columns else 0.0

    # ── Étape 1 : Filtre D1 ──────────────────────────────────────
    bull = (last_d1.ema5 > last_d1.ema9 > last_d1.ema13 > last_d1.ema20 > last_d1.ema50
            and last_d1.close > last_d1.ema20)

    bear = (last_d1.ema5 < last_d1.ema9 < last_d1.ema13 < last_d1.ema20 < last_d1.ema50
            and last_d1.close < last_d1.ema20)

    if not bull and not bear:
        result.error_msg = "Pas de tendance D1 claire"
        return result

    result.direction = "BUY" if bull else "SELL"

    # Pente EMA20/50 D1
    d1_slope = (
        (df_d1["ema20"].iloc[-1] > df_d1["ema20"].iloc[-2] and
         df_d1["ema50"].iloc[-1] > df_d1["ema50"].iloc[-2])
        if result.direction == "BUY"
        else
        (df_d1["ema20"].iloc[-1] < df_d1["ema20"].iloc[-2] and
         df_d1["ema50"].iloc[-1] < df_d1["ema50"].iloc[-2])
    )
    result.d1_ema_slope_ok = d1_slope

    # La pente D1 est bloquante : sans pente confirmée, pas de setup
    if not d1_slope:
        result.error_msg = "Pente EMA D1 non confirmée"
        return result

    aligned, align_bars = _count_d1_alignment(
        df_d1, result.direction, params["d1_alignment_min_candles"]
    )
    result.trend_d1          = aligned
    result.d1_alignment_bars = align_bars

    if not aligned:
        result.error_msg = f"Tendance D1 trop récente ({align_bars}b < {params['d1_alignment_min_candles']}b)"
        return result

    # ── Étape 2 : Compression H4 ─────────────────────────────────
    offset = max(params["expansion_min_candles"] + 1, 3)
    df_h4_comp = df_h4.iloc[:-offset]

    comp_ok, atr_contracted, comp_bars = _detect_compression(df_h4_comp, params)
    result.compression_detected = comp_ok
    result.atr_contracted       = atr_contracted
    result.compression_bars     = comp_bars

    if not comp_ok:
        result.error_msg = f"Compression insuffisante ({comp_bars}b)"
        return result

    # ── Étape 3 : Expansion H4 ───────────────────────────────────
    exp_ok, exp_bars, exp_age, vol_confirmed = _detect_expansion(df_h4, params, result.direction)
    result.expansion_detected = exp_ok
    result.expansion_bars     = exp_bars
    result.expansion_age      = exp_age
    result.volume_confirmed   = vol_confirmed

    if not exp_ok:
        if exp_age > params["expansion_max_age"] and exp_bars >= params["expansion_min_candles"]:
            result.error_msg = f"Expansion trop ancienne ({exp_age}b > {params['expansion_max_age']}b)"
        else:
            result.error_msg = "Expansion absente"
        return result

    # ── Étape 4 : Zone de pullback ───────────────────────────────
    pb_pending, pb_active = _detect_pullback_zone(df_h4, result.direction, params)
    result.pullback_pending  = pb_pending
    result.in_pullback_zone  = pb_active

    # ── Étape 5 : Score ──────────────────────────────────────────
    result.score, result.score_details = _calc_score(result, df_h4, params)
    result.signal = result.score >= params["min_score"]

    return result


# ═══════════════════════════════════════════════════════════════════
# § 5 — SCANNER
# ═══════════════════════════════════════════════════════════════════

def run_scanner(
    access_token:       str,
    environment:        str,
    instruments_list:   list,
    params:             dict,
    progress_callback=None,
) -> list:
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
        time.sleep(0.35)

    if progress_callback:
        progress_callback(1.0, "Scan terminé.")

    # Tri : signaux d'abord, puis pullback actif, puis score
    results.sort(
        key=lambda r: (r.signal, r.in_pullback_zone, r.score),
        reverse=True,
    )
    return results


# ═══════════════════════════════════════════════════════════════════
# § 6 — STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

def _score_badge(score: int) -> str:
    if score >= 6: return f"🟢 {score}/8"
    if score >= 4: return f"🟡 {score}/8"
    return f"🔴 {score}/8"

def _dir_emoji(direction: str) -> str:
    return {"BUY": "📈", "SELL": "📉"}.get(direction, "⚠️")

def _phase_badge(r: CEPResult) -> str:
    if r.in_pullback_zone:  return "🎯 Pullback en cours"
    if r.pullback_pending:  return "⏳ Attendre pullback"
    return ""


def main():
    st.set_page_config(
        page_title="CEP Detector v2",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg          = get_config()
    access_token = cfg["access_token"]
    environment  = cfg["environment"]
    tg_token     = cfg["telegram_token"]
    tg_chat_id   = cfg["telegram_chat_id"]

    st.title("📊 CEP Detector v2")
    st.markdown(
        "**Compression → Expansion → Pullback** — Signal de setup haute qualité.  "
        "Attendre le retour du prix sur EMA20/EMA50 avant toute entrée."
    )
    st.divider()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("📋 Instruments")
        instruments_input = st.text_area(
            "Un par ligne (format Oanda)",
            value="\n".join(DEFAULT_INSTRUMENTS),
            height=200,
        )
        instruments_list = [
            x.strip().upper()
            for x in instruments_input.strip().split("\n")
            if x.strip()
        ]

        st.divider()
        st.subheader("🎛️ Paramètres CEP")

        min_score = st.slider("Score minimum (sur 8)", 3, 8,
                              DEFAULT_CEP_PARAMS["min_score"])
        comp_ratio = st.slider("Seuil compression (× ATR)", 0.2, 1.0, step=0.05,
                               value=DEFAULT_CEP_PARAMS["compression_atr_ratio"])
        exp_ratio = st.slider("Seuil expansion (× ATR)", 0.5, 2.5, step=0.1,
                              value=DEFAULT_CEP_PARAMS["expansion_atr_ratio"])
        comp_min = st.slider("Min bougies compression", 3, 20,
                             DEFAULT_CEP_PARAMS["compression_min_candles"])
        atr_ratio = st.slider("Ratio contraction ATR", 0.5, 0.95, step=0.05,
                              value=DEFAULT_CEP_PARAMS["atr_contraction_ratio"],
                              help="ATR actuel < ATR passé × ce ratio = compression confirmée")
        exp_age = st.slider("Fraîcheur expansion max (bougies H4)", 2, 12,
                            DEFAULT_CEP_PARAMS["expansion_max_age"],
                            help="L'expansion doit être récente")
        vol_ratio = st.slider("Volume expansion min (× moyenne)", 1.0, 2.5, step=0.1,
                              value=DEFAULT_CEP_PARAMS["volume_ratio_min"])
        d1_min = st.slider("Min bougies tendance D1", 3, 20,
                           DEFAULT_CEP_PARAMS["d1_alignment_min_candles"])

        params = {
            **DEFAULT_CEP_PARAMS,
            "min_score":               min_score,
            "compression_atr_ratio":   comp_ratio,
            "expansion_atr_ratio":     exp_ratio,
            "compression_min_candles": comp_min,
            "atr_contraction_ratio":   atr_ratio,
            "expansion_max_age":       exp_age,
            "volume_ratio_min":        vol_ratio,
            "d1_alignment_min_candles": d1_min,
        }

        st.divider()
        st.subheader("⏱️ Scan automatique")
        auto_scan = st.toggle("Activer le scan automatique", value=False)
        refresh_interval = st.select_slider(
            "Intervalle", options=[5, 10, 15, 30, 60], value=15,
            format_func=lambda x: f"{x} min", disabled=not auto_scan,
        )
        if tg_token and tg_chat_id:
            st.success("🔔 Telegram configuré ✅")
        else:
            st.caption("💬 Ajoutez TELEGRAM_TOKEN et TELEGRAM_CHAT_ID dans les secrets.")

        st.divider()
        scan_btn = st.button(
            "🔍 Lancer le scan", type="primary",
            use_container_width=True, disabled=not bool(access_token),
        )
        if not access_token:
            st.warning("Credentials manquants.")

    # ── Auto-refresh ───────────────────────────────────────────
    if auto_scan and AUTOREFRESH_AVAILABLE and access_token:
        st_autorefresh(interval=refresh_interval * 60 * 1000, key="auto_scan_refresh")
        scan_btn = True
    elif auto_scan and not AUTOREFRESH_AVAILABLE:
        st.warning("`streamlit-autorefresh` non installé.")

    # ── Secrets manquants ──────────────────────────────────────
    if not access_token:
        st.error("⚠️ Credentials non configurés.")
        with st.expander("📖 Configuration des secrets"):
            st.code(
                'OANDA_ACCESS_TOKEN = "votre-token"\n'
                'OANDA_ACCOUNT_ID   = "votre-account-id"\n'
                'OANDA_ENVIRONMENT  = "practice"\n'
                'TELEGRAM_TOKEN     = "123456:ABC..."\n'
                'TELEGRAM_CHAT_ID   = "123456789"',
                language="toml",
            )
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

        # Telegram — nouveaux signaux uniquement
        new_signals      = [r for r in results if r.signal]
        prev_keys        = st.session_state.get("prev_signal_keys", set())
        current_keys     = {f"{r.instrument}_{r.direction}" for r in new_signals}
        fresh_signals    = [r for r in new_signals if f"{r.instrument}_{r.direction}" not in prev_keys]

        if fresh_signals and tg_token and tg_chat_id:
            for r in fresh_signals:
                send_telegram(tg_token, tg_chat_id, r.to_telegram())

        st.session_state["prev_signal_keys"] = current_keys
        st.session_state["results"]          = results
        st.session_state["scan_time"]        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.session_state["params"]           = params
        st.session_state["fresh_count"]      = len(fresh_signals)

    if "results" not in st.session_state:
        st.markdown("### ← Cliquez sur **Lancer le scan** pour démarrer.")
        return

    results     = st.session_state["results"]
    scan_time   = st.session_state.get("scan_time", "")
    fresh_count = st.session_state.get("fresh_count", 0)

    col_time, col_fresh = st.columns([3, 1])
    col_time.caption(f"Dernier scan : {scan_time}")
    if fresh_count:
        col_fresh.success(f"🔔 {fresh_count} nouveau(x) signal(aux) → Telegram")
    elif auto_scan:
        col_fresh.caption("Aucun nouveau signal.")

    # ── Métriques ──────────────────────────────────────────────
    signals      = [r for r in results if r.signal]
    buy_sig      = [r for r in signals if r.direction == "BUY"]
    sell_sig     = [r for r in signals if r.direction == "SELL"]
    pullback_now = [r for r in signals if r.in_pullback_zone]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Analysés",         len(results))
    c2.metric("🔔 Signaux",       len(signals))
    c3.metric("📈 BUY",           len(buy_sig))
    c4.metric("📉 SELL",          len(sell_sig))
    c5.metric("🎯 Pullback actif", len(pullback_now))

    st.divider()

    # ── Cartes signaux ─────────────────────────────────────────
    if signals:
        st.subheader(f"🔔 Setups CEP actifs — score ≥ {params['min_score']}/8")

        for r in signals:
            phase  = _phase_badge(r)
            header = (
                f"{_dir_emoji(r.direction)} **{r.instrument}**  ·  "
                f"{_score_badge(r.score)}  ·  {r.direction}  "
                + (f"  ·  {phase}" if phase else "")
            )
            with st.expander(header, expanded=True):

                # Alerte visuelle si pullback actif
                if r.in_pullback_zone:
                    st.success("🎯 **PULLBACK EN COURS** — Surveiller une entrée imminente sur EMA20/50")
                elif r.pullback_pending:
                    st.info("⏳ Expansion confirmée — Attendre le retour du prix sur EMA20 ou EMA50")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Prix actuel",  f"{r.current_price:.5f}")
                col2.metric("EMA20 H4",     f"{r.ema20_h4:.5f}")
                col3.metric("EMA50 H4",     f"{r.ema50_h4:.5f}")
                col4.metric("ATR H4",       f"{r.atr_h4:.5f}")
                col5.metric("Volume ×",     f"{r.volume_ratio:.1f}×")

                st.markdown("---")
                ca, cb = st.columns(2)

                with ca:
                    st.markdown("**Conditions :**")
                    st.write(f"{'✅' if r.trend_d1 else '❌'}  Tendance D1 ({r.d1_alignment_bars}b)")
                    st.write(f"{'✅' if r.d1_ema_slope_ok else '❌'}  Pente EMA D1")
                    st.write(f"{'✅' if r.compression_detected else '❌'}  Compression H4 ({r.compression_bars}b)")
                    st.write(f"{'✅' if r.atr_contracted else '❌'}  ATR contracté")
                    st.write(f"{'✅' if r.expansion_detected else '❌'}  Expansion H4 ({r.expansion_bars}b, âge {r.expansion_age}b)")
                    st.write(f"{'✅' if r.volume_confirmed else '❌'}  Volume confirmé ({r.volume_ratio:.1f}×)")
                    st.write(f"{'✅' if r.ema_slope_ok else '❌'}  Pente EMA H4")

                with cb:
                    st.markdown("**Score détaillé :**")
                    for label, pts in r.score_details:
                        icon = f"`+{pts}`" if pts > 0 else "`  0`"
                        st.write(f"  {icon}  {label}")
    else:
        st.info(
            f"Aucun setup CEP avec score ≥ {params['min_score']}/8.  \n"
            "Abaissez le score minimum ou attendez un prochain cycle."
        )

    # ── Tableau complet ────────────────────────────────────────
    st.divider()
    with st.expander("📋 Tous les résultats du scan", expanded=False):
        rows = []
        for r in results:
            rows.append({
                "Instrument":   r.instrument,
                "Direction":    r.direction,
                "Score":        r.score,
                "Signal":       "🔔" if r.signal else "—",
                "Phase":        _phase_badge(r) if r.signal else "—",
                "D1 Trend":     f"✅ {r.d1_alignment_bars}b" if r.trend_d1 else "❌",
                "D1 Pente":     "✅" if r.d1_ema_slope_ok else "❌",
                "Compression":  f"✅ {r.compression_bars}b" if r.compression_detected else "❌",
                "ATR Contracté":"✅" if r.atr_contracted else "❌",
                "Expansion":    f"✅ {r.expansion_bars}b (âge {r.expansion_age}b)" if r.expansion_detected else "❌",
                "Volume":       f"{r.volume_ratio:.1f}×",
                "EMA20 H4":     f"{r.ema20_h4:.5f}" if r.ema20_h4 else "—",
                "EMA50 H4":     f"{r.ema50_h4:.5f}" if r.ema50_h4 else "—",
                "ATR H4":       f"{r.atr_h4:.5f}" if r.atr_h4 else "—",
                "Info":         r.error_msg or "OK",
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


if __name__ == "__main__":
    main()
