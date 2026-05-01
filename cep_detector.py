#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   CEP DETECTOR v3  —  Compression → Expansion → Pullback        ║
║   Single-file · Oanda API · Streamlit-ready · GitHub-safe       ║
╠══════════════════════════════════════════════════════════════════╣
║  Corrections v3 (audit final) :                                  ║
║   FIX 1  d1_atr_trending toujours False → supprimé              ║
║   FIX 2  dist_ema50 calculé mais jamais utilisé → EMA50 active  ║
║   FIX 3  Pullback zone vérifie la direction de mouvement        ║
║   FIX 4  2ème alerte Telegram : pending → pullback actif        ║
║   FIX 5  Client Oanda créé une seule fois par scan              ║
║   FIX 6  Variable `last` inutilisée dans _calc_score → retirée ║
║   FIX 7  Score capé à 8 explicitement                           ║
║   FIX 8  Tableau avec filtre direction + tri interactif         ║
║   FIX 9  Import Optional inutilisé → retiré                     ║
╚══════════════════════════════════════════════════════════════════╝

Déploiement Streamlit Cloud :
  → Settings > Secrets :
      OANDA_ACCESS_TOKEN = "votre-token"
      OANDA_ACCOUNT_ID   = "votre-compte"
      OANDA_ENVIRONMENT  = "practice"
      TELEGRAM_TOKEN     = "123456:ABC..."
      TELEGRAM_CHAT_ID   = "123456789"

Usage local :
  streamlit run cep_detector.py
"""

__version__ = "3.0.0"
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

# ── Paramètres CEP ───────────────────────────────────────────────
DEFAULT_CEP_PARAMS = {
    # Compression
    "atr_contraction_ratio":    0.75,   # ATR actuel < ATR passé × ratio
    "compression_atr_ratio":    0.50,   # spread_norm < seuil
    "compression_min_candles":  6,
    "atr_lookback":             20,     # bougies H4 pour mesurer contraction ATR
    # Expansion
    "expansion_atr_ratio":      1.00,   # spread_norm > seuil
    "expansion_min_candles":    2,
    "expansion_max_age":        6,      # fraîcheur max (bougies H4)
    "volume_ratio_min":         1.2,    # volume > moyenne × ratio
    # Tendance D1
    "d1_alignment_min_candles": 5,
    # Pullback
    "pullback_distance_min":    0.15,   # distance min EMA20 en × ATR (pending)
    "pullback_ema50_zone":      3.0,    # prix dans N×ATR autour de EMA50 = zone profonde
    "pullback_direction_bars":  2,      # N dernières bougies doivent aller vers l'EMA
    # Signal
    "min_score":    5,
    "candles_d1":   120,
    "candles_h4":   200,
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
    """
    FIX 5 : le client est passé en paramètre (créé une seule fois
    par scan dans run_scanner) au lieu d'être recréé à chaque appel.
    """
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
) -> tuple:
    df_d1 = fetch_candles(client, instrument, "D",  params["candles_d1"])
    df_h4 = fetch_candles(client, instrument, "H4", params["candles_h4"])
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colonnes ajoutées :
      ema5/9/13/20/50  — alignement tendance
      atr14            — volatilité courante
      spread           — |ema5 - ema50| brut
      spread_norm      — spread / atr14  (comparable cross-assets)
      volume_ma20      — moyenne volume 20 périodes
      volume_ratio     — volume / volume_ma20
    """
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
# § 4 — CEP ENGINE v3
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CEPResult:
    """
    FIX 1 : d1_atr_trending supprimé (était déclaré mais jamais renseigné).
    FIX 9 : Optional retiré des imports (inutilisé).
    """
    instrument:           str
    direction:            str    # "BUY" | "SELL" | "NONE" | "ERROR"
    score:                int    # 0–8  (capé explicitement)
    signal:               bool

    # ── Tendance D1 ─────────────────────────────────────────────
    trend_d1:             bool  = False
    d1_alignment_bars:    int   = 0
    d1_ema_slope_ok:      bool  = False

    # ── Compression H4 ──────────────────────────────────────────
    compression_detected: bool  = False
    compression_bars:     int   = 0
    atr_contracted:       bool  = False

    # ── Expansion H4 ────────────────────────────────────────────
    expansion_detected:   bool  = False
    expansion_bars:       int   = 0
    expansion_age:        int   = 0
    volume_confirmed:     bool  = False

    # ── Zone de pullback ────────────────────────────────────────
    pullback_pending:     bool  = False  # expansion confirmée, prix loin des EMAs
    in_pullback_zone:     bool  = False  # prix revient vers EMA20 (direction confirmée)
    near_ema50_zone:      bool  = False  # FIX 2 : prix proche EMA50 (pullback profond)

    # ── Pentes ──────────────────────────────────────────────────
    ema_slope_ok:         bool  = False  # pente EMA H4
    ema_h4_slope_ok:      bool  = False  # alias explicite utilisé dans le score

    # ── Niveaux ─────────────────────────────────────────────────
    ema20_h4:      float = 0.0
    ema50_h4:      float = 0.0
    current_price: float = 0.0
    spread_norm:   float = 0.0
    atr_h4:        float = 0.0
    volume_ratio:  float = 0.0

    score_details: list  = field(default_factory=list)
    error_msg:     str   = ""

    # ── Telegram ────────────────────────────────────────────────
    def to_telegram_signal(self) -> str:
        """Message initial : nouveau signal CEP détecté."""
        if not self.signal:
            return ""
        emoji   = "📈" if self.direction == "BUY" else "📉"
        phase   = "🎯 Pullback en cours" if self.in_pullback_zone else "⏳ Attendre le pullback"
        age_txt = f"{self.expansion_age}b H4" if self.expansion_age else "—"
        lines = [
            f"🔔 *NOUVEAU SETUP CEP — {self.instrument}*",
            f"Direction : {emoji} *{self.direction}*  |  Score : *{self.score}/8*",
            f"Phase     : {phase}",
            "",
            f"{'✅' if self.trend_d1 else '❌'} Tendance D1 ({self.d1_alignment_bars}b) {'+ pente ✅' if self.d1_ema_slope_ok else ''}",
            f"{'✅' if self.compression_detected else '❌'} Compression ({self.compression_bars}b) {'+ ATR ✅' if self.atr_contracted else ''}",
            f"{'✅' if self.expansion_detected else '❌'} Expansion ({self.expansion_bars}b, âge {age_txt})",
            f"{'✅' if self.volume_confirmed else '❌'} Volume : {self.volume_ratio:.1f}× moyenne",
            f"{'✅' if self.ema_slope_ok else '❌'} Pente EMA H4",
            "",
            f"📍 Prix    : `{self.current_price:.5f}`",
            f"🎯 EMA20 H4 : `{self.ema20_h4:.5f}`",
            f"🎯 EMA50 H4 : `{self.ema50_h4:.5f}`",
            f"📊 ATR H4  : `{self.atr_h4:.5f}`",
            "",
            "_Attendre le pullback sur EMA20 ou EMA50 avant d'entrer._",
        ]
        return "\n".join(lines)

    def to_telegram_pullback(self) -> str:
        """
        FIX 4 : 2ème alerte envoyée quand le pullback devient actif
        (transition pending → in_pullback_zone ou near_ema50_zone).
        C'est le moment d'entrer en alerte maximale sur le graphique.
        """
        if not self.signal:
            return ""
        emoji = "📈" if self.direction == "BUY" else "📉"
        zone  = "EMA50 (pullback profond)" if self.near_ema50_zone else "EMA20"
        lines = [
            f"🎯 *PULLBACK EN COURS — {self.instrument}* {emoji}",
            f"Le prix approche la zone d'entrée : *{zone}*",
            "",
            f"📍 Prix actuel  : `{self.current_price:.5f}`",
            f"🎯 EMA20 H4    : `{self.ema20_h4:.5f}`",
            f"🎯 EMA50 H4    : `{self.ema50_h4:.5f}`",
            f"📊 ATR H4      : `{self.atr_h4:.5f}`",
            f"Score setup    : *{self.score}/8*",
            "",
            "⚡ *Ouvrir le graphique et surveiller une bougie de confirmation.*",
        ]
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────

def _count_d1_alignment(df_d1: pd.DataFrame, direction: str, min_bars: int) -> tuple:
    """Compte les bougies D1 consécutives avec les EMAs alignées."""
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
    Compression double confirmation :
      1. spread_norm < seuil (EMAs serrées)
      2. Spread progressivement décroissant (≥ 55% des pas)
      3. Durée ≥ min_candles
      4. ATR actuel < ATR de N bougies en arrière × ratio (volatilité contractante)

    Returns (detected: bool, atr_contracted: bool, bar_count: int)
    """
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
    """
    Expansion avec confirmation ATR, volume et fraîcheur.

    Returns (detected: bool, bar_count: int, age: int, volume_confirmed: bool)
    """
    threshold = params["expansion_atr_ratio"]
    min_bars  = params["expansion_min_candles"]
    max_age   = params["expansion_max_age"]
    vol_min   = params["volume_ratio_min"]

    count = 0
    for i in range(len(df) - 1, max(len(df) - max_age - 5, 1), -1):
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1]

        spread_ok = row.spread_norm > threshold
        atr_ok    = row.atr14 >= prev_row.atr14  # ATR stable ou croissant

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


def _detect_pullback_zone(
    df:        pd.DataFrame,
    direction: str,
    params:    dict,
) -> tuple:
    """
    FIX 2 + FIX 3 — Détection de zone de pullback améliorée.

    Trois états possibles :
      pullback_pending   : expansion fraîche, prix encore loin des EMAs
      in_pullback_zone   : prix s'approche de EMA20 ET se déplace VERS elle
      near_ema50_zone    : FIX 2 — prix s'approche de EMA50 (pullback profond)

    FIX 3 : on vérifie que les N dernières bougies vont dans la bonne direction
    (vers l'EMA, pas en s'en éloignant).

    Returns (pullback_pending: bool, in_pullback_zone: bool, near_ema50_zone: bool)
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
    dist_ema50 = abs(price - ema50)   # FIX 2 : maintenant utilisé

    # ── FIX 3 : direction de mouvement des N dernières bougies ──
    # BUY  : après expansion haussière, le pullback = prix baisse
    # SELL : après expansion baissière, le pullback = prix monte
    moving_toward_ema = False
    if len(df) >= n_bars + 1:
        recent_closes = df["close"].iloc[-(n_bars + 1):]
        if direction == "BUY":
            # Prix doit baisser (closes décroissants)
            moving_toward_ema = recent_closes.iloc[-1] < recent_closes.iloc[0]
        else:
            # Prix doit monter (closes croissants)
            moving_toward_ema = recent_closes.iloc[-1] > recent_closes.iloc[0]

    if direction == "BUY":
        above_ema20 = price > ema20
        # Prix loin d'EMA20, mouvement non encore engagé
        pullback_pending = above_ema20 and dist_ema20 > atr * dist_min

        # FIX 3 : zone EMA20 ET mouvement confirmé vers elle
        in_pullback_zone = (
            above_ema20
            and dist_ema20 <= atr * 1.5
            and dist_ema20 >= atr * 0.05
            and moving_toward_ema
        )

        # FIX 2 : prix entre EMA20 et EMA50, pullback profond
        near_ema50_zone = (
            price <= ema20                     # a dépassé EMA20 vers le bas
            and dist_ema50 <= atr * ema50_thr  # mais encore proche d'EMA50
            and moving_toward_ema
        )

    else:  # SELL
        below_ema20 = price < ema20
        pullback_pending = below_ema20 and dist_ema20 > atr * dist_min

        # FIX 3 : zone EMA20 ET mouvement confirmé vers elle
        in_pullback_zone = (
            below_ema20
            and dist_ema20 <= atr * 1.5
            and dist_ema20 >= atr * 0.05
            and moving_toward_ema
        )

        # FIX 2 : pullback profond côté vendeur
        near_ema50_zone = (
            price >= ema20
            and dist_ema50 <= atr * ema50_thr
            and moving_toward_ema
        )

    return pullback_pending, in_pullback_zone, near_ema50_zone


def _calc_score(result: "CEPResult", df_h4: pd.DataFrame, params: dict) -> tuple:
    """
    Score 0–8, sans redondance.

    +1  Tendance D1 longue (≥ 10 bougies)
    +1  Pente EMA20/50 D1 confirmée
    +1  Compression confirmée spread + ATR contracté
    +1  Compression très longue (≥ 2× min_candles) = coil puissant
    +1  Expansion fraîche (âge ≤ 3 bougies)
    +1  Volume expansion > moyenne
    +1  Pente EMA20/50 H4 confirmée
    +1  Prix en zone pullback actif (EMA20 ou EMA50)

    FIX 6 : variable `last` supprimée (était assignée mais jamais utilisée).
    FIX 7 : score capé à 8 explicitement avec min().
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
    elif result.compression_detected:
        details.append((f"Compression spread seule ({result.compression_bars}b) — ATR non contracté", 0))

    # +1 Compression longue
    if result.compression_bars >= params["compression_min_candles"] * 2:
        score += 1
        details.append((f"Compression longue ({result.compression_bars}b ≥ {params['compression_min_candles'] * 2}b)", 1))

    # +1 Expansion fraîche
    if result.expansion_detected and result.expansion_age <= 3:
        score += 1
        details.append((f"Expansion fraîche ({result.expansion_age}b)", 1))
    elif result.expansion_detected:
        details.append((f"Expansion âgée ({result.expansion_age}b > 3) — setup moins frais", 0))

    # +1 Volume
    if result.volume_confirmed:
        score += 1
        details.append((f"Volume élevé ({result.volume_ratio:.1f}× moyenne)", 1))

    # +1 Pente H4  — FIX 6 : plus de variable `last` inutilisée
    if result.direction == "BUY":
        slope_ok = (df_h4["ema20"].iloc[-1] > df_h4["ema20"].iloc[-2] and
                    df_h4["ema50"].iloc[-1] > df_h4["ema50"].iloc[-2])
    else:
        slope_ok = (df_h4["ema20"].iloc[-1] < df_h4["ema20"].iloc[-2] and
                    df_h4["ema50"].iloc[-1] < df_h4["ema50"].iloc[-2])
    result.ema_slope_ok    = slope_ok
    result.ema_h4_slope_ok = slope_ok
    if slope_ok:
        score += 1
        details.append(("Pente EMA20/50 H4 confirmée", 1))

    # +1 Zone pullback actif (EMA20 ou EMA50)
    if result.in_pullback_zone or result.near_ema50_zone:
        score += 1
        zone_lbl = "EMA50 (profond)" if result.near_ema50_zone else "EMA20"
        details.append((f"Pullback actif vers {zone_lbl} ← EN ALERTE", 1))
    elif result.pullback_pending:
        details.append(("Pullback pas encore amorcé — surveiller", 0))

    # FIX 7 : cap explicite à 8
    score = min(score, 8)

    return score, details


# ── Moteur principal ─────────────────────────────────────────────

def run_cep_engine(
    instrument: str,
    df_d1:      pd.DataFrame,
    df_h4:      pd.DataFrame,
    params:     dict,
) -> CEPResult:
    """
    Pipeline CEP v3 :
      1. Filtre D1 (alignement + pente bloquante + durée)
      2. Compression H4 (spread + ATR contractant)
      3. Expansion H4 (spread + ATR + volume + fraîcheur)
      4. Zone pullback (EMA20, EMA50, direction de mouvement)
      5. Score 0-8 non redondant, capé
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
    result.volume_ratio  = float(last_h4.get("volume_ratio", 0.0))

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
        result.error_msg = f"Tendance D1 trop récente ({align_bars}b < {params['d1_alignment_min_candles']}b)"
        return result

    # ── 2. Compression H4 ────────────────────────────────────────
    offset     = max(params["expansion_min_candles"] + 1, 3)
    df_h4_comp = df_h4.iloc[:-offset]
    comp_ok, atr_contracted, comp_bars = _detect_compression(df_h4_comp, params)
    result.compression_detected = comp_ok
    result.atr_contracted       = atr_contracted
    result.compression_bars     = comp_bars
    if not comp_ok:
        result.error_msg = f"Compression insuffisante ({comp_bars}b)"
        return result

    # ── 3. Expansion H4 ──────────────────────────────────────────
    exp_ok, exp_bars, exp_age, vol_ok = _detect_expansion(df_h4, params, result.direction)
    result.expansion_detected = exp_ok
    result.expansion_bars     = exp_bars
    result.expansion_age      = exp_age
    result.volume_confirmed   = vol_ok
    if not exp_ok:
        if exp_age > params["expansion_max_age"] and exp_bars >= params["expansion_min_candles"]:
            result.error_msg = f"Expansion trop ancienne ({exp_age}b > {params['expansion_max_age']}b)"
        else:
            result.error_msg = "Expansion absente"
        return result

    # ── 4. Zone pullback ─────────────────────────────────────────
    pb_pending, pb_active, pb_ema50 = _detect_pullback_zone(df_h4, result.direction, params)
    result.pullback_pending  = pb_pending
    result.in_pullback_zone  = pb_active
    result.near_ema50_zone   = pb_ema50

    # ── 5. Score ─────────────────────────────────────────────────
    result.score, result.score_details = _calc_score(result, df_h4, params)
    result.signal = result.score >= params["min_score"]

    return result


# ═══════════════════════════════════════════════════════════════════
# § 5 — SCANNER
# ═══════════════════════════════════════════════════════════════════

def run_scanner(
    access_token:     str,
    environment:      str,
    instruments_list: list,
    params:           dict,
    progress_callback=None,
) -> list:
    """
    FIX 5 : le client Oanda est créé UNE SEULE FOIS ici
    puis passé à chaque appel fetch_candles (vs 64 créations avant).
    """
    results = []
    total   = len(instruments_list)

    # FIX 5 : instance unique du client
    client = oandapyV20.API(access_token=access_token, environment=environment)

    for i, instrument in enumerate(instruments_list):
        if progress_callback:
            progress_callback(i / total, f"Analyse de {instrument}…")
        try:
            df_d1, df_h4 = fetch_multi_timeframe(client, instrument, params)
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
        time.sleep(0.35)  # respect rate-limit Oanda (~3 req/s)

    if progress_callback:
        progress_callback(1.0, "Scan terminé.")

    # Tri : pullback actif > signal > score
    results.sort(
        key=lambda r: (
            r.in_pullback_zone or r.near_ema50_zone,
            r.signal,
            r.score,
        ),
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
    if r.near_ema50_zone:   return "🎯 Pullback EMA50"
    if r.in_pullback_zone:  return "🎯 Pullback EMA20"
    if r.pullback_pending:  return "⏳ Attendre pullback"
    return ""


def main():
    st.set_page_config(
        page_title="CEP Detector",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg          = get_config()
    access_token = cfg["access_token"]
    environment  = cfg["environment"]
    tg_token     = cfg["telegram_token"]
    tg_chat_id   = cfg["telegram_chat_id"]

    st.title("📊 CEP Detector v3")
    st.markdown(
        "**Compression → Expansion → Pullback** — Signal haute qualité.  "
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
        min_score  = st.slider("Score minimum (sur 8)", 3, 8,
                               DEFAULT_CEP_PARAMS["min_score"])
        comp_ratio = st.slider("Seuil compression (× ATR)", 0.2, 1.0, step=0.05,
                               value=DEFAULT_CEP_PARAMS["compression_atr_ratio"])
        exp_ratio  = st.slider("Seuil expansion (× ATR)", 0.5, 2.5, step=0.1,
                               value=DEFAULT_CEP_PARAMS["expansion_atr_ratio"])
        comp_min   = st.slider("Min bougies compression", 3, 20,
                               DEFAULT_CEP_PARAMS["compression_min_candles"])
        atr_ratio  = st.slider("Ratio contraction ATR", 0.5, 0.95, step=0.05,
                               value=DEFAULT_CEP_PARAMS["atr_contraction_ratio"],
                               help="ATR actuel < ATR passé × ratio = compression réelle")
        exp_age    = st.slider("Fraîcheur expansion max (H4)", 2, 12,
                               DEFAULT_CEP_PARAMS["expansion_max_age"])
        vol_ratio  = st.slider("Volume min (× moyenne)", 1.0, 2.5, step=0.1,
                               value=DEFAULT_CEP_PARAMS["volume_ratio_min"])
        d1_min     = st.slider("Min bougies tendance D1", 3, 20,
                               DEFAULT_CEP_PARAMS["d1_alignment_min_candles"])

        params = {
            **DEFAULT_CEP_PARAMS,
            "min_score":                min_score,
            "compression_atr_ratio":    comp_ratio,
            "expansion_atr_ratio":      exp_ratio,
            "compression_min_candles":  comp_min,
            "atr_contraction_ratio":    atr_ratio,
            "expansion_max_age":        exp_age,
            "volume_ratio_min":         vol_ratio,
            "d1_alignment_min_candles": d1_min,
        }

        st.divider()
        st.subheader("⏱️ Scan automatique")
        auto_scan        = st.toggle("Activer le scan automatique", value=False)
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
        st_autorefresh(interval=refresh_interval * 60 * 1000, key="cep_autorefresh")
        scan_btn = True
    elif auto_scan and not AUTOREFRESH_AVAILABLE:
        st.warning("`streamlit-autorefresh` non installé. Ajouter dans requirements.txt.")

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

        # ── Telegram — logique FIX 4 ───────────────────────────
        # Clé signal   : f"{instrument}_{direction}"
        # Clé pullback : f"{instrument}_{direction}_pb"
        # → 2 alertes distinctes, aucun doublon

        new_signals  = [r for r in results if r.signal]
        prev_sig     = st.session_state.get("prev_signal_keys",   set())
        prev_pb      = st.session_state.get("prev_pullback_keys", set())

        cur_sig = {f"{r.instrument}_{r.direction}" for r in new_signals}
        cur_pb  = {
            f"{r.instrument}_{r.direction}_pb"
            for r in new_signals
            if r.in_pullback_zone or r.near_ema50_zone
        }

        # Alerte 1 : nouveau signal
        fresh_signals  = [r for r in new_signals
                          if f"{r.instrument}_{r.direction}" not in prev_sig]
        # Alerte 2 : pullback devenu actif (FIX 4)
        fresh_pullbacks = [r for r in new_signals
                           if f"{r.instrument}_{r.direction}_pb" not in prev_pb
                           and (r.in_pullback_zone or r.near_ema50_zone)]

        if tg_token and tg_chat_id:
            for r in fresh_signals:
                send_telegram(tg_token, tg_chat_id, r.to_telegram_signal())
            for r in fresh_pullbacks:
                # Éviter doublon si c'est aussi un nouveau signal
                if r not in fresh_signals:
                    send_telegram(tg_token, tg_chat_id, r.to_telegram_pullback())

        st.session_state["prev_signal_keys"]   = cur_sig
        st.session_state["prev_pullback_keys"] = cur_pb
        st.session_state["results"]            = results
        st.session_state["scan_time"]          = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.session_state["params"]             = params
        st.session_state["fresh_sig_count"]    = len(fresh_signals)
        st.session_state["fresh_pb_count"]     = len(fresh_pullbacks)

    if "results" not in st.session_state:
        st.markdown("### ← Cliquez sur **Lancer le scan** pour démarrer.")
        return

    results        = st.session_state["results"]
    scan_time      = st.session_state.get("scan_time", "")
    fresh_sig_cnt  = st.session_state.get("fresh_sig_count", 0)
    fresh_pb_cnt   = st.session_state.get("fresh_pb_count", 0)

    col_time, col_sig, col_pb = st.columns([2, 1, 1])
    col_time.caption(f"Dernier scan : {scan_time}")
    if fresh_sig_cnt:
        col_sig.success(f"🔔 {fresh_sig_cnt} nouveau(x) signal(aux)")
    if fresh_pb_cnt:
        col_pb.warning(f"🎯 {fresh_pb_cnt} pullback(s) actif(s) → Telegram")

    # ── Métriques ──────────────────────────────────────────────
    signals      = [r for r in results if r.signal]
    buy_sig      = [r for r in signals if r.direction == "BUY"]
    sell_sig     = [r for r in signals if r.direction == "SELL"]
    pullback_now = [r for r in signals if r.in_pullback_zone or r.near_ema50_zone]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Analysés",          len(results))
    c2.metric("🔔 Signaux",        len(signals))
    c3.metric("📈 BUY",            len(buy_sig))
    c4.metric("📉 SELL",           len(sell_sig))
    c5.metric("🎯 Pullback actif",  len(pullback_now))

    st.divider()

    # ── Cartes signaux ─────────────────────────────────────────
    if signals:
        st.subheader(f"🔔 Setups CEP — score ≥ {params['min_score']}/8")

        for r in signals:
            phase  = _phase_badge(r)
            header = (
                f"{_dir_emoji(r.direction)} **{r.instrument}**  ·  "
                f"{_score_badge(r.score)}  ·  {r.direction}"
                + (f"  ·  {phase}" if phase else "")
            )
            with st.expander(header, expanded=True):

                if r.near_ema50_zone:
                    st.warning("🎯 **PULLBACK EMA50** — Zone d'entrée profonde atteinte. Vérifier sur graphique.")
                elif r.in_pullback_zone:
                    st.success("🎯 **PULLBACK EMA20 EN COURS** — Prix en retour confirmé. Surveiller l'entrée.")
                elif r.pullback_pending:
                    st.info("⏳ Expansion confirmée — Attendre le retour du prix sur EMA20 ou EMA50.")

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
                    st.write(f"{'✅' if r.volume_confirmed else '❌'}  Volume ({r.volume_ratio:.1f}×)")
                    st.write(f"{'✅' if r.ema_slope_ok else '❌'}  Pente EMA H4")

                with cb:
                    st.markdown("**Score détaillé :**")
                    for label, pts in r.score_details:
                        icon = f"`+{pts}`" if pts > 0 else "` 0`"
                        st.write(f"  {icon}  {label}")

    else:
        st.info(
            f"Aucun setup CEP avec score ≥ {params['min_score']}/8.  \n"
            "Abaissez le score minimum ou attendez un prochain cycle."
        )

    # ── Tableau complet — FIX 8 : filtre + tri interactif ──────
    st.divider()
    with st.expander("📋 Tous les résultats du scan", expanded=False):

        # FIX 8 : filtres interactifs
        fc1, fc2 = st.columns(2)
        dir_filter   = fc1.selectbox("Filtrer direction",
                                     ["Tous", "BUY", "SELL", "NONE", "ERROR"])
        phase_filter = fc2.selectbox("Filtrer phase",
                                     ["Tous", "Signal", "Pullback actif", "En attente"])

        rows = []
        for r in results:
            # Application des filtres
            if dir_filter != "Tous" and r.direction != dir_filter:
                continue
            if phase_filter == "Signal" and not r.signal:
                continue
            if phase_filter == "Pullback actif" and not (r.in_pullback_zone or r.near_ema50_zone):
                continue
            if phase_filter == "En attente" and not r.pullback_pending:
                continue

            rows.append({
                "Instrument":    r.instrument,
                "Direction":     r.direction,
                "Score":         r.score,
                "Signal":        "🔔" if r.signal else "—",
                "Phase":         _phase_badge(r) if r.signal else "—",
                "D1 Trend":      f"✅ {r.d1_alignment_bars}b" if r.trend_d1 else "❌",
                "D1 Pente":      "✅" if r.d1_ema_slope_ok else "❌",
                "Compression":   f"✅ {r.compression_bars}b" if r.compression_detected else "❌",
                "ATR Contracté": "✅" if r.atr_contracted else "❌",
                "Expansion":     f"✅ {r.expansion_bars}b (âge {r.expansion_age}b)" if r.expansion_detected else "❌",
                "Volume":        f"{r.volume_ratio:.1f}×",
                "EMA20 H4":      f"{r.ema20_h4:.5f}" if r.ema20_h4 else "—",
                "EMA50 H4":      f"{r.ema50_h4:.5f}" if r.ema50_h4 else "—",
                "ATR H4":        f"{r.atr_h4:.5f}" if r.atr_h4 else "—",
                "Info":          r.error_msg or "OK",
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
        else:
            st.caption("Aucun résultat correspondant aux filtres.")


if __name__ == "__main__":
    main()
  
