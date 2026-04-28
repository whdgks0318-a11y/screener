"""
pattern_detector.py
====================
추세추종 패턴 감지 모듈
윌리엄 오닐 + 마크 미너비니 방법론 기반

감지 패턴:
  1. VCP (변동성 축소 패턴)
  2. 컵앤핸들 (Cup & Handle)
  3. 이중 바닥형 (Double Bottom)
  4. 플랫 베이스 (Flat Base)
  5. 어센딩 베이스 (Ascending Base)
  6. 하이 타이트 플래그 (High Tight Flag)
  7. 소서 앤 핸들 (Saucer with Handle)
  8. RS Line 신고가 선행
  9. 어닝 갭업 후 유지 (Earnings Gap Up)
 10. 파워 플레이 (Power Play)
 11. 거래량 급증 돌파 (Volume Breakout)
 12. S1→S2 전환 감지

함정 필터:
  - V자형 급반등 (매물 소화 실패)
  - 하단 핸들 (Lower Half Handle)
  - 거래량 없는 돌파 (False Breakout)
"""

import numpy as np
import pandas as pd
from typing import Optional


def detect_all_patterns(df: pd.DataFrame) -> dict:
    """
    전체 패턴 감지 실행
    df: OHLCV DataFrame (close, high, low, open, volume 컬럼 필요)
    반환: 각 패턴별 감지 결과 dict
    """
    if df is None or len(df) < 50:
        return _empty_result()

    close  = df["Close"].astype(float).values
    high   = df["High"].astype(float).values
    low    = df["Low"].astype(float).values
    volume = df["Volume"].astype(float).values
    # open이 없을 수도 있으니 안전하게 처리
    open_  = df["Open"].astype(float).values if "Open" in df.columns else close.copy()

    n = len(close)

    results = {
        "vcp":            detect_vcp(close, high, low, volume, n),
        "cup_handle":     detect_cup_handle(close, high, low, volume, n),
        "double_bottom":  detect_double_bottom(close, low, volume, n),
        "flat_base":      detect_flat_base(close, high, low, volume, n),
        "ascending_base": detect_ascending_base(close, low, n),
        "htf":            detect_htf(close, high, low, volume, n),
        "saucer_handle":  detect_saucer_handle(close, low, volume, n),
        "rs_line_lead":   detect_rs_line_lead(close, high, n),
        "earnings_gap":   detect_earnings_gap(close, high, low, open_, volume, n),
        "power_play":     detect_power_play(close, open_, volume, n),
        "vol_breakout":   detect_vol_breakout(close, high, volume, n),
        "s1_to_s2":       detect_s1_to_s2(close, high, low, volume, n),
    }

    # 감지된 패턴 수
    detected = [k for k, v in results.items() if v.get("detected")]
    results["pattern_count"] = len(detected)
    results["patterns"]      = detected
    results["best_pattern"]  = detected[0] if detected else None

    return results


def _empty_result() -> dict:
    return {
        "vcp": {"detected": False}, "cup_handle": {"detected": False},
        "double_bottom": {"detected": False}, "flat_base": {"detected": False},
        "ascending_base": {"detected": False}, "htf": {"detected": False},
        "saucer_handle": {"detected": False}, "rs_line_lead": {"detected": False},
        "earnings_gap": {"detected": False}, "power_play": {"detected": False},
        "vol_breakout": {"detected": False}, "s1_to_s2": {"detected": False},
        "pattern_count": 0, "patterns": [], "best_pattern": None,
    }


# ── 1. VCP (변동성 축소 패턴) ─────────────────────────────────────────────────

def detect_vcp(close, high, low, volume, n, lookback=60) -> dict:
    """
    변동성이 점점 줄어드는 2~4회 조정 패턴
    조건: T1 > T2 > T3 (조정폭 감소), 거래량 감소
    """
    result = {"detected": False, "name": "VCP", "signal": "vcp"}
    if n < lookback:
        return result

    c = close[-lookback:]
    h = high[-lookback:]
    l = low[-lookback:]
    v = volume[-lookback:]

    # 고점/저점 찾기
    pivots = _find_pivots(c, min_pct=3.0)
    if len(pivots) < 4:
        return result

    # 조정 구간 추출
    contractions = []
    i = 0
    while i < len(pivots) - 1:
        if pivots[i]["type"] == "high" and i + 1 < len(pivots) and pivots[i+1]["type"] == "low":
            depth = (pivots[i]["val"] - pivots[i+1]["val"]) / pivots[i]["val"] * 100
            vol_avg = np.mean(v[pivots[i]["idx"]:pivots[i+1]["idx"]+1]) if pivots[i+1]["idx"] > pivots[i]["idx"] else 0
            contractions.append({"depth": depth, "vol": vol_avg, "idx": pivots[i]["idx"]})
        i += 1

    if len(contractions) < 2:
        return result

    # 조정폭이 감소하는지 확인
    depths = [ct["depth"] for ct in contractions[-3:]]
    vols   = [ct["vol"]   for ct in contractions[-3:]]

    depth_shrinking = all(depths[i] > depths[i+1] for i in range(len(depths)-1))
    vol_shrinking   = all(vols[i] > vols[i+1] * 0.9 for i in range(len(vols)-1))

    # 마지막 조정이 3% 이내 (수렴)
    last_depth = depths[-1] if depths else 999
    tightness  = last_depth < 10

    # Stage 2 선행 조건: 패턴 전 상승 추세
    pre_trend = c[0] < c[-1] * 0.85 if len(c) > 20 else False

    if depth_shrinking and vol_shrinking and tightness:
        result["detected"] = True
        result["contractions"] = len(contractions)
        result["last_depth"]   = round(last_depth, 1)
        result["quality"]      = "high" if (pre_trend and len(contractions) >= 3) else "medium"
        result["desc"]         = f"VCP {len(contractions)}회 수렴 / 마지막 조정 {round(last_depth,1)}%"

    return result


# ── 2. 컵앤핸들 ───────────────────────────────────────────────────────────────

def detect_cup_handle(close, high, low, volume, n, lookback=130) -> dict:
    """
    컵앤핸들 패턴 감지
    컵: 7~65주, 깊이 20~50% (V자형 제외)
    핸들: 상단 절반에서 형성, 깊이 10~15%, 1주 이상
    """
    result = {"detected": False, "name": "컵앤핸들", "signal": "cup_handle"}
    if n < 50:
        return result

    lb = min(lookback, n)
    c  = close[-lb:]
    h  = high[-lb:]
    l  = low[-lb:]
    v  = volume[-lb:]
    m  = len(c)

    # 왼쪽 고점 탐색 (전체 구간의 앞 30%)
    left_end   = int(m * 0.35)
    left_high_idx = int(np.argmax(h[:left_end]))
    left_high_val = h[left_high_idx]

    # 컵 바닥 탐색 (중간 구간)
    cup_start = left_high_idx
    cup_end   = int(m * 0.75)
    if cup_end <= cup_start + 10:
        return result

    cup_low_idx = int(np.argmin(l[cup_start:cup_end])) + cup_start
    cup_low_val = l[cup_low_idx]

    # 컵 깊이 20~50%
    cup_depth = (left_high_val - cup_low_val) / left_high_val * 100
    if not (15 <= cup_depth <= 55):
        return result

    # V자형 제외: 바닥 구간이 최소 7개 봉 이상 평탄
    bottom_range = c[cup_low_idx:cup_low_idx+10]
    if len(bottom_range) < 5:
        return result
    bottom_std = np.std(bottom_range) / np.mean(bottom_range) * 100
    if bottom_std > 8:  # 너무 V자형이면 제외
        return result

    # 오른쪽 회복: 왼쪽 고점 근처까지 회복
    right_start = cup_low_idx
    right_end   = min(m - 5, int(m * 0.88))
    if right_end <= right_start:
        return result

    right_high_val = np.max(h[right_start:right_end])
    recovery = (right_high_val - cup_low_val) / (left_high_val - cup_low_val)
    if recovery < 0.75:  # 75% 이상 회복해야 컵 인정
        return result

    # 핸들 탐색 (마지막 10~15% 구간)
    handle_start = right_end
    handle_data  = c[handle_start:]
    if len(handle_data) < 5:
        return result

    handle_high = np.max(handle_data)
    handle_low  = np.min(handle_data)
    handle_depth = (handle_high - handle_low) / handle_high * 100

    # 핸들 깊이 5~20%
    if not (3 <= handle_depth <= 20):
        return result

    # 핸들이 베이스 상단 절반에 위치 (Lower Half 제외)
    base_mid = cup_low_val + (left_high_val - cup_low_val) * 0.5
    if handle_low < base_mid:  # 하단 핸들 → 실패 확률 높음
        result["detected"] = False
        result["pitfall"] = "하단핸들"
        return result

    # 핸들에서 거래량 감소 확인
    handle_vol = np.mean(v[handle_start:])
    cup_vol    = np.mean(v[cup_start:handle_start])
    vol_dry_up = handle_vol < cup_vol * 0.8

    # 현재 피봇 돌파 여부 (최근 2봉이 핸들 고점 근처)
    near_pivot = c[-1] >= handle_high * 0.97

    result["detected"]    = True
    result["cup_depth"]   = round(cup_depth, 1)
    result["handle_depth"]= round(handle_depth, 1)
    result["vol_dry_up"]  = vol_dry_up
    result["near_pivot"]  = near_pivot
    result["quality"]     = "high" if (vol_dry_up and near_pivot) else "medium"
    result["desc"]        = f"컵앤핸들 / 컵깊이 {round(cup_depth,1)}% / 핸들깊이 {round(handle_depth,1)}%"

    return result


# ── 3. 이중 바닥형 (Double Bottom) ───────────────────────────────────────────

def detect_double_bottom(close, low, volume, n, lookback=80) -> dict:
    """
    W형 패턴 — 두 번째 바닥이 첫 번째보다 낮아야 함 (언더컷)
    중간 고점 돌파 시 매수 신호
    """
    result = {"detected": False, "name": "이중바닥", "signal": "double_bottom"}
    if n < 30:
        return result

    lb = min(lookback, n)
    c  = close[-lb:]
    l  = low[-lb:]
    v  = volume[-lb:]
    m  = len(c)

    # 첫 번째 바닥 탐색 (앞 40%)
    first_half = int(m * 0.45)
    b1_idx = int(np.argmin(l[:first_half]))
    b1_val = l[b1_idx]

    # 중간 고점 (두 바닥 사이)
    mid_start = b1_idx + 3
    mid_end   = int(m * 0.75)
    if mid_end <= mid_start:
        return result

    mid_high_idx = int(np.argmax(c[mid_start:mid_end])) + mid_start
    mid_high_val = c[mid_high_idx]

    # 두 번째 바닥 탐색
    b2_start = mid_high_idx + 2
    b2_end   = m - 3
    if b2_end <= b2_start:
        return result

    b2_idx = int(np.argmin(l[b2_start:b2_end])) + b2_start
    b2_val = l[b2_idx]

    # 언더컷 필수: 두 번째 바닥 < 첫 번째 바닥
    undercut = b2_val < b1_val * 0.99
    if not undercut:
        return result

    # 두 바닥 깊이 비교 (너무 차이 나면 무효)
    depth_ratio = b2_val / b1_val
    if depth_ratio < 0.85:  # 너무 많이 언더컷하면 무효
        return result

    # 중간 고점 돌파 여부
    near_midpeak = c[-1] >= mid_high_val * 0.97

    # 두 번째 바닥 후 거래량 증가
    b2_vol = np.mean(v[b2_idx:b2_idx+5]) if b2_idx + 5 < m else np.mean(v[b2_idx:])
    base_vol = np.mean(v[b1_idx:b2_idx])
    vol_expand = b2_vol > base_vol * 0.8

    result["detected"]     = True
    result["b1_val"]       = round(b1_val, 2)
    result["b2_val"]       = round(b2_val, 2)
    result["undercut_pct"] = round((b1_val - b2_val) / b1_val * 100, 1)
    result["near_midpeak"] = near_midpeak
    result["quality"]      = "high" if near_midpeak else "medium"
    result["desc"]         = f"이중바닥 / 언더컷 {round((b1_val-b2_val)/b1_val*100,1)}%"

    return result


# ── 4. 플랫 베이스 ────────────────────────────────────────────────────────────

def detect_flat_base(close, high, low, volume, n, lookback=50) -> dict:
    """
    5주(25봉) 이상, 깊이 15% 미만 횡보
    시장 약세에도 버티는 Super Strength
    """
    result = {"detected": False, "name": "플랫베이스", "signal": "flat_base"}
    if n < 30:
        return result

    lb = min(lookback, n)
    c  = close[-lb:]
    h  = high[-lb:]
    l  = low[-lb:]
    v  = volume[-lb:]

    base_high = np.max(h)
    base_low  = np.min(l)
    depth     = (base_high - base_low) / base_high * 100

    # 깊이 15% 미만
    if depth > 15:
        return result

    # 기간 최소 25봉 (5주)
    if lb < 25:
        return result

    # 횡보 판단: 종가 표준편차 / 평균 < 5%
    price_cv = np.std(c) / np.mean(c) * 100
    if price_cv > 6:
        return result

    # 선행 상승 추세 확인 (패턴 전 상승했어야 함)
    if n > lb + 20:
        prior = close[-(lb+20):-lb]
        prior_trend = prior[-1] > prior[0] * 1.15
    else:
        prior_trend = True

    # 현재 고점 근처인지 (돌파 임박)
    near_high = close[-1] >= base_high * 0.97

    # 거래량 감소 (공급 고갈)
    early_vol = np.mean(v[:len(v)//2])
    late_vol  = np.mean(v[len(v)//2:])
    vol_dry   = late_vol < early_vol * 0.85

    result["detected"]  = True
    result["depth"]     = round(depth, 1)
    result["weeks"]     = round(lb / 5, 1)
    result["near_high"] = near_high
    result["vol_dry"]   = vol_dry
    result["quality"]   = "high" if (near_high and vol_dry and prior_trend) else "medium"
    result["desc"]      = f"플랫베이스 / 깊이 {round(depth,1)}% / {round(lb/5,1)}주"

    return result


# ── 5. 어센딩 베이스 ──────────────────────────────────────────────────────────

def detect_ascending_base(close, low, n, lookback=80) -> dict:
    """
    저점이 높아지는 3번의 계단식 상승
    각 풀백 3~5주, 10~20% 조정
    """
    result = {"detected": False, "name": "어센딩베이스", "signal": "ascending_base"}
    if n < 40:
        return result

    lb = min(lookback, n)
    c  = close[-lb:]
    l  = low[-lb:]

    # 피봇 포인트 탐색
    pivots = _find_pivots(c, min_pct=5.0)
    lows = [p for p in pivots if p["type"] == "low"]

    if len(lows) < 3:
        return result

    # 최근 3개 저점이 높아지는지 확인
    recent_lows = lows[-3:]
    ascending = all(
        recent_lows[i]["val"] < recent_lows[i+1]["val"]
        for i in range(len(recent_lows)-1)
    )

    if not ascending:
        return result

    # 각 조정 깊이 계산
    depths = []
    for i in range(len(recent_lows)):
        if i < len(pivots) - 1:
            for j in range(len(pivots)):
                if pivots[j] == recent_lows[i] and j > 0:
                    prior_high = pivots[j-1]["val"]
                    depth = (prior_high - recent_lows[i]["val"]) / prior_high * 100
                    depths.append(depth)
                    break

    # 조정 깊이 10~25% 범위
    valid_depths = all(8 <= d <= 30 for d in depths) if depths else True

    result["detected"]    = True
    result["low_count"]   = len(recent_lows)
    result["valid_depths"]= valid_depths
    result["quality"]     = "high" if valid_depths else "medium"
    result["desc"]        = f"어센딩베이스 / 저점 {len(recent_lows)}회 상승"

    return result


# ── 6. 하이 타이트 플래그 (HTF) ───────────────────────────────────────────────

def detect_htf(close, high, low, volume, n, lookback=60) -> dict:
    """
    8주(40봉) 이내 +100% 급등 후 10~25% 좁은 횡보
    가장 폭발적인 패턴 (댄 쟁거)
    """
    result = {"detected": False, "name": "하이타이트플래그", "signal": "htf"}
    if n < 20:
        return result

    lb = min(lookback, n)
    c  = close[-lb:]
    h  = high[-lb:]
    l  = low[-lb:]
    v  = volume[-lb:]

    # 최근 40봉(8주) 내 급등 구간 탐색
    for pole_len in range(10, min(42, lb-8)):
        pole_low  = np.min(l[:pole_len])
        pole_high = np.max(h[:pole_len])
        pole_gain = (pole_high - pole_low) / pole_low * 100

        if pole_gain < 80:  # 80% 이상 급등
            continue

        # 플래그 구간 (급등 후 나머지)
        flag_data = c[pole_len:]
        if len(flag_data) < 5:
            continue

        flag_high = np.max(flag_data)
        flag_low  = np.min(flag_data)
        flag_depth = (flag_high - flag_low) / flag_high * 100

        # 플래그 깊이 10~25%
        if not (5 <= flag_depth <= 30):
            continue

        # 플래그 기간 3~10주
        flag_weeks = len(flag_data) / 5
        if not (2 <= flag_weeks <= 12):
            continue

        # 플래그 거래량 감소
        pole_vol = np.mean(v[:pole_len])
        flag_vol = np.mean(v[pole_len:])
        vol_dry  = flag_vol < pole_vol * 0.6

        result["detected"]   = True
        result["pole_gain"]  = round(pole_gain, 1)
        result["flag_depth"] = round(flag_depth, 1)
        result["flag_weeks"] = round(flag_weeks, 1)
        result["vol_dry"]    = vol_dry
        result["quality"]    = "high" if (pole_gain >= 100 and vol_dry) else "medium"
        result["desc"]       = f"하이타이트플래그 / 급등 {round(pole_gain,1)}% / 플래그 {round(flag_depth,1)}%"
        break

    return result


# ── 7. 소서 앤 핸들 ───────────────────────────────────────────────────────────

def detect_saucer_handle(close, low, volume, n, lookback=200) -> dict:
    """
    컵앤핸들보다 훨씬 얕고 긴 U자형 (접시 모양)
    주로 대형 우량주에서 발생
    """
    result = {"detected": False, "name": "소서앤핸들", "signal": "saucer_handle"}
    if n < 80:
        return result

    lb = min(lookback, n)
    c  = close[-lb:]
    l  = low[-lb:]
    v  = volume[-lb:]
    m  = len(c)

    # 소서: 전체 구간의 앞 80%에서 완만한 U자형
    saucer_end = int(m * 0.82)
    saucer_c   = c[:saucer_end]

    saucer_high = max(saucer_c[0], saucer_c[-1])
    saucer_low  = np.min(l[:saucer_end])
    depth       = (saucer_high - saucer_low) / saucer_high * 100

    # 소서 깊이 10~30% (컵보다 얕음)
    if not (5 <= depth <= 35):
        return result

    # 소서 기간 최소 40봉(8주)
    if saucer_end < 40:
        return result

    # U자형 확인: 중간 구간이 양쪽보다 낮아야
    third = saucer_end // 3
    left_avg  = np.mean(saucer_c[:third])
    mid_avg   = np.mean(saucer_c[third:2*third])
    right_avg = np.mean(saucer_c[2*third:])
    is_u_shape = mid_avg < left_avg * 0.98 and mid_avg < right_avg * 0.98

    if not is_u_shape:
        return result

    # 핸들 (마지막 15~20%)
    handle_data = c[saucer_end:]
    if len(handle_data) < 5:
        return result

    handle_high = np.max(handle_data)
    handle_low  = np.min(handle_data)
    handle_depth = (handle_high - handle_low) / handle_high * 100

    # 핸들 깊이 5~15%
    if handle_depth > 20:
        return result

    result["detected"]     = True
    result["saucer_depth"] = round(depth, 1)
    result["saucer_weeks"] = round(saucer_end / 5, 1)
    result["handle_depth"] = round(handle_depth, 1)
    result["quality"]      = "high" if depth < 20 else "medium"
    result["desc"]         = f"소서앤핸들 / 깊이 {round(depth,1)}% / {round(saucer_end/5,1)}주"

    return result


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _find_pivots(data: np.ndarray, min_pct: float = 3.0, window: int = 5) -> list:
    """
    고점/저점 피봇 포인트 탐색
    min_pct: 최소 변동폭 (%)
    """
    pivots = []
    n = len(data)

    for i in range(window, n - window):
        left  = data[i-window:i]
        right = data[i+1:i+window+1]

        # 고점
        if data[i] == np.max(data[i-window:i+window+1]):
            if not pivots or pivots[-1]["type"] == "low":
                change = abs(data[i] - (pivots[-1]["val"] if pivots else data[0])) / max(data[i], 0.01) * 100
                if change >= min_pct or not pivots:
                    pivots.append({"type": "high", "idx": i, "val": float(data[i])})

        # 저점
        elif data[i] == np.min(data[i-window:i+window+1]):
            if not pivots or pivots[-1]["type"] == "high":
                change = abs(data[i] - (pivots[-1]["val"] if pivots else data[0])) / max(pivots[-1]["val"] if pivots else data[0], 0.01) * 100
                if change >= min_pct or not pivots:
                    pivots.append({"type": "low", "idx": i, "val": float(data[i])})

    return pivots


# ── 8. RS Line 신고가 선행 ────────────────────────────────────────────────────

def detect_rs_line_lead(close, high, n, lookback=60, spy_beta=1.0) -> dict:
    """
    주가가 아직 신고가를 못 찍었는데 RS Line이 먼저 신고가를 기록
    RS Line ≈ 종목 가격 / SPY 가격 (여기선 상대 모멘텀으로 근사)
    """
    result = {"detected": False, "name": "RS Line 선행", "signal": "rs_line_lead"}
    if n < lookback:
        return result

    c  = close[-lookback:]
    h  = high[-lookback:]

    # RS Line 근사: 가격 / 이동평균 비율 (SPY 없으면 자체 모멘텀으로 대체)
    ma50 = np.convolve(c, np.ones(50)/50, mode='valid')
    if len(ma50) < 5:
        return result

    rs_line = c[49:] / ma50  # 최근 RS Line 값들

    # RS Line 신고가 (최근 5일 기준)
    rs_52w_high  = np.max(rs_line)
    rs_recent    = rs_line[-1]
    rs_at_high   = rs_recent >= rs_52w_high * 0.99

    # 주가는 아직 52주 신고가가 아님 (-3% ~ -20% 사이)
    price_52w_high = np.max(h)
    price_from_high = (close[-1] - price_52w_high) / price_52w_high * 100
    price_not_at_high = -20 <= price_from_high <= -2

    if rs_at_high and price_not_at_high:
        result["detected"]        = True
        result["price_from_high"] = round(price_from_high, 1)
        result["quality"]         = "high" if price_from_high >= -10 else "medium"
        result["desc"]            = f"RS Line 선행 / 주가 고점 대비 {round(price_from_high,1)}%"

    return result


# ── 9. 어닝 갭업 후 유지 ─────────────────────────────────────────────────────

def detect_earnings_gap(close, high, low, open_, volume, n, lookback=30) -> dict:
    """
    갭업(+5% 이상) 후 갭을 메우지 않고 유지
    기관이 계속 매수하고 있다는 신호
    """
    result = {"detected": False, "name": "어닝갭업", "signal": "earnings_gap"}
    if n < 10:
        return result

    lb = min(lookback, n - 2)

    # 최근 30봉 내에서 갭업 탐색
    for i in range(1, lb):
        idx = n - lb + i
        if idx <= 0 or idx >= n:
            continue

        prev_close = close[idx - 1]
        curr_open  = open_[idx]
        curr_close = close[idx]

        if prev_close <= 0:
            continue

        gap_pct = (curr_open - prev_close) / prev_close * 100

        # 갭업 +5% 이상
        if gap_pct < 5:
            continue

        # 갭업 당일 거래량 급증
        avg_vol = np.mean(volume[max(0, idx-20):idx])
        gap_vol_ratio = volume[idx] / avg_vol if avg_vol > 0 else 1

        if gap_vol_ratio < 1.5:
            continue

        # 갭업 후 갭 유지 여부 (갭 = prev_close ~ curr_open 사이)
        gap_bottom  = prev_close
        days_since  = n - idx
        post_data   = low[idx:]

        # 갭을 메우지 않았는지 (저가가 갭 하단을 터치하지 않음)
        gap_held = all(l >= gap_bottom * 0.98 for l in post_data)

        if gap_held and days_since <= 15:
            result["detected"]      = True
            result["gap_pct"]       = round(gap_pct, 1)
            result["days_since"]    = days_since
            result["vol_ratio"]     = round(gap_vol_ratio, 1)
            result["quality"]       = "high" if (gap_pct >= 8 and gap_vol_ratio >= 2) else "medium"
            result["desc"]          = f"갭업 +{round(gap_pct,1)}% / {days_since}일째 유지 / 거래량 {round(gap_vol_ratio,1)}배"
            break

    return result


# ── 10. 파워 플레이 (Power Play) ──────────────────────────────────────────────

def detect_power_play(close, open_, volume, n, min_days=3) -> dict:
    """
    3일 이상 연속 양봉 + 거래량 연속 증가
    기관의 대규모 매집 진행 중인 모멘텀 신호
    """
    result = {"detected": False, "name": "파워플레이", "signal": "power_play"}
    if n < min_days + 2:
        return result

    # 최근 10봉에서 연속 상승 구간 탐색
    best_streak = 0
    best_end    = 0

    for end in range(n-1, max(n-15, min_days), -1):
        streak = 0
        vol_increasing = True
        for i in range(end, max(end - 8, 0), -1):
            # 양봉 조건: 종가 > 시가
            if close[i] > open_[i] and close[i] > close[i-1]:
                streak += 1
                # 거래량 증가 확인
                if streak > 1 and volume[i] < volume[i-1] * 0.85:
                    vol_increasing = False
            else:
                break
        if streak >= min_days and vol_increasing:
            if streak > best_streak:
                best_streak = streak
                best_end    = end

    if best_streak >= min_days:
        # 연속 상승 기간 수익률
        start_idx  = best_end - best_streak + 1
        streak_gain = (close[best_end] - close[start_idx - 1]) / close[start_idx - 1] * 100 if start_idx > 0 else 0

        result["detected"]     = True
        result["streak_days"]  = best_streak
        result["streak_gain"]  = round(streak_gain, 1)
        result["quality"]      = "high" if best_streak >= 4 else "medium"
        result["desc"]         = f"파워플레이 {best_streak}일 연속 / 기간 수익 {round(streak_gain,1)}%"

    return result


# ── 11. 거래량 급증 돌파 ─────────────────────────────────────────────────────

def detect_vol_breakout(close, high, volume, n, lookback=60) -> dict:
    """
    저항선을 거래량 급증(평소 대비 150%+)하며 돌파
    피봇 포인트 진입 신호
    """
    result = {"detected": False, "name": "거래량돌파", "signal": "vol_breakout"}
    if n < 20:
        return result

    lb = min(lookback, n)
    c  = close[-lb:]
    h  = high[-lb:]
    v  = volume[-lb:]
    m  = len(c)

    # 52주 고점 (저항선)
    resistance = np.max(h[:-3]) if m > 5 else np.max(h)

    # 최근 3봉에서 돌파 여부
    recent_high  = np.max(h[-3:])
    near_breakout = recent_high >= resistance * 0.98

    if not near_breakout:
        return result

    # 돌파 시 거래량 확인
    avg_vol      = np.mean(v[:-5]) if m > 5 else np.mean(v)
    recent_vol   = np.max(v[-3:])
    vol_ratio    = recent_vol / avg_vol if avg_vol > 0 else 0

    # 거래량 150% 이상
    if vol_ratio < 1.4:
        return result

    # 양봉 확인 (종가 > 시가)
    bullish_candle = c[-1] > c[-2]

    result["detected"]   = True
    result["vol_ratio"]  = round(vol_ratio, 1)
    result["resistance"] = round(resistance, 2)
    result["quality"]    = "high" if (vol_ratio >= 1.5 and bullish_candle) else "medium"
    result["desc"]       = f"거래량 돌파 / {round(vol_ratio,1)}배 급증"

    return result


# ── 12. S1→S2 전환 감지 ──────────────────────────────────────────────────────

def detect_s1_to_s2(close, high, low, volume, n, lookback=60) -> dict:
    """
    Stage 1(횡보)에서 Stage 2(상승)로 막 전환
    최근 4주 이내 200일선 상향 돌파 + 200일선 기울기 전환
    """
    result = {"detected": False, "name": "S1→S2 전환", "signal": "s1_to_s2"}
    if n < 210:
        return result

    c  = close
    ma200 = np.convolve(c, np.ones(200)/200, mode='valid')
    if len(ma200) < 22:
        return result

    ma200_now      = ma200[-1]
    ma200_4w_ago   = ma200[-21] if len(ma200) >= 21 else ma200[0]
    ma200_8w_ago   = ma200[-41] if len(ma200) >= 41 else ma200[0]

    price_now      = c[-1]
    price_4w_ago   = c[-21] if n >= 21 else c[0]

    # 200일선 상향 기울기로 전환 (최근 4주간 상승)
    ma200_turning_up = ma200_now > ma200_4w_ago and ma200_4w_ago >= ma200_8w_ago * 0.999

    # 현재 주가가 200일선 위
    price_above_ma200 = price_now > ma200_now

    # 4주 전엔 200일선 아래였거나 경계에 있었음 (최근 돌파)
    recently_crossed = price_4w_ago <= ma200_4w_ago * 1.05

    # 거래량 증가하며 돌파
    avg_vol_old = np.mean(volume[-60:-20]) if n >= 60 else np.mean(volume[:-20])
    avg_vol_new = np.mean(volume[-20:])
    vol_expand  = avg_vol_new > avg_vol_old * 0.9

    if ma200_turning_up and price_above_ma200 and recently_crossed:
        ma200_slope = round((ma200_now - ma200_4w_ago) / ma200_4w_ago * 100, 2)
        result["detected"]    = True
        result["ma200_slope"] = ma200_slope
        result["vol_expand"]  = vol_expand
        result["quality"]     = "high" if (vol_expand and ma200_slope > 0.5) else "medium"
        result["desc"]        = f"S1→S2 전환 / 200일선 기울기 +{ma200_slope}%"

    return result


def get_pattern_label(pattern_key: str) -> str:
    """패턴 키를 한국어 레이블로 변환"""
    labels = {
        "vcp":            "VCP",
        "cup_handle":     "컵앤핸들",
        "double_bottom":  "이중바닥",
        "flat_base":      "플랫베이스",
        "ascending_base": "어센딩베이스",
        "htf":            "하이타이트플래그",
        "saucer_handle":  "소서앤핸들",
        "rs_line_lead":   "RS Line 선행",
        "earnings_gap":   "어닝갭업",
        "power_play":     "파워플레이",
        "vol_breakout":   "거래량돌파",
        "s1_to_s2":       "S1→S2전환",
    }
    return labels.get(pattern_key, pattern_key)


def get_pattern_emoji(pattern_key: str) -> str:
    """패턴 키를 이모지로 변환"""
    emojis = {
        "vcp":            "🎯",
        "cup_handle":     "🏆",
        "double_bottom":  "W",
        "flat_base":      "📦",
        "ascending_base": "📈",
        "htf":            "🚀",
        "saucer_handle":  "🍽",
        "rs_line_lead":   "📡",
        "earnings_gap":   "⚡",
        "power_play":     "💪",
        "vol_breakout":   "💥",
        "s1_to_s2":       "🌱",
    }
    return emojis.get(pattern_key, "📊")
