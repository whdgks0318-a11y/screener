#!/usr/bin/env python3
"""
enrich_sectors.py — Yahoo Finance에서 GICS 섹터/업종/시가총액 정보를 가져와
screener_data.json 에 병합하는 보조 스크립트.

사용법:
    pip install yfinance
    python enrich_sectors.py

자동으로 다음 경로들을 탐색하여 가장 최근 파일을 선택합니다:
  1. ./screener_data.json (현재 폴더)
  2. ./output/screener_data.json (output 하위 폴더)
  3. ../output/screener_data.json (부모의 output 폴더)

캐시 파일: sector_cache.json (티커별 섹터 정보 저장, 다음 실행 시 재사용)
첫 실행은 1838개 종목 기준 10~20분 걸릴 수 있음.
이후 실행은 캐시 덕분에 1분 내 완료.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("❌ yfinance 라이브러리가 없습니다.")
    print("   설치: pip install yfinance")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
CANDIDATE_PATHS = [
    Path("screener_data.json"),
    Path("output") / "screener_data.json",
    Path("..") / "output" / "screener_data.json",
]
CACHE_FILE = Path("sector_cache.json")
CACHE_TTL_DAYS = 30                  # 캐시 유효기간
REQUEST_DELAY = 0.08                 # 과다 요청 방지 (초)
SAVE_CHECKPOINT_EVERY = 50           # N개마다 캐시 중간 저장


def find_target_files():
    """존재하는 모든 후보 경로를 모두 반환 (복수 파일 동시 업데이트)."""
    existing = [p for p in CANDIDATE_PATHS if p.exists()]
    if not existing:
        print("❌ screener_data.json 파일을 찾을 수 없습니다.")
        print("   다음 경로들을 확인해보세요:")
        for p in CANDIDATE_PATHS:
            print(f"     - {p.absolute()}")
        sys.exit(1)

    print(f"📂 발견된 screener_data.json: {len(existing)}개")
    for p in existing:
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        size_kb = p.stat().st_size / 1024
        print(f"   - {p}  ({size_kb:,.0f} KB, 수정: {mtime:%Y-%m-%d %H:%M})")
    return existing


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ 캐시 로드 실패 ({e}), 새 캐시로 시작")
        return {}


def save_cache(cache):
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def is_cache_fresh(entry):
    if not entry or "fetched_at" not in entry:
        return False
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
        return (datetime.now() - fetched) < timedelta(days=CACHE_TTL_DAYS)
    except Exception:
        return False


def fetch_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        if not info or not isinstance(info, dict):
            return None
        return {
            "sector":     info.get("sector") or "",
            "industry":   info.get("industry") or "",
            "marketCap":  info.get("marketCap") or 0,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as e:
        return {
            "sector":     "",
            "industry":   "",
            "marketCap":  0,
            "error":      str(e)[:100],
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }


def enrich_file(target_file, cache):
    """주어진 파일에 sector/industry/mktcap 병합하고 저장."""
    backup_file = target_file.with_suffix(target_file.suffix + ".bak")

    print(f"\n📂 처리 중: {target_file}")
    with target_file.open(encoding="utf-8") as f:
        data = json.load(f)
    stocks = data.get("stocks", [])
    print(f"   {len(stocks)}개 종목")

    if not backup_file.exists():
        print(f"💾 원본 백업: {backup_file.name}")
        backup_file.write_text(target_file.read_text(encoding="utf-8"), encoding="utf-8")

    hit, miss = 0, 0
    for s in stocks:
        t = s.get("ticker", "").strip()
        info = cache.get(t)
        if info and (info.get("sector") or info.get("industry")):
            s["sector"]    = info.get("sector", "")
            s["industry"]  = info.get("industry", "")
            s["mktcap"]    = info.get("marketCap", 0)
            hit += 1
        else:
            s.setdefault("sector", "")
            s.setdefault("industry", "")
            s.setdefault("mktcap", 0)
            miss += 1

    with target_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"   ✓ 매칭: {hit}개, 매칭 실패: {miss}개 → 저장됨")
    return stocks


def main():
    target_files = find_target_files()

    # 모든 파일에서 티커 수집
    all_tickers = set()
    for tf in target_files:
        try:
            with tf.open(encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("stocks", []):
                t = s.get("ticker", "").strip()
                if t:
                    all_tickers.add(t)
        except Exception as e:
            print(f"⚠ {tf} 로드 실패: {e}")

    print(f"\n📊 전체 유니크 티커: {len(all_tickers)}개")

    cache = load_cache()
    print(f"   캐시: {len(cache)}개 티커 저장됨")

    need_fetch = [t for t in all_tickers
                  if t not in cache or not is_cache_fresh(cache[t])]
    print(f"🌐 {len(need_fetch)}개 티커 Yahoo 조회 필요")

    if need_fetch:
        t0 = time.time()
        for i, ticker in enumerate(need_fetch, 1):
            cache[ticker] = fetch_stock_info(ticker) or {}
            time.sleep(REQUEST_DELAY)

            if i % 20 == 0 or i == len(need_fetch):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.1)
                remaining = (len(need_fetch) - i) / max(rate, 0.01)
                eta_sec = int(remaining)
                bar = "█" * (i * 30 // len(need_fetch)) + "░" * (30 - i * 30 // len(need_fetch))
                print(f"   [{bar}] {i}/{len(need_fetch)}  "
                      f"({rate:.1f}/s · ETA {eta_sec//60}m {eta_sec%60}s)", end="\r")

            if i % SAVE_CHECKPOINT_EVERY == 0:
                save_cache(cache)

        print()
        save_cache(cache)
        print(f"✅ Yahoo 조회 완료 ({time.time() - t0:.1f}s)")

    # 모든 대상 파일에 병합
    print("\n" + "=" * 60)
    print("📝 모든 대상 파일에 sector/industry/mktcap 병합")
    print("=" * 60)
    all_stocks = []
    for tf in target_files:
        stocks = enrich_file(tf, cache)
        if tf == target_files[0]:  # 첫 파일 기준 통계
            all_stocks = stocks

    # 섹터 분포 통계 (첫 파일 기준)
    sec_dist = {}
    for s in all_stocks:
        key = s.get("sector") or "(비어있음)"
        sec_dist[key] = sec_dist.get(key, 0) + 1

    print(f"\n📊 섹터 분포 ({target_files[0].name}):")
    total = len(all_stocks)
    for sec, cnt in sorted(sec_dist.items(), key=lambda x: -x[1]):
        pct = cnt * 100 / max(total, 1)
        bar = "█" * int(pct // 2)
        print(f"   {sec:30s}  {cnt:4d} ({pct:5.1f}%)  {bar}")

    print(f"\n✅ 완료. 업데이트된 파일 목록:")
    for tf in target_files:
        print(f"   - {tf.absolute()}")


if __name__ == "__main__":
    main()
