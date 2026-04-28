"""
github_upload.py v3 — US + KR + Dashboard 동시 업로드
screener_data.json + kr_screener_output.json + dashboard.html
"""

import json, base64, os, sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("[오류] requests 미설치. pip install requests")
    sys.exit(1)

# ── 설정 ─────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "YOUR_TOKEN_HERE")
REPO_OWNER   = os.environ.get("GITHUB_OWNER", "trend-pilot")
REPO_NAME    = os.environ.get("GITHUB_REPO",  "screener")
BRANCH       = "main"

# 업로드할 파일 (로컬 경로 후보들, GitHub 경로)
UPLOAD_FILES = [
    (["output/screener_data.json", "screener_data.json"],  "screener_data.json"),
    (["kr_screener_output.json"],                           "kr_screener_output.json"),
    (["dashboard.html"],                                    "dashboard.html"),
]

API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
HEADERS  = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
}


def get_sha(path):
    r = requests.get(f"{API_BASE}/{path}", headers=HEADERS)
    return r.json().get("sha") if r.status_code == 200 else None


def upload_file(local_candidates, remote):
    # 존재하는 로컬 파일 탐색
    local = next((p for p in local_candidates if os.path.exists(p)), None)
    if not local:
        print(f"  [스킵] {remote} — 로컬 파일 없음 (찾은 경로: {local_candidates})")
        return False

    with open(local, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    sha  = get_sha(remote)
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    size = os.path.getsize(local) / 1024

    payload = {
        "message": f"🤖 Update {remote} [{now}]",
        "content": content,
        "branch":  BRANCH,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(f"{API_BASE}/{remote}", headers=HEADERS, json=payload)
    if r.status_code in (200, 201):
        action = "수정" if sha else "신규"
        print(f"  ✅ {remote} {action} ({size:.1f} KB)  ← {local}")
        return True
    else:
        print(f"  ❌ {remote} 실패: {r.status_code} — {r.text[:120]}")
        return False


def main():
    print("=" * 55)
    print("🚀 TrendPilot GitHub 업로드 v3")
    print(f"   저장소: {REPO_OWNER}/{REPO_NAME} [{BRANCH}]")
    print("=" * 55)

    if GITHUB_TOKEN == "YOUR_TOKEN_HERE":
        print("\n[오류] GITHUB_TOKEN 환경변수를 설정하세요.")
        print("  Windows: set GITHUB_TOKEN=ghp_xxxxxxxxxxxx")
        print("  Mac/Linux: export GITHUB_TOKEN=ghp_xxxxxxxxxxxx")
        sys.exit(1)

    ok = 0
    for local_candidates, remote in UPLOAD_FILES:
        print(f"\n📤 {remote}")
        if upload_file(local_candidates, remote):
            ok += 1

    print(f"\n{'='*55}")
    print(f"결과: {ok}/{len(UPLOAD_FILES)}개 업로드 완료")

    if ok > 0:
        url = f"https://{REPO_OWNER}.github.io/{REPO_NAME}/dashboard.html"
        print(f"\n🌐 대시보드: {url}")
        print("\n📊 데이터 현황:")
        for candidates, remote in UPLOAD_FILES:
            path = next((p for p in candidates if os.path.exists(p)), None)
            if path and path.endswith(".json"):
                try:
                    with open(path, encoding="utf-8") as f:
                        d = json.load(f)
                    meta  = d.get("meta", {})
                    mkt   = meta.get("market", "?")
                    total = (d.get("summary", {}).get("total_stocks")
                             or meta.get("total", "?"))
                    demo  = "⚠️ 데모" if meta.get("demo") else "✅ 실제"
                    ts    = (meta.get("updated_at") or meta.get("generated_at",""))[:16]
                    print(f"  {remote}: {mkt} {total}종목 [{demo}] {ts}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
