"""
Run this on your Windows machine to find which URL works for LM Studio.
    python test_lm_studio.py
"""

import requests

print("Testing LM Studio connections...\n")

urls_to_try = [
    "http://127.0.0.1:1234/v1/models",
    "http://localhost:1234/v1/models",
    "http://127.0.0.1:1234/api/v1/models",
    "http://localhost:1234/api/v1/models",
]

working = None
for url in urls_to_try:
    try:
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            print(f"✅ WORKS: {url}")
            print(f"   Response: {r.text[:300]}\n")
            working = url
        else:
            print(f"⚠️  {url} → HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ {url} → {e}")

print()
if working:
    base = working.replace("/v1/models", "").replace("/api/v1/models", "")
    print(f"✔ Add this to your .env file:")
    print(f"  LM_STUDIO_BASE_URL={base}")
else:
    print("No working URL found. Make sure LM Studio server is started.")
    print("In LM Studio → Local Server tab → click 'Start Server'")