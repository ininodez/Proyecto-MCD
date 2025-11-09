import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

VT_API_KEY = os.getenv("VT_API_KEY")
if not VT_API_KEY:
    raise ValueError("VT_API_KEY not found in environment variables. Please set it in .env file.")

ip = "144.48.241.166"  # cambia a la IP que quieras probar

def vt_ip_lookup(ip, api_key):
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    headers = {"x-apikey": api_key}
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "text": r.text}
    return r.json()

data = vt_ip_lookup(ip, VT_API_KEY)
if "error" in data:
    print("Error:", data["error"])
else:
    attrs = data.get("data", {}).get("attributes", {})
    stats = attrs.get("last_analysis_stats", {})
    results = attrs.get("last_analysis_results", {})
    reputation = attrs.get("reputation")
    country = attrs.get("country")
    as_owner = attrs.get("as_owner")
    asn = attrs.get("asn")
    last_analysis_date = attrs.get("last_analysis_date")
    # passive dns / resoluciones
    resolutions = attrs.get("resolutions", [])
    # urls detectadas
    detected_urls = attrs.get("detected_urls", [])
    # muestras/samples relacionadas
    communicating_files = attrs.get("communicating_files") or attrs.get("observed_in_files") or []

    print("=== Resumen rápido ===")
    print("IP:", ip)
    print("Country:", country)
    print("ASN:", asn, "| AS Owner:", as_owner)
    print("Reputation:", reputation)
    print("Last analysis date:", last_analysis_date)
    print("Last analysis stats:", json.dumps(stats, indent=2, ensure_ascii=False))

    # Mostrar motores que marcaron como malicious/suspicious
    malicious_engines = []
    for engine, info in results.items():
        category = info.get("category")
        if category in ("malicious", "suspicious"):
            malicious_engines.append({"engine": engine, "category": category, "result": info.get("result")})
    print("\nMotores que marcaron la IP como malicious/suspicious:")
    print(json.dumps(malicious_engines, indent=2, ensure_ascii=False))

    print("\nResolutions (passive DNS) (máx 10):")
    for r in resolutions[:10]:
        print("-", r)

    print("\nDetected URLs (máx 10):")
    for u in detected_urls[:10]:
        print("-", u)

    print("\nCommunicating / observed files (máx 10):")
    for f in communicating_files[:10]:
        print("-", f)

    # opcional: guardar raw
    with open("vt_ip_raw.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print("\nRaw guardado en vt_ip_raw.json")
