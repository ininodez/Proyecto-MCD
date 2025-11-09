import pandas as pd
import requests
import time
import json
import torch
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

input_path = r"C:\Users\DmP\Desktop\Gemma-MCD\input\Empresa Sanitaria IPS.csv"
output_path = r"C:\Users\DmP\Desktop\Gemma-MCD\output\ips_resumen_llm.csv"
cache_ip_file = r"C:\Users\DmP\Desktop\Gemma-MCD\cache_ipinfo.json"
cache_vt_file = r"C:\Users\DmP\Desktop\Gemma-MCD\cache_virustotal.json"

os.makedirs(os.path.dirname(cache_ip_file), exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if not os.path.exists(cache_ip_file):
    with open(cache_ip_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

if not os.path.exists(cache_vt_file):
    with open(cache_vt_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

VT_API_KEY = os.getenv("VT_API_KEY")
if not VT_API_KEY:
    raise ValueError("VT_API_KEY not found in environment variables. Please set it in .env file.")

MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

with open(cache_ip_file, "r", encoding="utf-8") as f:
    cache_ip = json.load(f)

with open(cache_vt_file, "r", encoding="utf-8") as f:
    cache_vt = json.load(f)


def save_cache():
    with open(cache_ip_file, "w", encoding="utf-8") as f:
        json.dump(cache_ip, f, ensure_ascii=False, indent=2)
    with open(cache_vt_file, "w", encoding="utf-8") as f:
        json.dump(cache_vt, f, ensure_ascii=False, indent=2)


def enrich_ip(ip):
    if ip in cache_ip:
        return cache_ip[ip]

    url = f"http://ip-api.com/json/{ip}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("status") == "success":
            cache_ip[ip] = data
        else:
            cache_ip[ip] = {"error": data.get("message")}
    except Exception as e:
        cache_ip[ip] = {"error": str(e)}

    save_cache()
    return cache_ip[ip]


def enrich_virustotal(ip):
    if ip in cache_vt:
        return cache_vt[ip]

    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    headers = {"x-apikey": VT_API_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            attributes = data.get("data", {}).get("attributes", {})
            malicious = attributes.get("last_analysis_stats", {}).get("malicious", 0)
            suspicious = attributes.get("last_analysis_stats", {}).get("suspicious", 0)
            harmless = attributes.get("last_analysis_stats", {}).get("harmless", 0)
            reputation = attributes.get("reputation", 0)
            country = attributes.get("country", "N/A")
            as_owner = attributes.get("as_owner", "N/A")
            last_analysis_date = attributes.get("last_analysis_date", "N/A")

            result = {
                "malicious_count": malicious,
                "suspicious_count": suspicious,
                "harmless_count": harmless,
                "reputation_score": reputation,
                "country_vt": country,
                "as_owner": as_owner,
                "last_analysis_date": last_analysis_date,
                "raw": attributes
            }
            cache_vt[ip] = result
        else:
            cache_vt[ip] = {"error": f"Status code {r.status_code}", "raw_response": r.text}
    except Exception as e:
        cache_vt[ip] = {"error": str(e)}

    save_cache()
    return cache_vt[ip]


def summarize_with_gemma(ip, ip_info, vt_info):
    data_ip_json = json.dumps(ip_info, indent=2, ensure_ascii=False)
    data_vt_json = json.dumps(vt_info, indent=2, ensure_ascii=False)

    prompt_text = f"""
Eres un analista SOC experto en ciberseguridad. 
Analiza la siguiente información combinada de una IP: datos de geolocalización pública y reputación en VirusTotal.

Instrucciones:
- Resume país, ASN, ISP y si la IP podría estar asociada a actividad maliciosa.
- Considera indicadores como: país de alto riesgo, VPN/datacenter, ISP residencial, reportes de VirusTotal (malicious/suspicious), reputación.
- Explica de forma técnica y clara, en un solo párrafo breve.

Información IP Pública (ip-api):
{data_ip_json}

Reputación VirusTotal:
{data_vt_json}
"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Eres un experto SOC y analista de ciberseguridad."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False,
            temperature=0.0
        )
        generation = generation[0][input_len:]

    response = processor.decode(generation, skip_special_tokens=True).strip()
    return response


def process_ip(ip):
    ip_data = enrich_ip(ip)
    vt_data = enrich_virustotal(ip)
    resumen = summarize_with_gemma(ip, ip_data, vt_data)

    return {
        "source_ip": ip,
        "ip_country": ip_data.get("country", "N/A"),
        "ip_region": ip_data.get("regionName", "N/A"),
        "ip_city": ip_data.get("city", "N/A"),
        "ip_org": ip_data.get("org", "N/A"),
        "ip_isp": ip_data.get("isp", "N/A"),
        "ip_as": ip_data.get("as", "N/A"),
        "ip_lat": ip_data.get("lat", None),
        "ip_lon": ip_data.get("lon", None),
        "vt_country": vt_data.get("country_vt", "N/A"),
        "vt_as_owner": vt_data.get("as_owner", "N/A"),
        "vt_malicious": vt_data.get("malicious_count", 0),
        "vt_suspicious": vt_data.get("suspicious_count", 0),
        "vt_harmless": vt_data.get("harmless_count", 0),
        "vt_reputation": vt_data.get("reputation_score", 0),
        "vt_last_analysis_date": vt_data.get("last_analysis_date", "N/A"),
        "resumen_gemma": resumen
    }


df = pd.read_csv(input_path)
ips_input = set(df["source_ip"].dropna().unique())

ips_procesadas = set()
if os.path.exists(output_path):
    prev_df = pd.read_csv(output_path)
    if "source_ip" in prev_df.columns:
        ips_procesadas = set(prev_df["source_ip"].astype(str))

ips_nuevas = ips_input - ips_procesadas

resultados_nuevos = []
if ips_nuevas:
    MAX_WORKERS = 3
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ip = {executor.submit(process_ip, ip): ip for ip in ips_nuevas}
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                data = future.result()
                resultados_nuevos.append(data)
            except Exception as e:
                print(f"Error procesando IP {ip}: {e}")
            time.sleep(1)

if resultados_nuevos:
    nuevos_df = pd.DataFrame(resultados_nuevos)
    if os.path.exists(output_path):
        final_df = pd.concat([prev_df, nuevos_df], ignore_index=True)
    else:
        final_df = nuevos_df

    final_df.drop_duplicates(subset=["source_ip"], inplace=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8")
