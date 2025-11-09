import os
import json
import time
import re
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)

CSV_INPUT = r"C:\Users\DmP\Desktop\Gemma-MCD\input\Empresa Sanitaria IPS_final.csv"
CSV_OUTPUT = r"C:\Users\DmP\Desktop\Gemma-MCD\output\analisis_gemini_final.csv"

cols_interes = [
    "source_ip_hashed",
    "destination_hashed",
    "server_name",
    "conectado_a",
    "severity",
    "signature_name",
    "signature_cat",
    "ip_country",
    "vt_malicious",
    "vt_reputation",
    "resumen_gemma",
]


def build_prompt(fila):
    return f"""
You are an expert cybersecurity analyst. Given the following incident data, produce a VALID JSON object and NOTHING else.
The JSON must contain these keys:
- "prioridad": integer from 1 to 10 (10 = highest criticality).
- "acciones_remediacion": array of short strings (immediate remediation steps).
- "acciones_prevencion": array of short strings (future prevention).
- "diagnostico": short string (technical diagnosis, consider connected servers).

Incident data:
server_name: {fila['server_name']}
conectado_a: {fila['conectado_a_x']}
severity: {fila['severity']}
signature_name: {fila['signature_name']}
signature_cat: {fila['signature_cat']}
ip_country_y: {fila['ip_country']}
vt_malicious_y: {fila['vt_malicious']}
vt_reputation_y: {fila['vt_reputation']}
resumen_gemma_y: {fila['resumen_gemma']}

Important: Return only well-formed JSON and nothing else.
"""


def extract_json_from_text(text):
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m2 = re.search(r'(\[.*\])', text, flags=re.DOTALL)
    if m2:
        try:
            parsed = json.loads(m2.group(1))
            return {"wrapped_array": parsed}
        except Exception:
            pass

    return None


def ask_reformat(prev_text, fila):
    repair_prompt = f"""
The assistant previously returned invalid output:
---
{prev_text}
---
Please return ONLY a VALID JSON object with keys: "prioridad", "acciones_remediacion", "acciones_prevencion", "diagnostico".
If any field is unknown, use integer between 1 and 10 for prioridad, arrays empty, and diagnostico as "N/A".
Incident data:
server_name: {fila['server_name']}, conectado_a: {fila['conectado_a']},
severity: {fila['severity']}, signature_name: {fila['signature_name']}, signature_cat: {fila['signature_cat']}.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=repair_prompt
    )
    return response.text


df = pd.read_csv(CSV_INPUT).fillna("N/A")
data = df[cols_interes].reset_index(drop=True)

if os.path.exists(CSV_OUTPUT):
    old_df = pd.read_csv(CSV_OUTPUT)
    processed_pairs = set(zip(old_df["source_ip_hashed"], old_df["destination_hashed"]))
else:
    old_df = pd.DataFrame()
    processed_pairs = set()

data["pair"] = list(zip(data["source_ip_hashed"], data["destination_hashed"]))
data_new = data[~data["pair"].isin(processed_pairs)].drop(columns=["pair"])

if data_new.empty:
    exit(0)

outputs = []

for i, fila in data_new.iterrows():
    prompt = build_prompt(fila)
    success, attempts, parsed, last_resp_text = False, 0, None, ""

    while attempts < 10 and not success:
        attempts += 1
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            last_resp_text = resp.text or ""
            parsed = extract_json_from_text(last_resp_text)
            if parsed:
                success = True
                break
            else:
                repair_text = ask_reformat(last_resp_text, fila)
                parsed = extract_json_from_text(repair_text)
                if parsed:
                    success = True
                    break
        except Exception as e:
            pass
        time.sleep(min(2 * attempts, 30))

    if not success:
        parsed = {
            "prioridad": None,
            "acciones_remediacion": ["Error en análisis o formato no válido."],
            "acciones_prevencion": [],
            "diagnostico": f"No se pudo procesar correctamente. Última respuesta: {last_resp_text[:300]}"
        }

    outputs.append({
        "source_ip_hashed": fila["source_ip_hashed"],
        "destination_hashed": fila["destination_hashed"],
        "prioridad": parsed.get("prioridad"),
        "acciones_remediacion": json.dumps(parsed.get("acciones_remediacion", []), ensure_ascii=False),
        "acciones_prevencion": json.dumps(parsed.get("acciones_prevencion", []), ensure_ascii=False),
        "diagnostico": parsed.get("diagnostico", "")
    })

    time.sleep(0.5)

if outputs:
    new_df = pd.DataFrame(outputs)
    final_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
        subset=["source_ip_hashed", "destination_hashed"], keep="last"
    )
else:
    final_df = old_df

final_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
