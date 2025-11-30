import os
import json
import time
import re
import random
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

CSV_INPUT = "Empresa Sanitaria IPS_final_priorizado.csv"
CSV_OUTPUT = "analisis_parallel_comparison.csv"
MODELS_TO_RUN = {
    "gpt-5-mini": {
        "class": ChatOpenAI,
        "env_key": "OPENAI_API_KEY",
        "params": {"model": "gpt-5-mini-2025-08-07", "temperature": 0}
    },
    "claude-sonnet-4.5": {
        "class": ChatAnthropic,
        "env_key": "ANTHROPIC_API_KEY",
        "params": {"model": "claude-sonnet-4-5", "temperature": 0}
    },
    "gemini-2.5-flash": {
        "class": ChatGoogleGenerativeAI,
        "env_key": "GEMINI_API_KEY",
        "params": {"model": "gemini-2.5-flash", "temperature": 0}
    }
}

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


class SecurityAnalysis(BaseModel):
    prioridad: int = Field(description="Priority score from 1 to 10 (10 = highest criticality)", ge=1, le=10)
    acciones_remediacion: List[str] = Field(description="Array of immediate remediation steps")
    acciones_prevencion: List[str] = Field(description="Array of future prevention steps")
    diagnostico: str = Field(description="Technical diagnosis considering connected servers")


def build_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert cybersecurity analyst. Given the following incident data, produce a VALID JSON object and NOTHING else."),
        ("user", """You are an expert cybersecurity analyst. Given the following incident data, produce a VALID JSON object and NOTHING else.
The JSON must contain these keys:
- "prioridad": integer from 1 to 10 (10 = highest criticality). Use the priority criteria below.
- "acciones_remediacion": array of short strings (immediate remediation steps).
- "acciones_prevencion": array of short strings (future prevention).
- "diagnostico": short string (technical diagnosis, consider connected servers).

PRIORITY CRITERIA (prioridad field):
Apply the HIGHEST matching priority level:

Priority 10 (CRITICAL):
- Remote Command Execution (RCE) or Code Execution vulnerabilities on Production servers
- Attacks on Jump Box/Bastion Host, Primary SQL Database, or Production Web Server
- vt_reputation <= -5 AND vt_malicious >= 5 AND severity >= 4
- Any attack on critical infrastructure with high malicious reputation

Priority 9 (VERY HIGH):
- RCE/Code Execution on Staging or Development servers
- Command Injection, SQL Injection, or Directory Traversal on Production servers
- vt_reputation <= -3 AND vt_malicious >= 3 AND severity >= 4
- Attacks on Production servers with connected critical systems (Monitoring, Admin Portal)

Priority 8 (HIGH):
- RCE/Code Execution on non-production servers
- Command Injection, SQL Injection on Staging/Development
- vt_reputation <= -2 AND vt_malicious >= 2 AND severity >= 3
- Production server attacks with moderate threat indicators

Priority 7 (MODERATE-HIGH):
- Directory Traversal, Path Traversal on Production servers
- DoS attacks on Production or critical infrastructure
- vt_reputation <= -1 AND severity >= 3
- Attacks on Production with low malicious count but negative reputation

Priority 6 (MODERATE):
- Web threats (XSS, CSRF, etc.) on Production servers
- DoS attacks on Staging/Development servers
- vt_reputation <= 0 AND severity >= 3
- Attacks on non-production with moderate indicators

Priority 5 (MEDIUM):
- Web threats on Staging/Development servers
- Severity >= 3 with neutral reputation (vt_reputation > 0)
- Low malicious count (vt_malicious < 2) but severity >= 3

Priority 4 (LOW-MEDIUM):
- Severity = 2 with any negative reputation
- Low severity (severity < 3) but negative reputation (vt_reputation < 0)

Priority 3 (LOW):
- Severity = 2 with neutral/positive reputation
- Low threat indicators across all fields

Priority 2 (VERY LOW):
- Severity = 1
- All indicators suggest low risk

Priority 1 (MINIMAL):
- Severity = 1 AND vt_reputation > 0 AND vt_malicious = 0
- No significant threat indicators

Note: When multiple criteria apply, use the HIGHEST priority. Consider connected servers (conectado_a) - if a server connects to critical infrastructure, increase priority by 1 level.

Incident data:
server_name: {server_name}
conectado_a: {conectado_a}
severity: {severity}
signature_name: {signature_name}
signature_cat: {signature_cat}
ip_country: {ip_country}
vt_malicious: {vt_malicious}
vt_reputation: {vt_reputation}
resumen_gemma: {resumen_gemma}

Important: Return only well-formed JSON and nothing else.""")
    ])


def normalize_data_for_validation(data):
    if isinstance(data, dict):
        normalized = {}
        for key, value in data.items():
            if key in ["acciones_remediacion", "acciones_prevencion"]:
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            normalized[key] = parsed
                        else:
                            normalized[key] = [parsed] if parsed else []
                    except (json.JSONDecodeError, TypeError):
                        normalized[key] = [value] if value else []
                elif isinstance(value, list):
                    normalized[key] = value
                else:
                    normalized[key] = [str(value)] if value else []
            else:
                normalized[key] = value
        return normalized
    return data


def parse_response_to_pydantic(response_text, model_name):
    text = response_text.strip()
    if not text:
        return None
    
    if isinstance(text, SecurityAnalysis):
        return text.model_dump()
    
    try:
        parsed = json.loads(text)
        parsed = normalize_data_for_validation(parsed)
        validated = SecurityAnalysis(**parsed)
        return validated.model_dump()
    except (json.JSONDecodeError, Exception):
        pass
    
    m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            parsed = normalize_data_for_validation(parsed)
            validated = SecurityAnalysis(**parsed)
            return validated.model_dump()
        except Exception:
            pass
    
    m2 = re.search(r'(\[.*\])', text, flags=re.DOTALL)
    if m2:
        try:
            parsed = json.loads(m2.group(1))
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = normalize_data_for_validation(parsed[0])
                validated = SecurityAnalysis(**parsed)
                return validated.model_dump()
        except Exception:
            pass
    
    return None


def get_model_instance(model_name, config):
    api_key = os.getenv(config["env_key"])
    
    if not api_key:
        raise ValueError(f"{config['env_key']} not found in environment variables. Please set it in .env file.")
    
    model = config["class"](api_key=api_key, **config["params"])
    
    if model_name == "gemini-2.5-flash":
        try:
            return model.with_structured_output(SecurityAnalysis, method="json_schema")
        except Exception as e:
            print(f"Warning: Could not enable structured output for {model_name}: {e}")
            return model
    elif model_name in ["gpt-5-mini", "claude-sonnet-4.5"]:
        try:
            return model.with_structured_output(SecurityAnalysis)
        except Exception as e:
            print(f"Warning: Could not enable structured output for {model_name}: {e}")
            return model
    
    return model


def is_rate_limit_error(error):
    error_str = str(error).lower()
    
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return True
    
    if "capacity" in error_str or "overloaded" in error_str or "quota" in error_str:
        return True
    
    if hasattr(error, 'status_code') and error.status_code == 429:
        return True
    
    if hasattr(error, 'response'):
        if hasattr(error.response, 'status_code') and error.response.status_code == 429:
            return True
        if hasattr(error.response, 'status') and error.response.status == 429:
            return True
    
    return False


def exponential_backoff_sleep(attempt, base_delay=1, max_delay=300, jitter=True):
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    if jitter:
        delay = delay + random.uniform(0, delay * 0.1)
    return delay


def process_row_with_model(model_name, fila, prompt_template, max_attempts=10):
    config = MODELS_TO_RUN[model_name]
    model = get_model_instance(model_name, config)
    chain = prompt_template | model
    
    success = False
    attempts = 0
    parsed = None
    last_resp_text = ""
    
    while not success:
        attempts += 1
        if attempts > 1:
            print(f"    [{model_name}] Retry {attempts}...")
        try:
            input_data = {
                "server_name": str(fila.get('server_name', 'N/A')),
                "conectado_a": str(fila.get('conectado_a', 'N/A')),
                "severity": str(fila.get('severity', 'N/A')),
                "signature_name": str(fila.get('signature_name', 'N/A')),
                "signature_cat": str(fila.get('signature_cat', 'N/A')),
                "ip_country": str(fila.get('ip_country', 'N/A')),
                "vt_malicious": str(fila.get('vt_malicious', 'N/A')),
                "vt_reputation": str(fila.get('vt_reputation', 'N/A')),
                "resumen_gemma": str(fila.get('resumen_gemma', 'N/A'))
            }
            
            response = chain.invoke(input_data)
            
            if isinstance(response, SecurityAnalysis):
                parsed = response.model_dump()
                print(f"    [{model_name}] OK")
                success = True
                break
            elif isinstance(response, dict):
                try:
                    normalized = normalize_data_for_validation(response)
                    validated = SecurityAnalysis(**normalized)
                    parsed = validated.model_dump()
                    success = True
                    break
                except Exception as e:
                    try:
                        normalized = normalize_data_for_validation(response)
                        validated = SecurityAnalysis(**normalized)
                        parsed = validated.model_dump()
                        success = True
                        break
                    except Exception:
                        parsed = response
                        success = True
                        break
            else:
                if hasattr(response, 'content'):
                    last_resp_text = response.content
                elif isinstance(response, str):
                    last_resp_text = response
                else:
                    last_resp_text = str(response)
                
                parsed = parse_response_to_pydantic(last_resp_text, model_name)
                
                if parsed:
                    print(f"    [{model_name}] OK")
                    success = True
                    break
                else:
                    print(f"    [{model_name}] Parse failed, retrying...")
            
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_msg = f"{error_msg} | Response: {e.response.text[:200]}"
            elif hasattr(e, 'body'):
                error_msg = f"{error_msg} | Body: {str(e.body)[:200]}"
            
            error_display = error_msg[:150] if len(error_msg) > 150 else error_msg
            
            if is_rate_limit_error(e):
                backoff_delay = exponential_backoff_sleep(attempts, base_delay=2, max_delay=300)
                print(f"    [{model_name}] Rate limit (429), waiting {backoff_delay:.1f}s...")
                time.sleep(backoff_delay)
                continue
            
            if attempts >= max_attempts:
                print(f"    [{model_name}] Failed after {max_attempts} attempts: {error_display}")
                break
            else:
                print(f"    [{model_name}] Attempt {attempts} failed: {error_display}")
                time.sleep(min(2 * attempts, 30))
    
    if not success:
        try:
            default_analysis = SecurityAnalysis(
                prioridad=1,
                acciones_remediacion=["Error en análisis o formato no válido."],
                acciones_prevencion=[],
                diagnostico=f"No se pudo procesar correctamente. Última respuesta: {last_resp_text[:300]}"
            )
            parsed = default_analysis.model_dump()
        except Exception:
            parsed = {
                "prioridad": 1,
                "acciones_remediacion": ["Error en análisis o formato no válido."],
                "acciones_prevencion": [],
                "diagnostico": f"No se pudo procesar correctamente. Última respuesta: {last_resp_text[:300]}"
            }
    
    return model_name, parsed


def get_embedding(text, model_name=""):
    attempts = 0
    
    while True:
        attempts += 1
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            if 'embedding' in result:
                embedding = result['embedding']
                return embedding
            elif 'values' in result:
                embedding = result['values']
                return embedding
            elif isinstance(result, list):
                return result
            else:
                return None
        except Exception as e:
            if is_rate_limit_error(e):
                backoff_delay = exponential_backoff_sleep(attempts, base_delay=2, max_delay=300)
                print(f"    [Embedding] Rate limit for {model_name}, waiting {backoff_delay:.1f}s...")
                time.sleep(backoff_delay)
                continue
            else:
                print(f"    [Embedding] Error for {model_name}: {str(e)[:100]}")
                return None


def combine_analysis_text(parsed_data):
    prioridad = parsed_data.get("prioridad", "N/A")
    acciones_remediacion = parsed_data.get("acciones_remediacion", [])
    acciones_prevencion = parsed_data.get("acciones_prevencion", [])
    diagnostico = parsed_data.get("diagnostico", "N/A")
    
    if isinstance(acciones_remediacion, list):
        acciones_remediacion = "; ".join([str(a) for a in acciones_remediacion])
    if isinstance(acciones_prevencion, list):
        acciones_prevencion = "; ".join([str(a) for a in acciones_prevencion])
    
    combined = f"Prioridad: {prioridad}. Diagnóstico: {diagnostico}. Acciones de remediación: {acciones_remediacion}. Acciones de prevención: {acciones_prevencion}"
    return combined


def calculate_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None
    
    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return float(similarity)


def compare_texts_with_model(text1, text2, field_name, model_type="gemini"):
    # Format lists as strings if needed
    if isinstance(text1, list):
        text1_formatted = "; ".join([str(item) for item in text1])
    else:
        text1_formatted = str(text1)
    
    if isinstance(text2, list):
        text2_formatted = "; ".join([str(item) for item in text2])
    else:
        text2_formatted = str(text2)
    
    prompt = f"""You are a cybersecurity expert. Compare the following two {field_name} responses from different AI models analyzing the same security incident.

Response 1:
{text1_formatted}

Response 2:
{text2_formatted}

Task: Determine if these two responses are SIMILAR in meaning and offer similar recommendations/analysis.

Consider:
- Do they address the same security concerns?
- Do they propose similar actions or solutions?
- Are the key points and priorities aligned?

Respond with ONLY a single word: "true" if they are similar, or "false" if they are significantly different.

Your response (true/false only):"""

    if model_type == "gemini":
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        result_text = response.text.strip().lower()
    elif model_type == "gpt":
        from langchain_openai import ChatOpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found")
        llm = ChatOpenAI(api_key=openai_key, model="gpt-5-mini-2025-08-07", temperature=0)
        response = llm.invoke(prompt)
        result_text = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
    elif model_type == "claude":
        from langchain_anthropic import ChatAnthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        llm = ChatAnthropic(api_key=anthropic_key, model="claude-sonnet-4-5", temperature=0)
        response = llm.invoke(prompt)
        result_text = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Parse boolean response
    if "true" in result_text:
        return True
    elif "false" in result_text:
        return False
    else:
        # Fallback: try to extract boolean
        if "yes" in result_text or "similar" in result_text:
            return True
        else:
            return False


def compare_texts_with_gemini(text1, text2, field_name):
    attempts = 0
    models_to_try = [
        ("gemini", "Gemini 2.5 Flash"),
        ("gpt", "GPT-5-mini"),
        ("claude", "Claude Sonnet 4.5")
    ]
    current_model_idx = 0
    
    while current_model_idx < len(models_to_try):
        model_type, model_name = models_to_try[current_model_idx]
        attempts += 1
        
        try:
            result = compare_texts_with_model(text1, text2, field_name, model_type)
            if current_model_idx > 0:
                print(f"    [Comparison] {field_name} using {model_name} (fallback)")
            return result
                    
        except Exception as e:
            if is_rate_limit_error(e):
                if current_model_idx < len(models_to_try) - 1:
                    print(f"    [Comparison] Rate limit with {model_name} for {field_name}, trying next model...")
                    current_model_idx += 1
                    attempts = 0
                    time.sleep(1)
                    continue
                else:
                    backoff_delay = exponential_backoff_sleep(attempts, base_delay=2, max_delay=300)
                    print(f"    [Comparison] Rate limit with {model_name} for {field_name}, waiting {backoff_delay:.1f}s...")
                    time.sleep(backoff_delay)
                    continue
            else:
                if current_model_idx < len(models_to_try) - 1:
                    print(f"    [Comparison] Error with {model_name} for {field_name}, trying fallback...")
                    current_model_idx += 1
                    attempts = 0
                    time.sleep(1)
                    continue
                else:
                    print(f"    [Comparison] All models failed for {field_name}: {str(e)[:100]}")
                    return None
    
    return None


def compare_all_models(results):
    gpt_data = results.get("gpt-5-mini", {})
    claude_data = results.get("claude-sonnet-4.5", {})
    gemini_data = results.get("gemini-2.5-flash", {})
    
    comparison_results = {}
    comparison_tasks = []
    if gpt_data.get("acciones_remediacion") and claude_data.get("acciones_remediacion"):
        comparison_tasks.append(("acciones_remediacion_gpt_vs_claude", 
                                gpt_data.get("acciones_remediacion", []),
                                claude_data.get("acciones_remediacion", []),
                                "acciones_remediacion"))
    if gpt_data.get("acciones_remediacion") and gemini_data.get("acciones_remediacion"):
        comparison_tasks.append(("acciones_remediacion_gpt_vs_gemini",
                                gpt_data.get("acciones_remediacion", []),
                                gemini_data.get("acciones_remediacion", []),
                                "acciones_remediacion"))
    if claude_data.get("acciones_remediacion") and gemini_data.get("acciones_remediacion"):
        comparison_tasks.append(("acciones_remediacion_claude_vs_gemini",
                                claude_data.get("acciones_remediacion", []),
                                gemini_data.get("acciones_remediacion", []),
                                "acciones_remediacion"))
    
    if gpt_data.get("acciones_prevencion") and claude_data.get("acciones_prevencion"):
        comparison_tasks.append(("acciones_prevencion_gpt_vs_claude",
                                gpt_data.get("acciones_prevencion", []),
                                claude_data.get("acciones_prevencion", []),
                                "acciones_prevencion"))
    if gpt_data.get("acciones_prevencion") and gemini_data.get("acciones_prevencion"):
        comparison_tasks.append(("acciones_prevencion_gpt_vs_gemini",
                                gpt_data.get("acciones_prevencion", []),
                                gemini_data.get("acciones_prevencion", []),
                                "acciones_prevencion"))
    if claude_data.get("acciones_prevencion") and gemini_data.get("acciones_prevencion"):
        comparison_tasks.append(("acciones_prevencion_claude_vs_gemini",
                                claude_data.get("acciones_prevencion", []),
                                gemini_data.get("acciones_prevencion", []),
                                "acciones_prevencion"))
    if gpt_data.get("diagnostico") and claude_data.get("diagnostico"):
        comparison_tasks.append(("diagnostico_gpt_vs_claude",
                                gpt_data.get("diagnostico", ""),
                                claude_data.get("diagnostico", ""),
                                "diagnostico"))
    if gpt_data.get("diagnostico") and gemini_data.get("diagnostico"):
        comparison_tasks.append(("diagnostico_gpt_vs_gemini",
                                gpt_data.get("diagnostico", ""),
                                gemini_data.get("diagnostico", ""),
                                "diagnostico"))
    if claude_data.get("diagnostico") and gemini_data.get("diagnostico"):
        comparison_tasks.append(("diagnostico_claude_vs_gemini",
                                claude_data.get("diagnostico", ""),
                                gemini_data.get("diagnostico", ""),
                                "diagnostico"))
    
    if comparison_tasks:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(compare_texts_with_gemini, text1, text2, field_name): task_name
                for task_name, text1, text2, field_name in comparison_tasks
            }
            
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                    comparison_results[task_name] = result
                except Exception as e:
                    print(f"    [Comparison] Error in {task_name}: {str(e)[:100]}")
                    comparison_results[task_name] = None
    
    return comparison_results


def calculate_prioridad_similarity(results):
    prioridades = []
    
    for model_name in ["gpt-5-mini", "claude-sonnet-4.5", "gemini-2.5-flash"]:
        model_data = results.get(model_name, {})
        prioridad = model_data.get("prioridad")
        if prioridad is not None:
            try:
                prioridades.append(int(prioridad))
            except (ValueError, TypeError):
                pass
    
    if len(prioridades) < 2:
        return {
            "prioridad_std_dev": None,
            "prioridad_range": None,
            "prioridad_mean": None,
            "prioridad_coefficient_variation": None
        }
    
    prioridades_array = np.array(prioridades)
    
    std_dev = float(np.std(prioridades_array))
    range_val = float(np.max(prioridades_array) - np.min(prioridades_array))
    mean_val = float(np.mean(prioridades_array))
    
    coefficient_variation = std_dev / mean_val if mean_val > 0 else None
    
    return {
        "prioridad_std_dev": std_dev,
        "prioridad_range": range_val,
        "prioridad_mean": mean_val,
        "prioridad_coefficient_variation": coefficient_variation
    }


def process_single_row(row_data):
    idx, fila = row_data
    prompt_template = build_prompt_template()
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_row_with_model, model_name, fila, prompt_template): model_name
            for model_name in MODELS_TO_RUN.keys()
        }
        
        completed_models = []
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                model_name_result, parsed = future.result()
                results[model_name_result] = parsed
                completed_models.append(model_name_result)
            except Exception as e:
                print(f"    [Models] Error processing {model_name}: {str(e)[:100]}")
                results[model_name] = {
                    "prioridad": None,
                    "acciones_remediacion": [],
                    "acciones_prevencion": [],
                    "diagnostico": f"Error: {str(e)[:200]}"
                }
                completed_models.append(model_name)
    
    return idx, results


def main():
    print("=" * 80)
    print("Parallel Multi-Model Security Analysis")
    print("=" * 80)
    print()
    
    google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
    
    genai.configure(api_key=google_api_key)
    
    for model_name, config in MODELS_TO_RUN.items():
        api_key = os.getenv(config["env_key"])
        if not api_key:
            raise ValueError(f"{model_name}: {config['env_key']} not found in .env file")
    
    if not os.path.exists(CSV_INPUT):
        print(f"Error: CSV file not found: {CSV_INPUT}")
        return
    
    df = pd.read_csv(CSV_INPUT).fillna("N/A")
    
    missing_cols = [col for col in cols_interes if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        cols_interes_filtered = [col for col in cols_interes if col in df.columns]
        data = df[cols_interes_filtered].reset_index(drop=True)
    else:
        data = df[cols_interes].reset_index(drop=True)
    
    if os.path.exists(CSV_OUTPUT):
        try:
            old_df = pd.read_csv(CSV_OUTPUT)
            if len(old_df) == 0 or old_df.empty:
                old_df = pd.DataFrame()
                processed_pairs = set()
            elif "source_ip_hashed" in old_df.columns and "destination_hashed" in old_df.columns:
                processed_pairs = set(zip(old_df["source_ip_hashed"], old_df["destination_hashed"]))
                print(f"Found {len(old_df)} processed rows, resuming...")
            else:
                old_df = pd.DataFrame()
                processed_pairs = set()
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            old_df = pd.DataFrame()
            processed_pairs = set()
    else:
        old_df = pd.DataFrame()
        processed_pairs = set()
    
    if "source_ip_hashed" in data.columns and "destination_hashed" in data.columns:
        data["pair"] = list(zip(data["source_ip_hashed"], data["destination_hashed"]))
        data_new = data[~data["pair"].isin(processed_pairs)].drop(columns=["pair"])
    else:
        data_new = data
    
    if data_new.empty:
        print("No new rows to process.")
        return
    
    print(f"Processing {len(data_new)} new rows...\n")
    
    all_results = []
    total = len(data_new)
    start_time = time.time()
    
    for idx, (i, fila) in enumerate(data_new.iterrows(), 1):
        row_start_time = time.time()
        pair_id = fila.get('source_ip_hashed', 'N/A')[:8] if fila.get('source_ip_hashed') else 'N/A'
        print(f"\n[{idx}/{total}] Processing row (pair: {pair_id}...)")
        print(f"  Server: {fila.get('server_name', 'N/A')}")
        print(f"  Signature: {fila.get('signature_name', 'N/A')[:50]}...")
        
        try:
            row_idx, results = process_single_row((i, fila))
            
            embeddings = {}
            combined_texts = {}
            
            for model_name, parsed_data in results.items():
                combined_text = combine_analysis_text(parsed_data)
                combined_texts[model_name] = combined_text
                embedding = get_embedding(combined_text, model_name)
                embeddings[model_name] = embedding
                time.sleep(0.1)
            
            gemini_embedding = embeddings.get("gemini-2.5-flash")
            similarity_scores = {}
            
            if gemini_embedding:
                for model_name in ["gpt-5-mini", "claude-sonnet-4.5"]:
                    other_embedding = embeddings.get(model_name)
                    if other_embedding:
                        similarity = calculate_similarity(gemini_embedding, other_embedding)
                        similarity_scores[f"similarity_gemini_vs_{model_name.replace('-', '_')}"] = similarity
            
            comparison_results = compare_all_models(results)
            prioridad_metrics = calculate_prioridad_similarity(results)
            
            output_row = {
                "source_ip_hashed": fila.get("source_ip_hashed", ""),
                "destination_hashed": fila.get("destination_hashed", ""),
            }
            
            for model_name in MODELS_TO_RUN.keys():
                model_key = model_name.replace("-", "_")
                parsed = results.get(model_name, {})
                
                output_row[f"{model_key}_prioridad"] = parsed.get("prioridad")
                output_row[f"{model_key}_acciones_remediacion"] = json.dumps(
                    parsed.get("acciones_remediacion", []), ensure_ascii=False
                )
                output_row[f"{model_key}_acciones_prevencion"] = json.dumps(
                    parsed.get("acciones_prevencion", []), ensure_ascii=False
                )
                output_row[f"{model_key}_diagnostico"] = parsed.get("diagnostico", "")
                output_row[f"{model_key}_combined_text"] = combined_texts.get(model_name, "")
            
            output_row.update(similarity_scores)
            output_row.update(comparison_results)
            output_row.update(prioridad_metrics)
            
            all_results.append(output_row)
            
            try:
                new_df = pd.DataFrame(all_results)
                final_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
                    subset=["source_ip_hashed", "destination_hashed"], keep="last"
                )
                final_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
            except Exception as save_error:
                print(f"    Save error: {str(save_error)[:100]}")
            
            row_time = time.time() - row_start_time
            elapsed_total = time.time() - start_time
            avg_time_per_row = elapsed_total / idx
            remaining_rows = total - idx
            estimated_time_remaining = avg_time_per_row * remaining_rows
            
            print(f"  Row {idx} completed in {row_time:.1f}s")
            if similarity_scores:
                sim_str = ", ".join([f"{k.split('_')[-1]}: {v:.3f}" for k, v in similarity_scores.items()])
                print(f"    Similarities: {sim_str}")
            print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%) | ETA: {estimated_time_remaining/60:.1f} min")
            
        except Exception as e:
            print(f"  Error processing row {idx}: {str(e)}")
        
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes")
    print(f"Rows processed: {len(all_results)}")
    
    if all_results:
        new_df = pd.DataFrame(all_results)
        final_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
            subset=["source_ip_hashed", "destination_hashed"], keep="last"
        )
        final_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"Results saved to {CSV_OUTPUT}")
        print(f"Total rows: {len(final_df)}")
        
        if "similarity_gemini_vs_gpt_5_mini" in final_df.columns:
            avg_sim_gpt = final_df["similarity_gemini_vs_gpt_5_mini"].mean()
            print(f"Gemini vs GPT-5-mini avg similarity: {avg_sim_gpt:.4f}")
        
        if "similarity_gemini_vs_claude_sonnet_4_5" in final_df.columns:
            avg_sim_claude = final_df["similarity_gemini_vs_claude_sonnet_4_5"].mean()
            print(f"Gemini vs Claude Sonnet 4.5 avg similarity: {avg_sim_claude:.4f}")


if __name__ == "__main__":
    main()

