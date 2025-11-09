from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import glob
import pandas as pd
import json
import hashlib
import subprocess

WIN_INPUT_DIR = r"C:\Users\DmP\Desktop\Gemma-MCD\input"
WSL_INPUT_DIR = "/mnt/c/Users/DmP/Desktop/Gemma-MCD/input"

PYTHON_ENV = r"C:\Users\DmP\Desktop\Gemma-MCD\gemma_env2\Scripts\python.exe"
LLM_SCRIPT = r"C:\Users\DmP\Desktop\Gemma-MCD\gemma_v2.py"
ENRICH_SCRIPT = r"C:\Users\DmP\Desktop\Gemma-MCD\enriched_ip.py"
GEMINI_SCRIPT = r"C:\Users\DmP\Desktop\Gemma-MCD\gemini.py"

TOPOLOGY_FILE = os.path.join(WSL_INPUT_DIR, "topology.json")
ENRICHED_IP_FILE = "/mnt/c/Users/DmP/Desktop/Gemma-MCD/output/ips_resumen_llm.csv"
GEMINI_OUTPUT = "/mnt/c/Users/DmP/Desktop/Gemma-MCD/output/analisis_gemini_final.csv"


def get_latest_csv_file():
    files = glob.glob(os.path.join(WSL_INPUT_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError("No se encontró ningún archivo CSV en la carpeta de entrada.")
    return max(files, key=os.path.getctime)


def run_external_script(script_path):
    python_path_quoted = f'"{PYTHON_ENV}"'
    script_file_quoted = f'"{script_path}"'
    full_command_str = (
        f'/mnt/c/Windows/System32/cmd.exe /c '
        f'{python_path_quoted} {script_file_quoted}'
    )

    result = subprocess.run(
        full_command_str,
        capture_output=True,
        text=True,
        encoding='latin-1',
        cwd=WSL_INPUT_DIR,
        shell=True
    )

    if result.returncode != 0:
        raise Exception(f"Error al ejecutar {script_path}: {result.stderr}")


def enrich_ips_task(**kwargs):
    cached_ips = set()
    if os.path.exists(ENRICHED_IP_FILE):
        enriched_df = pd.read_csv(ENRICHED_IP_FILE)
        if "source_ip" in enriched_df.columns:
            cached_ips = set(enriched_df["source_ip"].astype(str))

    latest_file = get_latest_csv_file()
    input_df = pd.read_csv(latest_file)
    if "source_ip" not in input_df.columns:
        raise ValueError(f"El archivo {latest_file} no contiene la columna 'source_ip'")

    current_ips = set(input_df["source_ip"].astype(str))
    new_ips = current_ips - cached_ips

    if len(new_ips) == 0:
        return

    run_external_script(ENRICH_SCRIPT)
    if not os.path.exists(ENRICHED_IP_FILE):
        raise FileNotFoundError(f"No se encontró {ENRICHED_IP_FILE} después de ejecutar enriched_ip.py")


def run_llm_script(**kwargs):
    run_external_script(LLM_SCRIPT)
    if not os.path.exists(TOPOLOGY_FILE):
        raise FileNotFoundError(f"No se encontró {TOPOLOGY_FILE} después de ejecutar gemma_v2.py")


def process_csv(**kwargs):
    try:
        input_file = get_latest_csv_file()
        df = pd.read_csv(input_file)

        with open(TOPOLOGY_FILE, "r", encoding="utf-8") as f:
            topology_data = json.load(f)
        topo_df = pd.DataFrame(topology_data)

        df['destination'] = (
            df['destination']
            .astype(str)
            .str.strip()
            .str.split(':')
            .str[0]
        )
        topo_df['ip'] = topo_df['ip'].astype(str).str.strip()

        df = df.merge(
            topo_df[['ip', 'server', 'conectado_a']],
            how='left',
            left_on='destination',
            right_on='ip'
        )
        df['server_name'] = df['server'].fillna('server_not_found')
        df.drop(columns=['ip', 'server'], inplace=True)

        if os.path.exists(ENRICHED_IP_FILE):
            enriched_df = pd.read_csv(ENRICHED_IP_FILE)
            df = df.merge(enriched_df, how='left', on='source_ip')

        def hash_value(x):
            return hashlib.sha256(str(x).encode()).hexdigest()

        df['source_ip_hashed'] = df['source_ip'].apply(hash_value)
        df['destination_hashed'] = df['destination'].apply(hash_value)

        base_name = os.path.basename(input_file)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(WSL_INPUT_DIR, f"{name}_final{ext}")
        df.to_csv(output_file, index=False)

    except Exception as e:
        raise


def run_gemini_script(**kwargs):
    run_external_script(GEMINI_SCRIPT)
    if not os.path.exists(GEMINI_OUTPUT):
        raise FileNotFoundError(f"No se encontró {GEMINI_OUTPUT} después de ejecutar gemini.py")


def merge_final_output(**kwargs):
    latest_final = max(glob.glob(os.path.join(WSL_INPUT_DIR, "*_final.csv")), key=os.path.getctime)
    df_main = pd.read_csv(latest_final)
    df_gemini = pd.read_csv(GEMINI_OUTPUT)

    df_merged = df_main.merge(
        df_gemini,
        how="left",
        on=["source_ip_hashed", "destination_hashed"]
    )

    base_name = os.path.basename(latest_final)
    name, ext = os.path.splitext(base_name)
    output_final = os.path.join(WSL_INPUT_DIR, f"{name}_priorizado{ext}")
    df_merged.to_csv(output_final, index=False, encoding="utf-8-sig")


with DAG(
    dag_id="local_llm_pipeline",
    description="Pipeline que enriquece IPs, ejecuta Gemma y procesa CSV",
    start_date=datetime(2025, 10, 11),
    schedule_interval="@daily",
    catchup=False,
    tags=["llm", "local", "pipeline", "ip-enrichment"]
) as dag:

    t0 = PythonOperator(task_id="enrich_ips", python_callable=enrich_ips_task)
    t1 = PythonOperator(task_id="run_llm", python_callable=run_llm_script)
    t2 = PythonOperator(task_id="process_csv", python_callable=process_csv)
    t3 = PythonOperator(task_id="run_gemini", python_callable=run_gemini_script)
    t4 = PythonOperator(task_id="merge_final", python_callable=merge_final_output)

    t0 >> t1 >> t2 >> t3 >> t4
