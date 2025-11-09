import os
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import json
import re
from pydantic import BaseModel, Field, IPvAnyAddress, ValidationError, RootModel
from typing import List

MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = r"C:\Users\DmP\Desktop\Gemma-MCD\diagram_2.png"
TOPOLOGY_FILE = r"C:\Users\DmP\Desktop\Gemma-MCD\input\topology.json"


class ServerNode(BaseModel):
    server: str = Field(description="Nombre del servidor")
    ip: IPvAnyAddress = Field(description="IP valida en formato IPv4")
    conectado_a: List[str] = Field(description="Lista de servidores conectados")

class Topology(RootModel[List[ServerNode]]):
    pass


def sanitize_ip(ip_str: str) -> str:
    octets = ip_str.split(".")
    if len(octets) != 4:
        return "0.0.0.0"
    try:
        octets_int = [int(o) for o in octets]
        if all(0 <= o <= 255 for o in octets_int):
            return ip_str
        else:
            return "0.0.0.0"
    except ValueError:
        return "0.0.0.0"


if os.path.exists(TOPOLOGY_FILE):
    exit(0)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "Eres un asistente experto en redes. Genera JSON válido de topologías."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                "Extrae de esta imagen todos los servidores y sus IPs. "
                "Genera un JSON válido estricto con la topología de red.\n\n"
                "INSTRUCCIONES ESTRICTAS:\n"
                "- Solo incluye servidores con IP válida en formato IPv4.\n"
                "- Cada servidor debe aparecer una sola vez.\n"
                "- Cada objeto JSON debe tener exactamente estas claves: 'server', 'ip', 'conectado_a'.\n"
                "- 'conectado_a' debe ser una lista de servidores a los que está conectado.\n"
                "- No agregues texto adicional ni explicaciones.\n"
                "- La salida debe estar encerrada únicamente entre ```json y ```.\n"
                "- Asegúrate de que el último carácter sea ']' seguido de ```.\n"
                "- Si no puedes incluir toda la topología, no respondas."
            )}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]


def generate_response(max_tokens):
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0
        )
    generation = generation[0][input_len:]
    response = processor.decode(generation, skip_special_tokens=True)
    return response


max_tokens = 2000
response = generate_response(max_tokens)

if not response.strip().endswith("```"):
    max_tokens = 4000
    response = generate_response(max_tokens)

match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
json_str = match.group(1) if match else ""

try:
    data = json.loads(json_str)

    for item in data:
        if "ip" in item:
            if ":" in item["ip"]:
                item["ip"] = item["ip"].split(":")[0]
            item["ip"] = sanitize_ip(item["ip"])

    validated = Topology.model_validate(data)

    with open(TOPOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(validated.model_dump_json(indent=2))

except (json.JSONDecodeError, ValidationError):
    with open(r"C:\Users\DmP\Desktop\Gemma-MCD\topology_raw.txt", "w", encoding="utf-8") as f:
        f.write(response)
