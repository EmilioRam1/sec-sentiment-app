from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re
from bs4 import BeautifulSoup

app = Flask(__name__)
MAX_CHARS = 3000

print("Cargando Qwen2-0.5B-Instruct…")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", torch_dtype="auto", device_map="auto"
)
print("Modelo listo. Abre http://127.0.0.1:5001")


def parse_file(file_obj):
    name = file_obj.filename.lower()
    raw = file_obj.read()
    if name.endswith((".html", ".htm")):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return re.sub(r" {2,}", " ", text).strip()
    return raw.decode("utf-8", errors="ignore")


def run_sentiment(text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst. Analyze the sentiment of the SEC filing excerpt "
                "provided by the user. Begin your reply with exactly one word — 'Positive', "
                "'Negative', or 'Neutral' — followed by a single concise sentence explaining why."
            ),
        },
        {"role": "user", "content": text[:MAX_CHARS]},
    ]
    ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    low = reply.lower()
    if low.startswith("positive"):
        label = "Positive"
    elif low.startswith("negative"):
        label = "Negative"
    else:
        label = "Neutral"
    return label, reply


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Sin archivo"}), 400
    try:
        return jsonify({"text": parse_file(f)[:10000]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    text = (request.get_json() or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Sin texto"}), 400
    try:
        label, explanation = run_sentiment(text)
        return jsonify({"label": label, "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5001)
