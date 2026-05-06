from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re
from bs4 import BeautifulSoup

app = Flask(__name__)
MAX_CHARS = 3000

SECTION_NAMES = {
    '1': 'Business',
    '1A': 'Risk Factors',
    '1B': 'Unresolved Staff Comments',
    '2': 'Properties',
    '3': 'Legal Proceedings',
    '4': 'Mine Safety',
    '5': 'Market for Stock',
    '6': 'Reserved',
    '7': "Management's Discussion",
    '7A': 'Market Risk',
    '8': 'Financial Statements',
    '9': 'Accounting Disagreements',
    '9A': 'Controls & Procedures',
    '9B': 'Other Information',
    '10': 'Directors & Officers',
    '11': 'Executive Compensation',
    '12': 'Security Ownership',
    '13': 'Related Transactions',
    '14': 'Accountant Fees',
    '15': 'Exhibits',
}

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


def detect_sections(text):
    pattern = re.compile(
        r'(?:^|\n)((?:ITEM|Item)\s+(\d{1,2}[A-Za-z]?)\.?\s*[—–\-]?\s*([^\n]{0,80}))',
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    sections = []
    for i, m in enumerate(matches):
        item_num = m.group(2).strip().upper()
        raw_title = m.group(3).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(start + 8000, len(text))
        body = text[start:end].strip()
        if len(body) < 50:
            continue
        known = SECTION_NAMES.get(item_num, "")
        title = known or raw_title[:50] or f"Item {item_num}"
        sections.append({"key": f"Item {item_num}", "title": title, "text": body})

    seen, unique = set(), []
    for s in sections:
        if s["key"] not in seen:
            seen.add(s["key"])
            unique.append(s)
    return unique[:15]


def run_sentiment(text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst. Analyze the sentiment of the SEC filing excerpt. "
                "Begin your reply with exactly one word — 'Positive', 'Negative', or 'Neutral' — "
                "followed by a single concise sentence explaining why."
            ),
        },
        {"role": "user", "content": text[:MAX_CHARS]},
    ]
    ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    low = reply.lower()
    label = "Positive" if low.startswith("positive") else "Negative" if low.startswith("negative") else "Neutral"
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
        text = parse_file(f)
        sections = detect_sections(text)
        return jsonify({"text": text[:10000], "sections": sections})
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
