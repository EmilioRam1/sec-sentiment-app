# SEC Sentiment Analyzer

🌐 **Demo en vivo:** https://emilioram1.github.io/sec-sentiment-app/
<img width="1669" height="1305" alt="image" src="https://github.com/user-attachments/assets/3a4464b9-202c-476d-affd-18e9a6d995b7" />

App de análisis de sentimiento para archivos de la SEC (10-K, 10-Q, 8-K), usando **Qwen2-0.5B-Instruct** localmente y **Transformers.js** en el navegador.

Curso: **CD3002C.601 — Modelos de IA para Datos No Estructurados**

---

## ¿Para qué sirve?

Los reportes de la SEC revelan el tono de la gerencia sobre el estado de la empresa:

- **10-K** (anual): estrategia, riesgos, perspectivas
- **10-Q** (trimestral): resultados y cambios en el negocio
- **8-K** (eventos relevantes): adquisiciones, cambios ejecutivos, resultados sorpresa

Analizar el sentimiento permite a inversores y analistas **detectar alertas o señales positivas** sin leer cientos de páginas. Un cambio de tono entre trimestres puede anticipar movimientos en el precio de la acción.

> Ejemplo: el reporte 10-K de Apple Inc. está disponible en
> [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL&type=10-K)

---

## Versión local (Flask + Qwen2-0.5B-Instruct)

### Instalación

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Correr la app

```bash
python app.py
```

Abre `http://127.0.0.1:5001` en el navegador.

### Cómo funciona

```
[ Browser ]
   │  sube archivo SEC (.html / .txt)
   ▼
[ Flask app.py ]
   │  extrae texto con BeautifulSoup
   │  aplica chat template → system prompt financiero
   ▼
[ Qwen/Qwen2-0.5B-Instruct ]
   │  generate(max_new_tokens=150, do_sample=False)
   ▼
[ Browser ]   { label: "Positive", explanation: "..." }
```

Resultado: **Positive / Negative / Neutral** + explicación en una oración.

---

## Versión estática (GitHub Pages · Transformers.js)

`index.html` usa **Transformers.js** con `distilbert-base-uncased` corriendo en el navegador:
- Sin servidor, sin instalación
- El modelo (~25 MB) se descarga la primera vez y queda en caché
- Sube un archivo o pega texto directamente

---

## Archivos

```
sec-sentiment-app/
├── app.py              # backend Flask + Qwen2-0.5B-Instruct
├── templates/
│   └── index.html      # UI para versión local
├── index.html          # versión estática (GitHub Pages)
├── requirements.txt
└── README.md
```
