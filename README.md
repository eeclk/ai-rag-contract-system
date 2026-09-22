# 🚀 SSE Chat PoC — FastAPI + Next.js

Uçtan uca (E2E) çalışan bir **Server-Sent Events (SSE) streaming chat** uygulaması.  
Backend mesajları kelime kelime yayınlar; Frontend her kelimeyi anlık ekrana basar.

```
ai-rag-contract-system/
├── backend/
│   ├── main.py            ← FastAPI + SSE endpoint
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── app/
    │   ├── page.tsx       ← Chat UI + stream tüketici
    │   ├── layout.tsx
    │   └── globals.css
    └── package.json
```

---

## ⚡ Hızlı Başlangıç

### 1. Backend (FastAPI)

```bash
# Proje kök dizininden:
cd backend

# Sanal ortam oluştur ve aktifleştir
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Bağımlılıkları yükle
pip install -r requirements.txt

# Sunucuyu başlat
uvicorn main:app --reload --port 8000
```

> Swagger UI: http://localhost:8000/docs

### 2. Frontend (Next.js)

```bash
# Proje kök dizininden (yeni terminal):
cd frontend

# Bağımlılıkları yükle (create-next-app sırasında zaten kuruldu)
npm install

# Geliştirme sunucusunu başlat
npm run dev
```

> Uygulama: http://localhost:3000

---

## 🏗️ Mimari

```
Kullanıcı
  │  (Enter / Gönder)
  ▼
Next.js (port 3000)
  │  POST /chat/stream
  │  { "message": "Merhaba Dünya" }
  ▼
FastAPI (port 8000)
  │  CORSMiddleware ← allow_origins=["http://localhost:3000"]
  │
  └─► async event_generator(message)
        │  kelime kelime yield
        │  delay: 0.3–0.5s
        ▼
      StreamingResponse (text/event-stream)
        data: {"chunk": "Merhaba"}\n\n
        data: {"chunk": " Dünya"}\n\n
        data: [DONE]\n\n
  ▼
Next.js ReadableStream tüketicisi
  response.body.getReader() + TextDecoder
  → Her chunk'ta setMessages() güncellenir
  → Kullanıcı kelime kelime görür
```

---

## 🔌 API Referansı

### `POST /chat/stream`
```json
// Request body
{ "message": "Merhaba, beni kelime kelime anlat." }

// Response stream (text/event-stream)
data: {"chunk": "Merhaba,"}\n\n
data: {"chunk": " beni"}\n\n
data: {"chunk": " kelime"}\n\n
...
data: [DONE]\n\n
```

### `GET /chat/stream?message=...` *(tarayıcı testi)*
Adres çubuğundan doğrudan test edilebilir:  
`http://localhost:8000/chat/stream?message=Merhaba+Dünya`

### `GET /health`
```json
{ "status": "ok" }
```

---

## 🧩 Önemli Teknik Detaylar

| Konu | Detay |
|------|-------|
| **SSE Formatı** | `data: <JSON>\n\n` — çift newline her mesajı sonlandırır |
| **Buffer yönetimi** | Frontend `\n\n` sınırına kadar buffer'lar, tam mesajları parse eder |
| **CORS** | Backend'de `CORSMiddleware` ile `http://localhost:3000` izinli |
| **Abort** | "Durdur" butonu `AbortController` ile stream'i temiz kapatır |
| **Otomatik scroll** | Her yeni token sonrası `scrollIntoView` çağrılır |
| **Yazma imleci** | Stream aktifken yanıp sönen `│` imleci gösterilir |
| **Hata yönetimi** | `try/catch` + hata banner'ı + başarısız mesajı kaldırır |

---

## 🔮 Sonraki Adımlar (Gerçek LLM Entegrasyonu)

```python
# main.py içinde event_generator'ı şununla değiştirin:
import openai

async def event_generator(message: str):
    client = openai.AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield f"data: {json.dumps({'chunk': delta})}\n\n"
    yield "data: [DONE]\n\n"
```
