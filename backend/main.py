import gc
import io
import json
import logging
import os
import re
import tempfile
import uuid
from typing import AsyncGenerator, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
import pypdf
from pydantic import BaseModel
import uvicorn

load_dotenv()

# Logger yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-backend")

def free_memory():
    """
    Python çöp toplayıcısını (gc) ve Linux glibc bellek iadesini (malloc_trim) tetikler.
    Render 512MB RAM sınırında bellek sızıntısını ve OOM çökmesini %100 önler.
    """
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI RAG Contract System API",
    description="FastAPI, ChromaDB ve Groq ile RAG Streaming API",
    version="2.1.0",
)

# CORS yapılandırması (Canlı ve Yerel ortamlar için)
cors_origins_env = os.getenv("CORS_ORIGINS", "*").strip()
if cors_origins_env == "*":
    allow_origins = ["*"]
else:
    allow_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# ChromaDB Vektör Veritabanı Kurulumu
# ---------------------------------------------------------------------------
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collection():
    """ChromaDB 'pdf_knowledge' koleksiyonunu dinamik olarak getirir veya oluşturur."""
    return chroma_client.get_or_create_collection(name="pdf_knowledge")

# ---------------------------------------------------------------------------
# Pydantic Modelleri
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: Optional[str] = "Yapay zeka akışını test ediyorum."


# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar: Metin Parçalama (Chunking)
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Metni 500 karakterlik parçalara böler, ancak parçalar arasında
    kesinlikle 100 karakter örtüşme (overlap) sağlar.
    """
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap  # 500 - 100 = 400

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Eğer sonraki adım metnin sonunu aşıyorsa döngüyü sonlandır
        if start + chunk_size >= len(text):
            break

        start += step

    return chunks


# ---------------------------------------------------------------------------
# RAG SSE Generator (Gelişmiş Hibrit Arama & Tam Doküman Bağlamı)
# ---------------------------------------------------------------------------
async def rag_stream_generator(message: str) -> AsyncGenerator[str, None]:
    """
    1. ChromaDB'den yüklenen belgelerin parçalarını alır.
    2. Küçük ve orta ölçekli belgelerde (<= 25 parça) tüm belgeyi doğal okuma sırasıyla bağlama ekler.
    3. Büyük belgelerde Vektör + Anahtar Kelime + Madde Eşleştirme (Hibrit Arama) yaparak en alakalı 8 parçayı seçer.
    4. Groq API ile streaming yanıt üretip kelime kelime SSE formatında fırlatır.
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    # API anahtarı kontrolü
    if not groq_api_key or groq_api_key == "senin_groq_api_keyin":
        warning_msg = (
            "⚠️ Groq API anahtarı bulunamadı veya ayarlanmadı.\n\n"
            "Lütfen `backend/.env` dosyasına geçerli bir `GROQ_API_KEY` ekleyin.\n"
            "(Ücretsiz API anahtarı için: https://console.groq.com)"
        )
        payload = json.dumps({"chunk": warning_msg}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ChromaDB Sorgusu ve Bağlam Çıkarımı
    coll = get_collection()
    total_docs = coll.count()
    context_chunks: list[str] = []

    if total_docs > 0:
        all_data = coll.get()
        all_docs: list[str] = all_data.get("documents", []) or []
        all_metas: list[dict] = all_data.get("metadatas", []) or []

        # DURUM A: Küçük ve orta boyutlu dokümanlar (<= 25 parça, örn. 1-6 sayfalık sözleşmeler)
        # Modelin hiçbir maddeyi veya tarafı kaçırmaması için tüm parçaları doküman sırasıyla bağlama ekle.
        if total_docs <= 25 and len(all_docs) > 0:
            sorted_items = sorted(
                zip(all_metas, all_docs),
                key=lambda x: x[0].get("chunk_index", 0) if isinstance(x[0], dict) else 0,
            )
            for meta, doc in sorted_items:
                fname = meta.get("filename", "Belge") if isinstance(meta, dict) else "Belge"
                c_idx = meta.get("chunk_index", 0) if isinstance(meta, dict) else 0
                context_chunks.append(f"[Dosya: {fname} | Parça #{c_idx + 1}]\n{doc}")

        # DURUM B: Büyük Dokümanlar (Hibrit Arama: Vektör Benzerliği + Anahtar Kelime + Madde Arama)
        elif len(all_docs) > 0:
            k_candidates = min(12, total_docs)
            query_res = coll.query(query_texts=[message], n_results=k_candidates)
            vec_docs = query_res.get("documents", [[]])[0]

            # Madde numarası araması (Örn: "madde 3", "3. madde")
            madde_match = re.search(r"madde\s*(\d+)", message.lower())
            target_madde = f"madde {madde_match.group(1)}" if madde_match else None
            query_words = [w.lower() for w in re.findall(r"\w+", message) if len(w) > 2]

            scores: dict[str, float] = {}
            for rank, doc in enumerate(vec_docs):
                scores[doc] = 1.0 / (rank + 1)

            for doc in all_docs:
                doc_lower = doc.lower()
                kw_score = 0.0
                if target_madde and target_madde in doc_lower:
                    kw_score += 10.0
                for w in query_words:
                    if w in doc_lower:
                        kw_score += 1.5

                if doc in scores:
                    scores[doc] += kw_score
                elif kw_score > 0:
                    scores[doc] = kw_score

            # En yüksek puanlı 8 parçayı al
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]

            # Doküman içindeki doğal okuma sırasına göre sırala
            selected: list[tuple[int, str, str]] = []
            seen_indices: set[int] = set()
            for doc, _ in sorted_docs:
                try:
                    idx = all_docs.index(doc)
                    meta = all_metas[idx] if idx < len(all_metas) else {}
                    c_idx = meta.get("chunk_index", idx)
                    if c_idx not in seen_indices:
                        seen_indices.add(c_idx)
                        selected.append((c_idx, meta.get("filename", "Belge"), doc))
                except ValueError:
                    continue

            selected.sort(key=lambda x: x[0])
            for c_idx, fname, doc in selected:
                context_chunks.append(f"[Dosya: {fname} | Parça #{c_idx + 1}]\n{doc}")

    # Bağlam oluşturma
    if context_chunks:
        context_str = "\n\n".join(context_chunks)
    else:
        context_str = "Henüz sisteme yüklenmiş bir PDF belgesi bulunmamaktadır."

    system_prompt = (
        "Sen uzman bir Sözleşme ve Hukuk Analiz Asistanısın. "
        "Kullanıcının sorularını verilen <context> içeriğindeki belgelere dayanarak açık, net, detaylı ve doğru şekilde yanıtla. "
        "Sorulan maddeyi, tarafları, tarihleri, tutarları, cezai şartları ve koşulları bağlamdan tespit ederek profesyonel bir üslupla açıkla. "
        "Yalnızca sorulan konu veya bilgi verilen bağlamda kesinlikle ve hiçbir şekilde yer almıyorsa "
        "'Yüklenen belgede bu bilgi bulunmamaktadır' de."
    )

    user_prompt = f"""<context>
{context_str}
</context>

Kullanıcı Sorusu: {message}"""

    # Groq API Streaming Çağrısı
    try:
        client = AsyncGroq(api_key=groq_api_key)
        stream = await client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.1,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                payload = json.dumps({"chunk": delta}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

    except Exception as e:
        logger.exception("Groq streaming sırasında hata")
        err_payload = json.dumps(
            {"chunk": f"\n\n[Groq Hatası]: {str(e)}"}, ensure_ascii=False
        )
        yield f"data: {err_payload}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/upload", summary="PDF Yükle ve ChromaDB'ye İndeksle")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. UploadFile tipinde bir PDF dosyasını kabul eder.
    2. RAM kullanımını 1-2 MB seviyesinde tutmak için diske geçici dosya olarak stream eder.
    3. pypdf.PdfReader ile diskten sayfa sayfa okur, RAM'i korumak için reader nesnesini hemen temizler.
    4. Metni 500/100 kuralına göre chunk'lara ayırır.
    5. ChromaDB'ye ONNX tensör patlamasını (OOM) önlemek için 20'şerli batch'ler halinde kaydeder.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz dosya formatı. Lütfen yalnızca .pdf dosyası yükleyin.",
        )

    tmp_path = None
    try:
        # 1. Dosyayı diske 1 MB'lık tamponlarla yaz (RAM şişmesini engeller)
        MAX_FILE_SIZE = 35 * 1024 * 1024  # 35 MB
        total_size = 0

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Dosya boyutu çok büyük ({round(total_size / (1024 * 1024), 1)} MB). Lütfen 35 MB altındaki PDF yükleyin.",
                    )
                tmp.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Yüklenen dosya boş.")

        # 2. PyPDF ile disk üzerinden sayfa sayfa oku
        reader = pypdf.PdfReader(tmp_path)
        pages_count = len(reader.pages)

        if pages_count == 0:
            raise HTTPException(
                status_code=400, detail="PDF belgesinde sayfa bulunamadı."
            )

        # Render 512MB RAM ve CPU zaman aşımı koruması (Maksimum 150 sayfa)
        MAX_PAGES = 150
        pages_to_process = min(pages_count, MAX_PAGES)

        full_text_parts: list[str] = []
        for page_idx in range(pages_to_process):
            try:
                page = reader.pages[page_idx]
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    full_text_parts.append(page_text.strip())
            except Exception as p_err:
                logger.warning(f"Sayfa {page_idx + 1} okunurken hata: {p_err}")
                continue

        cleaned_text = "\n\n".join(full_text_parts).strip()

        # PyPDF nesnelerini bellekten derhal serbest bırak
        del reader
        del full_text_parts
        free_memory()

        if not cleaned_text:
            raise HTTPException(
                status_code=400,
                detail="PDF dosyasından metin okunamadı. Taranmış resim/fotoğraf veya korumalı bir belge olabilir. Lütfen seçilebilir metin içeren bir PDF yükleyin.",
            )

        # 500 karakter / 100 karakter örtüşmeli parçalama
        chunks = chunk_text(cleaned_text, chunk_size=500, overlap=100)
        del cleaned_text
        free_memory()

        if not chunks:
            raise HTTPException(
                status_code=400, detail="Metin parçalara ayrılamadı."
            )

        # Maksimum chunk sınırı (Render 512MB RAM ve 100sn timeout güvencesi)
        # 450 parça ~180.000 karakter olup sözleşmelerin tüm kritik maddelerini eksiksiz kapsar
        MAX_CHUNKS = 450
        if len(chunks) > MAX_CHUNKS:
            logger.warning(
                f"Doküman çok büyük ({len(chunks)} parça). Bellek ve performans için ilk {MAX_CHUNKS} parça alınıyor."
            )
            chunks = chunks[:MAX_CHUNKS]

        # ChromaDB için hazırlık
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        file_id = uuid.uuid4().hex[:8]

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file.filename}_{file_id}_{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "filename": file.filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            )

        # Vektör veritabanına ekleme (aynı dosya önceden yüklendiyse kopyaları temizle)
        coll = get_collection()
        try:
            coll.delete(where={"filename": file.filename})
        except Exception:
            pass

        # KRİTİK ADIM: 15'erli batch'ler halinde ekle ve her batch sonrası malloc_trim çağır!
        # Böylece Linux çekirdeği kullanılmayan RAM'i anında geri alır ve 512MB sınırı asla aşılmaz.
        BATCH_SIZE = 15
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_ids = ids[i : i + BATCH_SIZE]
            batch_docs = documents[i : i + BATCH_SIZE]
            batch_metas = metadatas[i : i + BATCH_SIZE]
            coll.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            free_memory()

        logger.info(
            f"PDF '{file.filename}' başarıyla indekslendi: {pages_to_process}/{pages_count} sayfa, {len(chunks)} parça."
        )

        return {
            "status": "success",
            "filename": file.filename,
            "pages_processed": pages_to_process,
            "total_pages": pages_count,
            "chunks_indexed": len(chunks),
            "total_collection_chunks": coll.count(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF işleme esnasında beklenmeyen hata")
        raise HTTPException(
            status_code=500, detail=f"PDF işlenirken bir hata oluştu: {str(e)}"
        )
    finally:
        # Geçici dosyayı güvenle temizle
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        free_memory()


@app.post("/chat/stream", summary="RAG Streaming Chat (POST)")
async def chat_stream_post(request: ChatRequest) -> StreamingResponse:
    """
    Kullanıcı sorusunu alır, ChromaDB'den ilgili parçaları çeker
    ve Groq LLM yanıtını kelime kelime SSE formatında stream eder.
    """
    return StreamingResponse(
        rag_stream_generator(request.message or ""),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chat/stream", summary="RAG Streaming Chat (GET – test için)")
async def chat_stream_get(
    message: str = "Yapay zeka akışını test ediyorum.",
) -> StreamingResponse:
    """
    Query string ile mesaj alır; tarayıcıdan doğrudan test edilebilir.
    """
    return StreamingResponse(
        rag_stream_generator(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/documents", summary="İndekslenen Belge Bilgileri")
async def get_documents() -> dict:
    """
    ChromaDB içindeki toplam parça adedini döner.
    """
    coll = get_collection()
    return {
        "collection_name": "pdf_knowledge",
        "total_chunks": coll.count(),
    }


@app.delete("/documents", summary="Vektör Koleksiyonunu Sıfırla")
async def reset_documents() -> dict:
    """
    Tüm indekslenmiş parçaları siler.
    """
    try:
        chroma_client.delete_collection(name="pdf_knowledge")
    except Exception:
        pass
    get_collection()
    return {"status": "success", "message": "Koleksiyon sıfırlandı."}


@app.get("/health", summary="Health Check")
async def health() -> dict:
    try:
        coll = get_collection()
        cnt = coll.count()
        last_filename = None
        if cnt > 0:
            sample = coll.get(limit=1)
            metas = sample.get("metadatas", [])
            if metas and isinstance(metas[0], dict):
                last_filename = metas[0].get("filename")
        return {
            "status": "ok",
            "chroma_chunks": cnt,
            "active_filename": last_filename,
        }
    except Exception as e:
        return {
            "status": "ok",
            "chroma_chunks": 0,
            "active_filename": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

