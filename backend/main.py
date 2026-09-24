import asyncio
from contextlib import asynccontextmanager
import gc
import io
import json
import logging
import os
import re
import tempfile
import uuid
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
import pypdf
from pydantic import BaseModel
import uvicorn

from session_manager import (
    RecursiveCharacterTextSplitter,
    SessionManager,
    free_memory,
)

load_dotenv()

# Logger yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-backend")

# ---------------------------------------------------------------------------
# Multi-Tenant Session Manager Kurulumu
# ---------------------------------------------------------------------------
# 2 saat (7200 sn) TTL ile oturum bazlı ChromaDB yöneticisi
session_manager = SessionManager(ttl_seconds=7200)

# Akıllı Parçalayıcı (RecursiveCharacterTextSplitter)
# 600 karakter boyut, 120 karakter örtüşme, paragraf ve tablo koruyucu ayırıcılar
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " "],
)


# ---------------------------------------------------------------------------
# Oturum ID Çözücü Dependency (X-Session-ID Header veya Parametre)
# ---------------------------------------------------------------------------
def get_session_id(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    session_id: Optional[str] = Query(None, alias="session_id"),
) -> str:
    """
    İstek başlıklarından (X-Session-ID) veya query parametresinden oturum UUID'sini alır.
    Oturum izolasyonu için zorunludur.
    """
    sid = x_session_id or session_id
    if not sid or not sid.strip():
        raise HTTPException(
            status_code=400,
            detail="Multi-tenant oturum izolasyonu için 'X-Session-ID' başlığı zorunludur. Lütfen geçerli bir oturum UUID'si belirtin.",
        )
    return sid.strip()


# ---------------------------------------------------------------------------
# FastAPI Lifespan (Arka Plan Temizleme Görevi)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama yaşam döngüsü:
    - Başlangıçta 2 saatten eski oturumları temizleyen asenkron arka plan görevini başlatır.
    - Kapanışta arka plan görevlerini temiz şekilde sonlandırır.
    """
    cleanup_task = asyncio.create_task(
        session_manager.start_background_cleanup(check_interval_seconds=600)
    )
    logger.info("DocuSense Multi-Tenant RAG Backend başlatıldı.")
    yield
    session_manager.stop_background_cleanup()
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("DocuSense Multi-Tenant RAG Backend sonlandırıldı.")


# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DocuSense AI - Multi-Tenant RAG Contract System API",
    description="FastAPI, Multi-Tenant ChromaDB İzolasyonu, Zengin Metadata Chunking ve Groq Streaming API",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS yapılandırması
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
# Pydantic Modelleri
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: Optional[str] = "Yapay zeka akışını test ediyorum."


# ---------------------------------------------------------------------------
# RAG SSE Generator (Multi-Tenant & Zengin Metadata Destekli)
# ---------------------------------------------------------------------------
async def rag_stream_generator(
    message: str, session_id: str
) -> AsyncGenerator[str, None]:
    """
    1. Sadece ilgili oturuma (session_id) ait izole ChromaDB koleksiyonundan parçaları çeker.
    2. A oturumunun yüklediği PDF sadece A oturumuna servis edilir; B oturumu ASLA göremez.
    3. Küçük ve orta ölçekli belgelerde doküman sırasıyla sayfa ve metadata bağlamını oluşturur.
    4. Büyük belgelerde Hibrit Arama (Vektör + Anahtar Kelime + Madde) ile en alakalı 8 parçayı seçer.
    5. Groq API ile streaming yanıt üretip SSE formatında istemciye aktarır.
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

    # İlgili oturuma özel ChromaDB koleksiyonunu al
    coll = session_manager.get_collection(session_id)
    total_docs = coll.count() if coll else 0
    context_chunks: list[str] = []

    # İZOLASYON KONTROLÜ: Oturumda henüz döküman yoksa
    if total_docs == 0:
        empty_msg = (
            f"ℹ️ Bu oturuma ({session_id}) ait yüklenmiş herhangi bir sözleşme veya belge bulunmamaktadır.\n"
            "Lütfen önce 'X-Session-ID' başlığı ile bir PDF belgesi yükleyin."
        )
        payload = json.dumps({"chunk": empty_msg}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
        return

    all_data = coll.get()
    all_docs: list[str] = all_data.get("documents", []) or []
    all_metas: list[dict] = all_data.get("metadatas", []) or []

    # DURUM A: Küçük ve orta boyutlu dokümanlar (<= 25 parça)
    # Hepsini sayfa numarası ve chunk_id sırasıyla bağlama ekle
    if total_docs <= 25 and len(all_docs) > 0:
        sorted_items = sorted(
            zip(all_metas, all_docs),
            key=lambda x: (
                x[0].get("page_number", 0) if isinstance(x[0], dict) else 0,
                x[0].get("chunk_id", "") if isinstance(x[0], dict) else "",
            ),
        )
        for meta, doc in sorted_items:
            fname = meta.get("source_file", "Belge") if isinstance(meta, dict) else "Belge"
            p_num = meta.get("page_number", "-") if isinstance(meta, dict) else "-"
            c_id = meta.get("chunk_id", "-") if isinstance(meta, dict) else "-"
            context_chunks.append(
                f"[Kaynak: {fname} | Sayfa: {p_num} | Parça ID: {c_id}]\n{doc}"
            )

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

        # En yüksek puanlı 8 parçayı seç
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]

        # Doküman içi sayfa ve parça sırasına göre hizala
        selected: list[tuple[int, str, str, str, str]] = []
        seen_chunks: set[str] = set()

        for doc, _ in sorted_docs:
            try:
                idx = all_docs.index(doc)
                meta = all_metas[idx] if idx < len(all_metas) else {}
                chunk_id = meta.get("chunk_id", str(idx))
                page_num = meta.get("page_number", 0)
                source_file = meta.get("source_file", "Belge")
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    selected.append((page_num, chunk_id, source_file, doc))
            except ValueError:
                continue

        selected.sort(key=lambda x: (x[0], x[1]))
        for page_num, chunk_id, fname, doc in selected:
            context_chunks.append(
                f"[Kaynak: {fname} | Sayfa: {page_num} | Parça ID: {chunk_id}]\n{doc}"
            )

    # Bağlam oluşturma
    context_str = (
        "\n\n".join(context_chunks)
        if context_chunks
        else "Bu oturumda indekslenmiş metin parçası bulunamadı."
    )

    system_prompt = (
        "Sen uzman bir Sözleşme ve Hukuk Analiz Asistanısın. "
        "Kullanıcının sorularını verilen <context> içeriğindeki belgelere dayanarak açık, net, detaylı ve doğru şekilde yanıtla. "
        "Cevap verirken referans verdiğin maddeleri ve sayfa numaralarını ('Sayfa X, Parça Y referansıyla') belirt. "
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
        logger.exception("Groq streaming sırasında hata oluştu")
        err_payload = json.dumps(
            {"chunk": f"\n\n[Groq Hatası]: {str(e)}"}, ensure_ascii=False
        )
        yield f"data: {err_payload}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/upload", summary="PDF Yükle, Zengin Metadata ile Parçala ve İndeksle")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
):
    """
    1. İstemciden zorunlu X-Session-ID alır.
    2. PDF'i okur, sayfa sayfa tarar.
    3. RecursiveCharacterTextSplitter ile (600 karakter, 120 overlap) böler.
    4. Her parçaya zorunlu metadata ekler:
       {
           "source_file": file.filename,
           "page_number": int,
           "chunk_id": f"{source_file}_p{page}_{index}",
           "char_length": int
       }
    5. Sadece o oturumun koleksiyonunu (tenant_{clean_session_id}) günceller.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz dosya formatı. Lütfen yalnızca .pdf dosyası yükleyin.",
        )

    tmp_path = None
    try:
        content = await file.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Yüklenen dosya boş.")

        MAX_FILE_SIZE = 35 * 1024 * 1024  # 35 MB
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Dosya boyutu çok büyük ({round(file_size / (1024 * 1024), 1)} MB). Lütfen 35 MB altındaki PDF yükleyin.",
            )

        # PyPDF ile PDF'i aç (bellek optimizasyonu)
        if file_size < 5 * 1024 * 1024:
            reader = pypdf.PdfReader(io.BytesIO(content))
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_path = tmp.name
                tmp.write(content)
            reader = pypdf.PdfReader(tmp_path)

        pages_count = len(reader.pages)
        if pages_count == 0:
            raise HTTPException(
                status_code=400, detail="PDF belgesinde okunabilir sayfa bulunamadı."
            )

        MAX_PAGES = 150
        pages_to_process = min(pages_count, MAX_PAGES)

        all_chunks: list[str] = []
        all_metadatas: list[dict] = []
        all_ids: list[str] = []

        # -------------------------------------------------------------------
        # ZENGİN METADATA DESTEKLİ PARÇALAMA (Sayfa Sayfa Taranır)
        # -------------------------------------------------------------------
        for page_idx in range(pages_to_process):
            page_num = page_idx + 1
            try:
                page = reader.pages[page_idx]
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                # Sayfa bazlı RecursiveCharacterTextSplitter çalıştırma
                page_chunks = text_splitter.split_text(page_text)

                for chunk_idx, chunk in enumerate(page_chunks):
                    clean_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename)
                    chunk_id = f"{clean_filename}_p{page_num}_{chunk_idx}"
                    char_len = len(chunk)

                    metadata = {
                        "source_file": file.filename,
                        "page_number": int(page_num),
                        "chunk_id": chunk_id,
                        "char_length": int(char_len),
                    }

                    all_chunks.append(chunk)
                    all_ids.append(chunk_id)
                    all_metadatas.append(metadata)

            except Exception as p_err:
                logger.warning(f"Sayfa {page_num} işlenirken hata: {p_err}")
                continue

        # Bellek tahliyesi
        del reader
        del content
        free_memory()

        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail="PDF dosyasından metin çıkarılamadı. Belge taranmış resim veya korumalı olabilir.",
            )

        # Maksimum chunk koruması
        MAX_CHUNKS = 450
        if len(all_chunks) > MAX_CHUNKS:
            logger.warning(
                f"Doküman çok büyük ({len(all_chunks)} parça). İlk {MAX_CHUNKS} parça alınıyor."
            )
            all_chunks = all_chunks[:MAX_CHUNKS]
            all_ids = all_ids[:MAX_CHUNKS]
            all_metadatas = all_metadatas[:MAX_CHUNKS]

        # -------------------------------------------------------------------
        # MULTI-TENANT OTURUM İZOLASYONU
        # Yalnızca bu session_id'ye ait koleksiyon sıfırlanıp yeniden yazılır!
        # Diğer kullanıcıların koleksiyonlarına kesinlikle dokunulmaz.
        # -------------------------------------------------------------------
        coll = session_manager.reset_session_collection(session_id)

        # Batch indeksleme (bellek ve CPU optimizasyonu)
        if len(all_chunks) <= 25:
            coll.add(ids=all_ids, documents=all_chunks, metadatas=all_metadatas)
        else:
            BATCH_SIZE = 15
            for i in range(0, len(all_chunks), BATCH_SIZE):
                coll.add(
                    ids=all_ids[i : i + BATCH_SIZE],
                    documents=all_chunks[i : i + BATCH_SIZE],
                    metadatas=all_metadatas[i : i + BATCH_SIZE],
                )
                free_memory()

        free_memory()

        logger.info(
            f"Oturum [{session_id}] için '{file.filename}' belgesi indekslendi: "
            f"{pages_to_process} sayfa, {len(all_chunks)} parça. Koleksiyon: {coll.name}"
        )

        return {
            "status": "success",
            "session_id": session_id,
            "collection_name": coll.name,
            "filename": file.filename,
            "pages_processed": pages_to_process,
            "total_pages": pages_count,
            "chunks_indexed": len(all_chunks),
            "sample_metadata": all_metadatas[0] if all_metadatas else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF yükleme sırasında hata")
        raise HTTPException(
            status_code=500, detail=f"PDF işlenirken bir hata oluştu: {str(e)}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        free_memory()


@app.post("/chat/stream", summary="Multi-Tenant RAG Streaming Chat (POST)")
async def chat_stream_post(
    request: ChatRequest,
    session_id: str = Depends(get_session_id),
) -> StreamingResponse:
    """
    Oturuma özel izole koleksiyondan arama yapar ve Groq LLM yanıtını
    kelime kelime Server-Sent Events (SSE) formatında stream eder.
    """
    return StreamingResponse(
        rag_stream_generator(request.message or "", session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id,
        },
    )


@app.get("/chat/stream", summary="Multi-Tenant RAG Streaming Chat (GET – Test)")
async def chat_stream_get(
    message: str = "Yapay zeka akışını test ediyorum.",
    session_id: str = Depends(get_session_id),
) -> StreamingResponse:
    """Tarayıcıdan veya SSE test araçlarından hızlı doğrulama için GET metodu."""
    return StreamingResponse(
        rag_stream_generator(message, session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id,
        },
    )


@app.get("/documents", summary="Oturuma Ait İndekslenen Belge Bilgileri")
async def get_documents(session_id: str = Depends(get_session_id)) -> dict:
    """
    Yalnızca istek yapan oturuma (X-Session-ID) ait parçaları ve dosya adını döner.
    """
    try:
        coll = session_manager.get_collection(session_id)
        if not coll:
            return {
                "status": "ok",
                "session_id": session_id,
                "total_chunks": 0,
                "active_filename": None,
            }

        cnt = coll.count()
        last_filename = None
        if cnt > 0:
            sample = coll.get(limit=1)
            metas = sample.get("metadatas", [])
            if metas and isinstance(metas[0], dict):
                last_filename = metas[0].get("source_file") or metas[0].get("filename")

        return {
            "status": "ok",
            "session_id": session_id,
            "collection_name": coll.name,
            "total_chunks": cnt,
            "active_filename": last_filename,
        }
    except Exception as e:
        return {"status": "error", "session_id": session_id, "total_chunks": 0, "error": str(e)}


@app.delete("/documents", summary="Oturumun Vektör Koleksiyonunu Sıfırla")
async def reset_documents(session_id: str = Depends(get_session_id)) -> dict:
    """
    Yalnızca istek yapan oturumun koleksiyonunu sıfırlar.
    Diğer oturumlar bundan etkilenmez.
    """
    session_manager.reset_session_collection(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "message": f"Oturuma ({session_id}) ait koleksiyon başarıyla temizlendi.",
    }


@app.get("/health", summary="Health Check")
async def health() -> dict:
    """Sunucu ve tenant yönetici sağlık kontrolü."""
    active_tenants = len(session_manager.last_accessed)
    return {
        "status": "ok",
        "active_tenants": active_tenants,
        "ttl_seconds": session_manager.ttl_seconds,
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
