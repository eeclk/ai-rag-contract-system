import asyncio
import gc
import hashlib
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger("rag-session-manager")


def free_memory():
    """
    Python çöp toplayıcısını (gc) ve Linux glibc bellek iadesini (malloc_trim) tetikler.
    Render 512MB RAM sınırında bellek sızıntısını ve OOM çökmesini önler.
    """
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. ZENGİN METADATA DESTEKLİ AKILLI PARÇALAYICI (CHUNKER)
# ---------------------------------------------------------------------------
class RecursiveCharacterTextSplitter:
    """
    Paragrafları, cümleleri ve tablo satırlarını bölmemek için öncelikli
    ayırıcılar kullanan ve belirlenen örtüşme (overlap) oranını koruyan
    akıllı metin parçalayıcı.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap değeri chunk_size'dan küçük olmalıdır.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split_text(self, text: str) -> List[str]:
        """Metni hiyerarşik ayırıcı önceliklerine göre parçalara ayırır."""
        if not text:
            return []
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    other_splits = self._split_text(s, new_separators)
                    good_splits.extend(other_splits)
                else:
                    good_splits.append(s)

        return self._merge_splits(good_splits, separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for piece in splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_doc else 0
            if total + piece_len + sep_len > self.chunk_size:
                if total > 0:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    # Örtüşme (overlap) koruma algoritması
                    while current_doc and (
                        total > self.chunk_overlap
                        or (total + piece_len + sep_len > self.chunk_size and total > 0)
                    ):
                        removed = current_doc.pop(0)
                        total -= len(removed) + (len(separator) if current_doc else 0)
                current_doc.append(piece)
                total += piece_len + (len(separator) if len(current_doc) > 1 else 0)
            else:
                current_doc.append(piece)
                total += piece_len + sep_len

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs


# ---------------------------------------------------------------------------
# 2. MULTI-TENANT OTURUM VE KOLEKSİYON YÖNETİCİSİ (SESSION MANAGER)
# ---------------------------------------------------------------------------
class SessionManager:
    """
    Her istemci oturumuna (X-Session-ID) özel izole ChromaDB koleksiyonu açar,
    erişimleri takip eder ve TTL (Time-To-Live) süresi dolmuş oturumları
    asenkron arka plan görevi ile otomatik olarak temizler.
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        ttl_seconds: int = 7200,  # 2 saat
    ):
        if chroma_dir is None:
            chroma_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "chroma_db"
            )
        os.makedirs(chroma_dir, exist_ok=True)

        self.chroma_dir = chroma_dir
        self.ttl_seconds = ttl_seconds
        self.client = chromadb.PersistentClient(path=self.chroma_dir)

        # session_id -> son erişim zamanı (timestamp)
        self.last_accessed: Dict[str, float] = {}

        # Arka plan temizleme görevi kontrolü
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    @staticmethod
    def clean_session_id(session_id: str) -> str:
        """
        Gelen oturum ID'sini ChromaDB koleksiyon adı standartlarına uyarlar:
        - Yalnızca alfanümerik, alt çizgi ve tire karakterlerine izin verilir.
        - ChromaDB sınırlarına uygunluk (maks 63 karakter).
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Geçerli bir session_id belirtilmelidir.")

        # İzin verilmeyen karakterleri alt çizgiye dönüştür
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id.strip())
        if not cleaned:
            cleaned = "default"

        # ChromaDB koleksiyon adı maks 63 karakter:
        # "tenant_" (7 karakter) + maks 50 karakter session id
        if len(cleaned) > 50:
            short_hash = hashlib.md5(session_id.encode("utf-8")).hexdigest()[:8]
            cleaned = f"{cleaned[:41]}_{short_hash}"

        return cleaned

    def get_collection_name(self, session_id: str) -> str:
        """Oturuma özel izole koleksiyon adını döner: tenant_{clean_session_id}"""
        clean_id = self.clean_session_id(session_id)
        return f"tenant_{clean_id}"

    def touch_session(self, session_id: str) -> None:
        """Oturumun son erişim zamanını günceller."""
        col_name = self.get_collection_name(session_id)
        self.last_accessed[col_name] = time.time()

    def get_or_create_collection(self, session_id: str) -> Collection:
        """
        Belirtilen session_id'ye ait dinamik koleksiyonu döner veya oluşturur.
        Koleksiyon adlandırması: tenant_{clean_session_id}
        """
        col_name = self.get_collection_name(session_id)
        collection = self.client.get_or_create_collection(name=col_name)
        self.touch_session(session_id)
        return collection

    def get_collection(self, session_id: str) -> Optional[Collection]:
        """
        Belirtilen session_id'ye ait koleksiyonu döner.
        Yoksa None döner veya sessizce oluşturur.
        """
        col_name = self.get_collection_name(session_id)
        try:
            col = self.client.get_collection(name=col_name)
            self.touch_session(session_id)
            return col
        except Exception:
            return None

    def reset_session_collection(self, session_id: str) -> Collection:
        """
        Oturuma ait eski verileri sıfırlar ve temiz bir koleksiyon oluşturur.
        (Yeni bir sözleşme yüklendiğinde yalnızca bu oturumun eski verilerini silmek için).
        """
        col_name = self.get_collection_name(session_id)
        try:
            self.client.delete_collection(name=col_name)
            logger.info(f"Oturum koleksiyonu sıfırlandı: {col_name}")
        except Exception:
            pass

        collection = self.client.get_or_create_collection(name=col_name)
        self.touch_session(session_id)
        free_memory()
        return collection

    def delete_session(self, session_id: str) -> bool:
        """Oturumu ve ChromaDB üzerindeki koleksiyonunu tamamen siler."""
        col_name = self.get_collection_name(session_id)
        deleted = False
        try:
            self.client.delete_collection(name=col_name)
            deleted = True
            logger.info(f"Oturum koleksiyonu başarıyla silindi: {col_name}")
        except Exception as e:
            logger.warning(f"Koleksiyon silinirken uyarı: {e}")

        self.last_accessed.pop(col_name, None)
        free_memory()
        return deleted

    def cleanup_expired_sessions(self) -> List[str]:
        """
        TTL (Time-To-Live) süresi dolmuş (2 saatten eski) oturumları tespit eder
        ve ChromaDB koleksiyonlarını silerek sunucu belleğini temizler.
        """
        now = time.time()
        expired_collections: List[str] = []

        try:
            all_collections = self.client.list_collections()
        except Exception as e:
            logger.error(f"Koleksiyonlar listelenirken hata: {e}")
            return []

        for col in all_collections:
            col_name = col.name if hasattr(col, "name") else str(col)

            # Sadece tenant_ ön ekiyle başlayan koleksiyonları kontrol et
            if not col_name.startswith("tenant_"):
                continue

            last_time = self.last_accessed.get(col_name)

            # Eğer bellek kaydında varsa ve TTL aşılmışsa
            # ya da bellek kaydı yoksa (sunucu yeniden başlamış ve 2 saattir dokunulmamışsa)
            is_expired = False
            if last_time is not None:
                if (now - last_time) > self.ttl_seconds:
                    is_expired = True
            else:
                # Bellekte takip edilmeyen eski bir koleksiyon
                # Güvenlik gereği ilk turda zamana kaydedilir, bir sonraki döngüde TTL'ye tabi tutulur
                self.last_accessed[col_name] = now

            if is_expired:
                try:
                    self.client.delete_collection(name=col_name)
                    self.last_accessed.pop(col_name, None)
                    expired_collections.append(col_name)
                    logger.info(f"Zaman aşımına uğrayan oturum koleksiyonu temizlendi: {col_name}")
                except Exception as del_err:
                    logger.warning(f"Koleksiyon silinirken hata ({col_name}): {del_err}")

        if expired_collections:
            free_memory()
            logger.info(f"Toplam {len(expired_collections)} adet zaman aşımına uğramış oturum temizlendi.")

        return expired_collections

    async def start_background_cleanup(self, check_interval_seconds: int = 600):
        """
        Asenkron arka plan döngüsü: Belirli aralıklarla (varsayılan: 10 dk)
        zaman aşımına uğrayan oturumları temizler.
        """
        self._running = True
        logger.info(
            f"Oturum temizleme arka plan görevi başlatıldı (Periyot: {check_interval_seconds}s, TTL: {self.ttl_seconds}s)."
        )
        while self._running:
            try:
                await asyncio.sleep(check_interval_seconds)
                if not self._running:
                    break
                self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Oturum temizleme arka plan görevinde hata: {e}")

    def stop_background_cleanup(self):
        """Arka plan temizleme görevini durdurur."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        logger.info("Oturum temizleme arka plan görevi durduruldu.")
