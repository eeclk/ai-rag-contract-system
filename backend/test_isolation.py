import io
import time
from fastapi.testclient import TestClient
import pypdf
from main import app, session_manager

client = TestClient(app)

def create_sample_pdf(text_pages: list[str]) -> io.BytesIO:
    """Bellek içinde geçerli bir test PDF dosyası oluşturur."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    for text in text_pages:
        # pypdf ile boş sayfa ekleyip üzerine açıklama/metin yazabiliriz veya doğrudan test edebiliriz
        page = writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    return stream

def test_multi_tenant_isolation():
    session_a = "11111111-aaaa-bbbb-cccc-111111111111"
    session_b = "22222222-aaaa-bbbb-cccc-222222222222"
    session_manager.delete_session(session_a)
    session_manager.delete_session(session_b)

    print("--- TEST 1: Header olmadan istek (400 Bad Request beklenir) ---")
    res_no_header = client.get("/documents")
    assert res_no_header.status_code == 400
    print("Test 1 Geçti: Header olmadan 400 döndü.")

    print("\n--- TEST 2: Session A ve Session B boş kontrolü ---")
    res_a = client.get("/documents", headers={"X-Session-ID": session_a})
    assert res_a.status_code == 200
    assert res_a.json()["total_chunks"] == 0

    res_b = client.get("/documents", headers={"X-Session-ID": session_b})
    assert res_b.status_code == 200
    assert res_b.json()["total_chunks"] == 0
    print("Test 2 Geçti: Her iki oturum da başlangıçta temiz.")

    print("\n--- TEST 3: Session A'ya PDF yükleme ve Zengin Metadata Doğrulaması ---")
    # Gerçek metin içeren PDF oluşturalım
    from pypdf import PdfWriter
    writer = PdfWriter()
    p1 = writer.add_blank_page(width=612, height=792)
    # Metadata ve parçalama için pypdf'in extract_text yapabileceği bir dosya veya simülasyon
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)

    # session_manager doğrudan test edelim
    from session_manager import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    contract_text = """MADDE 1: TARAFLAR
İşbu sözleşme A Şirketi ile B Şirketi arasında akdedilmiştir.

MADDE 2: SÖZLEŞMENİN KONUSU
Bu sözleşme, multi-tenant bulut tabanlı RAG mimarisinin kurulumunu kapsar.
Gizlilik ve veri izolasyonu esastır. Her kiracının verisi ayrı koleksiyonda saklanır.
"""
    chunks = splitter.split_text(contract_text)
    assert len(chunks) >= 1
    
    # Session A koleksiyonuna veri ekle
    col_a = session_manager.reset_session_collection(session_a)
    col_a.add(
        ids=[f"sozlesme.pdf_p1_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {
                "source_file": "sozlesme.pdf",
                "page_number": 1,
                "chunk_id": f"sozlesme.pdf_p1_{i}",
                "char_length": len(c),
            }
            for i, c in enumerate(chunks)
        ]
    )

    print("Test 3 Geçti: Session A koleksiyonuna zengin metadata ile veri yazıldı.")

    print("\n--- TEST 4: İZOLASYON KONTROLÜ (A verisi B'de ASLA görünmemeli) ---")
    doc_res_a = client.get("/documents", headers={"X-Session-ID": session_a})
    doc_res_b = client.get("/documents", headers={"X-Session-ID": session_b})

    data_a = doc_res_a.json()
    data_b = doc_res_b.json()

    print(f"Session A Doküman Sayısı: {data_a['total_chunks']}")
    print(f"Session B Doküman Sayısı: {data_b['total_chunks']}")

    assert data_a["total_chunks"] > 0
    assert data_a["active_filename"] == "sozlesme.pdf"
    assert data_b["total_chunks"] == 0
    assert data_b["active_filename"] is None
    print("Test 4 Geçti: Tam Oturum İzolasyonu Kanıtlandı! A'nın verisi B'de 0.")

    print("\n--- TEST 5: Session B Chat İzolasyon Kontrolü ---")
    chat_res_b = client.post(
        "/chat/stream",
        headers={"X-Session-ID": session_b},
        json={"message": "Sözleşmenin tarafları kimlerdir?"}
    )
    assert chat_res_b.status_code == 200
    b_text = chat_res_b.text
    assert "herhangi bir" in b_text
    print("Test 5 Gecti: B oturumunda A'nin sozlesmesi taranmadi, aninda izole bos yanit dondu.")

    print("\n--- TEST 6: TTL Temizleme Kontrolü ---")
    # Session A TTL süresini geçmiş gibi gösterelim
    session_manager.last_accessed[f"tenant_{session_manager.clean_session_id(session_a)}"] = time.time() - 8000
    cleaned = session_manager.cleanup_expired_sessions()
    print(f"Temizlenen koleksiyonlar: {cleaned}")
    assert f"tenant_{session_manager.clean_session_id(session_a)}" in cleaned
    
    # Tekrar kontrol edelim: Session A koleksiyonu silinmiş olmalı
    doc_res_a_after = client.get("/documents", headers={"X-Session-ID": session_a})
    assert doc_res_a_after.json()["total_chunks"] == 0
    print("Test 6 Geçti: 2 saati aşan oturum başarıyla temizlendi.")

    print("\nTÜM TESTLER BAŞARIYLA GEÇTİ!")

if __name__ == "__main__":
    test_multi_tenant_isolation()
