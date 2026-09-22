"use client";

import { useState, useRef, useEffect } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface UploadedDoc {
  filename: string;
  pages: number;
  chunks: number;
}

// ---------------------------------------------------------------------------
// Dynamic API URLs (Local & Production via NEXT_PUBLIC_API_URL)
// ---------------------------------------------------------------------------
function getApiBaseUrl(): string {
  let url = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").trim();
  url = url.replace(/\/+$/, "");

  // Eğer protokol girilmediyse (örn: my-api.onrender.com)
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    if (url.includes("localhost") || url.includes("127.0.0.1")) {
      url = `http://${url}`;
    } else {
      url = `https://${url}`;
    }
  }

  // Canlı ortamda http:// kaldıysa Mixed Content hatasını önlemek için kesinlikle https:// yap
  if (
    url.startsWith("http://") &&
    !url.includes("localhost") &&
    !url.includes("127.0.0.1")
  ) {
    url = url.replace(/^http:\/\//i, "https://");
  }

  return url;
}

const API_BASE_URL = getApiBaseUrl();
const CHAT_STREAM_URL = `${API_BASE_URL}/chat/stream`;
const UPLOAD_URL = `${API_BASE_URL}/upload`;
const HEALTH_URL = `${API_BASE_URL}/health`;


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function parseSSEChunk(raw: string): string {
  /**
   * SSE formatı: "data: {\"chunk\": \"kelime\"}\n\n"
   * [DONE] özel tokenını filtreler.
   */
  const lines = raw.split("\n");
  let result = "";

  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    const payload = line.slice(6).trim();
    if (payload === "[DONE]") break;
    try {
      const parsed = JSON.parse(payload) as { chunk?: string };
      if (parsed.chunk) result += parsed.chunk;
    } catch {
      // Bozuk satırları sessizce atla
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Render Cold Start / Server Health State
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "waking">("checking");

  // PDF Upload & Status States
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeDoc, setActiveDoc] = useState<UploadedDoc | null>(null);


  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Yeni mesaj geldiğinde otomatik scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Render.com Cold Start & Sunucu Sağlık Kontrolü
  useEffect(() => {
    let isMounted = true;
    const checkServer = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);
        const res = await fetch(HEALTH_URL, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (res.ok && isMounted) {
          setServerStatus("online");
        } else if (isMounted) {
          setServerStatus("waking");
        }
      } catch {
        if (isMounted) setServerStatus("waking");
      }
    };

    checkServer();
    const interval = setInterval(checkServer, 20000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);


  // ---------------------------------------------------------------------------
  // PDF Upload Logic
  // ---------------------------------------------------------------------------
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Geçersiz dosya. Lütfen yalnızca bir PDF dosyası seçin.");
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(UPLOAD_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Dosya yüklenirken bir hata oluştu.");
      }

      setActiveDoc({
        filename: data.filename || file.name,
        pages: data.pages_processed || 1,
        chunks: data.chunks_indexed || 0,
      });

      setUploadMessage("PDF hazır, artık soru sorabilirsiniz!");
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "PDF yüklenirken bir hata oluştu.";
      setUploadError(msg);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // ---------------------------------------------------------------------------
  // Stream logic
  // ---------------------------------------------------------------------------
  const sendMessage = async () => {
    const userMessage = input.trim();
    if (!userMessage || isStreaming) return;

    setInput("");
    setError(null);

    // Kullanıcı mesajını geçmişe ekle
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: userMessage,
    };

    // Asistan için boş bir yer tutucu oluştur
    const assistantId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    abortRef.current = new AbortController();

    try {
      const response = await fetch(CHAT_STREAM_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error("ReadableStream desteklenmiyor.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        const textChunk = parts
          .map((part) => parseSSEChunk(part + "\n\n"))
          .join("");

        if (textChunk) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: msg.content + textChunk }
                : msg,
            ),
          );
        }
      }

      if (buffer) {
        const remaining = parseSSEChunk(buffer);
        if (remaining) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: msg.content + remaining }
                : msg,
            ),
          );
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        // Kullanıcı iptal etti
      } else {
        const message =
          err instanceof Error ? err.message : "Bilinmeyen bir hata oluştu.";
        setError(message);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-2 sm:p-4">
      <div className="w-full max-w-2xl flex flex-col h-[94dvh] sm:h-[88vh] rounded-2xl shadow-2xl overflow-hidden border border-slate-700 bg-slate-900/90 backdrop-blur-md">
        {/* ── Header ── */}
        <header className="flex items-center justify-between px-4 sm:px-6 py-3.5 sm:py-4 border-b border-slate-700/80 bg-slate-800/70">
          <div className="flex items-center gap-2.5 sm:gap-3">
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-lg sm:text-xl shadow-md shrink-0">
              ⚡
            </div>
            <div>
              <div className="flex items-center gap-1.5 sm:gap-2">
                <h1 className="text-white font-semibold text-base sm:text-lg leading-tight">
                  AI RAG Sözleşme Asistanı
                </h1>
                <span className="text-[9px] sm:text-[10px] uppercase font-bold tracking-wider px-1.5 sm:px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">
                  Groq LLM
                </span>
              </div>
              <p className="text-slate-400 text-[11px] sm:text-xs mt-0.5 truncate max-w-[200px] sm:max-w-none">
                FastAPI · ChromaDB · PyPDF · Production
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Sunucu Durumu Rozeti */}
            <div
              title={
                serverStatus === "online"
                  ? "Sunucu çevrimiçi ve hazır"
                  : serverStatus === "waking"
                    ? "Render sunucusu uykudan uyanıyor (~45 sn)"
                    : "Sunucu bağlantısı kontrol ediliyor"
              }
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-700/50 border border-slate-600/40 text-[11px]"
            >
              <span
                className={`w-2 h-2 rounded-full inline-block ${
                  serverStatus === "online"
                    ? "bg-emerald-400"
                    : serverStatus === "waking"
                      ? "bg-amber-400 animate-ping"
                      : "bg-blue-400 animate-pulse"
                }`}
              />
              <span className="text-slate-300 hidden sm:inline font-medium">
                {serverStatus === "online"
                  ? "API Çevrimiçi"
                  : serverStatus === "waking"
                    ? "Uyanıyor…"
                    : "Bağlanıyor…"}
              </span>
            </div>

            {activeDoc && (
              <div
                title={`${activeDoc.filename} (${activeDoc.pages} sayfa, ${activeDoc.chunks} parça)`}
                className="hidden md:flex items-center gap-1.5 px-3 py-1.5 bg-indigo-950/60 border border-indigo-600/40 rounded-lg text-indigo-300 text-xs"
              >
                <span>📄</span>
                <span className="max-w-[110px] truncate font-medium">
                  {activeDoc.filename}
                </span>
                <span className="text-indigo-400 font-semibold text-[11px]">
                  ({activeDoc.chunks})
                </span>
              </div>
            )}

            {isStreaming && (
              <span className="flex items-center gap-1 px-2 sm:px-2.5 py-1 rounded-md bg-emerald-950/60 border border-emerald-500/30 text-emerald-400 text-xs font-medium animate-pulse shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                <span className="hidden sm:inline">Yanıtlanıyor…</span>
              </span>
            )}
          </div>
        </header>

        {/* ── Render Cold Start Uyarısı ── */}
        {serverStatus === "waking" && (
          <div className="bg-amber-950/80 border-b border-amber-600/40 px-4 py-2 flex items-center justify-between text-xs text-amber-200 shadow-sm animate-pulse">
            <div className="flex items-center gap-2">
              <span className="shrink-0 text-sm">⏳</span>
              <span>
                <strong>Render Sunucusu Başlatılıyor:</strong> Ücretsiz plandaki uyku modu nedeniyle ilk bağlantı ~45 saniye sürebilir, lütfen bekleyin...
              </span>
            </div>
            <button
              onClick={() => setServerStatus("checking")}
              className="ml-2 text-amber-400 hover:text-amber-200 text-[11px] underline shrink-0 font-medium"
            >
              Yenile
            </button>
          </div>
        )}

        {/* ── Messages ── */}
        <section className="flex-1 overflow-y-auto px-4 py-5 space-y-4 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-slate-400 px-6 select-none">
              <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-3xl mb-4 shadow-inner">
                📑
              </div>
              <h2 className="text-base font-medium text-slate-200">
                {activeDoc
                  ? `"${activeDoc.filename}" Belgesi Hazır!`
                  : "PDF Belgesi Yükleyin & Soru Sorun"}
              </h2>
              <p className="text-xs mt-1.5 max-w-md text-slate-400 leading-relaxed">
                {activeDoc
                  ? "Belgeniz ChromaDB vektör veritabanına indekslendi. Artık Groq LLM modeli üzerinden belge içeriğiyle ilgili sorularınızı sorabilirsiniz."
                  : "Sözleşme veya dokümanınızı ataş ikonuna tıklayarak yükleyin. Sistem metni 500 karakterlik parçalara bölüp semantik olarak analiz edecektir."}
              </p>

              {!activeDoc && (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  className="mt-4 px-4 py-2 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-xs font-medium transition flex items-center gap-2"
                >
                  <span>📎</span>
                  <span>Bir PDF Dosyası Seç</span>
                </button>
              )}
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center text-sm mr-2.5 mt-1 shrink-0">
                  🤖
                </div>
              )}
              <div
                className={`
                  max-w-[82%] px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap break-words shadow-md
                  ${
                    msg.role === "user"
                      ? "bg-indigo-600 text-white rounded-br-sm"
                      : "bg-slate-800/90 text-slate-100 border border-slate-700 rounded-bl-sm"
                  }
                `}
              >
                {msg.content}
                {msg.role === "assistant" &&
                  isStreaming &&
                  msg.content === "" && (
                    <span className="inline-block w-2 h-4 bg-slate-300 animate-pulse rounded-sm ml-0.5 align-middle" />
                  )}
                {msg.role === "assistant" &&
                  isStreaming &&
                  msg.content !== "" && (
                    <span className="inline-block w-0.5 h-4 bg-indigo-400 animate-pulse rounded ml-0.5 align-middle" />
                  )}
              </div>
              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-lg bg-slate-700 border border-slate-600 flex items-center justify-center text-sm ml-2.5 mt-1 shrink-0">
                  👤
                </div>
              )}
            </div>
          ))}

          {error && (
            <div className="mx-2 px-4 py-3 rounded-xl bg-red-950/50 border border-red-700/60 text-red-300 text-sm flex items-start gap-2.5 shadow">
              <span className="shrink-0 text-base">⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </section>

        {/* ── Status Notifications (Upload/Indexing) ── */}
        <div className="px-4 space-y-2">
          {isUploading && (
            <div className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl bg-indigo-950/60 border border-indigo-600/40 text-indigo-200 text-xs shadow animate-pulse">
              <span className="w-3.5 h-3.5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin shrink-0" />
              <span className="font-medium">
                Belge indeksleniyor... Sayfalar okunuyor ve ChromaDB&apos;ye
                aktarılıyor.
              </span>
            </div>
          )}

          {uploadMessage && !isUploading && (
            <div className="flex items-center justify-between px-4 py-2.5 rounded-xl bg-emerald-950/60 border border-emerald-600/40 text-emerald-200 text-xs shadow">
              <div className="flex items-center gap-2">
                <span className="text-emerald-400 font-bold">✓</span>
                <span>{uploadMessage}</span>
                {activeDoc && (
                  <span className="text-emerald-400 font-medium">
                    (Belge: {activeDoc.filename}, {activeDoc.chunks} parça)
                  </span>
                )}
              </div>
              <button
                onClick={() => setUploadMessage(null)}
                className="text-emerald-400/60 hover:text-emerald-300 text-sm font-bold ml-2"
              >
                ✕
              </button>
            </div>
          )}

          {uploadError && (
            <div className="flex items-center justify-between px-4 py-2.5 rounded-xl bg-red-950/60 border border-red-600/40 text-red-200 text-xs shadow">
              <div className="flex items-center gap-2">
                <span className="text-red-400">⚠️</span>
                <span>{uploadError}</span>
              </div>
              <button
                onClick={() => setUploadError(null)}
                className="text-red-400/60 hover:text-red-300 text-sm font-bold ml-2"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* ── Input & Attachment Bar ── */}
        <footer className="p-4 border-t border-slate-700/80 bg-slate-800/60">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={handleFileUpload}
          />

          <div className="flex items-end gap-2.5">
            {/* Ataş / PDF Yükle Butonu */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || isStreaming}
              title="PDF Belgesi Yükle"
              className="
                shrink-0 h-[46px] w-[46px] rounded-xl bg-slate-700/70 hover:bg-slate-700
                border border-slate-600 text-slate-300 hover:text-indigo-400
                focus:outline-none focus:ring-2 focus:ring-indigo-500
                disabled:opacity-40 disabled:cursor-not-allowed transition
                flex items-center justify-center group shadow
              "
            >
              <svg
                className="w-5 h-5 transition-transform group-hover:scale-110"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                />
              </svg>
            </button>

            {/* Mesaj Girdisi */}
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height =
                  Math.min(e.target.scrollHeight, 120) + "px";
              }}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              placeholder={
                activeDoc
                  ? `"${activeDoc.filename}" hakkında soru sorun… (Enter ile gönderin)`
                  : "Bir soru yazın veya sol taraftan PDF yükleyin…"
              }
              className="
                flex-1 resize-none rounded-xl bg-slate-700/60 border border-slate-600
                text-slate-100 placeholder-slate-400 text-base sm:text-sm px-4 py-3
                focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                disabled:opacity-50 disabled:cursor-not-allowed transition
                min-h-[46px] max-h-[120px]
              "
            />

            {/* Gönder / Durdur Butonları */}
            {isStreaming ? (
              <button
                onClick={handleStop}
                className="
                  shrink-0 h-[46px] px-4 rounded-xl bg-red-600 hover:bg-red-500
                  text-white text-sm font-medium transition shadow
                  flex items-center gap-1.5
                "
              >
                <span>⏹</span>
                <span>Durdur</span>
              </button>
            ) : (
              <button
                onClick={sendMessage}
                disabled={!input.trim()}
                className="
                  shrink-0 h-[46px] px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500
                  disabled:opacity-40 disabled:cursor-not-allowed
                  text-white text-sm font-medium transition shadow
                  flex items-center gap-1.5
                "
              >
                <span>Gönder</span>
                <span>➤</span>
              </button>
            )}
          </div>
        </footer>
      </div>
    </main>
  );
}

