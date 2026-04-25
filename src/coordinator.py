# src/coordinator.py
import psutil
import traceback
from typing import Optional, Dict, Any, Callable
import re
import time
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool
from indexer import build_index
from rag import get_rag_chain, generate_suggested_questions
from config import MODEL_NAME, EMBEDDING_MODEL, get_llm_settings


class IndexingSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)


class AskSignals(QObject):
    result = pyqtSignal(dict)


class IndexingRunnable(QRunnable):
    def __init__(self, coordinator):
        super().__init__()
        self.coordinator = coordinator
        self.signals = IndexingSignals()
        self.setAutoDelete(False)

    def run(self):
        try:
            if getattr(self.coordinator, "closing", False):
                return

            def _progress_callback(current_files: int, total_files: int):
                try:
                    percent = int((current_files * 100) / total_files) if total_files else 0
                    self.coordinator.indexing_progress.emit(current_files, total_files, percent)
                except Exception:
                    pass

            vectorstore = build_index(
                self.coordinator.folder_path,
                EMBEDDING_MODEL,
                progress_callback=_progress_callback,
            )

            if vectorstore:
                self.coordinator.vectorstore = vectorstore
                self.coordinator._rebuild_qa_chain()

                # Generate suggested questions for the folder (non-blocking within this background runnable)
                try:
                    suggestions = generate_suggested_questions(
                        vectorstore,
                        MODEL_NAME,
                        use_gpu=self.coordinator.use_gpu,
                        folder_path=self.coordinator.folder_path,
                        max_q=6,
                    )
                    self.coordinator.suggested_questions = suggestions or []
                except Exception:
                    self.coordinator.suggested_questions = []

            try:
                if not getattr(self.coordinator, "closing", False):
                    self.signals.finished.emit()
            except RuntimeError:
                pass

        except Exception:
            try:
                if not getattr(self.coordinator, "closing", False):
                    self.signals.error.emit(traceback.format_exc())
            except RuntimeError:
                pass

        finally:
            if hasattr(self.coordinator, "active_runnables"):
                self.coordinator.active_runnables = [
                    r for r in self.coordinator.active_runnables if r != self
                ]


class AskRunnable(QRunnable):
    def __init__(self, coordinator, query, file_filter):
        super().__init__()
        self.coordinator = coordinator
        self.query = query
        self.file_filter = file_filter
        self.signals = AskSignals()
        self.setAutoDelete(False)

    def run(self):
        try:
            if getattr(self.coordinator, "closing", False):
                return
            if not self.coordinator.qa_chain:
                try:
                    self.signals.result.emit({"result": "Индексация не завершена.", "sources": ""})
                except RuntimeError:
                    pass
                return

            # Вызываем цепочку — она может вернуть dict (синхронно) или iterable (streaming)
            try:
                resp = self.coordinator.qa_chain(self.query, file_filter=self.file_filter)
            except Exception as e:
                try:
                    self.signals.result.emit({"result": f"Ошибка: {e}", "sources": ""})
                except RuntimeError:
                    pass
                return

            # Если ответ — итерируемый объект (stream), то пробежим по нему и эмитим частичные обновления
            if hasattr(resp, "__iter__") and not isinstance(resp, dict) and not isinstance(resp, str):
                try:
                    cum = ""
                    for item in resp:
                        if getattr(self.coordinator, "closing", False):
                            return

                        # If the stream yields dicts with structure produced by rag._stream_generator
                        if isinstance(item, dict):
                            # If it's a delta chunk, emit delta directly
                            if "delta" in item and item.get("delta"):
                                delta = item.get("delta")
                                cum += delta
                                try:
                                    if not getattr(self.coordinator, "closing", False):
                                        self.signals.result.emit({"delta": delta, "final": False})
                                except RuntimeError:
                                    pass
                                continue

                            # If it's a partial cumulative (legacy), handle gracefully
                            if "partial" in item and item.get("partial"):
                                partial = item.get("partial")
                                cum = partial
                                try:
                                    if not getattr(self.coordinator, "closing", False):
                                        self.signals.result.emit({"partial": partial, "final": False})
                                except RuntimeError:
                                    pass
                                continue

                            # If item indicates final result
                            if item.get("final") or item.get("done") or item.get("is_final"):
                                final_text = item.get("result") or item.get("partial") or item.get("content") or ""
                                sources = item.get("sources") if isinstance(item.get("sources"), (str, list, set)) else ""
                                highlight_chunks = item.get("highlight_chunks", [])
                                keywords = item.get("keywords", [])
                                formatted_context = item.get("formatted_context", "")
                                if isinstance(sources, set):
                                    sources = ", ".join([s for s in sources if s])
                                try:
                                    if not getattr(self.coordinator, "closing", False):
                                        self.signals.result.emit({
                                            "result": final_text or cum,
                                            "sources": sources,
                                            "highlight_chunks": highlight_chunks,
                                            "keywords": keywords,
                                            "formatted_context": formatted_context,
                                            "final": True,
                                        })
                                except RuntimeError:
                                    pass
                                return

                            # Fallback: maybe item contains 'result'
                            if "result" in item and item.get("result"):
                                final_text = item.get("result")
                                try:
                                    if not getattr(self.coordinator, "closing", False):
                                        self.signals.result.emit({"result": final_text, "final": True})
                                except RuntimeError:
                                    pass
                                return

                            # otherwise ignore metadata-only dicts
                            continue

                        # If plain string chunk — treat as delta
                        if isinstance(item, str) and item.strip():
                            delta = item
                            cum += delta
                            try:
                                if not getattr(self.coordinator, "closing", False):
                                    self.signals.result.emit({"delta": delta, "final": False})
                            except RuntimeError:
                                pass
                            continue

                    # Exhausted iterator — emit final
                    try:
                        if not getattr(self.coordinator, "closing", False):
                            self.signals.result.emit({"result": cum, "final": True})
                    except RuntimeError:
                        pass
                    return
                except Exception:
                    # fall through to non-stream handling
                    pass

            # Обычный (синхронный) ответ — это dict
            response = resp if isinstance(resp, dict) else {}
            sources = response.get("sources", "")
            if isinstance(sources, set):
                sources = ", ".join([s for s in sources if s])
            elif not sources:
                sources = "Неизвестно"

            final_text = response.get("result", "") or ""
            highlight_chunks = response.get("highlight_chunks", [])

            # Разбиваем ответ на предложения для имитации поточной генерации
            parts = [p.strip() for p in re.split(r'(?<=[\.!\?])\s+', final_text) if p.strip()]
            if not parts and final_text:
                parts = [final_text]

            cum = ""
            try:
                for i, part in enumerate(parts):
                    if getattr(self.coordinator, "closing", False):
                        return
                    if cum:
                        cum = f"{cum} {part}"
                    else:
                        cum = part
                    partial_payload = {"partial": cum, "final": False}
                    try:
                        self.signals.result.emit(partial_payload)
                    except RuntimeError:
                        pass
                    time.sleep(0.05)

                final_payload = {
                    "result": final_text,
                    "sources": sources,
                    "highlight_chunks": highlight_chunks,
                    "keywords": response.get("keywords", []),
                    "formatted_context": response.get("formatted_context", ""),
                    "final": True,
                }
                try:
                    if not getattr(self.coordinator, "closing", False):
                        self.signals.result.emit(final_payload)
                except RuntimeError:
                    pass
            except RuntimeError:
                pass

        except Exception as e:
            try:
                if not getattr(self.coordinator, "closing", False):
                    self.signals.result.emit({"result": f"Ошибка: {e}", "sources": ""})
            except RuntimeError:
                pass

        finally:
            if hasattr(self.coordinator, "active_runnables"):
                self.coordinator.active_runnables = [
                    r for r in self.coordinator.active_runnables if r != self
                ]


class RAGCoordinator(QObject):
    indexing_started = pyqtSignal()
    indexing_finished = pyqtSignal()
    indexing_error = pyqtSignal(str)
    indexing_progress = pyqtSignal(int, int, int)

    _instance = None

    @staticmethod
    def instance(folder_path=None):
        if not RAGCoordinator._instance:
            if folder_path is None:
                raise ValueError("folder_path required")
            RAGCoordinator._instance = RAGCoordinator(folder_path)
        return RAGCoordinator._instance

    def __init__(self, folder_path: str):
        super().__init__()
        if hasattr(self, "initialized"):
            return

        self.folder_path = folder_path
        self.vectorstore = None
        self.qa_chain = None
        self.is_indexing = False
        self.closing = False
        self.use_gpu = self._detect_gpu()
        self.threadpool = QThreadPool.globalInstance()
        self.active_runnables = []

        self.start_indexing()
        self.initialized = True

    def _detect_gpu(self):
        try:
            battery = psutil.sensors_battery()
            return battery is None or battery.power_plugged
        except Exception:
            return True

    def _rebuild_qa_chain(self):
        """Пересоздаёт qa_chain с актуальными настройками LLM из config."""
        if not self.vectorstore:
            return
        s = get_llm_settings()
        self.qa_chain = get_rag_chain(
            self.vectorstore,
            model_name=s["ollama_model"],
            use_gpu=self.use_gpu,
            folder_path=self.folder_path,
            llm_provider=s["provider"],
            openrouter_api_key=s["openrouter_key"],
            openrouter_model=s["openrouter_model"],
        )

    def apply_llm_settings(self):
        """Вызывается из UI после сохранения настроек — перезапускает LLM без переиндексации."""
        self._rebuild_qa_chain()

    def start_indexing(self):
        if self.is_indexing:
            return
        self.is_indexing = True
        self.indexing_started.emit()

        runnable = IndexingRunnable(self)
        runnable.signals.finished.connect(self._indexing_done)
        runnable.signals.error.connect(self._indexing_error)
        self.active_runnables.append(runnable)
        self.threadpool.start(runnable)

    def close(self):
        try:
            self.closing = True
            self.is_indexing = False
            self.active_runnables = []
            self.qa_chain = None
            self.vectorstore = None
            try:
                self.threadpool.clear()
            except Exception:
                pass
            try:
                self.threadpool.waitForDone(1000)
            except Exception:
                pass
        except Exception:
            pass

    def _indexing_done(self):
        self.is_indexing = False
        self.indexing_finished.emit()

    def _indexing_error(self, msg):
        self.is_indexing = False
        self.indexing_error.emit(msg)

    def ask_async(self, query: str, file_filter: Optional[str], callback: Callable[[Dict], None]):
        if self.is_indexing:
            callback({"result": "Индексация в процессе...", "sources": ""})
            return
        if not self.qa_chain:
            callback({"result": "Модель не загружена.", "sources": ""})
            return

        runnable = AskRunnable(self, query, file_filter)
        runnable.signals.result.connect(callback)
        self.active_runnables.append(runnable)
        self.threadpool.start(runnable)
