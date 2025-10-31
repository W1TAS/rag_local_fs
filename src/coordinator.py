# src/coordinator.py
import os
import psutil
import traceback
from typing import Optional, Dict, Any, Callable
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool
from src.indexer import build_index
from src.rag import get_rag_chain
from src.config import MODEL_NAME, EMBEDDING_MODEL


class IndexingSignals(QObject):
    finished = Signal()
    error = Signal(str)

class AskSignals(QObject):
    result = Signal(dict)


class IndexingRunnable(QRunnable):
    def __init__(self, coordinator):
        super().__init__()
        self.coordinator = coordinator
        self.signals = IndexingSignals()
        self.setAutoDelete(False)

    def run(self):
        try:
            vectorstore = build_index(self.coordinator.folder_path, EMBEDDING_MODEL)
            if vectorstore:
                self.coordinator.vectorstore = vectorstore
                self.coordinator.qa_chain = get_rag_chain(
                    vectorstore, MODEL_NAME,
                    use_gpu=self.coordinator.use_gpu,
                    folder_path=self.coordinator.folder_path
                )
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(traceback.format_exc())
        finally:
            if hasattr(self.coordinator, 'active_runnables'):
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
            if not self.coordinator.qa_chain:
                self.signals.result.emit({"result": "Индексация не завершена.", "sources": ""})
                return
            response = self.coordinator.qa_chain(self.query, file_filter=self.file_filter)
            sources = response.get("sources", "")
            if isinstance(sources, set):
                sources = ", ".join([s for s in sources if s])
            elif not sources:
                sources = "Неизвестно"
            response["sources"] = sources
            self.signals.result.emit(response)
        except Exception as e:
            self.signals.result.emit({"result": f"Ошибка: {e}", "sources": ""})
        finally:
            if hasattr(self.coordinator, 'active_runnables'):
                self.coordinator.active_runnables = [
                    r for r in self.coordinator.active_runnables if r != self
                ]


class RAGCoordinator(QObject):
    indexing_started = Signal()
    indexing_finished = Signal()
    indexing_error = Signal(str)

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
        if hasattr(self, "initialized"): return

        self.folder_path = folder_path
        self.vectorstore = None
        self.qa_chain = None
        self.is_indexing = False
        self.use_gpu = self._detect_gpu()
        self.threadpool = QThreadPool.globalInstance()
        self.active_runnables = []

        self.start_indexing()
        self.initialized = True

    def _detect_gpu(self):
        try:
            battery = psutil.sensors_battery()
            return battery is None or battery.power_plugged
        except:
            return True

    def start_indexing(self):
        if self.is_indexing: return
        self.is_indexing = True
        self.indexing_started.emit()

        runnable = IndexingRunnable(self)
        runnable.signals.finished.connect(self._indexing_done)
        runnable.signals.error.connect(self._indexing_error)
        self.active_runnables.append(runnable)
        self.threadpool.start(runnable)

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