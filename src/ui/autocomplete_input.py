"""
AutocompleteLineEdit — автодополнение на базе статистической модели.

Обучается на:
  1. Базовом частотном словаре русского языка (data/ru_freq_dict.txt)
  2. Документах пользователя из RAG-папки
  3. Истории чата

Кэш хранится в <project>/cache/autocomplete/
"""

from __future__ import annotations

import os
import re
import pickle
import hashlib
import threading
from collections import Counter, defaultdict
from typing import List, Optional, Tuple, Dict, Set
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets


# ─── Пути проекта ──────────────────────────────────────────────────────

def _get_project_root() -> str:
    """Корень проекта (где лежит src/)."""
    here = os.path.dirname(os.path.abspath(__file__))  # src/ui/
    src_dir = os.path.dirname(here)                     # src/
    return os.path.dirname(src_dir)                     # project root


def _get_cache_dir() -> str:
    """Директория кэша: <project>/cache/autocomplete/"""
    d = os.path.join(_get_project_root(), "cache", "autocomplete")
    os.makedirs(d, exist_ok=True)
    return d


def _get_dict_path() -> str:
    """Путь к словарю: <project>/data/ru_freq_dict.txt"""
    return os.path.join(_get_project_root(), "data", "ru_freq_dict.txt")


# ─── Trie ───────────────────────────────────────────────────────────────

class _TrieNode:
    __slots__ = ("children", "word")

    def __init__(self):
        self.children: Dict[str, _TrieNode] = {}
        self.word: Optional[str] = None


class PrefixTrie:
    """Префиксное дерево для мгновенного поиска слов по началу."""

    def __init__(self):
        self.root = _TrieNode()
        self.size = 0

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        if node.word is None:
            self.size += 1
        node.word = word

    def search(self, prefix: str, limit: int = 30) -> List[str]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: List[str] = []
        self._collect(node, results, limit)
        return results

    def _collect(self, node: _TrieNode, out: List[str], limit: int) -> None:
        if len(out) >= limit:
            return
        if node.word is not None:
            out.append(node.word)
        for ch in sorted(node.children):
            self._collect(node.children[ch], out, limit)
            if len(out) >= limit:
                return


# ─── Языковая модель ────────────────────────────────────────────────────

class StatLanguageModel:
    """
    N-gram языковая модель с Trie-индексом.

    Поддерживает:
    - Дополнение слов по префиксу (complete_word)
    - Предсказание следующего слова (predict_next)
    - Обучение на произвольных текстах (train)
    - Загрузку частотного словаря (load_dictionary)
    """

    def __init__(self):
        self.trie = PrefixTrie()
        self.word_freq: Counter = Counter()
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)
        self.trigrams: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.total_words: int = 0
        self._known_hashes: Set[int] = set()
        self._lock = threading.Lock()

    # ── Загрузка словаря ────────────────────────────────────────────────

    def load_dictionary(self, dict_path: str) -> int:
        """
        Загрузить частотный словарь из файла.
        Формат: слово<tab>частота (по одному на строку).
        Возвращает количество загруженных слов.
        """
        if not os.path.exists(dict_path):
            return 0

        count = 0
        with self._lock:
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) != 2:
                        continue
                    word, freq_str = parts[0].strip(), parts[1].strip()
                    if not word or not freq_str.isdigit():
                        continue

                    wl = word.lower()
                    freq = int(freq_str)

                    if self.word_freq[wl] == 0:
                        self.trie.insert(wl)
                    self.word_freq[wl] += freq
                    self.total_words += freq
                    count += 1

        return count

    # ── Обучение ────────────────────────────────────────────────────────

    def train(self, text: str) -> int:
        """Обучить на тексте. Дубликаты игнорируются."""
        h = hash(text[:300])
        if h in self._known_hashes:
            return 0
        self._known_hashes.add(h)

        added = 0
        with self._lock:
            for sentence in self._split_sentences(text):
                words = self._tokenize(sentence)
                if len(words) < 2:
                    continue

                for w in words:
                    wl = w.lower()
                    if self.word_freq[wl] == 0:
                        self.trie.insert(wl)
                    self.word_freq[wl] += 1
                    added += 1

                self.total_words += len(words)

                for i in range(len(words) - 1):
                    a = words[i].lower()
                    b = words[i + 1].lower()
                    self.bigrams[a][b] += 1

                for i in range(len(words) - 2):
                    a = words[i].lower()
                    b = words[i + 1].lower()
                    c = words[i + 2].lower()
                    self.trigrams[(a, b)][c] += 1

        return added

    # ── Предсказание ────────────────────────────────────────────────────

    def complete_word(
        self, prefix: str, context: List[str], limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Дополнить слово по префиксу с учётом контекста."""
        pl = prefix.lower()
        candidates = self.trie.search(pl, limit=50)
        candidates = [c for c in candidates if c != pl]

        if not candidates:
            return []

        ctx = [w.lower() for w in context[-3:]]
        scored: List[Tuple[str, float]] = []

        for word in candidates:
            score = 0.0

            # Триграмма
            if len(ctx) >= 2:
                tri = self.trigrams.get((ctx[-2], ctx[-1]))
                if tri:
                    score += tri.get(word, 0) * 10.0

            # Биграмма
            if len(ctx) >= 1:
                bi = self.bigrams.get(ctx[-1])
                if bi:
                    score += bi.get(word, 0) * 3.0

            # Частота слова
            score += self.word_freq.get(word, 0) * 0.01

            # Бонус за длину (короткие слова чуть приоритетнее)
            score += 0.001 / (len(word) + 1)

            scored.append((word, score))

        scored.sort(key=lambda x: (-x[1], len(x[0])))
        return scored[:limit]

    def predict_next(
        self, context: List[str], prefix: str = "", limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Предсказать следующее слово."""
        if not context:
            top = self.word_freq.most_common(limit * 2)
            if prefix:
                top = [(w, c) for w, c in top if w.startswith(prefix.lower())]
            return top[:limit]

        ctx = [w.lower() for w in context[-3:]]
        pl = prefix.lower()
        candidates: Dict[str, float] = {}

        # Триграммы (вес 10)
        if len(ctx) >= 2:
            tri = self.trigrams.get((ctx[-2], ctx[-1]))
            if tri:
                for word, count in tri.most_common(limit * 5):
                    if not pl or word.startswith(pl):
                        candidates[word] = candidates.get(word, 0) + count * 10.0

        # Биграммы (вес 3)
        if len(ctx) >= 1:
            bi = self.bigrams.get(ctx[-1])
            if bi:
                for word, count in bi.most_common(limit * 5):
                    if not pl or word.startswith(pl):
                        candidates[word] = candidates.get(word, 0) + count * 3.0

        # Фоллбэк на униграммы
        if not candidates:
            for word, count in self.word_freq.most_common(limit * 3):
                if not pl or word.startswith(pl):
                    candidates[word] = count * 0.01

        scored = sorted(candidates.items(), key=lambda x: -x[1])
        return scored[:limit]

    # ── Утилиты ─────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return re.split(r"[.!?\n;]+", text)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w for w in re.findall(r"[А-Яа-яЁёA-Za-z0-9']+", text) if len(w) >= 2]

    # ── Сохранение / загрузка ───────────────────────────────────────────

    def save(self, path: str) -> None:
        with self._lock:
            tmp = path + ".tmp"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, "wb") as f:
                pickle.dump(
                    {
                        "word_freq": self.word_freq,
                        "bigrams": dict(self.bigrams),
                        "trigrams": dict(self.trigrams),
                        "total_words": self.total_words,
                        "known_hashes": self._known_hashes,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)

    @classmethod
    def load(cls, path: str) -> StatLanguageModel:
        model = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        model.word_freq = data["word_freq"]
        model.bigrams = defaultdict(Counter, data["bigrams"])
        model.trigrams = defaultdict(Counter, data["trigrams"])
        model.total_words = data["total_words"]
        model._known_hashes = data.get("known_hashes", set())

        # Восстанавливаем Trie из word_freq
        for word in model.word_freq:
            model.trie.insert(word)

        return model


# ─── Чтение документов ──────────────────────────────────────────────────

_MAX_TEXT = 100_000
_SUPPORTED_EXT = {"txt", "md", "csv", "html", "pdf", "docx", "log", "json"}


def _read_file_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    try:
        if ext in ("txt", "md", "csv", "json", "log"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(_MAX_TEXT)
        if ext == "html":
            try:
                import bs4
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return bs4.BeautifulSoup(f.read(_MAX_TEXT), "html.parser").get_text()
            except ImportError:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return re.sub(r"<[^>]+>", " ", f.read(_MAX_TEXT))
        if ext == "pdf":
            try:
                import PyPDF2
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join(
                        (p.extract_text() or "") for p in reader.pages[:50]
                    )[:_MAX_TEXT]
            except ImportError:
                return ""
        if ext == "docx":
            try:
                from docx import Document as DocxDoc
                doc = DocxDoc(path)
                return "\n".join(p.text for p in doc.paragraphs)[:_MAX_TEXT]
            except ImportError:
                return ""
    except Exception:
        pass
    return ""


# ─── Виджет ─────────────────────────────────────────────────────────────

class AutocompleteLineEdit(QtWidgets.QLineEdit):
    """
    QLineEdit с умным автодополнением.

    Обучается на:
      1. Базовом частотном словаре (data/ru_freq_dict.txt)
      2. Документах из RAG-папки (train_on_folder)
      3. Истории чата (set_history)

    Кэш: <project>/cache/autocomplete/lm_<hash>.pkl

    Управление:
      Tab / Shift+Tab  — переключить вариант
      → / Enter        — принять
      Escape           — отклонить
    """

    _DEBOUNCE_MS = 120
    _MIN_PREFIX = 1

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._ghost: str = ""
        self._candidates: List[str] = []
        self._cand_idx: int = 0
        self._mode: str = ""
        self._suppress: bool = False
        self._training: bool = False

        # Модель
        self._model = StatLanguageModel()
        self._model_path: Optional[str] = None

        # Загружаем базовый словарь
        dict_path = _get_dict_path()
        if os.path.exists(dict_path):
            n = self._model.load_dictionary(dict_path)
            print(f"[Autocomplete] ✓ Dictionary loaded: {n} base words")
        else:
            print(f"[Autocomplete] Dictionary not found: {dict_path}")
            print(f"[Autocomplete] Run: python scripts/download_dictionary.py")

        # Debounce
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._compute)

        self.textChanged.connect(self._on_text)
        self.cursorPositionChanged.connect(self._on_cursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # ── Public API ──────────────────────────────────────────────────────

    def train_on_folder(self, folder_path: str) -> None:
        """Обучить модель на документах (фоновый поток)."""
        h = hashlib.md5(os.path.abspath(folder_path).encode()).hexdigest()[:12]
        self._model_path = os.path.join(_get_cache_dir(), f"lm_{h}.pkl")

        if os.path.exists(self._model_path):
            try:
                self._model = StatLanguageModel.load(self._model_path)
                # Досыпаем словарь поверх (слова уже есть → просто обновляет freq)
                dict_path = _get_dict_path()
                if os.path.exists(dict_path):
                    self._model.load_dictionary(dict_path)
                print(
                    f"[Autocomplete] ✓ Cached model: "
                    f"{self._model.trie.size} words"
                )
                return
            except Exception as e:
                print(f"[Autocomplete] Cache error: {e}")

        self._training = True
        threading.Thread(
            target=self._train_worker, args=(folder_path,), daemon=True
        ).start()

    def retrain_folder(self, folder_path: str) -> None:
        """Переобучить (сбросить кэш)."""
        # Сохраняем словарь
        dict_words = StatLanguageModel()
        dict_path = _get_dict_path()
        if os.path.exists(dict_path):
            dict_words.load_dictionary(dict_path)

        self._model = dict_words
        if self._model_path and os.path.exists(self._model_path):
            os.remove(self._model_path)
        self.train_on_folder(folder_path)

    def set_history(self, history: List[str]) -> None:
        """Обучить на истории чата."""
        for msg in history:
            if msg and len(msg.strip()) >= 5:
                self._model.train(msg.strip())

    def train_on_text(self, text: str) -> None:
        """Обучить на произвольном тексте."""
        self._model.train(text)

    # ── Фоновое обучение ────────────────────────────────────────────────

    def _train_worker(self, folder_path: str) -> None:
        try:
            print(f"[Autocomplete] Training on: {folder_path}")
            total = 0
            file_count = 0

            for root, _, files in os.walk(folder_path):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower().lstrip(".")
                    if ext not in _SUPPORTED_EXT:
                        continue
                    text = _read_file_text(os.path.join(root, fname))
                    if text and len(text.strip()) > 50:
                        added = self._model.train(text)
                        total += added
                        file_count += 1

            print(
                f"[Autocomplete] ✓ Trained: {file_count} files, "
                f"{self._model.trie.size} unique words"
            )

            if self._model_path:
                self._model.save(self._model_path)
                print(f"[Autocomplete] ✓ Cached: {self._model_path}")

        except Exception as e:
            print(f"[Autocomplete] Training error: {e}")
        finally:
            self._training = False

    # ── Keyboard ────────────────────────────────────────────────────────

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        key = ev.key()
        no_mod = ev.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier

        if key == QtCore.Qt.Key.Key_Tab and no_mod:
            if self._candidates:
                if len(self._candidates) > 1:
                    self._cand_idx = (self._cand_idx + 1) % len(self._candidates)
                    self._ghost = self._candidates[self._cand_idx]
                    self.update()
                else:
                    self._accept()
                ev.accept()
                return
            super().keyPressEvent(ev)
            return

        if key == QtCore.Qt.Key.Key_Backtab:
            if self._candidates and len(self._candidates) > 1:
                self._cand_idx = (self._cand_idx - 1) % len(self._candidates)
                self._ghost = self._candidates[self._cand_idx]
                self.update()
                ev.accept()
                return
            super().keyPressEvent(ev)
            return

        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and no_mod:
            if self._ghost:
                self._accept()
                ev.accept()
                return
            super().keyPressEvent(ev)
            return

        if (
            key == QtCore.Qt.Key.Key_Right
            and no_mod
            and self._ghost
            and self.cursorPosition() == len(self.text())
        ):
            self._accept()
            ev.accept()
            return

        if key == QtCore.Qt.Key.Key_Escape and no_mod and self._ghost:
            self._clear()
            ev.accept()
            return

        if key not in (
            QtCore.Qt.Key.Key_Shift,
            QtCore.Qt.Key.Key_Control,
            QtCore.Qt.Key.Key_Alt,
            QtCore.Qt.Key.Key_Meta,
            QtCore.Qt.Key.Key_CapsLock,
        ):
            self._clear()

        super().keyPressEvent(ev)

    def focusNextPrevChild(self, _next: bool) -> bool:
        return False if self._ghost else super().focusNextPrevChild(_next)

    # ── Paint ────────────────────────────────────────────────────────────

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        super().paintEvent(ev)
        if not self._ghost:
            return

        p = QtGui.QPainter(self)
        try:
            c = self.palette().color(QtGui.QPalette.ColorRole.Text)
            c.setAlphaF(0.45)
            p.setPen(c)
            p.setFont(self.font())

            opt = QtWidgets.QStyleOptionFrame()
            self.initStyleOption(opt)
            rect = self.style().subElementRect(
                QtWidgets.QStyle.SubElement.SE_LineEditContents, opt, self
            )
            fm = QtGui.QFontMetrics(self.font())
            x = rect.left() + fm.horizontalAdvance(self.text())
            gr = QtCore.QRect(x, rect.top(), rect.right() - x, rect.height())

            label = self._ghost
            if len(self._candidates) > 1:
                label += f"  [{self._cand_idx + 1}/{len(self._candidates)}]"

            p.drawText(
                gr,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                label,
            )
        finally:
            p.end()

    # ── Triggers ─────────────────────────────────────────────────────────

    def _on_text(self, _t: str) -> None:
        if self._suppress:
            return
        self._clear()
        self._timer.start(self._DEBOUNCE_MS)

    def _on_cursor(self, _o: int, _n: int) -> None:
        if not self._suppress and self._ghost and self.cursorPosition() < len(self.text()):
            self._clear()

    # ── Вычисление ───────────────────────────────────────────────────────

    def _compute(self) -> None:
        text = self.text()
        cur = self.cursorPosition()
        if cur != len(text):
            self._clear()
            return

        context_words, prefix = self._parse(text)

        if self._model.total_words == 0:
            self._clear()
            return

        self._candidates = []

        if prefix and len(prefix) >= self._MIN_PREFIX:
            self._mode = "word"

            # 1. Контекстно-зависимое предсказание
            if context_words:
                preds = self._model.predict_next(context_words, prefix=prefix, limit=5)
                for word, _ in preds:
                    suffix = word[len(prefix):]
                    if suffix and suffix not in self._candidates:
                        self._candidates.append(suffix)

            # 2. Trie-поиск по префиксу
            if len(self._candidates) < 5:
                wpreds = self._model.complete_word(prefix, context_words, limit=7)
                for word, _ in wpreds:
                    suffix = word[len(prefix):]
                    if suffix and suffix not in self._candidates:
                        self._candidates.append(suffix)
                        if len(self._candidates) >= 5:
                            break

        elif not prefix and context_words:
            self._mode = "next"
            preds = self._model.predict_next(context_words, limit=5)
            for word, _ in preds:
                cand = " " + word
                if cand not in self._candidates:
                    self._candidates.append(cand)

        if self._candidates:
            self._cand_idx = 0
            self._ghost = self._candidates[0]
            self.update()
        else:
            self._clear()

    @staticmethod
    def _parse(text: str) -> Tuple[List[str], str]:
        last_space = text.rfind(" ")
        if last_space == -1:
            return [], text.strip()

        ctx_text = text[:last_space].strip()
        prefix = text[last_space + 1:]
        ctx_words = [
            w.lower()
            for w in re.findall(r"[А-Яа-яЁёA-Za-z0-9']+", ctx_text)
            if len(w) >= 2
        ]
        return ctx_words, prefix

    # ── Accept / Clear ──────────────────────────────────────────────────

    def _accept(self) -> None:
        if not self._ghost:
            return
        self._suppress = True
        new = self.text() + self._ghost
        self.setText(new)
        self.setCursorPosition(len(new))
        self._suppress = False
        self._clear()

    def _clear(self) -> None:
        changed = bool(self._ghost)
        self._ghost = ""
        self._candidates = []
        self._cand_idx = 0
        self._mode = ""
        if changed:
            self.update()