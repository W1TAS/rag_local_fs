# src/ui/chat_model.py
from dataclasses import dataclass
from typing import List
from PyQt6 import QtCore


@dataclass
class ChatMessage:
    text: str
    isUser: bool
    timestamp: str
    kind: str = "message"


class ChatListModel(QtCore.QAbstractListModel):
    TextRole = QtCore.Qt.ItemDataRole.UserRole + 1
    IsUserRole = QtCore.Qt.ItemDataRole.UserRole + 2
    TimestampRole = QtCore.Qt.ItemDataRole.UserRole + 3
    KindRole = QtCore.Qt.ItemDataRole.UserRole + 4

    def __init__(self, messages: List[ChatMessage] = None):
        super().__init__()
        self._messages: List[ChatMessage] = messages or []

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._messages)

    def data(self, index: QtCore.QModelIndex, role: int):  # deprecated
        if not index.isValid():
            return None
        msg = self._messages[index.row()]
        if role == self.TextRole:
            return msg.text
        if role == self.IsUserRole:
            return msg.isUser
        if role == self.TimestampRole:
            return msg.timestamp
        if role == self.KindRole:
            return msg.kind
        return None

    def roleNames(self):  # deprecated
        return {
            self.TextRole: b"text",
            self.IsUserRole: b"isUser",
            self.TimestampRole: b"timestamp",
            self.KindRole: b"kind",
        }

    def appendMessage(self, msg: ChatMessage):
        self.beginInsertRows(QtCore.QModelIndex(), len(self._messages), len(self._messages))
        self._messages.append(msg)
        self.endInsertRows()

    def removeTyping(self):
        if self._messages and self._messages[-1].kind == "typing":
            last = len(self._messages) - 1
            self.beginRemoveRows(QtCore.QModelIndex(), last, last)
            self._messages.pop()
            self.endRemoveRows()

    def setTypingDots(self, dots: int):
        if self._messages and self._messages[-1].kind == "typing":
            self._messages[-1].text = "Генерация" + "." * dots
            idx = self.index(len(self._messages) - 1)
            self.dataChanged.emit(idx, idx, [self.TextRole])

    def updateLastMessageText(self, new_text: str):
        """Обновить текст последнего сообщения и уведомить view."""
        if not self._messages:
            return
        last_idx = len(self._messages) - 1
        self._messages[last_idx].text = new_text
        idx = self.index(last_idx)
        self.dataChanged.emit(idx, idx, [self.TextRole])

    def appendToLastMessage(self, delta: str):
        """Добавить дельту в конец текста последнего сообщения и обновить view."""
        if not self._messages:
            return
        last_idx = len(self._messages) - 1
        self._messages[last_idx].text = (self._messages[last_idx].text or "") + delta
        idx = self.index(last_idx)
        self.dataChanged.emit(idx, idx, [self.TextRole])

    def setLastMessageTimestamp(self, ts: str):
        """Установить timestamp последнего сообщения (например, при финализации стрима)."""
        if not self._messages:
            return
        last_idx = len(self._messages) - 1
        self._messages[last_idx].timestamp = ts
        idx = self.index(last_idx)
        self.dataChanged.emit(idx, idx, [self.TimestampRole])

    def replaceLastMessage(self, text: str, is_user: bool, ts: str, kind: str = "message"):
        """Заменить последнее сообщение полностью (текст, isUser, timestamp, kind).
        Используется для замены частичного сообщения финальным.
        """
        if not self._messages:
            # fallback: append
            self.appendMessage(ChatMessage(text=text, isUser=is_user, timestamp=ts, kind=kind))
            return
        last_idx = len(self._messages) - 1
        self._messages[last_idx].text = text
        self._messages[last_idx].isUser = is_user
        self._messages[last_idx].timestamp = ts
        self._messages[last_idx].kind = kind
        idx = self.index(last_idx)
        self.dataChanged.emit(idx, idx, [self.TextRole, self.IsUserRole, self.TimestampRole, self.KindRole])