from dataclasses import dataclass
from typing import List
from PySide6 import QtCore


@dataclass
class ChatMessage:
    text: str
    isUser: bool
    timestamp: str
    kind: str = "message"


class ChatListModel(QtCore.QAbstractListModel):
    TextRole = QtCore.Qt.UserRole + 1
    IsUserRole = QtCore.Qt.UserRole + 2
    TimestampRole = QtCore.Qt.UserRole + 3
    KindRole = QtCore.Qt.UserRole + 4

    def __init__(self, messages: List[ChatMessage] = None):
        super().__init__()
        self._messages: List[ChatMessage] = messages or []

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._messages)

    def data(self, index: QtCore.QModelIndex, role: int):
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

    def roleNames(self):
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


