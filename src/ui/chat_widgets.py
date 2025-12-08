from PySide6 import QtWidgets, QtCore, QtGui


class ChatMessageWidget(QtWidgets.QWidget):

    SIDE_PADDING = 12
    AVATAR_SPACE = 28 + 8
    BUBBLE_MAX_RATIO = 0.68

    def __init__(self, text: str, is_user: bool, timestamp: str):
        super().__init__()
        self._is_user = is_user
        self._text = text
        self._timestamp = timestamp

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(self.SIDE_PADDING, 6, self.SIDE_PADDING, 6)
        layout.setSpacing(8)

        if is_user:
            layout.addStretch(1)

        if not is_user:
            avatar = self._build_avatar("A")
            layout.addWidget(avatar)

        self.bubble = QtWidgets.QFrame()
        self.bubble.setObjectName("bubbleUser" if is_user else "bubbleAssistant")
        bubble_layout = QtWidgets.QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble_layout.setSpacing(6)

        self.content = QtWidgets.QTextBrowser()
        self.content.setOpenExternalLinks(True)
        self.content.setOpenLinks(True)
        self.content.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.content.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.content.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.content.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.content.setText(text)

        self.meta = QtWidgets.QLabel(timestamp)
        self.meta.setObjectName("bubbleMeta")

        bubble_layout.addWidget(self.content)
        bubble_layout.addWidget(self.meta, 0, QtCore.Qt.AlignRight)

        layout.addWidget(self.bubble, 0)

        if not is_user:
            layout.addStretch(1)

        self.setLayout(layout)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

    def _available_width(self, parent_width: int) -> int:
        available = int(parent_width * self.BUBBLE_MAX_RATIO)
        if not self._is_user:
            available = max(200, available - self.AVATAR_SPACE)
        return max(240, available)

    def reflow(self, parent_width: int):
        maxw = self._available_width(parent_width)
        doc = self.content.document()
        doc.setTextWidth(maxw - 28)
        doc.adjustSize()
        text_h = int(doc.size().height())
        self.content.setFixedWidth(maxw - 4)
        self.content.setFixedHeight(text_h + 4)
        self.bubble.setMaximumWidth(maxw)

    def _build_avatar(self, letter: str) -> QtWidgets.QLabel:
        avatar = QtWidgets.QLabel(letter)
        avatar.setFixedSize(28, 28)
        avatar.setAlignment(QtCore.Qt.AlignCenter)
        avatar.setObjectName("avatar")
        return avatar

    def sizeHint(self) -> QtCore.QSize:
        parent = self.parent()
        if parent and hasattr(parent, 'viewport'):
            width = parent.viewport().width()
        else:
            width = parent.width() if parent else 600
        maxw = self._available_width(width)
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(self.content.font())
        doc.setHtml(self._text)
        doc.setTextWidth(max(100, maxw - 28))
        text_height = int(doc.size().height())
        meta_height = self.meta.sizeHint().height()
        bubble_vert = 10 + 10 
        outer_vert = 6 + 6
        height = text_height + meta_height + bubble_vert + outer_vert + 4
        return QtCore.QSize(width, max(40, height))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        parent = self.parent()
        if parent and hasattr(parent, 'viewport'):
            width = parent.viewport().width()
        else:
            width = (parent.width() if parent else event.size().width())
        self.reflow(width)
        super().resizeEvent(event)


def add_message_item(list_widget: QtWidgets.QListWidget, widget: QtWidgets.QWidget) -> QtWidgets.QListWidgetItem:
    item = QtWidgets.QListWidgetItem(list_widget)
    item.setSizeHint(widget.sizeHint())
    list_widget.addItem(item)
    list_widget.setItemWidget(item, widget)
    return item


