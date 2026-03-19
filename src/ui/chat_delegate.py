# src/ui/chat_delegate.py
from PyQt6 import QtWidgets, QtCore, QtGui
from .chat_model import ChatListModel
import re


class ChatItemDelegate(QtWidgets.QStyledItemDelegate):
    PADDING = QtCore.QMargins(14, 10, 14, 10)
    OUTER = QtCore.QMargins(12, 6, 12, 6)
    BUBBLE_MAX_RATIO = 0.7
    AVATAR_W = 28
    AVATAR_SP = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        painter.save()
        r = option.rect.marginsRemoved(self.OUTER)
        text = index.data(ChatListModel.TextRole)
        is_user = bool(index.data(ChatListModel.IsUserRole))
        ts = index.data(ChatListModel.TimestampRole) or ""

        vpw = option.widget.viewport().width() if hasattr(option.widget, "viewport") else option.rect.width()
        maxw = int(vpw * self.BUBBLE_MAX_RATIO)
        if not is_user:
            maxw -= self.AVATAR_W + self.AVATAR_SP
        maxw = max(240, maxw)

        text_rect = QtCore.QRect(0, 0, maxw - (self.PADDING.left() + self.PADDING.right()), 10)
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(option.font)
        
        # Set text color based on bubble type
        if is_user:
            # User bubble: white text
            doc.setHtml(f"<span style='color: #ffffff;'>{text}</span>")
        else:
            # Assistant bubble: white text
            doc.setHtml(f"<span style='color: #ffffff;'>{text}</span>")
        
        doc.setTextWidth(text_rect.width())
        doc_size = doc.size().toSize()

        fm = QtGui.QFontMetrics(option.font)
        meta_h = fm.height()
        bubble_h = self.PADDING.top() + doc_size.height() + 6 + meta_h + self.PADDING.bottom()
        bubble_w = maxw

        if is_user:
            bubble_rect = QtCore.QRect(r.right() - bubble_w, r.top(), bubble_w, bubble_h)
        else:
            bubble_rect = QtCore.QRect(r.left() + self.AVATAR_W + self.AVATAR_SP, r.top(), bubble_w, bubble_h)

        if not is_user:
            avatar_rect = QtCore.QRect(r.left(), r.top(), self.AVATAR_W, self.AVATAR_W)
            painter.setBrush(QtGui.QColor("#262626"))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(avatar_rect)
            painter.setPen(QtGui.QPen(QtGui.QColor("#9e9e9e")))
            painter.drawText(avatar_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "A")

        radius = 14
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(bubble_rect), radius, radius)
        if is_user:
            # User bubble: gray background
            painter.fillPath(path, QtGui.QColor("#5a5a5a"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#6a6a6a")))
        else:
            # Assistant bubble: black background
            painter.fillPath(path, QtGui.QColor("#2a2a2a"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#3a3a3a")))
        painter.drawPath(path)

        painter.translate(bubble_rect.left() + self.PADDING.left(), bubble_rect.top() + self.PADDING.top())
        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        doc.documentLayout().draw(painter, ctx)

        painter.resetTransform()
        ts_rect = QtCore.QRect(
            bubble_rect.right() - self.PADDING.right() - 80,
            bubble_rect.bottom() - self.PADDING.bottom() - meta_h,
            80,
            meta_h,
        )
        painter.setPen(QtGui.QPen(QtGui.QColor("#9aa")))
        painter.drawText(ts_rect, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, ts)

        painter.restore()

    def editorEvent(self, event: QtCore.QEvent, model, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> bool:
        """Handle right-click context menu for copying text"""
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            mouse_event = event
            if mouse_event.button() == QtCore.Qt.MouseButton.RightButton:
                text = index.data(ChatListModel.TextRole) or ""
                # Remove HTML tags for display
                clean_text = re.sub('<[^<]+?>', '', text)
                
                menu = QtWidgets.QMenu()
                copy_action = menu.addAction("Copy")
                
                action = menu.exec(QtGui.QCursor.pos())
                if action == copy_action:
                    clipboard = QtWidgets.QApplication.clipboard()
                    clipboard.setText(clean_text)
                return True
        
        return super().editorEvent(event, model, option, index)

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
        vpw = option.widget.viewport().width() if hasattr(option.widget, "viewport") else option.rect.width()
        maxw = int(vpw * self.BUBBLE_MAX_RATIO)
        if not bool(index.data(ChatListModel.IsUserRole)):
            maxw -= self.AVATAR_W + self.AVATAR_SP
        maxw = max(240, maxw)

        text = index.data(ChatListModel.TextRole)
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(option.font)
        doc.setHtml(text)
        doc.setTextWidth(maxw - (self.PADDING.left() + self.PADDING.right()))
        doc_h = int(doc.size().height())

        fm = QtGui.QFontMetrics(option.font)
        meta_h = fm.height()
        total_h = self.OUTER.top() + (self.PADDING.top() + doc_h + 6 + meta_h + self.PADDING.bottom()) + self.OUTER.bottom()
        return QtCore.QSize(vpw, max(40, total_h))