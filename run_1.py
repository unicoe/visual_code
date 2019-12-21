import os
import sys
import time

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from ui_1 import Ui_MainWindow


class my_window(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(my_window, self).__init__()
        self.setupUi(self)
        self.label.setAutoFillBackground(True)
        global palette_red, palette_green, palette_blue, palette_yellow
        palette_red = QPalette()
        palette_green = QPalette()
        palette_blue = QPalette()
        palette_yellow = QPalette()
        palette_red.setColor(QPalette.Window, Qt.red)
        palette_green.setColor(QPalette.Window, Qt.green)
        palette_blue.setColor(QPalette.Window, Qt.blue)
        palette_yellow.setColor(QPalette.Window, Qt.yellow)
        self.btn_1.clicked.connect(self.btn_1_clicked)
        self.btn_2.clicked.connect(self.btn_2_clicked)
        self.btn_3.clicked.connect(self.btn_3_clicked)
        self.btn_4.clicked.connect(self.btn_4_clicked)
        self.btn_exit.clicked.connect(self.exit_systerm)
        self.setWindowTitle('通过按钮改变Label的背景颜色')

    def btn_1_clicked(self):
        self.label.setPalette(palette_red)

    def btn_2_clicked(self):
        self.label.setPalette(palette_blue)

    def btn_3_clicked(self):
        self.label.setPalette(palette_green)

    def btn_4_clicked(self):
        self.label.setPalette(palette_yellow)

    def exit_systerm(self):
        self.close()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = my_window()
    w.show()
    sys.exit(app.exec_())
