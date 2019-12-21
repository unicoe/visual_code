from PyQt5.Qt import *
import sys


# class Label(QLabel):
#    def mousePressEvent(self, QMouseEvent):
#        self.setStyleSheet("background-color:red;")

class Window(QWidget):
    def mousePressEvent(self, evt):
        local_x = evt.x()
        local_y = evt.y()
        sub_widget = self.childAt(local_x, local_y)
        if sub_widget is not None:
            sub_widget.setStyleSheet("background-color:red;")
        print("被点击了", local_x, local_y)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = Window()
    win.resize(500, 500)
    win.setWindowTitle("点击设置控件颜色")
    # win.resize(500.500)

    for i in range(1, 11):
        lable = QLabel(win)
        lable.setText("标签" + str(i))
        lable.move(40 * i, 40 * i)

    win.show()

    sys.exit(app.exec_())
