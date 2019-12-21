#_*_ coding:utf-8 _*_
import os

import cv2
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

from PIL import Image
import numpy as np
#from yolo_NMS import YOLO

var = {
    'model_path': "./model_data/trained_weights_final.h5"
    , 'anchors_path': "./model_data/yolo_anchors.txt"
    , 'classes_path': './model_data/coin_classes.txt'
    , 'score': 0.1
}


class ShowVideo(QtCore.QObject):




    #### camera ####
    #camera = cv2.VideoCapture(0)
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None



    @QtCore.pyqtSlot()
    def startVideo(self):


        #yolo = YOLO(**var)


        video_path = r'./video/IMG_4680.MOV'


        # os.makedirs('./video/IMG')
        capture = cv2.VideoCapture(video_path)

        rval, frame = capture.read()


        if rval:
            print 'sussess'
        else:
            print 'failed'


        run_video = True
        while rval:
            rval, frame = capture.read()

            #### detection ####
            #ret, frame = self.camera.read()
            #image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #output = yolo.detect_image(image)




            #### convert color ####             name
            print frame
            image = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = color_swapped_image.shape

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap(qt_image)
            qt_image = pixmap.scaled(750, 1000, QtCore.Qt.KeepAspectRatio)
            qt_image = QtGui.QImage(qt_image)


            #text = "Object list : \n"
            #for i in np.unique(output_class):
            #    text += i + " : " + str(output_class.count(i)) + "\n"


            #self.text_widget_Object.setText(text)
            #self.text_widget_Total.setText("Total object detected : \n" + str(len(output_class)))

            self.VideoSignal.emit(qt_image)
            #self.total_amount = len(output_class)



#### Image Load ####
class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("viewer dropped frame!")
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()




def create_QLabel(init_text,text_color=QtCore.Qt.black):
    label = QtWidgets.QLabel(init_text)
    #label.setAutoFillBackground(True)
    pe = QtGui.QPalette()
    pe.setColor(QtGui.QPalette.WindowText,text_color)
    pe.setColor(QtGui.QPalette.Background,QtCore.Qt.white)
    label.setPalette(pe)
    #label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
    label.setFont(QtGui.QFont("Roman times",10,QtGui.QFont.Bold))
    return label


class Main(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setWindowTitle("Coin Detection")

        self.button_run = QtWidgets.QPushButton('Run')
        self.button_run.setIcon(QtGui.QIcon('.\icon\\run.png'))
        self.button_stop = QtWidgets.QPushButton('Stop')
        self.button_stop.setIcon(QtGui.QIcon('.\icon\stop.png'))
        self.button_exit = QtWidgets.QPushButton('Exit')
        self.button_exit.setIcon(QtGui.QIcon('.\icon\exit.png'))

        self.radio_button_pafcn = QtWidgets.QRadioButton('PA-FCN')
        self.radio_button_pafcn.setChecked(True)
        self.radio_button_pafcntaa = QtWidgets.QRadioButton('PA-FCN-TAA')
        self.grid_layout = QtWidgets.QGridLayout()


        #self.button_select = QtWidgets.QPushButton('Select')
        #self.button_select.setIcon(QtGui.QIcon('.\icon\\run.png'))

        self.button_select = QtWidgets.QPushButton('Select')
        self.button_select.setIcon(QtGui.QIcon('.\icon\\run.png'))

        #self.click_action = QtWidgets.QAction(QtGui.QIcon(r'C:\Users\yang\PycharmProjects\CoinDetection\icon\exit.png'), 'open', self)
        #self.button_select.addAction(self.showDialog)


        self.real_path_label = create_QLabel('real_path_label',text_color=QtCore.Qt.black)
        #self.real_path_label.setFixedWidth(100)
        self.real_path_label.setFixedHeight(20)



        #### add image view ####
        self.image_viewer = ImageViewer()
        self.grid_layout.addWidget(self.image_viewer, 0, 0, 6, 8)

        #### add radio button ####
        self.grid_layout.addWidget(self.radio_button_pafcn, 0, 9, 1, 2)
        self.grid_layout.addWidget(self.radio_button_pafcntaa, 1, 9, 1, 2)


        self.grid_layout.addWidget(self.real_path_label, 2, 9, 2, 2)
        self.grid_layout.addWidget(self.button_select, 3, 9, 2, 2)


        #### add button ####
        self.grid_layout.addWidget(self.button_run, 4, 9, 1, 2)
        self.grid_layout.addWidget(self.button_stop, 5, 9, 1, 2)
        self.grid_layout.addWidget(self.button_exit, 6, 9, 1, 2)
        self.layout_widget = QtWidgets.QWidget()
        self.layout_widget.setLayout(self.grid_layout)

        #self.path = './video/IMG_4680.MOV'
        #### show video ####
        self.vid = ShowVideo()
        self.vid.VideoSignal.connect(self.image_viewer.setImage)

        self.thread = QtCore.QThread()
        self.thread.start()

    def reBind(self):

        #### init video_path & judge format illegal ####


        self.vid.startVideo()

        # if self.path != '':
        #
        #
        # else:
        #     print 'aaaa'

    def showDialog(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'open file', '/')
        if fname[0]:
            self.real_path_label.setText(fname[0])
            self.path=fname[0]

            print self.path

            self.vid = ShowVideo(self.path)
            self.vid.VideoSignal.connect(self.image_viewer.setImage)
            print 'aaa'
            print self.vid



            #print fname[0]
            # try:
            #     f = open(fname[0], 'r')
            #     with f:
            #         path = os.path.realpath(f)
            #         print path
            #
            #         #self.textEdit.setText(data)
            # except:
                #self.textEdit.setText("false")


    def closeApplication(self):
        choice = QtWidgets.QMessageBox.question(self, 'Message', 'Do you really want to exit?',
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()
        else:
            pass



    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_S:
            QtWidgets.QMessageBox.information(self, "Send_Batch", "Object amount : " + str(self.vid.total_amount))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main_window = Main()

    #### thread ####
    thread = QtCore.QThread()
    thread.start()
    main_window.vid.moveToThread(thread)

    main_window.button_run.clicked.connect(main_window.vid.startVideo)

    main_window.button_select.clicked.connect(main_window.showDialog)

    #main_window.button_stop.clicked.connect()
    main_window.button_exit.clicked.connect(main_window.closeApplication)

    main_window.setCentralWidget(main_window.layout_widget)
    main_window.resize(1000, 600)
    main_window.show()
    sys.exit(app.exec_())