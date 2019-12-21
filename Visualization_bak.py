import cv2
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.Qt import QTextEdit
import os
import time
from model.show_img_code import  get_img
import pdb

import numpy as np
#from yolo_NMS import YOLO

var = {
    'model_path': "./model_data/trained_weights_final.h5"
    , 'anchors_path': "./model_data/yolo_anchors.txt"
    , 'classes_path': './model_data/coin_classes.txt'
    , 'score': 0.1
}

stop_flag = True

def get_time():
    time_tup = time.localtime(time.time())

    format_time = '%Y-%m-%d_%a_%H-%M-%S'

    cur_time = time.strftime(format_time, time_tup)
    return cur_time

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))

class DataSet1(QtCore.QObject):
    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(DataSet1, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False

    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'PA-FCN':
            self.method = method
        print(get_time())
        print("当前使用的方法为：" + method)

    def clearImage(self):
        # image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480, 360, 3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)

    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True

        # yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/unicoe/pytorch-ssd-0/data/VOCdevkit/VOC0712/JPEGImages"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)
            return dirList

        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 Caltech 数据集......")
        for i_img in img_file:
            frame = cv2.imread(i_img)
            image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = color_swapped_image.shape

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap(qt_image)
            qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
            qt_image = QtGui.QImage(qt_image)

            time.sleep(0.5)
            self.VideoSignal.emit(qt_image)
            if not stop_flag:
                self.clearImage()
                stop_flag = False
                break
                # self.clearImage()

class DataSet2(QtCore.QObject):
    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(DataSet2, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False

    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'PA-FCN':
            self.method = method
        print("当前使用的方法为：" + method)

    def clearImage(self):
        # image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480, 360, 3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)

    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True

        # yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/data_set/MOT17Det/train/MOT17-02"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)

            return dirList

        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 MOT2017Det 数据集......")
        for i_img in img_file:
            frame = cv2.imread(i_img)
            frame = cv2.imread(i_img)
            image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = color_swapped_image.shape

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap(qt_image)
            qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
            qt_image = QtGui.QImage(qt_image)

            time.sleep(0.5)
            self.VideoSignal.emit(qt_image)
            if not stop_flag:
                self.clearImage()
                stop_flag = False
                break
                # self.clearImage()

class ShowImages1(QtCore.QObject):

    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowImages1, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False


    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'PA-FCN':
            self.method = method
        print("当前使用的方法为：" + method)

    def clearImage(self):
        #image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480,360,3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)



    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True
        #yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/draw_result/paper_result/select/2019_06_19_Wed_11_43_49/VOC0712/JPEGImages"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)

            return dirList
        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 方法： ad-SSD")
        for i_img in img_file:
           frame = cv2.imread(i_img)
           image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
           color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           height, width, _ = color_swapped_image.shape

           qt_image = QtGui.QImage(color_swapped_image.data,
                                   width,
                                   height,
                                   color_swapped_image.strides[0],
                                   QtGui.QImage.Format_RGB888)

           pixmap = QtGui.QPixmap(qt_image)
           qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
           qt_image = QtGui.QImage(qt_image)

           time.sleep(0.5)
           self.VideoSignal.emit(qt_image)
           if not stop_flag:
               self.clearImage()
               stop_flag = False
               break
        #self.clearImage()

class ShowImages2(QtCore.QObject):

    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowImages2, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False


    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'PA-FCN':
            self.method = method
        print('method: ' + method)

    def clearImage(self):
        #image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480,360,3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)



    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True
        #yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/draw_result/paper_result/select/2019_07_04_Thu_09_32_48/MOT17-11"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)

            return dirList
        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 方法： ad-SSD-seg")
        for i_img in img_file:
           frame = cv2.imread(i_img)
           image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
           color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           height, width, _ = color_swapped_image.shape

           qt_image = QtGui.QImage(color_swapped_image.data,
                                   width,
                                   height,
                                   color_swapped_image.strides[0],
                                   QtGui.QImage.Format_RGB888)

           pixmap = QtGui.QPixmap(qt_image)
           qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
           qt_image = QtGui.QImage(qt_image)

           time.sleep(0.5)
           self.VideoSignal.emit(qt_image)
           if not stop_flag:
               self.clearImage()
               stop_flag = False
               break
        #self.clearImage()

class ShowImages3(QtCore.QObject):

    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowImages3, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False


    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'ad-SSD1':
            self.method = method
        print('method: ' + method)

    def clearImage(self):
        #image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480,360,3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)



    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True
        #yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/draw_result/paper_result/select/2019_01_15_Tue_12_09_38_110/VOC0712/JPEGImages"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)

            return dirList
        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 方法： ad-SSD-seg-vis")
        for i_img in img_file:
           frame = cv2.imread(i_img)
           image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
           color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           height, width, _ = color_swapped_image.shape

           qt_image = QtGui.QImage(color_swapped_image.data,
                                   width,
                                   height,
                                   color_swapped_image.strides[0],
                                   QtGui.QImage.Format_RGB888)

           pixmap = QtGui.QPixmap(qt_image)
           qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
           qt_image = QtGui.QImage(qt_image)

           time.sleep(0.5)
           self.VideoSignal.emit(qt_image)
           if not stop_flag:
               self.clearImage()
               stop_flag = False
               break
        #self.clearImage()




class ShowImages4(QtCore.QObject):

    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowImages4, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False


    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'ad-SSD1':
            self.method = method
        print('method: ' + method)

    def clearImage(self):
        #image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480,360,3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)



    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True

        # pdb.set_trace()
        # yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/draw_result/paper_result/select/2019_01_15_Tue_12_09_38_110/VOC0712/JPEGImages"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)

            return dirList
        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 方法： ad-SSD-seg-vis")
        for i_img in img_file:
           frame = get_img()
           image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
           color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           height, width, _ = color_swapped_image.shape

           qt_image = QtGui.QImage(color_swapped_image.data,
                                   width,
                                   height,
                                   color_swapped_image.strides[0],
                                   QtGui.QImage.Format_RGB888)

           pixmap = QtGui.QPixmap(qt_image)
           qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
           qt_image = QtGui.QImage(qt_image)

           time.sleep(0.5)
           self.VideoSignal.emit(qt_image)
           if not stop_flag:
               self.clearImage()
               stop_flag = False
               break
        self.clearImage()

class ShowImages5(QtCore.QObject):

    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowImages3, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False


    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'ad-SSD1':
            self.method = method
        print('method: ' + method)

    def clearImage(self):
        #image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480,360,3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)



    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True
        #yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/draw_result/paper_result/select/2019_01_15_Tue_12_09_38_110/VOC0712/JPEGImages"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)

            return dirList
        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 方法： ad-SSD-seg-vis")
        for i_img in img_file:
           frame = cv2.imread(i_img)
           image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
           color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           height, width, _ = color_swapped_image.shape

           qt_image = QtGui.QImage(color_swapped_image.data,
                                   width,
                                   height,
                                   color_swapped_image.strides[0],
                                   QtGui.QImage.Format_RGB888)

           pixmap = QtGui.QPixmap(qt_image)
           qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
           qt_image = QtGui.QImage(qt_image)

           time.sleep(0.5)
           self.VideoSignal.emit(qt_image)
           if not stop_flag:
               self.clearImage()
               stop_flag = False
               break
        #self.clearImage()

class ShowImages6(QtCore.QObject):

    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowImages3, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ""

    def stop(self):
        self.stop_flag = False


    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'ad-SSD1':
            self.method = method
        print('method: ' + method)

    def clearImage(self):
        #image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480,360,3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)



    @QtCore.pyqtSlot()
    def startVideo(self):

        global stop_flag

        stop_flag = True
        #yolo = YOLO(**var)
        def get_file_name():
            cur_path = "/home/user/Disk1.8T/draw_result/paper_result/select/2019_01_15_Tue_12_09_38_110/VOC0712/JPEGImages"
            dirList = []
            files = os.listdir(cur_path)

            for file in files:
                dirList.append(cur_path + "/" + file)

            return dirList
        img_file = get_file_name()
        print(get_time())
        print("当前正在展示 方法： ad-SSD-seg-vis")
        for i_img in img_file:
           frame = cv2.imread(i_img)
           image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
           color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           height, width, _ = color_swapped_image.shape

           qt_image = QtGui.QImage(color_swapped_image.data,
                                   width,
                                   height,
                                   color_swapped_image.strides[0],
                                   QtGui.QImage.Format_RGB888)

           pixmap = QtGui.QPixmap(qt_image)
           qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
           qt_image = QtGui.QImage(qt_image)

           time.sleep(0.5)
           self.VideoSignal.emit(qt_image)
           if not stop_flag:
               self.clearImage()
               stop_flag = False
               break
        #self.clearImage()


class ShowVideo(QtCore.QObject):

    #### camera ####
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.do_it_or_not = False
        self.total_amount = None
        self.video_real_path = ''

    def stop(self):
        self.stop_flag = False

    def bindPath(self, path):
        self.video_real_path = path

    def bindMethod(self, method):
        if method == 'ad-SSD':
            self.method = method
        print(get_time())
        print("当前使用的方法为：" + method)

    def clearImage(self):
        # image = cv2.resize(frame * 0, (480, 360), interpolation=cv2.INTER_CUBIC)
        image = np.zeros([480, 360, 3])
        qt_image = QtGui.QImage(image.data,
                                480,
                                360,
                                image.strides[0],
                                QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)
        qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        qt_image = QtGui.QImage(qt_image)
        self.VideoSignal.emit(qt_image)

    @QtCore.pyqtSlot()
    def startVideo(self):

        # yolo = YOLO(**var)

        video_path = self.video_real_path  # r'./video/IMG_4680.MOV'
        capture = cv2.VideoCapture(video_path)
        rval, frame = capture.read()

        global stop_flag
        stop_flag = True

        if rval:
            print('sussess')
        else:
            print('failed')

        while rval and stop_flag:
            #### detection ####
            # ret, frame = self.camera.read()
            # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # output = yolo.detect_image(image)

            #### convert color ####
            # image = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2RGBA)
            image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = color_swapped_image.shape

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap(qt_image)
            qt_image = pixmap.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
            qt_image = QtGui.QImage(qt_image)

            self.VideoSignal.emit(qt_image)
            rval, frame = capture.read()
            if not stop_flag:
                self.clearImage()
                # self.clearImage()

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

class Main(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setWindowTitle("行人检测演示系统")

        self.wind_name1 = QtWidgets.QLabel("原始图像展示:")
        self.wind_name1.setFont(QtGui.QFont("Microsoft YaHei UI", 13))
        self.wind_name2 = QtWidgets.QLabel("检测结果展示:")
        self.wind_name2.setFont(QtGui.QFont("Microsoft YaHei UI",13))

        self.label_method = QtWidgets.QLabel("结果展示:")
        self.label_method.setFont(QtGui.QFont("Microsoft YaHei UI", 12))

        self.show_method = QtWidgets.QLabel("ad-ssd")
        self.show_method.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        self.show_method1 = QtWidgets.QLabel("ad-ssd-seg")
        self.show_method1.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        self.show_method2 = QtWidgets.QLabel("ad-ssd-seg-vis")
        self.show_method2.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.data_set = QtWidgets.QLabel("数据集选择:")
        self.data_set.setFont(QtGui.QFont("Microsoft YaHei UI", 12))

        self.data_set1 = QtWidgets.QRadioButton("Caltech")
        self.data_set1.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        self.data_set2 = QtWidgets.QRadioButton("MOT2017Det")
        self.data_set2.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        # self.label_method.setFixedSize(200,30)

        self.run_method = QtWidgets.QLabel("实时运行:")
        self.run_method.setFont(QtGui.QFont("Microsoft YaHei UI", 12))

        self.run_method0 = QtWidgets.QLabel("ad-ssd")
        self.run_method0.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        self.run_method1 = QtWidgets.QLabel("ad-ssd-seg")
        self.run_method1.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        self.run_method2 = QtWidgets.QLabel("ad-ssd-seg-vis")
        self.run_method2.setFont(QtGui.QFont("Microsoft YaHei UI", 10))


        # self.radio_button_pafcn = QtWidgets.QRadioButton('ad-SSD')
        # self.radio_button_pafcn.setFont(QtGui.QFont("Microsoft YaHei UI",10))
        #
        # self.radio_button_pafcnia = QtWidgets.QRadioButton('ad-SSD-IA')
        # self.radio_button_pafcnia.setFont(QtGui.QFont("Microsoft YaHei UI",10))
        #
        # self.radio_button_pafcn1 = QtWidgets.QRadioButton('ad-SSD11')
        # self.radio_button_pafcn1.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.grid_layout = QtWidgets.QGridLayout()

        self.app_state = QtWidgets.QLabel("执行状态:")
        self.app_state.setFont(QtGui.QFont("Microsoft YaHei UI", 12))
        # 定义文本框，用于显示执行内容
        self.textbox = QTextEdit()
        self.textbox.setObjectName("textEdit")
        self.textbox.setReadOnly(True)
        # 下面将输出重定向到textEdit中
        sys.stdout = EmittingStream(textWritten=self.outputWritten)
        sys.stderr = EmittingStream(textWritten=self.outputWritten)
        # 接收信号str的信号槽


        # self.button_file = QtWidgets.QPushButton('File')
        # self.button_file.setIcon(QtGui.QIcon('./icon/file.png'))
        # # self.button_file.setFixedSize(210, 30)
        # self.button_file.setFont(QtGui.QFont("Microsoft YaHei UI",10))

        self.button_run1 = QtWidgets.QPushButton('Run1')
        self.button_run1.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run1.setFont(QtGui.QFont("Microsoft YaHei UI",10))

        self.button_run2 = QtWidgets.QPushButton('Run2')
        self.button_run2.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run2.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.button_run3 = QtWidgets.QPushButton('Run3')
        self.button_run3.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run3.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.button_stop = QtWidgets.QPushButton('Stop')
        self.button_stop.setIcon(QtGui.QIcon('./icon/stop.png'))
        self.button_stop.setFont(QtGui.QFont("Microsoft YaHei UI",10))

        self.button_exit = QtWidgets.QPushButton('Exit')
        self.button_exit.setIcon(QtGui.QIcon('./icon/exit.png'))
        self.button_exit.setFont(QtGui.QFont("Microsoft YaHei UI",10))

        self.button_run4 = QtWidgets.QPushButton('Caltech数据集展示')
        self.button_run4.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run4.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.button_run5 = QtWidgets.QPushButton('MOT数据集展示')
        self.button_run5.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run5.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.button_run6 = QtWidgets.QPushButton('载入模型运行1')
        self.button_run6.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run6.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.button_run7 = QtWidgets.QPushButton('载入模型运行2')
        self.button_run7.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run7.setFont(QtGui.QFont("Microsoft YaHei UI", 10))

        self.button_run8 = QtWidgets.QPushButton('载入模型运行3')
        self.button_run8.setIcon(QtGui.QIcon('./icon/run.png'))
        self.button_run8.setFont(QtGui.QFont("Microsoft YaHei UI", 10))


        #### add image view ####
        self.grid_layout.addWidget(self.data_set,  17 + 0, 0, 1, 4)
        self.grid_layout.addWidget(self.data_set1, 17 + 1, 0, 1, 4)
        self.grid_layout.addWidget(self.data_set2, 17 + 2, 0, 1, 4)

        self.grid_layout.addWidget(self.label_method, 17,   12, 1, 4)
        self.grid_layout.addWidget(self.show_method,  18+0, 12, 1, 4)
        self.grid_layout.addWidget(self.show_method1, 18+1, 12, 1, 4)
        self.grid_layout.addWidget(self.show_method2, 18+2, 12, 1, 4)

        self.grid_layout.addWidget(self.run_method,  22,     0, 1, 4)
        self.grid_layout.addWidget(self.run_method0, 23 + 0, 0, 1, 4)
        self.grid_layout.addWidget(self.run_method1, 23 + 1, 0, 1, 4)
        self.grid_layout.addWidget(self.run_method2, 23 + 2, 0, 1, 4)
        self.grid_layout.addWidget(self.app_state,   22, 12, 1, 4)
        self.grid_layout.addWidget(self.wind_name1, 0,  0, 1, 4)
        self.grid_layout.addWidget(self.wind_name2, 0, 12, 1, 4)
        self.image_viewer = ImageViewer()
        self.grid_layout.addWidget(self.image_viewer, 1, 0, 12, 12)


        self.image_viewer1 = ImageViewer()
        self.grid_layout.addWidget(self.image_viewer1, 1, 12, 12, 12)
        #
        # self.image_viewer2 = ImageViewer()
        # self.grid_layout.addWidget(self.image_viewer2, 1, 24, 12, 12)

        #### add button ####
        # self.grid_layout.addWidget(self.button_file, 18+5, 0, 1, 4)
        self.grid_layout.addWidget(self.button_run4, 18 + 0, 8, 1, 4)
        self.grid_layout.addWidget(self.button_run5, 18 + 1, 8, 1, 4)
        self.grid_layout.addWidget(self.button_run1, 18+0, 20, 1, 4)
        self.grid_layout.addWidget(self.button_run2, 18+1, 20, 1, 4)
        self.grid_layout.addWidget(self.button_run3, 18+2, 20, 1, 4)
        self.grid_layout.addWidget(self.button_stop, 18+3, 8, 1, 4)
        self.grid_layout.addWidget(self.button_exit, 18+3, 20, 1, 4)
        self.grid_layout.addWidget(self.button_run6, 23 + 0, 8, 1, 4)
        self.grid_layout.addWidget(self.button_run7, 23 + 1, 8, 1, 4)
        self.grid_layout.addWidget(self.button_run8, 23 + 2, 8, 1, 4)
        self.grid_layout.addWidget(self.textbox,     23 + 0, 12, 3, 12)


        self.layout_widget = QtWidgets.QWidget()
        self.layout_widget.setLayout(self.grid_layout)

        #### show video ####
        self.vid0 = DataSet1()
        self.vid0.VideoSignal.connect(self.image_viewer.setImage)
        self.vid00 = DataSet2()
        self.vid00.VideoSignal.connect(self.image_viewer.setImage)

        self.vid1 = ShowImages1()
        self.vid1.VideoSignal.connect(self.image_viewer1.setImage)
        self.vid2 = ShowImages2()
        self.vid2.VideoSignal.connect(self.image_viewer1.setImage)
        self.vid3 = ShowImages3()
        self.vid3.VideoSignal.connect(self.image_viewer1.setImage)

        self.vid4 = ShowImages4()
        self.vid4.VideoSignal.connect(self.image_viewer1.setImage)

        self.thread = QtCore.QThread()
        self.thread.start()


        # self.judgeMethod()


    # def judgeMethod(self):
    #     self.vid0.bindMethod('ad-SSD')
    #     self.vid00.bindMethod('ad-SSD')
    #     if self.radio_button_pafcn.isChecked():
    #         self.vid1.bindMethod('ad-SSD')
    #     elif self.radio_button_pafcn1.isChecked():
    #         self.vid2.bindMethod('ad-SSD1')
    #     else:
    #         self.vid3.bindMethod('ad-SSD-IA')

    # # def showDialog(self):
    # #     fname = QtWidgets.QFileDialog.getOpenFileName(self, 'open file', '/')
    # #     if fname[0]:
    # #         file_name = list(fname)[0].split('/')[-1]
    # #         print(file_name)
    # #         self.button_file.setText('File: '+file_name)
    # #
    # #         self.vid.bindPath(fname[0])

    #         #print fname[0]
    #         # try:
    #         #     f = open(fname[0], 'r')
    #         #     with f:
    #         #         path = os.path.realpath(f)
    #         #         print path
    #         #
    #         #         #self.textEdit.setText(data)
    #         # except:
    #             #self.textEdit.setText("false")

    def stop(self):
        global stop_flag
        stop_flag = False
        self.vid0.clearImage()
        self.vid00.clearImage()
        self.vid1.clearImage()
        self.vid2.clearImage()
        self.vid3.clearImage()
        self.vid4.clearImage()
        print(get_time())
        print("演示停止.")

    def closeApplication(self):
        choice = QtWidgets.QMessageBox.question(self, 'Message', 'Do you want to exit?',
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()
        else:
            pass
    # 接收信号str的信号槽
    def outputWritten(self, text):
        cursor = self.textbox.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textbox.setTextCursor(cursor)
        self.textbox.ensureCursorVisible()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    main_window = Main()

    #### thread ####
    thread = QtCore.QThread()
    thread.start()
    thread1 = QtCore.QThread()
    thread1.start()
    thread2 = QtCore.QThread()
    thread2.start()

    main_window.vid0.moveToThread(thread)
    main_window.vid00.moveToThread(thread)

    main_window.vid1.moveToThread(thread1)
    main_window.vid2.moveToThread(thread1)
    main_window.vid3.moveToThread(thread1)
    main_window.vid4.moveToThread(thread2)

    #main_window.radio_button_pafcn.toggled.connect(main_window.judgeMethod)
    # main_window.button_file.clicked.connect(main_window.showDialog)
    main_window.button_run1.clicked.connect(main_window.vid1.startVideo)
    main_window.button_run2.clicked.connect(main_window.vid2.startVideo)
    main_window.button_run3.clicked.connect(main_window.vid3.startVideo)
    main_window.button_run4.clicked.connect(main_window.vid0.startVideo)
    main_window.button_run5.clicked.connect(main_window.vid00.startVideo)
    main_window.button_run6.clicked.connect(main_window.vid4.startVideo)


    main_window.button_stop.clicked.connect(main_window.stop)
    main_window.button_exit.clicked.connect(main_window.closeApplication)

    main_window.setCentralWidget(main_window.layout_widget)
    main_window.setFixedSize(980, 360+400)

    print(get_time())
    print("演示系统载入完成.")

    main_window.show()
    sys.exit(app.exec_())