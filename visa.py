import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget, QGraphicsPixmapItem, QFileDialog, QGraphicsScene, QApplication
from PyQt5.QtGui import QPixmap, QImage, QPainter
from Image import *
import cv2
import numpy as np
from graphics import GraphicsView, GraphicsPixmapItem


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("垃圾图片分类识别")
        MainWindow.setFixedSize(825, 570)
        center = QDesktopWidget().screenGeometry()
        MainWindow.move((center.width() - 825) / 2, (center.height() - 570) / 2)
        MainWindow.resize(825, 617)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(670, 190, 151, 341))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.graphicsView = GraphicsView(self.centralwidget)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setGeometry(QtCore.QRect(10, 180, 652, 352))
        self.graphicsView.setObjectName("graphicsView")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 30, 502, 52))
        self.textEdit.setObjectName("textBrowser")
        self.textEdit.setFontPointSize(15)
        #self.label = QtWidgets.QLabel(self.centralwidget)
        #self.label.setGeometry(QtCore.QRect(10, 100, 500, 50))
        #self.label.setObjectName("label")
        #self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        #self.radioButton.setGeometry(QtCore.QRect(550, 150, 112, 23))
        #self.radioButton.setObjectName("radioButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 825, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        #MainWindow.setStyleSheet("QMainWindow{background:#66B3FF}")
        self.textEdit.setStyleSheet("QGraphicsView{background:#66B3FF}")
       # self.textEdit.setStyleSheet("QGraphicsView{background:#66B3FF}")

        self.pushButton_3.setText(_translate("MainWindow", "打开图片"))
        self.pushButton_3.setIcon(QtGui.QIcon("icons/open.svg"))
        self.pushButton_3.setIconSize(QtCore.QSize(40, 20))
        self.pushButton_3.setStyleSheet("QPushButton{background:#16A085;border:none;color:#000000;font-size:15px;}"
                                        "QPushButton:hover{background-color:#008080;}")

        self.pushButton_2.setText(_translate("MainWindow", "图片识别"))
        self.pushButton_2.setIcon(QtGui.QIcon("icons/locate.svg"))
        self.pushButton_2.setIconSize(QtCore.QSize(40, 20))
        self.pushButton_2.setStyleSheet("QPushButton{background:#FFA500;border:none;color:#000000;font-size:15px;}"
                                        "QPushButton:hover{background-color:#D26900;}")
        self.pushButton.setText(_translate("MainWindow", "退出"))
        self.pushButton.setIcon(QtGui.QIcon("icons/close_.svg"))
        self.pushButton.setIconSize(QtCore.QSize(20, 20))
        self.pushButton.setStyleSheet("QPushButton{background:#CE0000;border:none;color:#000000;font-size:15px;}"
                                      "QPushButton:hover{background-color:#8B0000;}")
        #self.label.setText(_translate("MainWindow"))
        self.pushButton.clicked.connect(self.close)
        #打开图片
        self.pushButton_3.clicked.connect(self.clickOpen)
        #加载图片
        self.pushButton_2.clicked.connect(self.predict)
        #self.label.setText(_translate("MainWindow", "TextLabe1"))
        self.messageBox = QMessageBox()
        self.messageBox.setStyleSheet("QMessageBox{background-color:#CE0000;border:none;color:#000000;font-size:15px;}")

    def close(self):
        reply = self.messageBox.question(None, "Quit", "确定要关闭该程序？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit()
    c=''
    def clickOpen(self):
        imgName, imgType = QFileDialog.getOpenFileName(None, "打开图片", "",
                                                       "All Files(*)")  # "*.jpg;;*.png;;*.jpeg;;All Files(*)"
        img = cv2.imread(imgName)
        global c
        c=imgName
        self.image = Image(img)
        H, W, C = self.image.img.shape
        P = 3 * W
        qimage = QImage(self.image.img.data, W, H, P, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.graphicsView.setSceneRect(0, 0, 650, 350)
        self.graphicsView.setItem(pixmap)
        self.graphicsView.Scene()
        self.graphicsView.setStyleSheet("QGraphicsView{background-color:#66B3FF}")

        #if self.radioButton.isChecked() == True:
            #self.graphicsView.image_item.setStart(True)

    '''def clickLocation(self):
        if self.radioButton.isChecked() == False:
            self.graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
            Img = self.image.pos_img
            Img = cv2.resize(Img, (500, 50), cv2.INTER_NEAREST)
            H, W, _ = Img.shape
            P = 3 * W
            Img = QImage(Img.data, W, H, P, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(Img)
            Item = QGraphicsPixmapItem(pixmap)

            self.graphicsView_3Scene = QGraphicsScene()
            self.graphicsView_3Scene.addItem(Item)
            self.graphicsView_3.setSceneRect(0, 0, 500, 50)
            self.graphicsView_3.setStyleSheet("QGraphicsView{background-color:#66B3FF}")
            self.graphicsView_3.setScene(self.graphicsView_3Scene)
            backImg = self.image.remove_back_img.copy()
            cv2.rectangle(backImg, (self.image.W_start, self.image.H_start), (self.image.W_end, self.image.H_end),
                          (0, 0, 255), 2)
            backImg = cv2.resize(backImg, (650, 350), cv2.INTER_NEAREST)
            H, W, _ = backImg.shape
            P = 3 * W
            backImg = QImage(backImg.data, W, H, P, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(backImg)
            self.graphicsView.setItem(pixmap)
            self.graphicsView.Scene()
        else:
            backImg = self.image.remove_back_img.copy()
            Img = backImg[
                  int(self.graphicsView.image_item.start_point.y()):int(self.graphicsView.image_item.end_point.y()),
                  int(self.graphicsView.image_item.start_point.x()):int(self.graphicsView.image_item.end_point.x())]
            Img = cv2.resize(Img, (500, 50), cv2.INTER_NEAREST)
            Img = QImage(Img.data, 500, 50, 1500, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(Img)
            Item = QGraphicsPixmapItem(pixmap)
            self.graphicsView_3Scene = QGraphicsScene()
            self.graphicsView_3Scene.addItem(Item)
            self.graphicsView_3.setSceneRect(0, 0, 500, 50)
            self.graphicsView_3.setStyleSheet("QGraphicsView{background-color:#66B3FF}")
            self.graphicsView_3.setScene(self.graphicsView_3Scene)'''
    def predict(self):
        import torchvision.transforms as transforms
        import torch
        from PIL import Image
        from collections import OrderedDict
        import torch.nn.functional as F
        from efficientnet_pytorch import EfficientNet
        from torch import nn
        import os, time
        import torchvision.models as models
        from resnetxt_wsl import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        args = {}
        args['arch'] = 'resnext101_32x16d_wsl'
        args['pretrained'] = False
        args['num_classes'] = 43
        args['image_size'] = 288

        class classfication_service():
            def __init__(self, model_path):
                self.model = self.build_model(model_path)
                self.pre_img = self.preprocess_img()
                self.model.eval()
                self.device = torch.device('cpu')
                self.label_id_name_dict = \
                    {
                        "0": "其他垃圾/一次性快餐盒",
                        "1": "其他垃圾/污损塑料",
                        "2": "其他垃圾/烟蒂",
                        "3": "其他垃圾/牙签",
                        "4": "其他垃圾/破碎花盆及碟碗",
                        "5": "其他垃圾/竹筷",
                        "6": "厨余垃圾/剩饭剩菜",
                        "7": "厨余垃圾/大骨头",
                        "8": "厨余垃圾/水果果皮",
                        "9": "厨余垃圾/水果果肉",
                        "10": "厨余垃圾/茶叶渣",
                        "11": "厨余垃圾/菜叶菜根",
                        "12": "厨余垃圾/蛋壳",
                        "13": "厨余垃圾/鱼骨",
                        "14": "可回收物/充电宝",
                        "15": "可回收物/包",
                        "16": "可回收物/化妆品瓶",
                        "17": "可回收物/塑料玩具",
                        "18": "可回收物/塑料碗盆",
                        "19": "可回收物/塑料衣架",
                        "20": "可回收物/快递纸袋",
                        "21": "可回收物/插头电线",
                        "22": "可回收物/旧衣服",
                        "23": "可回收物/易拉罐",
                        "24": "可回收物/枕头",
                        "25": "可回收物/毛绒玩具",
                        "26": "可回收物/洗发水瓶",
                        "27": "可回收物/玻璃杯",
                        "28": "可回收物/皮鞋",
                        "29": "可回收物/砧板",
                        "30": "可回收物/纸板箱",
                        "31": "可回收物/调料瓶",
                        "32": "可回收物/酒瓶",
                        "33": "可回收物/金属食品罐",
                        "34": "可回收物/锅",
                        "35": "可回收物/食用油桶",
                        "36": "可回收物/饮料瓶",
                        "37": "有害垃圾/干电池",
                        "38": "有害垃圾/软膏",
                        "39": "有害垃圾/过期药物",
                        "40": "可回收物/毛巾",
                        "41": "可回收物/饮料盒",
                        "42": "可回收物/纸袋"
                    }

            def build_model(self, model_path):
                model = resnext101_32x16d_wsl(pretrained=False, progress=False)
                model.fc = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(2048, 43)
                )
                model.load_state_dict(torch.load(model_path))
                return model

            def preprocess_img(self):
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                infer_transformation = transforms.Compose([
                    Resize((args['image_size'], args['image_size'])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
                return infer_transformation

            def _preprocess(self, data):
                preprocessed_data = {}
                for k, v in data.items():
                    for file_name, file_content in v.items():
                        img = Image.open(file_content)
                        img = self.pre_img(img)
                        preprocessed_data[k] = img
                return preprocessed_data

            def _inference(self, data):
                """
                model inference function
                Here are a inference example of resnet, if you use another model, please modify this function
                """
                img = data['input_img']
                img = img.unsqueeze(0)
                img = img.to(self.device)
                with torch.no_grad():
                    pred_score = self.model(img)

                if pred_score is not None:
                    _, pred_label = torch.max(pred_score.data, 1)
                    result = {'result': self.label_id_name_dict[str(pred_label[0].item())]}
                else:
                    result = {'result': 'predict score is None'}

                return result

            def _postprocess(self, data):
                return data

        class Resize(object):
            def __init__(self, size, interpolation=Image.BILINEAR):
                self.size = size
                self.interpolation = interpolation

            def __call__(self, img):
                ratio = self.size[0] / self.size[1]
                w, h = img.size
                if w / h < ratio:
                    t = int(h * ratio)
                    w_padding = (t - w) // 2
                    img = img.crop((-w_padding, 0, w + w_padding, h))
                else:
                    t = int(w / ratio)
                    h_padding = (t - h) // 2
                    img = img.crop((0, -h_padding, w, h + h_padding))

                img = img.resize(self.size, self.interpolation)

                return img

        model_path = 'D:/Download/model_20_9982_9080.pth'
        infer = classfication_service(model_path)
        file_path= c
        img = Image.open(file_path)
        img = infer.pre_img(img)
        result = infer._inference({'input_img': img})
        a=str(result)
        self.textEdit.setText(a)
