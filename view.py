# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow #窗口控件
#from zzz import *
from yuancheng import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()#创建一个应用程序的窗口
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    #ui.paintEvent(MainWindow)
    MainWindow.setWindowTitle("垃圾图片分类识别")
    MainWindow.show()
    sys.exit(app.exec_())



        # self.pushButton_3.setText(_translate("MainWindow", "打开图片"))
        # self.pushButton_3.setIcon(QtGui.QIcon("icons/open.svg"))
        # self.pushButton_3.setIconSize(QtCore.QSize(20,20))
        # self.pushButton_3.setStyleSheet("QPushButton{background:#16A085;border:5px;color:#000000;font-size:15px;}"
        # "QPushButon:hover{background-color:#333333;}")

        # self.pushButton_2.setText(_translate("MainWindow", "卡号定位"))
        # self.pushButton_2.setIcon(QtGui.QIcon("icons/locate.svg"))
        # self.pushButton_2.setIconSize(QtCore.QSize(20,20))
        # self.pushButton_2.setStyleSheet("QPushButton{background:#D26900;border:3px;color:#000000;font-size:15px;}"
        # "QPushButton:hover{background-color:#333333;}")

        # self.pushButton_4.setText(_translate("MainWindow", "卡号识别"))
        # self.pushButton_4.setIcon(QtGui.QIcon("icons/bank_card.svg"))
        # self.pushButton_4.setIconSize(QtCore.QSize(20,20))
        # self.pushButton_4.setStyleSheet("QPushButton{background:#00AEAE;border:3px;color:#000000;font-size:15px}"
        # "QPushButton:hover{background-color:#333333;}")

        # self.pushButton.setText(_translate("MainWindow", "退        出"))
        # self.pushButton.setIcon(QtGui.QIcon("icons/close_.svg"))
        # self.pushButton.setIconSize(QtCore.QSize(20,20))
        # self.pushButton.setStyleSheet("QPushButton{background:#CE0000;border:3px;color:#000000;font-size:15px;}"
        # "QPushButton:hover{background-color:#333333;}")
