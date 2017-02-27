import math
import re
import sys
from os import path
from functools import partial
import numpy as np
from PyQt4 import QtGui, QtCore
import Dm3Reader3 as dm3
import Constants as const
import ImageSupport as imsup
import CrossCorr as cc
import Transform as tr

# --------------------------------------------------------

class TriangulateWidget(QtGui.QWidget):
    def __init__(self):
        super(TriangulateWidget, self).__init__()
        self.display = QtGui.QLabel()
        imagePath = QtGui.QFileDialog.getOpenFileName()
        self.image = LoadImageSeriesFromFirstFile(imagePath)
        self.pointSets = []
        self.createPixmap()
        self.initUI()

    def initUI(self):
        prevButton = QtGui.QPushButton(QtGui.QIcon('gui/prev.png'), '', self)
        prevButton.clicked.connect(partial(self.changePixmap, False))
        nextButton = QtGui.QPushButton(QtGui.QIcon('gui/next.png'), '', self)
        nextButton.clicked.connect(partial(self.changePixmap, True))
        doneButton = QtGui.QPushButton('Done', self)
        doneButton.clicked.connect(self.triangulate)

        hbox_nav = QtGui.QHBoxLayout()
        hbox_nav.addWidget(prevButton)
        hbox_nav.addWidget(nextButton)
        hbox_nav.addWidget(doneButton)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.display)
        vbox.addLayout(hbox_nav)
        self.setLayout(vbox)

        # self.statusBar().showMessage('Ready')
        self.move(250, 30)
        self.setWindowTitle('Triangulation window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

    def createPixmap(self):
        qImg = QtGui.QImage(imsup.ScaleImage(self.image.buffer, 0.0, 255.0).astype(np.uint8),
                            self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        # pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.display.setPixmap(pixmap)

    def changePixmap(self, toNext=True):
        newImage = self.image.next if toNext else self.image.prev
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
            # child.hide()
        if newImage is not None:
            newImage.ReIm2AmPh()
            self.image = newImage
            self.createPixmap()
            if len(self.pointSets) < self.image.numInSeries:
                return
            for pt, idx in zip(self.pointSets[self.image.numInSeries-1], range(1, len(self.pointSets[self.image.numInSeries-1])+1)):
                lab = QtGui.QLabel('{0}'.format(idx), self.display)
                lab.setStyleSheet('font-size:18pt; background-color:white; border:1px solid rgb(0, 0, 0);')
                lab.move(pt[1], pt[0] + lab.height() // 2)
                lab.show()

    # def mousePressEvent(self, QMouseEvent):
    #     print(QMouseEvent.pos())

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        currPos = [pos.y(), pos.x()]
        startPos = ((self.height() - self.image.height) // 2, (self.width() - self.image.width) // 2)
        endPos = (startPos[0] + self.image.height, startPos[1] + self.image.width)

        if startPos[0] < currPos[0] < endPos[0] and startPos[1] < currPos[1] < endPos[1]:
            currPos = [a - b for a, b in zip(currPos, startPos)]
            if len(self.pointSets) < self.image.numInSeries:
                self.pointSets.append([])
            self.pointSets[self.image.numInSeries-1].append(currPos)
            lab = QtGui.QLabel('{0}'.format(len(self.pointSets[self.image.numInSeries-1])), self.display)
            lab.setStyleSheet('font-size:18pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(currPos[1], currPos[0] + lab.height() // 2)
            lab.show()
            print(currPos)

        print(self.pointSets)

    def triangulate(self):
        tr1 = self.pointSets[0][:3]
        tr2 = self.pointSets[1][:3]

        r12 = CalcDistance(tr1[0], tr1[1])
        r13 = CalcDistance(tr1[0], tr1[2])
        r23 = CalcDistance(tr1[1], tr1[2])

        alpha1 = CalcInnerAngle(r12, r13, r23)
        alpha2 = CalcInnerAngle(r12, r23, r13)
        alpha3 = CalcInnerAngle(r13, r23, r12)

        betha1 = CalcOuterAngle(tr1[0], tr1[2])
        betha2 = CalcOuterAngle(tr1[0], tr1[1])
        betha3 = CalcOuterAngle(tr1[1], tr1[2])

        R12 = CalcDistance(tr2[0], tr2[1])
        R13 = CalcDistance(tr2[0], tr2[2])
        R23 = CalcDistance(tr2[1], tr2[2])

        Alpha1 = CalcInnerAngle(R12, R13, R23)
        Alpha2 = CalcInnerAngle(R12, R23, R13)
        Alpha3 = CalcInnerAngle(R13, R23, R12)

        Betha1 = CalcOuterAngle(tr2[0], tr2[2])
        Betha2 = CalcOuterAngle(tr2[0], tr2[1])
        Betha3 = CalcOuterAngle(tr2[1], tr2[2])

        dBetha1 = abs(betha1 - Betha1)
        dBetha2 = abs(betha2 - Betha2)
        dBetha3 = abs(betha3 - Betha3)

        magAvg = np.average([r12 / R12, r13 / R13, r23 / R23])
        dBethaAvg = np.average([dBetha1, dBetha2, dBetha3])

        print('---- Triangle 1 ----')
        print('R12 = {0:.2f} px\nR13 = {1:.2f} px\nR23 = {2:.2f} px\n---'.format(r12, r13, r23))
        print('a1 = {0:.0f} deg\na2 = {1:.0f} deg\na3 = {2:.0f} deg\n---'.format(alpha1, alpha2, alpha3))
        print('b1 = {0:.0f} deg\nb2 = {1:.0f} deg\nb3 = {2:.0f} deg'.format(betha1, betha2, betha3))
        print('---- Triangle 2 ----')
        print('R12 = {0:.2f} px\nR13 = {1:.2f} px\nR23 = {2:.2f} px\n---'.format(R12, R13, R23))
        print('a1 = {0:.0f} deg\na2 = {1:.0f} deg\na3 = {2:.0f} deg\n---'.format(Alpha1, Alpha2, Alpha3))
        print('b1 = {0:.0f} deg\nb2 = {1:.0f} deg\nb3 = {2:.0f} deg'.format(Betha1, Betha2, Betha3))
        print('---- Magnification ----')
        print('mag12 = {0:.2f}x\nmag13 = {0:.2f}x\nmag23 = {0:.2f}x'.format(r12 / R12, r13 / R13, r23 / R23))
        print('---- Rotation ----')
        print('db1 = {0:.0f} deg\ndb2 = {1:.0f} deg\ndb3 = {2:.0f} deg'.format(dBetha1, dBetha2, dBetha3))
        print('------------------')
        print('Average magnification = {0:.2f}x'.format(magAvg))
        print('Average rotation = {0:.2f} deg'.format(dBethaAvg))

        img1 = self.image.prev
        # img2Rot = imsup.RotateImageAndCrop(self.image, dBethaAvg)
        imsup.SaveAmpImage(img1, 'img1.png')
        imsup.SaveAmpImage(self.image, 'img2.png')
        img2Mag = tr.RescaleImageSki2(self.image, magAvg)
        imsup.SaveAmpImage(img2Mag, 'img2_mag.png')
        img2Rot = tr.RotateImageSki2(img2Mag, dBethaAvg)
        imsup.SaveAmpImage(img2Rot, 'img2_rot_crop.png')
        cropCoords = imsup.DetermineCropCoordsForNewWidth(img1.width, img2Rot.width)
        img1Crop = imsup.CropImageROICoords(img1, cropCoords)
        imsup.SaveAmpImage(img1Crop, 'img1_crop.png')

        imgs1H = imsup.LinkTwoImagesSmoothlyH(img1Crop, img1Crop)
        linkedImages1 = imsup.LinkTwoImagesSmoothlyV(imgs1H, imgs1H)
        imgs2H = imsup.LinkTwoImagesSmoothlyH(img2Rot, img2Rot)
        linkedImages2 = imsup.LinkTwoImagesSmoothlyV(imgs2H, imgs2H)

        img1Alg, img2Alg = cc.AlignTwoImages(linkedImages1, linkedImages2, [0, 1, 2])

        newCoords = imsup.DetermineCropCoords(img1Crop.width, img1Crop.height, img2Alg.shift)
        newSquareCoords = imsup.MakeSquareCoords(newCoords)
        print(newSquareCoords)
        newSquareCoords[2:4] = list(np.array(newSquareCoords[2:4]) - np.array(newSquareCoords[:2]))
        newSquareCoords[:2] = [0, 0]
        print(newSquareCoords)

        img1Res = imsup.CropImageROICoords(img1Alg, newSquareCoords)
        img2Res = imsup.CropImageROICoords(img2Alg, newSquareCoords)

        imsup.SaveAmpImage(img1Alg, 'holo1_big.png')
        imsup.SaveAmpImage(img2Alg, 'holo2_big.png')
        imsup.SaveAmpImage(img1Res, 'holo1.png')
        imsup.SaveAmpImage(img2Res, 'holo2.png')
        return

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        imgData = dm3.ReadDm3File(imgPath)
        imgMatrix = imsup.PrepareImageMatrix(imgData, const.dimSize)
        img = imsup.ImageWithBuffer(const.dimSize, const.dimSize, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'])
        img.LoadAmpData(np.sqrt(imgMatrix).astype(np.float32))
        # ---
        imsup.RemovePixelArtifacts(img, 1.3)
        img.UpdateBuffer()
        # ---
        img.numInSeries = imgNum
        imgList.append(img)

        imgNum += 1
        imgNumTextNew = imgNumText.replace(str(imgNum-1), str(imgNum))
        if imgNum == 10:
            imgNumTextNew = imgNumTextNew[1:]
        imgPath = RReplace(imgPath, imgNumText, imgNumTextNew, 1)
        imgNumText = imgNumTextNew

    imgList.UpdateLinks()
    return imgList[0]

# --------------------------------------------------------

def CalcDistance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

# --------------------------------------------------------

def CalcInnerAngle(a, b, c):
    alpha = np.arccos(np.abs((a*a + b*b - c*c) / (2*a*b)))
    return imsup.Degrees(alpha)

# --------------------------------------------------------

def CalcOuterAngle(p1, p2):
    dist = CalcDistance(p1, p2)
    betha = np.arcsin(np.abs(p1[0] - p2[0]) / dist)
    return imsup.Degrees(betha)

# --------------------------------------------------------

def RReplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

def RunTriangulationWindow():
    app = QtGui.QApplication(sys.argv)
    trWindow = TriangulateWidget()
    sys.exit(app.exec_())