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

class Triangle:
    pass

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
        prevButton = QtGui.QPushButton('Prev', self)
        nextButton = QtGui.QPushButton('Next', self)
        rotLeftButton = QtGui.QPushButton(QtGui.QIcon('gui/rot_left.png'), '', self)
        rotRightButton = QtGui.QPushButton(QtGui.QIcon('gui/rot_right.png'), '', self)
        resetButton = QtGui.QPushButton('Reset', self)
        applyButton = QtGui.QPushButton('Apply', self)

        rotLeftButton.setFixedWidth(self.display.width() // 6)
        rotRightButton.setFixedWidth(self.display.width() // 6)

        prevButton.clicked.connect(partial(self.changePixmap, False))
        nextButton.clicked.connect(partial(self.changePixmap, True))
        rotLeftButton.clicked.connect(self.rotateLeft)
        rotRightButton.clicked.connect(self.rotateRight)
        resetButton.clicked.connect(self.resetImage)
        applyButton.clicked.connect(self.applyChangesToImage)

        self.rotAngleEdit = QtGui.QLineEdit('5', self)
        self.rotAngleEdit.setFixedWidth(20)
        self.rotAngleEdit.setMaxLength(3)

        upButton = QtGui.QPushButton(QtGui.QIcon('gui/up.png'), '', self)
        downButton = QtGui.QPushButton(QtGui.QIcon('gui/down.png'), '', self)
        leftButton = QtGui.QPushButton(QtGui.QIcon('gui/left.png'), '', self)
        rightButton = QtGui.QPushButton(QtGui.QIcon('gui/right.png'), '', self)
        leftButton.setFixedWidth(self.display.width() // 6)
        rightButton.setFixedWidth(self.display.width() // 6)

        # upButton.clicked.connect(self.movePixmapUp)
        # downButton.clicked.connect(self.movePixmapDown)
        # leftButton.clicked.connect(self.movePixmapLeft)
        # rightButton.clicked.connect(self.movePixmapRight)

        self.shiftStepEdit = QtGui.QLineEdit('5', self)
        self.shiftStepEdit.setFixedWidth(20)
        self.shiftStepEdit.setMaxLength(3)

        alignButton = QtGui.QPushButton('Align', self)
        alignButton.clicked.connect(self.triangulate)
        cropButton = QtGui.QPushButton('Crop', self)
        cropButton.clicked.connect(self.cropFragment)
        exportButton = QtGui.QPushButton('Export', self)
        exportButton.clicked.connect(self.exportImage)

        self.basicRadioButton = QtGui.QRadioButton('Basic', self)
        self.basicRadioButton.setChecked(True)
        self.advancedRadioButton = QtGui.QRadioButton('Advanced', self)

        hbox_nav = QtGui.QHBoxLayout()
        hbox_nav.addWidget(prevButton)
        hbox_nav.addWidget(nextButton)

        hbox_rot = QtGui.QHBoxLayout()
        hbox_rot.addWidget(rotLeftButton)
        hbox_rot.addWidget(self.rotAngleEdit)
        hbox_rot.addWidget(rotRightButton)

        hbox_dec = QtGui.QHBoxLayout()
        hbox_dec.addWidget(resetButton)
        hbox_dec.addWidget(applyButton)

        vbox_nav = QtGui.QVBoxLayout()
        vbox_nav.addLayout(hbox_nav)
        vbox_nav.addLayout(hbox_rot)
        vbox_nav.addLayout(hbox_dec)

        hbox_mv_lr = QtGui.QHBoxLayout()
        hbox_mv_lr.addWidget(leftButton)
        hbox_mv_lr.addWidget(self.shiftStepEdit)
        hbox_mv_lr.addWidget(rightButton)

        vbox_mv = QtGui.QVBoxLayout()
        vbox_mv.addWidget(upButton)
        vbox_mv.addLayout(hbox_mv_lr)
        vbox_mv.addWidget(downButton)

        vbox_opt = QtGui.QVBoxLayout()
        vbox_opt.addWidget(alignButton)
        vbox_opt.addWidget(cropButton)
        vbox_opt.addWidget(exportButton)

        vbox_tr = QtGui.QVBoxLayout()
        vbox_tr.addWidget(self.basicRadioButton)
        vbox_tr.addWidget(self.advancedRadioButton)

        hbox_panel = QtGui.QHBoxLayout()
        hbox_panel.addLayout(vbox_nav)
        hbox_panel.addLayout(vbox_mv)
        hbox_panel.addLayout(vbox_opt)
        hbox_panel.addLayout(vbox_tr)

        vbox_main = QtGui.QVBoxLayout()
        vbox_main.addWidget(self.display)
        vbox_main.addLayout(hbox_panel)
        self.setLayout(vbox_main)

        # self.statusBar().showMessage('Ready')
        self.move(250, 5)
        self.setWindowTitle('Triangulation window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

    def createPixmap(self):
        paddedExitWave = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        qImg = QtGui.QImage(imsup.ScaleImage(paddedExitWave.buffer, 0.0, 255.0).astype(np.uint8),
                            paddedExitWave.width, paddedExitWave.height, QtGui.QImage.Format_Indexed8)
        # qImg = QtGui.QImage(imsup.ScaleImage(self.image.buffer, 0.0, 255.0).astype(np.uint8),
        #                     self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)    # !!!
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
                lab.move(pt[0], pt[1])
                lab.show()

    # def mousePressEvent(self, QMouseEvent):
    #     print(QMouseEvent.pos())

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        currPos = [pos.x(), pos.y()]
        startPos = [ self.display.pos().x(), self.display.pos().y() ]
        endPos = [ startPos[0] + self.display.width(), startPos[1] + self.display.height() ]

        if startPos[0] < currPos[0] < endPos[0] and startPos[1] < currPos[1] < endPos[1]:
            currPos = [ a - b for a, b in zip(currPos, startPos) ]
            if len(self.pointSets) < self.image.numInSeries:
                self.pointSets.append([])
            self.pointSets[self.image.numInSeries-1].append(currPos)
            lab = QtGui.QLabel('{0}'.format(len(self.pointSets[self.image.numInSeries-1])), self.display)
            lab.setStyleSheet('font-size:18pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(currPos[0], currPos[1])
            lab.show()

    def triangulate(self):
        if self.basicRadioButton.isChecked():
            self.triangulateBasic()
        elif self.advancedRadioButton.isChecked():
            self.triangulateAdvanced()

    def triangulateBasic(self):
        triangles = [ [ CalcRealCoords(const.dimSize, self.pointSets[trIdx][pIdx]) for pIdx in range(3) ] for trIdx in range(2) ]
        tr1Dists = [ CalcDistance(triangles[0][pIdx1], triangles[0][pIdx2]) for pIdx1, pIdx2 in zip([0, 0, 1], [1, 2, 2]) ]
        tr2Dists = [ CalcDistance(triangles[1][pIdx1], triangles[1][pIdx2]) for pIdx1, pIdx2 in zip([0, 0, 1], [1, 2, 2]) ]

        mags = [dist1 / dist2 for dist1, dist2 in zip(tr1Dists, tr2Dists)]

        rotAngles = []
        for idx, p1, p2 in zip(range(3), triangles[0], triangles[1]):
            rotAngles.append(CalcRotAngle(p1, p2))

        magAvg = np.average(mags)
        rotAngleAvg = np.average(rotAngles)

        img1 = imsup.CopyImage(self.image.prev)
        img2 = imsup.CopyImage(self.image)

        # magnification
        # img2Mag = tr.RescaleImageSki2(img2, magAvg)

        # rotation
        print('rotAngles = {0}'.format(rotAngles))
        print('rotAngleAvg = {0}'.format(rotAngleAvg))
        img2Rot = tr.RotateImageSki2(img2, rotAngleAvg, cut=False)
        img2RotCut = tr.RotateImageSki2(img2, rotAngleAvg, cut=True)
        cropCoords = imsup.DetermineCropCoordsForNewWidth(img1.width, img2RotCut.width)
        img1Crop = imsup.CropImageROICoords(img1, cropCoords)

        # x-y alignment (shift)
        imgs1H = imsup.LinkTwoImagesSmoothlyH(img1Crop, img1Crop)
        linkedImages1 = imsup.LinkTwoImagesSmoothlyV(imgs1H, imgs1H)
        imgs2H = imsup.LinkTwoImagesSmoothlyH(img2RotCut, img2RotCut)
        linkedImages2 = imsup.LinkTwoImagesSmoothlyV(imgs2H, imgs2H)

        img1Alg, img2Alg = cc.AlignTwoImages(linkedImages1, linkedImages2, [0, 1, 2])

        newSquareCoords = imsup.MakeSquareCoords(imsup.DetermineCropCoords(img1Crop.width, img1Crop.height, img2Alg.shift))
        newSquareCoords[2:4] = list(np.array(newSquareCoords[2:4]) - np.array(newSquareCoords[:2]))
        newSquareCoords[:2] = [0, 0]

        img1Res = imsup.CropImageROICoords(img1Alg, newSquareCoords)
        img2Res = imsup.CropImageROICoords(img2Alg, newSquareCoords)

        # ---
        img2RotShift = cc.ShiftImage(img2Rot, img2Alg.shift)
        newSquareCoords2 = imsup.MakeSquareCoords(imsup.DetermineCropCoords(img2RotShift.width, img2RotShift.height, img2Alg.shift))
        img1Crop2 = imsup.CropImageROICoords(img1, newSquareCoords2)
        img2RotCrop = imsup.CropImageROICoords(img2RotShift, newSquareCoords2)

        imsup.SaveAmpImage(img1Crop2, 'holo1.png')
        imsup.SaveAmpImage(img2RotCrop, 'holo2.png')
        # ---

        imsup.SaveAmpImage(img1Alg, 'holo1_big.png')
        imsup.SaveAmpImage(img2Alg, 'holo2_big.png')

        imsup.SaveAmpImage(img1Res, 'holo1_cut.png')
        imsup.SaveAmpImage(img2Res, 'holo2_cut.png')

    def triangulateAdvanced(self):
        triangles = [ [ CalcRealCoords(const.dimSize, self.pointSets[trIdx][pIdx]) for pIdx in range(3) ] for trIdx in range(2) ]

        tr1Dists = [ CalcDistance(triangles[0][pIdx1], triangles[0][pIdx2]) for pIdx1, pIdx2 in zip([0, 0, 1], [1, 2, 2]) ]
        tr2Dists = [ CalcDistance(triangles[1][pIdx1], triangles[1][pIdx2]) for pIdx1, pIdx2 in zip([0, 0, 1], [1, 2, 2]) ]

        rcSum = [0, 0]
        rotCenters = []
        for idx1 in range(3):
            for idx2 in range(idx1+1, 3):
                print(triangles[0][idx1], triangles[0][idx2])
                print(triangles[1][idx1], triangles[1][idx2])
                rotCenter = tr.FindRotationCenter([triangles[0][idx1], triangles[0][idx2]],
                                                  [triangles[1][idx1], triangles[1][idx2]])
                rotCenters.append(rotCenter)
                print(rotCenter)
                rcSum = list(np.array(rcSum) + np.array(rotCenter))

        rotCenterAvg = list(np.array(rcSum) / 3.0)
        print(rotCenterAvg)

        rcShift = [ -int(rc) for rc in rotCenterAvg ]
        rcShift.reverse()
        img1 = imsup.CopyImage(self.image.prev)
        img2 = imsup.CopyImage(self.image)
        img1Rc = cc.ShiftImage(img1, rcShift)
        img2Rc = cc.ShiftImage(img2, rcShift)
        img1Rc = imsup.CreateImageWithBufferFromImage(img1Rc)
        img2Rc = imsup.CreateImageWithBufferFromImage(img2Rc)

        rotAngles = []
        for idx, p1, p2 in zip(range(3), triangles[0], triangles[1]):
            p1New = CalcNewCoords(p1, rotCenterAvg)
            p2New = CalcNewCoords(p2, rotCenterAvg)
            triangles[0][idx] = p1New
            triangles[1][idx] = p2New
            rotAngles.append(CalcRotAngle(p1New, p2New))

        rotAngleAvg = np.average(rotAngles)

        mags = [ dist1 / dist2 for dist1, dist2 in zip(tr1Dists, tr2Dists) ]
        magAvg = np.average(mags)

        # tr1InnerAngles = [ CalcInnerAngle(a, b, c) for a, b, c in zip(tr1Dists, tr1Dists[-1:] + tr1Dists[:-1], tr1Dists[-2:] + tr1Dists[:-2]) ]
        # tr2InnerAngles = [ CalcInnerAngle(a, b, c) for a, b, c in zip(tr2Dists, tr2Dists[-1:] + tr2Dists[:-1], tr2Dists[-2:] + tr2Dists[:-2]) ]

        # print('---- Triangle 1 ----')
        # print([ 'R{0} = {1:.2f} px\n'.format(idx + 1, dist) for idx, dist in zip(range(3), tr1Dists) ])
        # print([ 'alpha{0} = {1:.0f} deg\n'.format(idx + 1, angle) for idx, angle in zip(range(3), tr1InnerAngles) ])
        # print('---- Triangle 2 ----')
        # print([ 'R{0} = {1:.2f} px\n'.format(idx + 1, dist) for idx, dist in zip(range(3), tr2Dists) ])
        # print([ 'alpha{0} = {1:.0f} deg\n'.format(idx + 1, angle) for idx, angle in zip(range(3), tr2InnerAngles) ])
        # print('---- Magnification ----')
        # print([ 'mag{0} = {1:.2f}x\n'.format(idx + 1, mag) for idx, mag in zip(range(3), mags) ])
        print('---- Rotation ----')
        print([ 'phi{0} = {1:.0f} deg\n'.format(idx + 1, angle) for idx, angle in zip(range(3), rotAngles) ])
        # print('---- Shifts ----')
        # print([ 'dxy{0} = ({1:.1f}, {2:.1f}) px\n'.format(idx + 1, sh[0], sh[1]) for idx, sh in zip(range(3), shifts) ])
        # print('------------------')
        # print('Average magnification = {0:.2f}x'.format(magAvg))
        print('Average rotation = {0:.2f} deg'.format(rotAngleAvg))
        # print('Average shift = ({0:.0f}, {1:.0f}) px'.format(shiftAvg[0], shiftAvg[1]))

        # img2Mag = tr.RescaleImageSki2(img2Rc, magAvg)
        img2Rot = tr.RotateImageSki2(img2Rc, rotAngleAvg, cut=False)

        # imsup.SaveAmpImage(img1Rc, 'holo1.png')
        # imsup.SaveAmpImage(img2Rot, 'holo2.png')

        img1Rc.MoveToCPU()
        img2Rot.MoveToCPU()
        img1Rc.UpdateBuffer()
        img2Rot.UpdateBuffer()
        self.image.next = img1Rc
        img1Rc.prev = self.image
        img1Rc.next = img2Rot
        img2Rot.prev = img1Rc
        img1Rc.numInSeries = self.image.numInSeries + 1
        img2Rot.numInSeries = img1Rc.numInSeries + 1
        self.pointSets.append([])
        self.changePixmap(True)

        # dodac mozliwosc zaznaczenia wiekszej niz 3 liczby punktow w celu dokladniejszego okreslenia srodka obrotu
        # dodac opcje do unwarpingu
        print('Triangulation complete!')

    def rotateManual(self):
        # img2Rot = tr.RotateImageSki2(self.image, self.image.rot, cut=False)
        img2Rot = imsup.RotateImage(self.image, self.image.rot)
        img2Rot.MoveToCPU()
        self.image.buffer = np.copy(img2Rot.amPh.am)
        self.createPixmap()

    def rotateRight(self):
        self.rotAngleEdit.text()
        self.image.rot -= int(self.rotAngleEdit.text())
        self.rotateManual()

    def rotateLeft(self):
        self.image.rot += int(self.rotAngleEdit.text())
        self.rotateManual()

    def resetImage(self):
        self.image.UpdateBuffer()
        self.image.shift = [0, 0]
        self.image.rot = 0
        self.image.defocus = 0.0
        self.createPixmap()

    def applyChangesToImage(self):
        self.image.UpdateImageFromBuffer()

    def cropFragment(self):
        [ pt1, pt2 ] = self.pointSets[self.image.numInSeries-1][:2]
        dispCropCoords = pt1 + pt2
        midCropCoords = CalcRealCoords(self.image.width, dispCropCoords)
        tlCropCoords = imsup.MakeSquareCoords(CalcTopLeftCoords(self.image.width, midCropCoords))
        img1 = self.image
        img1Crop = imsup.CropImageROICoords(img1, tlCropCoords)
        imsup.SaveAmpImage(img1Crop, 'crop1.png')

        if self.image.prev is not None:
            img2 = self.image.prev
            img2Crop = imsup.CropImageROICoords(img2, tlCropCoords)
            imsup.SaveAmpImage(img2Crop, 'crop2.png')

    def unWarpImage(self):
        pass

    def exportImage(self):
        fName = 'img{0}.png'.format(self.image.numInSeries)
        imsup.SaveAmpImage(self.image, fName)
        print('Saved image as "{0}"'.format(fName))

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
        imsup.RemovePixelArtifacts(img, const.badPxThreshold)
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

def CalcTopLeftCoords(imgWidth, midCoords):
    topLeftCoords = [ mc + imgWidth // 2 for mc in midCoords ]
    return topLeftCoords

# --------------------------------------------------------

def CalcRealCoords(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ (dc - const.ccWidgetDim // 2) * factor for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcDispCoords(dispWidth, realCoords):
    imgWidth = const.dimSize
    factor = dispWidth / imgWidth
    dispCoords = [ (rc * factor) + const.ccWidgetDim // 2 for rc in realCoords ]
    return dispCoords

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

def CalcNewCoords(p1, newCenter):
    p2 = [ px - cx for px, cx in zip(p1, newCenter) ]
    return p2

# --------------------------------------------------------

# tu jeszcze cos nie tak (03-04-2017)
def CalcRotAngle(p1, p2):
    z1 = np.complex(p1[0], p1[1])
    z2 = np.complex(p2[0], p2[1])
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)
    # print(imsup.Degrees(phi1), imsup.Degrees(phi2))
    rotAngle = np.abs(imsup.Degrees(phi2 - phi1))
    # if rotAngle < 0:
    #     rotAngle = 360 - np.abs(rotAngle)
    return rotAngle

# --------------------------------------------------------

def SwitchXY(xy):
    return [xy[1], xy[0]]

# --------------------------------------------------------

def RReplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

def RunTriangulationWindow():
    app = QtGui.QApplication(sys.argv)
    trWindow = TriangulateWidget()
    sys.exit(app.exec_())