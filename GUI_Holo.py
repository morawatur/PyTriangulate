import re
import sys
from os import path
from functools import partial
import numpy as np

from skimage import transform as tf
from PyQt4 import QtGui, QtCore
import Dm3Reader3_New as dm3
import Constants as const
import ImageSupport as imsup
import CrossCorr as cc
import Transform as tr
import Unwarp as uw
import Holo as holo

# --------------------------------------------------------

class Triangle:
    pass

# --------------------------------------------------------

class LabelExt(QtGui.QLabel):
    def __init__(self, parent, image=None):
        super(LabelExt, self).__init__(parent)
        self.image = image
        self.setImage()
        self.pointSets = [[]]
        self.show_lines = True
        self.show_labs = True
        # while image.next is not None:
        #    self.pointSets.append([])

    # prowizorka - stałe liczbowe do poprawy
    def paintEvent(self, event):
        super(LabelExt, self).paintEvent(event)
        linePen = QtGui.QPen(QtCore.Qt.yellow)
        linePen.setCapStyle(QtCore.Qt.RoundCap)
        linePen.setWidth(3)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        imgIdx = self.image.numInSeries - 1
        qp.setPen(linePen)
        qp.setBrush(QtCore.Qt.yellow)
        for pt in self.pointSets[imgIdx]:
            # rect = QtCore.QRect(pt[0]-3, pt[1]-3, 7, 7)
            # qp.drawArc(rect, 0, 16*360)
            qp.drawEllipse(pt[0]-3, pt[1]-3, 7, 7)
        if self.show_lines:
            linePen.setWidth(2)
            qp.setPen(linePen)
            for pt1, pt2 in zip(self.pointSets[imgIdx], self.pointSets[imgIdx][1:] + self.pointSets[imgIdx][:1]):
                line = QtCore.QLine(pt1[0], pt1[1], pt2[0], pt2[1])
                qp.drawLine(line)
        qp.end()

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        currPos = [pos.x(), pos.y()]
        self.pointSets[self.image.numInSeries - 1].append(currPos)
        self.repaint()

        if self.parent().show_labels_checkbox.isChecked():
            lab = QtGui.QLabel('{0}'.format(len(self.pointSets[self.image.numInSeries - 1])), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pos.x()+4, pos.y()+4)
            lab.show()

    def setImage(self, dispAmp=True):
        self.image.MoveToCPU()
        if dispAmp:
            self.image.buffer = np.copy(self.image.amPh.am)
        else:
            self.image.buffer = np.copy(self.image.amPh.ph)
        paddedImage = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        qImg = QtGui.QImage(imsup.ScaleImage(paddedImage.buffer, 0.0, 255.0).astype(np.uint8),
                            paddedImage.width, paddedImage.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.setPixmap(pixmap)
        self.repaint()

    def changeImage(self, toNext=True, dispAmp=True, dispLabs=True):
        newImage = self.image.next if toNext else self.image.prev
        if newImage is not None:
            newImage.ReIm2AmPh()
            self.image = newImage
            if len(self.pointSets) < self.image.numInSeries:
                self.pointSets.append([])
            self.setImage(dispAmp)

        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

        if dispLabs:
            self.show_labels()

    def hide_labels(self):
        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

    def show_labels(self):
        imgIdx = self.image.numInSeries - 1
        for pt, idx in zip(self.pointSets[imgIdx], range(1, len(self.pointSets[imgIdx]) + 1)):
            lab = QtGui.QLabel('{0}'.format(idx), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pt[0] + 4, pt[1] + 4)
            lab.show()

# --------------------------------------------------------

class TriangulateWidget(QtGui.QWidget):
    def __init__(self):
        super(TriangulateWidget, self).__init__()
        imagePath = QtGui.QFileDialog.getOpenFileName()
        image = LoadImageSeriesFromFirstFile(imagePath)
        self.display = LabelExt(self, image)
        self.initUI()

    def initUI(self):
        prevButton = QtGui.QPushButton('Prev', self)
        nextButton = QtGui.QPushButton('Next', self)

        prevButton.clicked.connect(self.goToPrevImage)
        nextButton.clicked.connect(self.goToNextImage)

        flipButton = QtGui.QPushButton('Flip', self)
        cropButton = QtGui.QPushButton('Crop', self)

        exportButton = QtGui.QPushButton('Export', self)
        deleteButton = QtGui.QPushButton('Delete', self)
        clearButton = QtGui.QPushButton('Clear', self)

        flipButton.clicked.connect(self.flip_image_h)
        cropButton.clicked.connect(self.cropFragment)

        exportButton.clicked.connect(self.exportImage)
        deleteButton.clicked.connect(self.deleteImage)
        clearButton.clicked.connect(self.clearImage)

        alignButton = QtGui.QPushButton('Align', self)
        warpButton = QtGui.QPushButton('Warp', self)

        alignButton.clicked.connect(self.triangulate)
        warpButton.clicked.connect(partial(self.warpImage, False))

        holo_no_ref_button = QtGui.QPushButton('Holo no ref', self)
        holo_with_ref_button = QtGui.QPushButton('Holo with ref', self)

        holo_no_ref_button.clicked.connect(self.rec_holo_no_ref)
        holo_with_ref_button.clicked.connect(self.rec_holo_with_ref)

        sum_button = QtGui.QPushButton('Sum', self)
        diff_button = QtGui.QPushButton('Difference', self)

        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)

        self.show_lines_checkbox = QtGui.QCheckBox('Show lines', self)
        self.show_lines_checkbox.setChecked(True)
        self.show_lines_checkbox.toggled.connect(self.toggle_lines)

        self.show_labels_checkbox = QtGui.QCheckBox('Show labels', self)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.toggle_labels)

        self.amp_radio_button = QtGui.QRadioButton('Amplitude', self)
        self.phs_radio_button = QtGui.QRadioButton('Phase', self)
        self.amp_radio_button.setChecked(True)

        self.amp_radio_button.toggled.connect(self.update_display)
        self.phs_radio_button.toggled.connect(self.update_display)

        hbox_nav = QtGui.QHBoxLayout()
        hbox_nav.addWidget(prevButton)
        hbox_nav.addWidget(nextButton)

        hbox_modify = QtGui.QHBoxLayout()
        hbox_modify.addWidget(flipButton)
        hbox_modify.addWidget(cropButton)

        hbox_imgop = QtGui.QHBoxLayout()
        hbox_imgop.addWidget(exportButton)
        hbox_imgop.addWidget(deleteButton)

        vbox_nav = QtGui.QVBoxLayout()
        vbox_nav.addLayout(hbox_nav)
        vbox_nav.addStretch(1)
        vbox_nav.addLayout(hbox_modify)
        vbox_nav.addStretch(1)
        vbox_nav.addLayout(hbox_imgop)

        vbox_check = QtGui.QVBoxLayout()
        vbox_check.addWidget(self.show_lines_checkbox)
        vbox_check.addStretch(1)
        vbox_check.addWidget(self.show_labels_checkbox)

        vbox_radio = QtGui.QVBoxLayout()
        vbox_radio.addWidget(self.amp_radio_button)
        vbox_radio.addStretch(1)
        vbox_radio.addWidget(self.phs_radio_button)

        hbox_disp = QtGui.QHBoxLayout()
        hbox_disp.addLayout(vbox_check)
        hbox_disp.addLayout(vbox_radio)

        vbox_disp = QtGui.QVBoxLayout()
        vbox_disp.addLayout(hbox_disp)
        vbox_disp.addStretch(1)
        vbox_disp.addWidget(clearButton)

        hbox_align = QtGui.QHBoxLayout()
        hbox_align.addWidget(alignButton)
        hbox_align.addWidget(warpButton)

        hbox_holo = QtGui.QHBoxLayout()
        hbox_holo.addWidget(holo_no_ref_button)
        hbox_holo.addWidget(holo_with_ref_button)

        hbox_calc = QtGui.QHBoxLayout()
        hbox_calc.addWidget(sum_button)
        hbox_calc.addWidget(diff_button)

        vbox_opt = QtGui.QVBoxLayout()
        vbox_opt.addLayout(hbox_align)
        vbox_opt.addStretch(1)
        vbox_opt.addLayout(hbox_holo)
        vbox_opt.addStretch(1)
        vbox_opt.addLayout(hbox_calc)

        hbox_panel = QtGui.QHBoxLayout()
        hbox_panel.addLayout(vbox_nav)
        hbox_panel.addLayout(vbox_disp)
        hbox_panel.addLayout(vbox_opt)

        vbox_main = QtGui.QVBoxLayout()
        vbox_main.addWidget(self.display)
        vbox_main.addLayout(hbox_panel)
        self.setLayout(vbox_main)

        self.move(250, 5)
        self.setWindowTitle('Triangulation window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

    def goToPrevImage(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
        self.display.changeImage(toNext=False, dispAmp=is_amp_checked, dispLabs=is_show_labels_checked)

    def goToNextImage(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
        self.display.changeImage(toNext=True, dispAmp=is_amp_checked, dispLabs=is_show_labels_checked)

    def flip_image_h(self):
        imsup.flip_image_h(self.display.image)
        self.display.setImage()

    def cropFragment(self):
        [pt1, pt2] = self.display.pointSets[self.display.image.numInSeries - 1][:2]
        dispCropCoords = pt1 + pt2
        realCropCoords = imsup.MakeSquareCoords(
            CalcRealTLCoordsForPaddedImage(self.display.image.width, dispCropCoords))

        imgCurr = self.display.image
        imgCurrCrop = imsup.CropImageROICoords(imgCurr, realCropCoords)
        imgCurrCrop = imsup.CreateImageWithBufferFromImage(imgCurrCrop)
        imgCurrCrop.MoveToCPU()

        if imgCurr.prev is not None:
            imgPrev = imgCurr.prev
            imgPrevCrop = imsup.CropImageROICoords(imgPrev, realCropCoords)
            imgPrevCrop = imsup.CreateImageWithBufferFromImage(imgPrevCrop)
            imgPrevCrop.MoveToCPU()
            cropImgList = imsup.ImageList([self.display.image, imgPrevCrop, imgCurrCrop])
            self.display.pointSets.append([])
        else:
            cropImgList = imsup.ImageList([self.display.image, imgCurrCrop])

        cropImgList.UpdateLinks()
        self.display.pointSets.append([])
        print(len(self.display.pointSets))
        self.goToNextImage()

    def exportImage(self):
        fName = 'img{0}.png'.format(self.display.image.numInSeries)
        imsup.SaveAmpImage(self.display.image, fName)
        print('Saved image as "{0}"'.format(fName))

    def deleteImage(self):
        # if len(self.display.pointSets) < 2:
        #     return

        curr_img = self.display.image
        if curr_img.prev is None and curr_img.next is None:
            return

        del self.display.pointSets[curr_img.numInSeries - 1]
        print(len(self.display.pointSets))

        if curr_img.prev is not None:
            curr_img.prev.next = curr_img.next

        if curr_img.next is not None:
            curr_img.next.prev = curr_img.prev
            tmp = curr_img.next
            while tmp is not None:
                tmp.numInSeries = tmp.prev.numInSeries + 1 if tmp.prev is not None else 1
                tmp = tmp.next

        if curr_img.prev is not None:
            self.goToPrevImage()
        else:
            self.goToNextImage()

        del curr_img

    def toggle_lines(self):
        self.display.show_lines = not self.display.show_lines
        self.display.repaint()

    def toggle_labels(self):
        self.display.show_labs = not self.display.show_labs
        if self.display.show_labs:
            self.display.show_labels()
        else:
            self.display.hide_labels()

    def update_display(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        self.display.setImage(dispAmp=is_amp_checked)

    def clearImage(self):
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
        self.display.pointSets[self.display.image.numInSeries - 1][:] = []
        self.display.repaint()

    # dodac mozliwosc zaznaczenia wiekszej niz 3 liczby punktow w celu dokladniejszego okreslenia srodka obrotu
    # powinna byc zmienna img_width zamiast dimSize
    def triangulate(self):
        img_width = self.display.image.width
        triangles = [ [ CalcRealCoords(img_width, self.display.pointSets[trIdx][pIdx]) for pIdx in range(3) ] for trIdx in range(2) ]

        tr1Dists = [ CalcDistance(triangles[0][pIdx1], triangles[0][pIdx2]) for pIdx1, pIdx2 in zip([0, 0, 1], [1, 2, 2]) ]
        tr2Dists = [ CalcDistance(triangles[1][pIdx1], triangles[1][pIdx2]) for pIdx1, pIdx2 in zip([0, 0, 1], [1, 2, 2]) ]

        rcSum = [0, 0]
        rotCenters = []
        for idx1 in range(3):
            for idx2 in range(idx1+1, 3):
                rotCenter = tr.FindRotationCenter([triangles[0][idx1], triangles[0][idx2]],
                                                  [triangles[1][idx1], triangles[1][idx2]])
                rotCenters.append(rotCenter)
                rcSum = list(np.array(rcSum) + np.array(rotCenter))

        rotCenterAvg = list(np.array(rcSum) / 3.0)
        rcShift = [ -int(rc) for rc in rotCenterAvg ]
        rcShift.reverse()
        img1 = imsup.CopyImage(self.display.image.prev)
        img2 = imsup.CopyImage(self.display.image)

        # ten padding trzeba jednak dodac ze wszystkich stron
        bufSz = max([abs(x) for x in rcShift])
        # dirV = 't-' if rcShift[1] > 0 else '-b'
        # dirH = 'l-' if rcShift[0] > 0 else '-r'
        dirs = 'tblr'
        # img1Pad = imsup.PadImage(img1, bufSz, 0.0, dirV+dirH)
        # img2Pad = imsup.PadImage(img2, bufSz, 0.0, dirV+dirH)
        img1Pad = imsup.PadImage(img1, bufSz, 0.0, dirs)
        img2Pad = imsup.PadImage(img2, bufSz, 0.0, dirs)

        img1Rc = cc.ShiftImage(img1Pad, rcShift)
        img2Rc = cc.ShiftImage(img2Pad, rcShift)
        # cropCoords = imsup.MakeSquareCoords(imsup.DetermineCropCoords(img1Rc.width, img1Rc.height, rcShift))
        # img1Rc = imsup.CropImageROICoords(img1Rc, cropCoords)
        # img2Rc = imsup.CropImageROICoords(img2Rc, cropCoords)
        img1Rc = imsup.CreateImageWithBufferFromImage(img1Rc)
        img2Rc = imsup.CreateImageWithBufferFromImage(img2Rc)
        imsup.SaveAmpImage(img1Rc, 'a.png')
        imsup.SaveAmpImage(img2Rc, 'b.png')

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
        print('---- Magnification ----')
        print([ 'mag{0} = {1:.2f}x\n'.format(idx + 1, mag) for idx, mag in zip(range(3), mags) ])
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
        # img2Rot = imsup.RotateImage(img2Rc, rotAngleAvg)
        padSz = (img2Rot.width - img1Rc.width) // 2
        img1RcPad = imsup.PadImage(img1Rc, padSz, 0.0, 'tblr')

        img1RcPad.MoveToCPU()
        img2Rot.MoveToCPU()
        img1RcPad.UpdateBuffer()
        img2Rot.UpdateBuffer()

        tmpImgList = imsup.ImageList([ self.display.image, img1RcPad, img2Rot ])
        tmpImgList.UpdateLinks()

        self.display.pointSets.append([])
        self.display.pointSets.append([])
        self.goToNextImage()

        print('Triangulation complete!')

    def warpImage(self, moreAccurate=False):
        currImg = self.display.image
        currIdx = self.display.image.numInSeries - 1
        realPoints1 = CalcRealCoordsForSetOfPoints(currImg.width, self.display.pointSets[currIdx-1])
        realPoints2 = CalcRealCoordsForSetOfPoints(currImg.width, self.display.pointSets[currIdx])
        userPoints1 = CalcTopLeftCoordsForSetOfPoints(currImg.width, realPoints1)
        userPoints2 = CalcTopLeftCoordsForSetOfPoints(currImg.width, realPoints2)

        if moreAccurate:
            nDiv = const.nDivForUnwarp
            fragDimSize = currImg.width // nDiv

            # points #1
            gridPoints1 = [ (b, a) for a in range(nDiv) for b in range(nDiv) ]
            gridPoints1 = [ [ gptx * fragDimSize for gptx in gpt ] for gpt in gridPoints1 ]

            for pt1 in userPoints1:
                closestNode = [ np.floor(x / fragDimSize) * fragDimSize for x in pt1 ]
                gridPoints1 = [ pt1 if gridNode == closestNode else gridNode for gridNode in gridPoints1 ]

            src = np.array(gridPoints1)

            # points #2
            gridPoints2 = [ (b, a) for a in range(nDiv) for b in range(nDiv) ]
            gridPoints2 = [ [ gptx * fragDimSize for gptx in gpt ] for gpt in gridPoints2 ]
            for pt2 in userPoints2:
                closestNode = [ np.floor(x / fragDimSize) * fragDimSize for x in pt2 ]
                gridPoints2 = [ pt2 if gridNode == closestNode else gridNode for gridNode in gridPoints2 ]

            dst = np.array(gridPoints2)

        else:
            src = np.array(userPoints1)
            dst = np.array(userPoints2)

        currImg.MoveToCPU()
        oldMin, oldMax = np.min(currImg.amPh.am), np.max(currImg.amPh.am)
        scaledArray = imsup.ScaleImage(currImg.amPh.am, -1.0, 1.0)

        tform3 = tf.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = tf.warp(scaledArray, tform3, output_shape=(currImg.height, currImg.width)).astype(np.float32)
        warpedScaledBack = imsup.ScaleImage(warped, oldMin, oldMax)

        imgWarped = imsup.ImageWithBuffer(currImg.height, currImg.width, num=currImg.numInSeries+1)
        imgWarped.amPh.am = np.copy(warpedScaledBack)
        imgWarped.UpdateBuffer()
        self.display.image.next = imgWarped
        imgWarped.prev = self.display.image
        self.display.pointSets.append([])
        self.goToNextImage()

    def rec_holo_no_ref(self):
        holo1 = self.display.image
        holo2 = self.display.image.prev

        rec_holo1 = holo.rec_holo_no_ref(holo1)
        if holo2 is not None:
            rec_holo2 = holo.rec_holo_no_ref(holo2)
            holo_img_list = imsup.ImageList([holo1, rec_holo2, rec_holo1])
        else:
            holo_img_list = imsup.ImageList([holo1, rec_holo1])

        holo_img_list.UpdateLinks()

        self.display.pointSets.append([])
        self.display.pointSets.append([])
        self.goToNextImage()

    def rec_holo_with_ref(self):
        pass

    def calc_phs_sum(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_sum = holo.calc_phase_sum(rec_holo1, rec_holo2)

        if rec_holo2.next is not None:
            tmp_img_list = imsup.ImageList([rec_holo2, phs_sum, rec_holo2.next])
        else:
            tmp_img_list = imsup.ImageList([rec_holo2, phs_sum])
        tmp_img_list.UpdateLinks()

        self.display.pointSets.insert(phs_sum.numInSeries-1, [])
        self.goToNextImage()

    def calc_phs_diff(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_diff = holo.calc_phase_diff(rec_holo1, rec_holo2)

        if rec_holo2.next is not None:
            tmp_img_list = imsup.ImageList([rec_holo2, phs_diff, rec_holo2.next])
        else:
            tmp_img_list = imsup.ImageList([rec_holo2, phs_diff])
        tmp_img_list.UpdateLinks()

        self.display.pointSets.insert(phs_diff.numInSeries - 1, [])
        self.goToNextImage()

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        imgData, pxDims = dm3.ReadDm3File(imgPath)
        imsup.Image.px_dim_default = pxDims[0]
        imgData = np.abs(imgData)
        img = imsup.ImageWithBuffer(imgData.shape[0], imgData.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'],
                                    num=imgNum, px_dim_sz=pxDims[0])
        img.LoadAmpData(np.sqrt(imgData).astype(np.float32))
        # ---
        # imsup.RemovePixelArtifacts(img, const.minPxThreshold, const.maxPxThreshold)
        imsup.RemovePixelArtifacts(img, 0.7, 1.3)
        img.UpdateBuffer()
        # ---
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

def CalcTopLeftCoordsForSetOfPoints(imgWidth, points):
    topLeftPoints = [ CalcTopLeftCoords(imgWidth, pt) for pt in points ]
    return topLeftPoints

# --------------------------------------------------------

def CalcRealCoords(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ int((dc - dispWidth // 2) * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoordsForSetOfPoints(imgWidth, points):
    realPoints = [ CalcRealCoords(imgWidth, pt) for pt in points ]
    return realPoints

# --------------------------------------------------------

def CalcRealTLCoordsForPaddedImage(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    padImgWidthReal = np.ceil(imgWidth / 512.0) * 512.0
    pad = (padImgWidthReal - imgWidth) / 2.0
    factor = padImgWidthReal / dispWidth
    # dispPad = pad / factor
    # realCoords = [ (dc - dispPad) * factor for dc in dispCoords ]
    realCoords = [ int(dc * factor - pad) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcDispCoords(dispWidth, imgWidth, realCoords):
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

def CalcRotAngle(p1, p2):
    z1 = np.complex(p1[0], p1[1])
    z2 = np.complex(p2[0], p2[1])
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)
    rotAngle = np.abs(imsup.Degrees(phi2 - phi1))
    if rotAngle > 180:
        rotAngle = 360 - rotAngle
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