import re
import sys
from os import path
from functools import partial
import numpy as np

from PyQt4 import QtGui, QtCore
import Dm3Reader3_New as dm3
import Constants as const
import ImageSupport as imsup
import CrossCorr as cc
import Transform as tr
import Holo as holo

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

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

    def setImage(self, dispAmp=True, logScale=False, color=False):
        self.image.MoveToCPU()

        if dispAmp:
            self.image.buffer = np.copy(self.image.amPh.am)
            if logScale:
                self.image.buffer = np.log(self.image.buffer)
        else:
            self.image.buffer = np.copy(self.image.amPh.ph)

        q_image = QtGui.QImage(imsup.ScaleImage(self.image.buffer, 0.0, 255.0).astype(np.uint8),
                               self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)

        if color:
            step = 3
            bcm = [ QtGui.qRgb(0, i, j) for i, j in zip(np.arange(0, 256, step), np.arange(255, -1, -step)) ]
            gcm = [ QtGui.qRgb(i, j, 0) for i, j in zip(np.arange(0, 256, step), np.arange(255, -1, -step)) ]
            rcm = [ QtGui.qRgb(j, 0, i) for i, j in zip(np.arange(0, 256, step), np.arange(255, -1, -step)) ]
            cm = bcm + gcm + rcm
            q_image.setColorTable(cm)

        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.setPixmap(pixmap)
        self.repaint()

    def changeImage(self, toNext=True, dispAmp=True, logScale=False, dispLabs=True, color=False):
        newImage = self.image.next if toNext else self.image.prev
        if newImage is None:
            return

        newImage.ReIm2AmPh()
        self.image = newImage

        if len(self.pointSets) < self.image.numInSeries:
            self.pointSets.append([])
        self.setImage(dispAmp, logScale, color)

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

class PlotWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.markedPoints = []
        self.markedPointsData = []
        self.canvas.mpl_connect('button_press_event', self.getXYDataOnClick)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, dataX, dataY, xlab='x', ylab='y'):
        self.figure.clear()
        self.markedPoints = []
        self.markedPointsData = []
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axis([ min(dataX)-0.5, max(dataX)+0.5, min(dataY)-0.5, max(dataY)+0.5 ])
        ax = self.figure.add_subplot(111)
        ax.plot(dataX, dataY, '.-')
        self.canvas.draw()

    def getXYDataOnClick(self, event):
        if len(self.markedPoints) == 2:
            for pt in self.markedPoints:
                pt.remove()
            self.markedPoints = []
            self.markedPointsData = []
        pt, = plt.plot(event.xdata, event.ydata, 'ro')
        self.markedPoints.append(pt)
        self.markedPointsData.append([event.xdata, event.ydata])

# --------------------------------------------------------

class TriangulateWidget(QtGui.QWidget):
    def __init__(self):
        super(TriangulateWidget, self).__init__()
        imagePath = QtGui.QFileDialog.getOpenFileName()
        image = LoadImageSeriesFromFirstFile(imagePath)
        self.display = LabelExt(self, image)
        self.plot_widget = PlotWidget()
        self.shift = [0, 0]
        self.rot_angle = 0
        self.initUI()

    def initUI(self):
        self.plot_widget.canvas.setFixedHeight(250)

        prevButton = QtGui.QPushButton('Prev', self)
        nextButton = QtGui.QPushButton('Next', self)

        prevButton.clicked.connect(self.goToPrevImage)
        nextButton.clicked.connect(self.goToNextImage)

        lswap_button = QtGui.QPushButton('L-Swap', self)
        rswap_button = QtGui.QPushButton('R-Swap', self)

        lswap_button.clicked.connect(self.swap_left)
        rswap_button.clicked.connect(self.swap_right)

        flipButton = QtGui.QPushButton('Flip', self)
        zoomButton = QtGui.QPushButton('Zoom', self)

        flipButton.clicked.connect(self.flip_image_h)
        zoomButton.clicked.connect(self.zoom_n_fragments)

        n_to_zoom_label = QtGui.QLabel('How many?', self)
        self.n_to_zoom_input = QtGui.QLineEdit('1', self)

        exportButton = QtGui.QPushButton('Export', self)
        deleteButton = QtGui.QPushButton('Delete', self)
        clearButton = QtGui.QPushButton('Clear', self)

        exportButton.clicked.connect(self.export_image)
        deleteButton.clicked.connect(self.deleteImage)
        clearButton.clicked.connect(self.clearImage)

        self.shift_radio_button = QtGui.QRadioButton('Shift', self)
        self.rot_radio_button = QtGui.QRadioButton('Rot', self)
        self.shift_radio_button.setChecked(True)

        shift_rot_group = QtGui.QButtonGroup(self)
        shift_rot_group.addButton(self.shift_radio_button)
        shift_rot_group.addButton(self.rot_radio_button)

        alignButton = QtGui.QPushButton('Align', self)
        reshift_button = QtGui.QPushButton('Re-Shift', self)
        warpButton = QtGui.QPushButton('Warp', self)
        rerot_button = QtGui.QPushButton('Re-Rot', self)

        alignButton.clicked.connect(self.align_images)
        reshift_button.clicked.connect(self.reshift)
        warpButton.clicked.connect(partial(self.warp_image, False))
        rerot_button.clicked.connect(self.rerotate)

        fname_label = QtGui.QLabel('File name', self)
        self.fname_input = QtGui.QLineEdit('img', self)

        holo_no_ref_1_button = QtGui.QPushButton('FFT', self)
        holo_no_ref_2_button = QtGui.QPushButton('Holo', self)
        holo_with_ref_2_button = QtGui.QPushButton('Holo+Ref', self)
        holo_no_ref_3_button = QtGui.QPushButton('IFFT', self)

        holo_no_ref_1_button.clicked.connect(self.rec_holo_no_ref_1)
        holo_no_ref_2_button.clicked.connect(self.rec_holo_no_ref_2)
        holo_with_ref_2_button.clicked.connect(self.rec_holo_with_ref_2)
        holo_no_ref_3_button.clicked.connect(self.rec_holo_no_ref_3)

        self.show_lines_checkbox = QtGui.QCheckBox('Show lines', self)
        self.show_lines_checkbox.setChecked(True)
        self.show_lines_checkbox.toggled.connect(self.toggle_lines)

        self.show_labels_checkbox = QtGui.QCheckBox('Show labels', self)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.toggle_labels)

        self.log_scale_checkbox = QtGui.QCheckBox('Log scale', self)
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.toggled.connect(self.update_display)

        self.phs_unwrap_checkbox = QtGui.QCheckBox('Unwrap phase', self)
        self.phs_unwrap_checkbox.setChecked(False)

        phs_uw_ok_button = QtGui.QPushButton('OK', self)
        phs_uw_ok_button.clicked.connect(self.unwrap_img_phase)

        self.amp_radio_button = QtGui.QRadioButton('Amplitude', self)
        self.phs_radio_button = QtGui.QRadioButton('Phase', self)
        self.amp_radio_button.setChecked(True)

        self.amp_radio_button.toggled.connect(self.update_display)
        self.phs_radio_button.toggled.connect(self.update_display)

        amp_phs_group = QtGui.QButtonGroup(self)
        amp_phs_group.addButton(self.amp_radio_button)
        amp_phs_group.addButton(self.phs_radio_button)

        aperture_label = QtGui.QLabel('Aperture [px]', self)
        self.aperture_input = QtGui.QLineEdit(str(const.aperture), self)

        hann_win_label = QtGui.QLabel('Hann window [px]', self)
        self.hann_win_input = QtGui.QLineEdit(str(const.hann_win), self)

        sum_button = QtGui.QPushButton('Sum', self)
        diff_button = QtGui.QPushButton('Diff', self)

        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)

        self.amp_factor_input = QtGui.QLineEdit('2.0', self)

        amplify_button = QtGui.QPushButton('Amplify', self)
        amplify_button.clicked.connect(self.amplify_phase)

        int_width_label = QtGui.QLabel('Profile width', self)
        self.int_width_input = QtGui.QLineEdit('1', self)

        plot_button = QtGui.QPushButton('Plot profile', self)
        plot_button.clicked.connect(self.plot_profile)

        sample_thick_label = QtGui.QLabel('Sample thickness [nm]', self)
        self.sample_thick_input = QtGui.QLineEdit('30', self)

        calc_B_button = QtGui.QPushButton('Calculate B', self)
        calc_grad_button = QtGui.QPushButton('Calculate gradient', self)

        calc_B_button.clicked.connect(self.calc_magnetic_field)
        calc_grad_button.clicked.connect(self.calc_phase_gradient)

        self.gray_radio_button = QtGui.QRadioButton('Grayscale', self)
        self.color_radio_button = QtGui.QRadioButton('Color', self)
        self.gray_radio_button.setChecked(True)

        self.gray_radio_button.toggled.connect(self.update_display)
        self.color_radio_button.toggled.connect(self.update_display)

        color_group = QtGui.QButtonGroup(self)
        color_group.addButton(self.gray_radio_button)
        color_group.addButton(self.color_radio_button)

        grid_nav = QtGui.QGridLayout()
        grid_nav.addWidget(prevButton, 1, 1)
        grid_nav.addWidget(nextButton, 1, 2)
        grid_nav.addWidget(lswap_button, 2, 1)
        grid_nav.addWidget(rswap_button, 2, 2)
        grid_nav.addWidget(flipButton, 3, 1)
        grid_nav.addWidget(clearButton, 3, 2)
        grid_nav.addWidget(exportButton, 4, 1)
        grid_nav.addWidget(deleteButton, 4, 2)

        grid_disp = QtGui.QGridLayout()
        grid_disp.addWidget(n_to_zoom_label, 1, 2)
        grid_disp.addWidget(self.n_to_zoom_input, 2, 2)
        grid_disp.addWidget(zoomButton, 2, 1)
        grid_disp.addWidget(self.show_lines_checkbox, 3, 1)
        grid_disp.addWidget(self.show_labels_checkbox, 4, 1)
        grid_disp.addWidget(self.log_scale_checkbox, 5, 1)
        grid_disp.addWidget(self.amp_radio_button, 3, 2)
        grid_disp.addWidget(self.phs_radio_button, 4, 2)
        grid_disp.addWidget(self.phs_unwrap_checkbox, 5, 2)
        grid_disp.addWidget(phs_uw_ok_button, 5, 3)
        self.n_to_zoom_input.setFixedWidth(120)

        grid_align = QtGui.QGridLayout()
        grid_align.addWidget(self.shift_radio_button, 1, 1)
        grid_align.addWidget(self.rot_radio_button, 2, 1)
        grid_align.addWidget(alignButton, 1, 2)
        grid_align.addWidget(warpButton, 2, 2)
        grid_align.addWidget(reshift_button, 1, 3)
        grid_align.addWidget(rerot_button, 2, 3)

        grid_holo = QtGui.QGridLayout()
        grid_holo.addWidget(fname_label, 1, 1)
        grid_holo.addWidget(self.fname_input, 2, 1)
        grid_holo.addWidget(aperture_label, 1, 2)
        grid_holo.addWidget(self.aperture_input, 2, 2)
        grid_holo.addWidget(hann_win_label, 1, 3)
        grid_holo.addWidget(self.hann_win_input, 2, 3)
        grid_holo.addWidget(holo_no_ref_1_button, 3, 1)
        grid_holo.addWidget(holo_no_ref_2_button, 3, 2)
        grid_holo.addWidget(holo_with_ref_2_button, 3, 3)
        grid_holo.addWidget(holo_no_ref_3_button, 3, 4)
        grid_holo.addWidget(sum_button, 4, 1)
        grid_holo.addWidget(diff_button, 4, 2)
        grid_holo.addWidget(self.amp_factor_input, 4, 3)
        grid_holo.addWidget(amplify_button, 4, 4)

        grid_plot = QtGui.QGridLayout()
        grid_plot.addWidget(int_width_label, 1, 1)
        grid_plot.addWidget(self.int_width_input, 2, 1)
        grid_plot.addWidget(sample_thick_label, 1, 2)
        grid_plot.addWidget(self.sample_thick_input, 2, 2)
        grid_plot.addWidget(self.gray_radio_button, 1, 3)
        grid_plot.addWidget(self.color_radio_button, 2, 3)
        grid_plot.addWidget(plot_button, 3, 1)
        grid_plot.addWidget(calc_B_button, 3, 2)
        grid_plot.addWidget(calc_grad_button, 3, 3)

        self.int_width_input.setFixedWidth(150)
        self.sample_thick_input.setFixedWidth(150)

        vbox_panel = QtGui.QVBoxLayout()
        vbox_panel.addLayout(grid_nav)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_disp)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_align)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_holo)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_plot)

        hbox_panel = QtGui.QHBoxLayout()
        hbox_panel.addWidget(self.display)
        hbox_panel.addLayout(vbox_panel)

        vbox_main = QtGui.QVBoxLayout()
        vbox_main.addLayout(hbox_panel)
        vbox_main.addWidget(self.plot_widget)
        self.setLayout(vbox_main)

        self.move(250, 5)
        self.setWindowTitle('Triangulation window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

    def goToPrevImage(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        self.display.changeImage(toNext=False, dispAmp=is_amp_checked, logScale=is_log_scale_checked, dispLabs=is_show_labels_checked, color=is_color_checked)

    def goToNextImage(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        self.display.changeImage(toNext=True, dispAmp=is_amp_checked, logScale=is_log_scale_checked, dispLabs=is_show_labels_checked, color=is_color_checked)

    def flip_image_h(self):
        imsup.flip_image_h(self.display.image)
        self.display.setImage()

    # def cropFragment(self):
    #     [pt1, pt2] = self.display.pointSets[self.display.image.numInSeries - 1][:2]
    #     dispCropCoords = pt1 + pt2
    #     realCropCoords = imsup.MakeSquareCoords(
    #         CalcRealTLCoordsForPaddedImage(self.display.image.width, dispCropCoords))
    #
    #     imgCurr = self.display.image
    #     imgCurrCrop = imsup.CropImageROICoords(imgCurr, realCropCoords)
    #     imgCurrCrop = imsup.CreateImageWithBufferFromImage(imgCurrCrop)
    #     imgCurrCrop.MoveToCPU()
    #
    #     curr_num = self.display.image.numInSeries
    #     tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
    #
    #     if imgCurr.prev is not None:
    #         imgPrev = imgCurr.prev
    #         imgPrevCrop = imsup.CropImageROICoords(imgPrev, realCropCoords)
    #         imgPrevCrop = imsup.CreateImageWithBufferFromImage(imgPrevCrop)
    #         imgPrevCrop.MoveToCPU()
    #         tmp_img_list.insert(1, imgPrevCrop)
    #         tmp_img_list.insert(2, imgCurrCrop)
    #         self.display.pointSets.insert(curr_num, [])
    #         self.display.pointSets.insert(curr_num+1, [])
    #     else:
    #         tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
    #         tmp_img_list.insert(1, imgCurrCrop)
    #         self.display.pointSets.insert(curr_num, [])
    #
    #     tmp_img_list.UpdateLinks()
    #     self.goToNextImage()

    def export_image(self):
        curr_num = self.display.image.numInSeries
        fname = self.fname_input.text()
        is_amp_checked = self.amp_radio_button.isChecked()

        if fname == '':
            fname = 'amp{0}'.format(curr_num) if is_amp_checked else 'phs{0}'.format(curr_num)

        if is_amp_checked:
            imsup.SaveAmpImage(self.display.image, '{0}.png'.format(fname))
        else:
            imsup.SavePhaseImage(self.display.image, '{0}.png'.format(fname))
        print('Saved image as "{0}.png"'.format(fname))

    def deleteImage(self):
        curr_img = self.display.image
        if curr_img.prev is None and curr_img.next is None:
            return

        curr_idx = curr_img.numInSeries - 1
        first_img = imsup.GetFirstImage(curr_img)
        tmp_img_list = imsup.CreateImageListFromFirstImage(first_img)

        if curr_img.prev is not None:
            curr_img.prev.next = None
            self.goToPrevImage()
        else:
            curr_img.next.prev = None
            self.goToNextImage()
            if curr_idx == 0:
                self.display.image.numInSeries = 1

        del tmp_img_list[curr_idx]
        del self.display.pointSets[curr_idx]
        tmp_img_list.UpdateLinks()
        del curr_img

        # curr_img = self.display.image
        # if curr_img.prev is None and curr_img.next is None:
        #     return
        #
        # del self.display.pointSets[curr_img.numInSeries - 1]
        #
        # if curr_img.prev is not None:
        #     curr_img.prev.next = curr_img.next
        #
        # if curr_img.next is not None:
        #     curr_img.next.prev = curr_img.prev
        #     tmp = curr_img.next
        #     while tmp is not None:
        #         tmp.numInSeries = tmp.prev.numInSeries + 1 if tmp.prev is not None else 1
        #         tmp = tmp.next
        #
        # if curr_img.prev is not None:
        #     self.goToPrevImage()
        # else:
        #     self.goToNextImage()
        #
        # del curr_img

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
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        self.display.setImage(dispAmp=is_amp_checked, logScale=is_log_scale_checked, color=is_color_checked)

    def unwrap_img_phase(self):
        curr_img = self.display.image
        is_phs_unwrap_checked = self.phs_unwrap_checkbox.isChecked()

        if is_phs_unwrap_checked:
            new_phs = tr.unwrap_phase(curr_img.amPh.ph)
        else:
            uw_min = np.min(curr_img.amPh.ph)
            if uw_min > 0:
                uw_min = 0
            new_phs = (curr_img.amPh.ph - uw_min) % (2 * np.pi) - np.pi

        curr_img.amPh.ph = np.copy(new_phs)
        self.update_display()

    # def zoom_two_fragments(self):
    #     curr_idx = self.display.image.numInSeries - 1
    #     if len(self.display.pointSets[curr_idx]) < 2:
    #         return
    #
    #     curr_img = self.display.image
    #     [pt1, pt2] = self.display.pointSets[curr_idx][:2]
    #     disp_crop_coords = pt1 + pt2
    #     real_crop_coords = imsup.MakeSquareCoords(CalcRealTLCoords(curr_img.width, disp_crop_coords))
    #
    #     if curr_img.prev is not None:
    #         zoom_fragment(curr_img.prev, real_crop_coords)
    #     zoom_fragment(curr_img, real_crop_coords)

    def zoom_n_fragments(self):
        curr_idx = self.display.image.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) < 2:
            return

        curr_img = self.display.image
        [pt1, pt2] = self.display.pointSets[curr_idx][:2]
        disp_crop_coords = pt1 + pt2
        real_crop_coords = imsup.MakeSquareCoords(CalcRealTLCoords(curr_img.width, disp_crop_coords))

        n_to_zoom = np.int(self.n_to_zoom_input.text())
        img_list = imsup.CreateImageListFromFirstImage(curr_img)
        img_list2 = img_list[:n_to_zoom]
        print(len(img_list2))
        idx1 = curr_img.numInSeries + n_to_zoom
        idx2 = idx1 + n_to_zoom
        for img, n in zip(img_list2, range(idx1, idx2)):
            frag = zoom_fragment(img, real_crop_coords)
            img_list.insert(n, frag)

        img_list.UpdateLinks()

    def clearImage(self):
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
        self.display.pointSets[self.display.image.numInSeries - 1][:] = []
        self.display.repaint()

    def align_images(self):
        if self.shift_radio_button.isChecked():
            self.align_shift()
        else:
            self.align_rot()

    def align_rot(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        img_width = curr_img.width

        points1 = self.display.pointSets[curr_idx-1]
        points2 = self.display.pointSets[curr_idx]
        n_points1 = len(points1)
        n_points2 = len(points2)

        if n_points1 != n_points2:
            print('Mark the same number of points on both images!')
            return

        poly1 = [ CalcRealCoords(img_width, pt1) for pt1 in points1 ]
        poly2 = [ CalcRealCoords(img_width, pt2) for pt2 in points2 ]

        poly1_dists = []
        poly2_dists = []
        for i in range(len(poly1)):
            for j in range(i+1, len(poly1)):
                poly1_dists.append(CalcDistance(poly1[i], poly1[j]))
                poly2_dists.append(CalcDistance(poly2[i], poly2[j]))

        rcSum = [0, 0]
        rotCenters = []
        for idx1 in range(len(poly1)):
            for idx2 in range(idx1+1, len(poly1)):
                rotCenter = tr.FindRotationCenter([poly1[idx1], poly1[idx2]],
                                                  [poly2[idx1], poly2[idx2]])
                rotCenters.append(rotCenter)
                print(rotCenter)
                rcSum = list(np.array(rcSum) + np.array(rotCenter))

        rotCenterAvg = list(np.array(rcSum) / n_points1)
        rcShift = [ -int(rc) for rc in rotCenterAvg ]
        rcShift.reverse()
        img1 = imsup.copy_am_ph_image(self.display.image.prev)
        img2 = imsup.copy_am_ph_image(self.display.image)

        bufSz = max([abs(x) for x in rcShift])
        dirs = 'tblr'
        img1Pad = imsup.PadImage(img1, bufSz, 0.0, dirs)
        img2Pad = imsup.PadImage(img2, bufSz, 0.0, dirs)

        img1Rc = cc.shift_am_ph_image(img1Pad, rcShift)
        img2Rc = cc.shift_am_ph_image(img2Pad, rcShift)

        img1Rc = imsup.create_imgbuf_from_img(img1Rc)
        img2Rc = imsup.create_imgbuf_from_img(img2Rc)

        rotAngles = []
        for idx, p1, p2 in zip(range(n_points1), poly1, poly2):
            p1New = CalcNewCoords(p1, rotCenterAvg)
            p2New = CalcNewCoords(p2, rotCenterAvg)
            poly1[idx] = p1New
            poly2[idx] = p2New
            rotAngles.append(CalcRotAngle(p1New, p2New))

        rotAngleAvg = np.average(rotAngles)

        mags = [ dist1 / dist2 for dist1, dist2 in zip(poly1_dists, poly2_dists) ]
        magAvg = np.average(mags)

        print('---- Magnification ----')
        print([ 'mag{0} = {1:.2f}x\n'.format(idx + 1, mag) for idx, mag in zip(range(len(mags)), mags) ])
        print('---- Rotation ----')
        print([ 'phi{0} = {1:.0f} deg\n'.format(idx + 1, angle) for idx, angle in zip(range(len(rotAngles)), rotAngles) ])
        # print('---- Shifts ----')
        # print([ 'dxy{0} = ({1:.1f}, {2:.1f}) px\n'.format(idx + 1, sh[0], sh[1]) for idx, sh in zip(range(3), shifts) ])
        # print('------------------')
        # print('Average magnification = {0:.2f}x'.format(magAvg))
        print('Average rotation = {0:.2f} deg'.format(rotAngleAvg))
        # print('Average shift = ({0:.0f}, {1:.0f}) px'.format(shiftAvg[0], shiftAvg[1]))

        self.shift = rcShift
        self.rot_angle = rotAngleAvg

        # img2Mag = tr.RescaleImageSki(img2Rc, magAvg)
        img2Rot = tr.RotateImageSki(img2Rc, rotAngleAvg)
        padSz = (img2Rot.width - img1Rc.width) // 2
        img1RcPad = imsup.PadImage(img1Rc, padSz, 0.0, 'tblr')

        img1RcPad.MoveToCPU()
        img2Rot.MoveToCPU()
        img1RcPad.UpdateBuffer()
        img2Rot.UpdateBuffer()

        mag_factor = curr_img.width / img1RcPad.width
        img1_mag = tr.RescaleImageSki(img1RcPad, mag_factor)
        img2_mag = tr.RescaleImageSki(img2Rot, mag_factor)

        curr_num = self.display.image.numInSeries
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.insert(1, img1_mag)
        tmp_img_list.insert(2, img2_mag)
        tmp_img_list.UpdateLinks()
        self.display.pointSets.insert(curr_num, [])
        self.display.pointSets.insert(curr_num+1, [])
        self.goToNextImage()

        print('Triangulation complete!')

    def align_shift(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        img_width = curr_img.width

        points1 = self.display.pointSets[curr_idx - 1]
        points2 = self.display.pointSets[curr_idx]
        n_points1 = len(points1)
        n_points2 = len(points2)

        if n_points1 != n_points2:
            print('Mark the same number of points on both images!')
            return

        set1 = [CalcRealCoords(img_width, pt1) for pt1 in points1]
        set2 = [CalcRealCoords(img_width, pt2) for pt2 in points2]

        shift_sum = np.zeros(2, dtype=np.int32)
        for pt1, pt2 in zip(set1, set2):
            shift = np.array(pt1) - np.array(pt2)
            shift_sum += shift

        shift_avg = list(shift_sum // n_points1)
        self.shift = shift_avg

        shifted_img2 = cc.shift_am_ph_image(curr_img, shift_avg)
        shifted_img2 = imsup.create_imgbuf_from_img(shifted_img2)
        self.insert_img_after_curr(shifted_img2)

    def reshift(self):
        curr_img = self.display.image
        shift = self.shift

        if self.shift_radio_button.isChecked():
            shifted_img = cc.shift_am_ph_image(curr_img, shift)
            shifted_img = imsup.create_imgbuf_from_img(shifted_img)
            self.insert_img_after_curr(shifted_img)
        else:
            bufSz = max([abs(x) for x in shift])
            dirs = 'tblr'
            padded_img = imsup.PadImage(curr_img, bufSz, 0.0, dirs)
            shifted_img = cc.shift_am_ph_image(padded_img, shift)
            shifted_img = imsup.create_imgbuf_from_img(shifted_img)

            resc_factor = curr_img.width / padded_img.width
            resc_img = tr.RescaleImageSki(shifted_img, resc_factor)
            self.insert_img_after_curr(resc_img)

    def rerotate(self):
        curr_img = self.display.image
        rot_angle = self.rot_angle
        rotated_img = tr.RotateImageSki(curr_img, rot_angle)
        self.insert_img_after_curr(rotated_img)

    def warp_image(self, more_accurate=False):
        curr_img = self.display.image
        curr_idx = self.display.image.numInSeries - 1
        real_points1 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx-1])
        real_points2 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx])
        user_points1 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points1)
        user_points2 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points2)

        if more_accurate:
            n_div = const.nDivForUnwarp
            frag_dim_size = curr_img.width // n_div

            # points #1
            grid_points1 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points1 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points1 ]

            for pt1 in user_points1:
                closest_node = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt1 ]
                grid_points1 = [ pt1 if grid_node == closest_node else grid_node for grid_node in grid_points1 ]

            src = np.array(grid_points1)

            # points #2
            grid_points2 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points2 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points2 ]
            for pt2 in user_points2:
                closestNode = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt2 ]
                grid_points2 = [ pt2 if gridNode == closestNode else gridNode for gridNode in grid_points2 ]

            dst = np.array(grid_points2)

        else:
            src = np.array(user_points1)
            dst = np.array(user_points2)

        img_warp = tr.WarpImage(curr_img, src, dst)

        curr_num = self.display.image.numInSeries
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.insert(1, img_warp)
        tmp_img_list.UpdateLinks()
        self.display.pointSets.insert(curr_num, [])
        self.goToNextImage()

    def insert_img_after_curr(self, img):
        curr_num = self.display.image.numInSeries
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.insert(1, img)
        self.display.pointSets.insert(curr_num, [])
        tmp_img_list.UpdateLinks()
        self.goToNextImage()

    def rec_holo_no_ref_1(self):
        holo_img = self.display.image
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        self.insert_img_after_curr(holo_fft)

    def rec_holo_no_ref_2(self):
        holo_fft = self.display.image
        [pt1, pt2] = self.display.pointSets[holo_fft.numInSeries-1][:2]
        dpts = pt1 + pt2
        rpts = CalcRealTLCoords(holo_fft.width, dpts)
        rpt1 = rpts[:2]     # konwencja x, y
        rpt2 = rpts[2:]

        sband = np.copy(holo_fft.amPh.am[rpt1[1]:rpt2[1], rpt1[0]:rpt2[0]])     # konwencja y, x
        sband_xy = holo.find_img_max(sband)     # konwencja y, x
        sband_xy.reverse()

        sband_xy = [ px + sx for px, sx in zip(rpt1, sband_xy) ]    # konwencja x, y

        mid = holo_fft.width // 2
        shift = [ mid - sband_xy[1], mid - sband_xy[0] ]    # konwencja x, y

        aperture = int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        sband_img_ap = holo.rec_holo_no_ref_2(holo_fft, shift, ap_sz=aperture, N_hann=hann_window)
        self.log_scale_checkbox.setChecked(False)
        self.insert_img_after_curr(sband_img_ap)

    def rec_holo_with_ref_2(self):
        ref_fft = self.display.image
        [pt1, pt2] = self.display.pointSets[ref_fft.numInSeries - 1][:2]
        dpts = pt1 + pt2
        rpts = CalcRealTLCoords(ref_fft.width, dpts)
        rpt1 = rpts[:2]  # konwencja x, y
        rpt2 = rpts[2:]

        sband = np.copy(ref_fft.amPh.am[rpt1[1]:rpt2[1], rpt1[0]:rpt2[0]])  # konwencja y, x
        sband_xy = holo.find_img_max(sband)  # konwencja y, x
        sband_xy.reverse()

        sband_xy = [px + sx for px, sx in zip(rpt1, sband_xy)]  # konwencja x, y

        mid = ref_fft.width // 2
        shift = [mid - sband_xy[1], mid - sband_xy[0]]  # konwencja x, y

        aperture = int(self.aperture_input.text())
        hann_window = int(self.hann_win_input.text())

        ref_sband_ap = holo.rec_holo_no_ref_2(ref_fft, shift, ap_sz=aperture, N_hann=hann_window)

        holo_img = self.display.image.next
        holo_fft = holo.rec_holo_no_ref_1(holo_img)
        holo_sband_ap = holo.rec_holo_no_ref_2(holo_fft, shift, ap_sz=aperture, N_hann=hann_window)

        self.log_scale_checkbox.setChecked(False)
        self.insert_img_after_curr(ref_sband_ap)
        self.insert_img_after_curr(holo_sband_ap)

    def rec_holo_no_ref_3(self):
        sband_img = self.display.image
        rec_holo = holo.rec_holo_no_ref_3(sband_img)
        self.insert_img_after_curr(rec_holo)

    # def rec_holo_no_ref(self):
    #     holo1 = self.display.image.prev
    #     holo2 = self.display.image
    #
    #     rec_holo2 = holo.rec_holo_no_ref(holo2)
    #
    #     curr_num = self.display.image.numInSeries
    #     tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
    #
    #     if holo1 is not None:
    #         rec_holo1 = holo.rec_holo_no_ref(holo1)
    #         tmp_img_list.insert(1, rec_holo1)
    #         tmp_img_list.insert(2, rec_holo2)
    #         self.display.pointSets.insert(curr_num, [])
    #         self.display.pointSets.insert(curr_num+1, [])
    #     else:
    #         tmp_img_list.insert(1, rec_holo2)
    #         self.display.pointSets.insert(curr_num, [])
    #
    #     tmp_img_list.UpdateLinks()
    #     self.goToNextImage()

    def rec_holo_with_ref(self):
        pass

    def calc_phs_sum(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_sum = holo.calc_phase_sum(rec_holo1, rec_holo2)
        self.insert_img_after_curr(phs_sum)

    def calc_phs_diff(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_diff = holo.calc_phase_diff(rec_holo1, rec_holo2)
        self.insert_img_after_curr(phs_diff)

    def amplify_phase(self):
        curr_img = self.display.image
        amp_factor = float(self.amp_factor_input.text())

        phs_amplified = imsup.copy_am_ph_image(curr_img)
        phs_amplified.amPh.ph = np.cos(amp_factor * curr_img.amPh.ph)
        self.insert_img_after_curr(phs_amplified)

    def swap_left(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            return
        curr_idx = curr_img.numInSeries - 1

        first_img = imsup.GetFirstImage(curr_img)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        imgs[curr_idx-1], imgs[curr_idx] = imgs[curr_idx], imgs[curr_idx-1]

        imgs[0].prev = None
        imgs[len(imgs)-1].next = None
        imgs[curr_idx-1].numInSeries = imgs[curr_idx].numInSeries
        imgs.UpdateLinks()

        ps = self.display.pointSets
        ps[curr_idx-1], ps[curr_idx] = ps[curr_idx], ps[curr_idx-1]
        self.goToNextImage()

    def swap_right(self):
        curr_img = self.display.image
        if curr_img.next is None:
            return
        curr_idx = curr_img.numInSeries - 1

        first_img = imsup.GetFirstImage(curr_img)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        imgs[curr_idx], imgs[curr_idx+1] = imgs[curr_idx+1], imgs[curr_idx]

        imgs[0].prev = None
        imgs[len(imgs)-1].next = None
        imgs[curr_idx].numInSeries = imgs[curr_idx+1].numInSeries
        imgs.UpdateLinks()

        ps = self.display.pointSets
        ps[curr_idx], ps[curr_idx+1] = ps[curr_idx+1], ps[curr_idx]
        self.goToPrevImage()

    def plot_profile(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        px_sz = curr_img.px_dim
        print(px_sz)
        points = self.display.pointSets[curr_idx][:2]
        points = np.array([ CalcRealCoords(curr_img.width, pt) for pt in points ])

        # find rotation center (center of the line)
        rot_center = np.average(points, 0).astype(np.int32)
        print('rotCenter = {0}'.format(rot_center))

        # find direction (angle) of the line
        dir_info = FindDirectionAngles(points[0], points[1])
        dir_angle = imsup.Degrees(dir_info[0])
        proj_dir = dir_info[2]
        print('dir angle = {0:.2f} deg'.format(dir_angle))

        # shift image by -center
        shift_to_rot_center = list(-rot_center)
        shift_to_rot_center.reverse()
        img_shifted = cc.shift_am_ph_image(curr_img, shift_to_rot_center)

        # rotate image by angle
        img_rot = tr.RotateImageSki(img_shifted, dir_angle)

        # crop fragment (height = distance between two points)
        pt_diffs = points[0] - points[1]
        frag_dim1 = int(np.sqrt(pt_diffs[0] ** 2 + pt_diffs[1] ** 2))
        frag_dim2 = int(self.int_width_input.text())

        if proj_dir == 0:
            frag_width, frag_height = frag_dim1, frag_dim2
        else:
            frag_width, frag_height = frag_dim2, frag_dim1

        frag_coords = imsup.DetermineCropCoordsForNewDims(img_rot.width, img_rot.height, frag_width, frag_height)
        print('Frag dims = {0}, {1}'.format(frag_width, frag_height))
        print('Frag coords = {0}'.format(frag_coords))
        img_cropped = imsup.crop_am_ph_roi_cpu(img_rot, frag_coords)

        # calculate projection of intensity
        if self.amp_radio_button.isChecked():
            int_matrix = np.copy(img_cropped.amPh.am)
        else:
            ph_min = np.min(img_cropped.amPh.ph)
            ph_fix = -ph_min if ph_min < 0 else 0
            img_cropped.amPh.ph += ph_fix
            int_matrix = np.copy(img_cropped.amPh.ph)
        int_profile = np.sum(int_matrix, proj_dir)  # 0 - horizontal projection, 1 - vertical projection
        dists = np.arange(0, int_profile.shape[0], 1) * px_sz
        dists *= 1e9

        self.plot_widget.plot(dists, int_profile, 'Distance [nm]', 'Intensity [a.u.]')

    def calc_phase_gradient(self):
        curr_img = self.display.image
        dx_img = imsup.copy_am_ph_image(curr_img)
        dy_img = imsup.copy_am_ph_image(curr_img)
        grad_img = imsup.copy_am_ph_image(curr_img)
        dx, dy = np.gradient(curr_img.amPh.ph)
        dr = np.sqrt(dx * dx + dy * dy)
        dphi = np.arctan2(dy, dx)
        dx_img.amPh.ph = np.copy(dx)
        dy_img.amPh.ph = np.copy(dy)
        grad_img.amPh.am = np.copy(dr)
        grad_img.amPh.ph = np.copy(dphi)
        self.insert_img_after_curr(dx_img)
        self.insert_img_after_curr(dy_img)
        self.insert_img_after_curr(grad_img)

    def calc_magnetic_field(self):
        pt1, pt2 = self.plot_widget.markedPointsData
        d_dist = np.abs(pt1[0] - pt2[0]) * 1e-9
        d_phase = np.abs(pt1[1] - pt2[1])
        sample_thickness = float(self.sample_thick_input.text()) * 1e-9
        B_in_plane = (const.planck_const / sample_thickness) * (d_phase / d_dist)
        print('B = {0:.2f} T'.format(B_in_plane))

    # def plot_profile(self):
    #     curr_img = self.display.image
    #     curr_idx = curr_img.numInSeries - 1
    #     px_sz = curr_img.px_dim
    #     p1, p2 = self.display.pointSets[curr_idx][:2]
    #     p1 = CalcRealCoords(curr_img.width, p1)
    #     p2 = CalcRealCoords(curr_img.width, p2)
    #
    #     x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
    #     y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
    #     x_dist = x2 - x1
    #     y_dist = y2 - y1
    #
    #     if x_dist > y_dist:
    #         x_range = list(range(x1, x2))
    #         a_coeff = (p2[1] - p1[1]) / (p2[0] - p1[0])
    #         b_coeff = p1[1] - a_coeff * p1[0]
    #         y_range = [ int(a_coeff * x + b_coeff) for x in x_range ]
    #     else:
    #         y_range = list(range(y1, y2))
    #         a_coeff = (p2[0] - p1[0]) / (p2[1] - p1[1])
    #         b_coeff = p1[0] - a_coeff * p1[1]
    #         x_range = [ int(a_coeff * y + b_coeff) for y in y_range ]
    #
    #     print(len(x_range), len(y_range))
    #     profile = curr_img.amPh.am[x_range, y_range]
    #     dists = np.arange(0, profile.shape[0], 1) * px_sz
    #     dists *= 1e9
    #     self.plot_widget.plot(dists, profile, 'Distance [nm]', 'Intensity [a.u.]')

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

def zoom_fragment(img, coords):
    crop_img = imsup.crop_am_ph_roi(img, coords)
    crop_img = imsup.create_imgbuf_from_img(crop_img)
    crop_img.MoveToCPU()

    orig_width = img.width
    crop_width = np.abs(coords[2] - coords[0])
    zoom_factor = orig_width / crop_width
    zoom_img = tr.RescaleImageSki(crop_img, zoom_factor)
    zoom_img.px_dim *= zoom_factor
    # self.insert_img_after_curr(zoom_img)
    return zoom_img

# --------------------------------------------------------

def FindDirectionAngles(p1, p2):
    lpt = p1[:] if p1[0] < p2[0] else p2[:]     # left point
    rpt = p1[:] if p1[0] > p2[0] else p2[:]     # right point
    dx = np.abs(rpt[0] - lpt[0])
    dy = np.abs(rpt[1] - lpt[1])
    sign = 1 if rpt[1] < lpt[1] else -1
    projDir = 1         # projection on y axis
    if dx > dy:
        sign *= -1
        projDir = 0     # projection on x axis
    diff1 = dx if dx < dy else dy
    diff2 = dx if dx > dy else dy
    ang1 = np.arctan2(diff1, diff2)
    ang2 = np.pi / 2 - ang1
    ang1 *= sign
    ang2 *= (-sign)
    return ang1, ang2, projDir

# --------------------------------------------------------

def CalcTopLeftCoords(imgWidth, midCoords):
    topLeftCoords = [ mc + imgWidth // 2 for mc in midCoords ]
    return topLeftCoords

# --------------------------------------------------------

def CalcTopLeftCoordsForSetOfPoints(imgWidth, points):
    topLeftPoints = [ CalcTopLeftCoords(imgWidth, pt) for pt in points ]
    return topLeftPoints

# --------------------------------------------------------

def CalcRealTLCoords(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ int(dc * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealTLCoordsForSetOfPoints(imgWidth, points):
    realCoords = [ CalcRealTLCoords(imgWidth, pt) for pt in points ]
    return realCoords

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
    # rotAngle = np.abs(imsup.Degrees(phi2 - phi1))
    rotAngle = imsup.Degrees(phi2 - phi1)
    if np.abs(rotAngle) > 180:
        rotAngle = -np.sign(rotAngle) * (360 - np.abs(rotAngle))
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