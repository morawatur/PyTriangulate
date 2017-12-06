import numpy as np
import ImageSupport as imsup
from skimage import transform as tr

#-------------------------------------------------------------------

def RotateImageSki(img, angle, mode='constant'):
    mt = img.memType
    dt = img.cmpRepr
    img.ReIm2AmPh()
    img.MoveToCPU()

    limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    ampScaled = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)
    ampRot = tr.rotate(ampScaled, angle, mode=mode).astype(np.float32)
    ampRotRescaled = imsup.ScaleImage(ampRot, limits[0], limits[1])

    imgRot = imsup.ImageWithBuffer(ampRot.shape[0], ampRot.shape[1], defocus=img.defocus, num=img.numInSeries)
    imgRot.LoadAmpData(ampRotRescaled)

    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)
    imgRot.ChangeMemoryType(mt)
    imgRot.ChangeComplexRepr(dt)

    return imgRot

#-------------------------------------------------------------------

def RescaleImageSki(img, factor):
    mt = img.memType
    dt = img.cmpRepr
    img.ReIm2AmPh()
    img.MoveToCPU()

    amp_limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    phs_limits = [np.min(img.amPh.ph), np.max(img.amPh.ph)]
    amp_scaled = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)
    phs_scaled = imsup.ScaleImage(img.amPh.ph, -1.0, 1.0)
    amp_mag = tr.rescale(amp_scaled, scale=factor).astype(np.float32)
    phs_mag = tr.rescale(phs_scaled, scale=factor).astype(np.float32)
    amp_mag_rescaled = imsup.ScaleImage(amp_mag, amp_limits[0], amp_limits[1])
    phs_mag_rescaled = imsup.ScaleImage(phs_mag, phs_limits[0], phs_limits[1])

    img_mag = imsup.ImageWithBuffer(amp_mag.shape[0], amp_mag.shape[1], defocus=img.defocus, num=img.numInSeries)
    img_mag.LoadAmpData(amp_mag_rescaled)
    img_mag.LoadPhsData(phs_mag_rescaled)

    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)
    img_mag.ChangeMemoryType(mt)
    img_mag.ChangeComplexRepr(dt)

    return img_mag

#-------------------------------------------------------------------

def RotateAndMagnifyWrapper(img, todo='mr', factor=1.0, angle=0.0):
    mt = img.memType
    dt = img.cmpRepr
    img.ReIm2AmPh()
    img.MoveToCPU()

    limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    ampMod = np.copy(imsup.ScaleImage(img.amPh.am, -1.0, 1.0))
    ampMod[ampMod == -1] = 0.0

    if 'r' in todo:
        ampMod = tr.rotate(ampMod, angle=angle).astype(np.float32)
    if 'm' in todo:
        ampMod = tr.rescale(ampMod, scale=factor).astype(np.float32)

    ampModRescaled = imsup.ScaleImage(ampMod, limits[0], limits[1])
    imgMod = imsup.ImageWithBuffer(ampMod.shape[0], ampMod.shape[1], defocus=img.defocus, num=img.numInSeries)
    imgMod.LoadAmpData(ampModRescaled)

    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)
    imgMod.ChangeMemoryType(mt)
    imgMod.ChangeComplexRepr(dt)

    return imgMod

#-------------------------------------------------------------------

def DetermineCropCoordsAfterSkiRotation(oldDim, angle):
    return imsup.DetermineCropCoordsAfterRotation(oldDim, oldDim, angle)

#-------------------------------------------------------------------

def RotateImageSki2(img, angle, cut=False):
    imgRot = RotateAndMagnifyWrapper(img, 'r', angle=angle)
    if cut:
        cropCoords = DetermineCropCoordsAfterSkiRotation(img.width, angle)
        imgRot = imsup.CropImageROICoords(imgRot, cropCoords)
    return imgRot

#-------------------------------------------------------------------

def RescaleImageSki2(img, factor):
    return RotateAndMagnifyWrapper(img, 'm', factor=factor)

#-------------------------------------------------------------------

class Line:
    def __init__(self, a_coeff, b_coeff):
        self.a = a_coeff
        self.b = b_coeff

    def getFromPoints(self, p1, p2):
        self.a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = p1[1] - self.a * p1[0]

    def getFromDirCoeffAndPoint(self, a_coeff, p1):
        self.a = a_coeff
        self.b = p1[1] - self.a * p1[0]

# -------------------------------------------------------------------

def FindPerpendicularLine(line, point):
    linePerp = Line(-1 / line.a, 0)
    linePerp.getFromDirCoeffAndPoint(linePerp.a, point)
    return linePerp

#-------------------------------------------------------------------

# def FindRotationCenter(tr1, tr2):
    # A1, B1 = tr1[1:3]
    # A2, B2 = tr2[1:3]

def FindRotationCenter(pts1, pts2):
    A1, B1 = pts1
    A2, B2 = pts2

    Am = [np.average([A1[0], A2[0]]), np.average([A1[1], A2[1]])]
    Bm = [np.average([B1[0], B2[0]]), np.average([B1[1], B2[1]])]

    aLine = Line(0, 0)
    bLine = Line(0, 0)
    aLine.getFromPoints(A1, A2)
    bLine.getFromPoints(B1, B2)

    aLinePerp = FindPerpendicularLine(aLine, Am)
    bLinePerp = FindPerpendicularLine(bLine, Bm)

    rotCenterX = (bLinePerp.b - aLinePerp.b) / (aLinePerp.a - bLinePerp.a)
    rotCenterY = aLinePerp.a * rotCenterX + aLinePerp.b

    return [rotCenterX, rotCenterY]

#-------------------------------------------------------------------

def RotatePoint(p1, angle):
    z1 = np.complex(p1[0], p1[1])
    r = np.abs(z1)
    phi = np.angle(z1) + imsup.Radians(angle)
    p2 = [r * np.cos(phi), r * np.sin(phi)]
    return p2