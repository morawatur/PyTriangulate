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

    limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    ampScaled = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)
    ampMag = tr.rescale(ampScaled, scale=factor).astype(np.float32)
    ampMagRescaled = imsup.ScaleImage(ampMag, limits[0], limits[1])

    imgMag = imsup.ImageWithBuffer(ampMag.shape[0], ampMag.shape[1], defocus=img.defocus, num=img.numInSeries)
    imgMag.LoadAmpData(ampMagRescaled)

    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)
    imgMag.ChangeMemoryType(mt)
    imgMag.ChangeComplexRepr(dt)

    return imgMag

#-------------------------------------------------------------------

def RotateAndMagnifyWrapper(img, todo='mr', factor=1.0, angle=0.0):
    mt = img.memType
    dt = img.cmpRepr
    img.ReIm2AmPh()
    img.MoveToCPU()

    limits = [np.min(img.amPh.am), np.max(img.amPh.am)]
    ampMod = np.copy(imsup.ScaleImage(img.amPh.am, -1.0, 1.0))

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
    radAngle = imsup.Radians(np.abs(angle) % 45)
    newDim = oldDim / (np.cos(radAngle) + np.sin(radAngle))
    start = (oldDim - newDim) / 2.0
    end = oldDim - start
    cropCoords = [int(np.ceil(start))] * 2 + [int(np.floor(end))] * 2
    return cropCoords

#-------------------------------------------------------------------

def RotateImageSki2(img, angle, cut=False):
    imgRot = RotateAndMagnifyWrapper(img, 'r', angle=angle)
    imsup.SaveAmpImage(imgRot, 'img2_rot.png')
    if cut:     # wycinanie zle dziala
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

    print('A1 = ({0:.0f}, {1:.0f})'.format(A1[0], A1[1]))
    print('A2 = ({0:.0f}, {1:.0f})'.format(A2[0], A2[1]))
    print('B1 = ({0:.0f}, {1:.0f})'.format(B1[0], B1[1]))
    print('B2 = ({0:.0f}, {1:.0f})'.format(B2[0], B2[1]))
    # print('Am = ({0:.0f}, {1:.0f})'.format(Am[0], Am[1]))
    # print('aLine: a = {0:.1f}, b = {1:.1f}'.format(aLine.a, aLine.b))
    # print('aLinePerp: a = {0:.1f}, b = {1:.1f}'.format(aLinePerp.a, aLinePerp.b))

    rotCenterX = (bLinePerp.b - aLinePerp.b) / (aLinePerp.a - bLinePerp.a)
    rotCenterY = aLinePerp.a * rotCenterX + aLinePerp.b

    return [rotCenterX, rotCenterY]

#-------------------------------------------------------------------

def RotatePoint(p1, angle):
    z1 = np.complex(p1[0], p1[1])
    r = np.abs(z1)
    phi = np.angle(z1) + imsup.Radians(angle)
    p2 = [r * np.cos(phi), r * np.sin(phi)]
    print('######')
    print(imsup.Degrees(np.angle(z1)))
    print(angle)
    print('({0:.0f}, {1:.0f})'.format(p1[0], p1[1]))
    print('({0:.0f}, {1:.0f})'.format(p2[0], p2[1]))
    print('######')
    return p2