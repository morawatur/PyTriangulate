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
    newDim = oldDim / (np.cos(imsup.Radians(angle)) + np.sin(imsup.Radians(angle)))
    start = (oldDim - newDim) / 2.0
    end = oldDim - start
    cropCoords = [int(np.ceil(start))] * 2 + [int(np.floor(end))] * 2
    return cropCoords

#-------------------------------------------------------------------

def RotateImageSki2(img, angle):
    imgRot = RotateAndMagnifyWrapper(img, 'r', angle=angle)
    imsup.SaveAmpImage(imgRot, 'img2_rot.png')
    cropCoords = DetermineCropCoordsAfterSkiRotation(img.width, angle)
    imgRotCrop = imsup.CropImageROICoords(imgRot, cropCoords)
    return imgRotCrop

#-------------------------------------------------------------------

def RescaleImageSki2(img, factor):
    return RotateAndMagnifyWrapper(img, 'm', factor=factor)