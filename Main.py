from numba import cuda
import Constants as const
import GUI as gui
import ImageSupport as imsup
import CrossCorr as cc

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# img1 = imsup.FileToImage('holo1.dm3')
# imgs1H = imsup.LinkTwoImagesSmoothlyH(img1, img1)
# linkedImages1 = imsup.LinkTwoImagesSmoothlyV(imgs1H, imgs1H)
#
# img2 = imsup.FileToImage('holo2.dm3')
# imgs2H = imsup.LinkTwoImagesSmoothlyH(img2, img2)
# linkedImages2 = imsup.LinkTwoImagesSmoothlyV(imgs2H, imgs2H)

gui.RunTriangulationWindow()