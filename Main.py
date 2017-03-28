from numba import cuda
import Constants as const
import GUI2 as gui
import ImageSupport as imsup
import CrossCorr as cc
import time

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# img1 = imsup.FileToImage('holo-1.dm3')
# imgs1H = imsup.LinkTwoImagesSmoothlyH(img1, img1)
# linkedImages1 = imsup.LinkTwoImagesSmoothlyV(imgs1H, imgs1H)
#
# img2 = imsup.FileToImage('holo-2.dm3')
# imgs2H = imsup.LinkTwoImagesSmoothlyH(img2, img2)
# linkedImages2 = imsup.LinkTwoImagesSmoothlyV(imgs2H, imgs2H)

gui.RunTriangulationWindow()

# w koncu program wyznacza dobrze srodek obrotu
# pociagnac to dalej i uzyc do obracania i zsuwania obrazow (na podstawie trzech punktow, a nie MCF)

# img1.MoveToCPU()
# start = time.clock()
# for n in range(1000):
#     fft = cc.FFT(img1)
# end = time.clock()
# print('Elapsed time: {0:.3f} s'.format(end - start))