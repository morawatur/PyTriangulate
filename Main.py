from numba import cuda
import GUI_Holo as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# Potrzebne jeszcze:
# - przycisk Delete do usuwania obrazow z kolejki
# - panel z parametrami rekonstrukcji holo (wielkosc maski itp.)

gui.RunTriangulationWindow()