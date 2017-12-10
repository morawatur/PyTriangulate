from numba import cuda
import GUI_Holo as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# Potrzebne jeszcze:
#
# - poprawić funkcję PadImage tak, aby brała pod uwagę fazę
# - panel z parametrami rekonstrukcji holo (wielkosc maski itp.)
# - dodac mozliwosc zaznaczenia wiecej niz trzech punktow do triangulacji

gui.RunTriangulationWindow()