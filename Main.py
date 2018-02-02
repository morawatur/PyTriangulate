from numba import cuda
import GUI_Holo as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# Potrzebne jeszcze:
# - podczas zoomowania obrazu powinienem zmieniac zapisana wartosc wielkosci piksela
# - flip nie dziala na amplitude (?)
# - zrobic re-warp tak samo jak re-shift i re-rotate
# - kontrola jasnosci, gammy i kontrastu

gui.RunTriangulationWindow()

# juz wiem, co bylo nie tak - winna byla konwersja z re_im do am_ph, ktora powoduje wrapping fazy
# - funkcje polar i rect z cmath dzialaja zle na GPU, jezeli faza przekracza 2pi