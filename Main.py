from numba import cuda
import GUI_Holo as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# Potrzebne jeszcze:
# - znak kata obrotu nie zawsze jest poprawny
# - flip nie dziala na amplitude (?)
# - zrobic re-warp tak samo jak re-shift i re-rotate
# - kontrola jasnosci, gammy i kontrastu

gui.RunTriangulationWindow()

# juz wiem, co bylo nie tak - winna byla konwersja z re_im do am_ph, ktora powoduje wrapping fazy
# - funkcje polar i rect z cmath dzialaja zle na GPU, jezeli faza przekracza 2pi

# reshift powoduje, ze mozna zoomowac obrazy fazowe (przed reshiftem cos nie dziala)
# prawdopodobnie w trakcie reshifta nastepuje zmiana typu Image na ImageWithBuffer