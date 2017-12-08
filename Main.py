from numba import cuda
import GUI_Holo as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# Potrzebne jeszcze:
# - jezeli po sumie wykona sie roznice, to suma znika, a powinna sie przesunac o jeden obraz dalej
# - panel z parametrami rekonstrukcji holo (wielkosc maski itp.)
# - dodac mozliwosc zaznaczenia wiecej niz trzech punktow do triangulacji
# - przycisk Delete powinien tez usuwac zapisane zestawy punktow

gui.RunTriangulationWindow()