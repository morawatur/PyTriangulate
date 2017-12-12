from numba import cuda
import GUI_Holo as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

# Potrzebne jeszcze:
# - unwrapping fazy?
# - holo z referencja
# - kontrola jasnosci, gammy i kontrastu

gui.RunTriangulationWindow()