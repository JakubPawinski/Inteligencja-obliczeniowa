import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
import math
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history
# Set the style for the plot


def endurance(params):
    x, y, z, u, v, w = params
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

def f(x):
    """
    Funkcja obiektywna dla całego roju.
    
    Args:
        x (numpy.ndarray): Rój cząstek o wymiarach (n_particles, dimensions)
    
    Returns:
        numpy.ndarray: Wartości funkcji -endurance dla każdej cząstki
        (minimalizujemy -endurance, co jest równoważne maksymalizacji endurance)
    """
    n_particles = x.shape[0]
    j = np.zeros(n_particles)
    
    for i in range(n_particles):
        j[i] = -endurance(x[i])  # Odwracamy znak, bo PSO minimalizuje
    
    return j

x_max = np.ones(6)
x_min = np.zeros(6)

my_bounds = (x_min, x_max)

bounds=my_bounds

options = {'c1': 0.8, 'c2': 0.8, 'w':0.9}
# c1 => współczynnik poznawczy
# c2 => współczynnik społeczy
# w => współczynnik bezwładności

best_cost = 1000
best_pos = None
for run in range(1000):

    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6,
    options=options, bounds=bounds)

    cost, pos = optimizer.optimize(f, iters=200)

    if cost < best_cost:
        best_cost = cost
        best_pos = pos

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")

cost_history = optimizer.cost_history


plot_cost_history(cost_history)
# plt.show()