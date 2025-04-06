import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import heapq

class InsuranceRiskSimulation:
    def __init__(self, nu, mu, lambd, c, a0, n0, T, claim_dist):
        self.nu = nu          # Frecuencia de llegada de nuevos clientes
        self.mu = mu          # Frecuencia de abandono de clientes
        self.lambd = lambd    # Frecuencia de reclamos por cliente
        self.c = c            # Ingreso por cliente por unidad de tiempo
        self.capital = a0     # Capital inicial
        self.n = n0           # Clientes iniciales
        self.T = T            # Tiempo máximo
        self.claim_dist = claim_dist  # Función que genera montos de reclamos 
        self.t = 0            # Tiempo actual
        self.events = []      # Cola de eventos
        self.I = 1            # Variable de salida (1 si capital ≥ 0 hasta T)