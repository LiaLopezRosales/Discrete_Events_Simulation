import numpy as np
from scipy import stats
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
        self.events = []      # Cola de eventos con prioridad
        self.history = [(0.0, a0)]  # Registro de (tiempo, capital)
        self.ruin_time = None  # Tiempo de ruina
        self.I = 1            # Variable de salida (1 si capital ≥ 0 hasta T)
        
        self.schedule_next_event()
    
    def schedule_next_event(self):
        total_rate = self.nu + self.n * (self.mu + self.lambd)
        if total_rate <= 0:
            return
        X = np.random.exponential(scale=1/total_rate)
        heapq.heappush(self.events, (self.t + X, 'event'))

    def process_event(self):
        if not self.events:
            return
        
        # Extraer el próximo evento de la cola
        event_time, _ = heapq.heappop(self.events)
        
        # Caso 1: El evento ocurre después de T
        if event_time > self.T:
            delta_t = self.T - self.t
            self.capital += self.n * self.c * delta_t
            self.t = self.T
            self.history.append((self.t, self.capital))
            self.I = 1
            return
        
        # Caso 2: El evento ocurre antes o en T
        delta_t = event_time - self.t
        self.capital += self.n * self.c * delta_t
        self.t = event_time
        self.history.append((self.t, self.capital))
        
        # Determinar tipo de evento
        total_rate = self.nu + self.n * (self.mu + self.lambd)
        if total_rate <= 0:
            self.schedule_next_event()
            return
        
        rates = [self.nu, self.n * self.mu, self.n * self.lambd]
        total = sum(rates)
        event_type = np.random.choice([1, 2, 3], p=np.array(rates)/total)
        
        # Actualizar estado según el evento
        if event_type == 1:
            self.n += 1
        elif event_type == 2:
            self.n = max(0, self.n - 1)
        else:
            Y = self.claim_dist()
            if Y > self.capital:
                self.I = 0
                self.ruin_time = self.t
                return
            self.capital -= Y
        
        # Programar próximo evento
        self.schedule_next_event()
    
    def run(self):
        while self.I == 1:
            if not self.events:
                # Actualizar capital hasta T si no hay más eventos
                delta_t = self.T - self.t
                self.capital += self.n * self.c * delta_t
                self.t = self.T
                break
            
            event_time = self.events[0][0]
            if event_time > self.T:
                delta_t = self.T - self.t
                self.capital += self.n * self.c * delta_t
                self.t = self.T
                break
                
            self.process_event()
            
        return self.I


