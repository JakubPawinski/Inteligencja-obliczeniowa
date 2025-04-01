from random import randint
import matplotlib.pyplot as plt
import numpy as np
import math

INITIAL_VELOCITY = 50
INITIAL_HEIGHT = 100
GRAVITY = 9.81

class ShootingGame:
    def __init__(self, velocity=INITIAL_VELOCITY, height=INITIAL_HEIGHT):
        self.velocity = velocity
        self.height = height
        self.target = self.get_target()
        self.shots = []
        self.training_mode = False
    
    def get_target(self):
        return randint(50, 340)
    
    def calculate_range(self, angle):
        theta_rad = math.radians(angle)
        # Obliczenie czasu lotu
        
        # Składowe prędkości początkowej
        v0x = self.velocity * math.cos(theta_rad)
        v0y = self.velocity * math.sin(theta_rad)
        
        # Dla ujemnych kątów, potrzebujemy innego podejścia do obliczania czasu lotu
        # Rozwiązujemy równanie kwadratowe: h + v0y*t - 0.5*g*t^2 = 0 (kiedy y = 0)
        a = -0.5 * GRAVITY
        b = v0y
        c = self.height
        
        # Wyznaczamy delta
        delta = b**2 - 4*a*c
        
        if delta < 0:
            # Gdy delta < 0, pocisk nigdy nie osiągnie ziemi
            return 0
        
        # Obliczamy dwa możliwe czasy
        t1 = (-b + math.sqrt(delta)) / (2*a)
        t2 = (-b - math.sqrt(delta)) / (2*a)
        
        # Wybieramy dodatnią wartość czasu
        t = max(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2)
        
        if t <= 0:
            # Pocisk nie osiągnie ziemi (np. gdy strzelamy zbyt mocno w dół)
            return 0
        
        # Obliczenie zasięgu poziomego
        R = v0x * t
        
        return round(R, 2)
    
    def check_if_hit(self, shot):
        return self.target - 5 <= shot <= self.target + 5
    
    def take_shot(self, angle):
        shot = self.calculate_range(angle)
        self.shots.append((angle, shot))
        return shot
    
    def plot_all_trajectories(self):
        plt.figure(figsize=(12, 7))
        
        # Rysuj wszystkie poprzednie strzały (szarymi liniami)
        for i, (angle, distance) in enumerate(self.shots):
            is_last = i == len(self.shots) - 1
            color = 'b' if is_last else 'gray'
            alpha = 1.0 if is_last else 0.3
            linewidth = 2 if is_last else 1
            self.plot_single_trajectory(angle, color, alpha, linewidth)
        
        # Dodanie celu i innych elementów
        plt.scatter(self.target, 0, color='red', s=100, label='Cel')
        plt.axvspan(self.target-5, self.target+5, color='red', alpha=0.3, label='Strefa trafienia')
        plt.scatter(0, self.height, color='green', s=100, label='Start')
        
        # Dodanie poziomej linii dla ziemi
        plt.axhline(y=0, color='k', linestyle='-')
        
        # Etykiety i tytuł
        plt.xlabel('Odległość [m]')
        plt.ylabel('Wysokość [m]')
        plt.title(f'Historia strzałów (cel: {self.target} m)')
        plt.grid(True)
        plt.legend()
        
        # Ustawienia osi i wyświetlenie wykresu
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.tight_layout()
        plt.show()
    
    def plot_single_trajectory(self, angle, color='b', alpha=1.0, linewidth=2):
        theta_rad = math.radians(angle)
        
        # Obliczanie prędkości początkowych w osiach x i y
        v0x = self.velocity * math.cos(theta_rad)
        v0y = self.velocity * math.sin(theta_rad)
        
        # Obliczanie czasu lotu (dla ujemnych i dodatnich kątów)
        a = -0.5 * GRAVITY
        b = v0y
        c = self.height
        
        delta = b**2 - 4*a*c
        
        if delta < 0:
            # Trajektoria nigdy nie osiągnie ziemi - ustalamy jakiś maksymalny czas
            t_total = 10  # Arbitralnie wybrana wartość
        else:
            t1 = (-b + math.sqrt(delta)) / (2*a)
            t2 = (-b - math.sqrt(delta)) / (2*a)
            t_total = max(t1, t2) if t1 > 0 or t2 > 0 else 0
            
            if t_total <= 0:
                # Jeśli czas jest ujemny lub zero, ustalamy minimalny czas
                t_total = 0.1
        
        # Tworzenie tablicy czasu dla wykresu
        t = np.linspace(0, t_total, 100)
        
        # Obliczanie współrzędnych x i y w każdym punkcie czasu
        x = v0x * t
        y = self.height + v0y * t - 0.5 * GRAVITY * t**2
        
        # Przycinamy trajektorię, gdy osiągnie ziemię
        above_ground = y >= 0
        if not all(above_ground):
            # Znajdujemy pierwszy punkt, w którym trajektoria przecina ziemię
            first_ground_idx = np.where(~above_ground)[0][0]
            if first_ground_idx > 0:
                # Interpolujemy dokładny punkt przecięcia z ziemią
                i = first_ground_idx - 1
                j = first_ground_idx
                if y[j] != y[i]:  # Unikamy dzielenia przez zero
                    ratio = -y[i] / (y[j] - y[i])
                    x_ground = x[i] + ratio * (x[j] - x[i])
                    # Dodajemy punkt przecięcia z ziemią
                    x = np.append(x[:first_ground_idx], x_ground)
                    y = np.append(y[:first_ground_idx], 0)
                else:
                    x = x[:first_ground_idx]
                    y = y[:first_ground_idx]
            else:
                # Jeśli pierwszy punkt jest już pod ziemią, skracamy do pustej tablicy
                x = np.array([])
                y = np.array([])
        
        # Rysowanie trajektorii
        plt.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
    
    def suggest_angle(self):
        # Dla uproszczenia, ta metoda zwraca tylko dodatni kąt
        # W przypadku ujemnych kątów analityczne rozwiązanie jest bardziej złożone
        g = GRAVITY
        v0 = self.velocity
        h = self.height
        x = self.target
        
        # Równanie kwadratowe dla dodatnich kątów
        a = x**2
        b = 2 * x * math.sqrt(v0**2 * h)
        c = -v0**2 * x**2
        
        # Rozwiązanie równania kwadratowego
        delta = b**2 - 4*a*c
        if delta < 0:
            return None  # Nie ma rozwiązania
        
        sin_theta1 = (-b + math.sqrt(delta)) / (2*a)
        sin_theta2 = (-b - math.sqrt(delta)) / (2*a)
        
        angle1 = None
        angle2 = None
        
        # Wybieramy rozwiązania w zakresie [-1,1] dla sin(θ)
        if -1 <= sin_theta1 <= 1:
            angle1 = math.degrees(math.asin(sin_theta1))
        
        if -1 <= sin_theta2 <= 1:
            angle2 = math.degrees(math.asin(sin_theta2))
        
        # Jeśli mamy dwa rozwiązania, wybieramy to o mniejszej wartości bezwzględnej
        if angle1 is not None and angle2 is not None:
            return round(angle1, 1) if abs(angle1) < abs(angle2) else round(angle2, 1)
        elif angle1 is not None:
            return round(angle1, 1)
        elif angle2 is not None:
            return round(angle2, 1)
        else:
            return None  # Nie ma rozwiązania

def main():
    print("Witaj w symulatorze strzelnicy!")
    print("1. Tryb standardowy")
    print("2. Tryb treningowy (z podpowiedziami)")
    
    choice = input("Wybierz tryb gry (1/2): ")
    
    game = ShootingGame()
    if choice == '2':
        game.training_mode = True
    
    print(f"\nCel znajduje się w odległości {game.target} metrów.")
    
    n = 1
    while True:
        if game.training_mode and n > 1:
            last_angle, last_shot = game.shots[-1]
            if last_shot < game.target:
                print("Podpowiedź: Spróbuj zwiększyć kąt (lub użyć bardziej ujemnego).")
            elif last_shot > game.target:
                print("Podpowiedź: Spróbuj zmniejszyć kąt (lub użyć bardziej dodatniego).")
                
            if n % 3 == 0:  # Co trzy próby dajemy dokładniejszą podpowiedź
                suggested_angle = game.suggest_angle()
                if suggested_angle:
                    print(f"Podpowiedź: Spróbuj kąt około {suggested_angle}°")
        
        try:
            angle = float(input(f"Strzał #{n}. Podaj kąt wyrzutu (w stopniach, może być ujemny): "))
            
            if angle < -90 or angle > 90:
                print("Kąt musi być między -90° a 90°.")
                continue
                
            shot = game.take_shot(angle)
            if shot == 0:
                print("Ten strzał nie osiągnął ziemi lub poleciał w złą stronę!")
                n += 1
                continue
                
            print(f"Twój strzał wylądował w odległości {shot} m od startu.")
            
            if game.check_if_hit(shot):
                print(f"Trafienie! Udało Ci się trafić w cel po {n} próbach.")
                game.plot_all_trajectories()
                break
            else:
                if shot < game.target:
                    print(f"Za blisko! Cel jest dalej o {game.target - shot} m.")
                else:
                    print(f"Za daleko! Cel jest bliżej o {shot - game.target} m.")
                print("\n" + "="*30)
                n += 1
        except ValueError:
            print("Proszę podać poprawną wartość liczbową dla kąta.")

if __name__ == "__main__":
    main()