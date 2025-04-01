from random import randint
import matplotlib.pyplot as plt
import numpy as np

INITIAL_VELOCITY = 50
INITIAL_HEIGHT = 100

import math

def calculate_range(v0, theta, h):
    g = 9.81  
    theta_rad = math.radians(theta) 

    t = (v0 * math.sin(theta_rad) + math.sqrt((v0 * math.sin(theta_rad))**2 + 2 * g * h)) / g

    R = v0 * math.cos(theta_rad) * t

    return round(R, 2)

def plot_trajectory(v0, theta, h, target):
    g = 9.81 
    theta_rad = math.radians(theta) 

    v0x = v0 * math.cos(theta_rad)
    v0y = v0 * math.sin(theta_rad)

    t_total = (v0y + math.sqrt(v0y**2 + 2*g*h)) / g

    t = np.linspace(0, t_total, 100)

    x = v0x * t
    y = h + v0y * t - 0.5 * g * t**2

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)

    plt.scatter(0, h, color='green', s=100, label='Start')
    plt.scatter(target, 0, color='red', s=100, label='Target')
    plt.axvspan(target-5, target+5, color='red', alpha=0.3, label='Hit zone')

    plt.axhline(y=0, color='k', linestyle='-')

    plt.xlabel('Odległość [m]')
    plt.ylabel('Wysokość [m]')
    plt.title(f'Trajektoria pocisku (kąt: {theta}°, prędkość początkowa: {v0} m/s)')
    plt.grid(True)
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()

def get_target():
    return randint(50, 340)

def check_if_hit(target, shot):
    if target - 5 <= shot <= target + 5:
        return True
    else:
        return False
    

def main():
    print("Welcome to the shooting range!")
    
    n = 1
    target = get_target()
    
    while True:
        print(f"Target is at {target}")
        angle = int(input("Enter your angle: "))
        
        shot = calculate_range(INITIAL_VELOCITY, angle, INITIAL_HEIGHT)
        print(f"Shot is at {shot}")
        if check_if_hit(target, shot):
            print("Hit!")
            print(f"You have hit the target in {n} shots.")
            plot_trajectory(INITIAL_VELOCITY, angle, INITIAL_HEIGHT, target)
            break
        else:
            print("Miss!\nTry again.\n========")
            n += 1

if __name__ == "__main__":
    main()