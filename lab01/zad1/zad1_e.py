import datetime
import math

#1 min

def oblicz_biorytm(dni, okres):
    """Oblicza wartość biorytmu dla danej liczby dni i okresu"""
    return math.sin(2 * math.pi * dni / okres)

def main():
    # Pobranie danych od użytkownika
    imie = input("Podaj swoje imię: ")
    rok_urodzenia = int(input("Podaj rok urodzenia: "))
    miesiac_urodzenia = int(input("Podaj miesiąc urodzenia (1-12): "))
    dzien_urodzenia = int(input("Podaj dzień urodzenia: "))
    
    # Obliczenie dnia życia
    data_urodzenia = datetime.date(rok_urodzenia, miesiac_urodzenia, dzien_urodzenia)
    dzisiaj = datetime.date.today()
    dni_zycia = (dzisiaj - data_urodzenia).days
    
    # Obliczenie biorytmów
    biorytm_fizyczny = oblicz_biorytm(dni_zycia, 23)
    biorytm_emocjonalny = oblicz_biorytm(dni_zycia, 28)
    biorytm_intelektualny = oblicz_biorytm(dni_zycia, 33)
    
    # Wyświetlenie wyników
    print(f"\nCześć {imie}!")
    print(f"Dziś jest {dzisiaj.strftime('%d.%m.%Y')}, co oznacza, że żyjesz już {dni_zycia} dni.")
    print("\nTwoje biorytmy na dziś wynoszą:")
    print(f"Fizyczny: {biorytm_fizyczny:.4f}")
    print(f"Emocjonalny: {biorytm_emocjonalny:.4f}")
    print(f"Intelektualny: {biorytm_intelektualny:.4f}")
    
    # Dodatkowe informacje - punkt b
    for nazwa, wartosc in [("Fizyczny", biorytm_fizyczny), 
                           ("Emocjonalny", biorytm_emocjonalny), 
                           ("Intelektualny", biorytm_intelektualny)]:
        if wartosc > 0.5:
            print(f"\nGratulacje! Twój biorytm {nazwa.lower()} jest wysoki ({wartosc:.4f}).")
        elif wartosc < -0.5:
            # Sprawdzenie, czy jutro będzie lepiej
            if nazwa == "Fizyczny":
                jutro = oblicz_biorytm(dni_zycia + 1, 23)
            elif nazwa == "Emocjonalny":
                jutro = oblicz_biorytm(dni_zycia + 1, 28)
            else:  # Intelektualny
                jutro = oblicz_biorytm(dni_zycia + 1, 33)
                
            print(f"\nTwój biorytm {nazwa.lower()} jest dziś niski ({wartosc:.4f}).")
            if jutro > wartosc:
                print(f"Nie martw się! Jutro będzie lepiej - biorytm wzrośnie do {jutro:.4f}.")
            else:
                print(f"Niestety, jutro może być trudniejszy dzień - biorytm spadnie do {jutro:.4f}.")

if __name__ == "__main__":
    main()