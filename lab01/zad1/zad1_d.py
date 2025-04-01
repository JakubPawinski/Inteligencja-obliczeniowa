import datetime
import math



def calculate_age(year, month, day):
    try:
        today = datetime.datetime.now()
        birth_date = datetime.datetime(year, month, day)
        age = today - birth_date
        return age.days
    except ValueError:
        print("Błąd: Nieprawidłowa data. Upewnij się, że podana data faktycznie istnieje.")
        return None
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        return None

def calculate_emotional_wave(age):
    return math.sin(2 * math.pi / 23 * age)

def calculate_physical_wave(age):
    return math.sin(2 * math.pi / 28 * age)

def calculate_intellectual_wave(age):
    return math.sin(2 * math.pi / 33 * age)

def check_wave(wave, type, age):
    if wave < -0.5:
        msg = 'Today is not your day!'
        match type:
            case "emotional":
                msg = f'{msg}\nTomorrow you will have {calculate_emotional_wave(age + 1):.2f} emotional wave'
            case "physical":
                msg = f'{msg}\nTomorrow you will have {calculate_physical_wave(age + 1):.2f} physical wave'
            case "intellectual":
                msg = f'{msg}\nTomorrow you will have {calculate_intellectual_wave(age + 1):.2f} intellectual wave'
        return msg
    elif wave > 0.5:
        return "Today is your day!"
    else:
        return "Today is a normal day."

def get_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Proszę wprowadzić liczbę całkowitą.")

def main():
    print("Program do obliczania biotrytmów")
    name = input("Wprowadź swoje imię: ")
    surname = input("Wprowadź swoje nazwisko: ")
    
    year_of_birth = get_int_input("Wprowadź rok urodzenia: ")
    month_of_birth = get_int_input("Wprowadź miesiąc urodzenia (1-12): ")
    day_of_birth = get_int_input("Wprowadź dzień urodzenia: ")

    age = calculate_age(year_of_birth, month_of_birth, day_of_birth)
    
    if age is None:
        return
    
    if age < 0:
        print("Błąd: Data urodzenia jest w przyszłości!")
        return

    emotional_wave = calculate_emotional_wave(age)
    physical_wave = calculate_physical_wave(age)
    intellectual_wave = calculate_intellectual_wave(age)

    msg = f'''
Hello, {name} {surname}!
Today is your {age} day of life.
Your results are:
Emotional: {emotional_wave:.2f} - {check_wave(emotional_wave, "emotional", age)}
Physical: {physical_wave:.2f} - {check_wave(physical_wave, "physical", age)}
Intellectual: {intellectual_wave:.2f} - {check_wave(intellectual_wave, "intellectual", age)}
'''
    print(msg)

if __name__ == "__main__":
    main()