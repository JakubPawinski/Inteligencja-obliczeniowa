import datetime
import math

#Około 25 min

def calculate_age(year, month, day):
    today = datetime.datetime.now()
    birth_date = datetime.datetime(year, month, day)
    age = today - birth_date
    return age.days

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
                msg = f'{msg}\nTomorrow you will have {calculate_emotional_wave(age + 1)} emotional wave'
            case "physical":
                msg = f'{msg}\nTomorrow you will have {calculate_physical_wave(age + 1)} physical wave'
            case "intellectual":
                msg = f'{msg}\nTomorrow you will have {calculate_intellectual_wave(age + 1)} intellectual wave'
        return msg
    elif wave > 0.5:
        return "Today is your day!"
    else:
        return "Today is a normal day."


def main():
    print("Hello, World!")
    name = input("Enter your name: ")
    surname = input("Enter your surname: ")
    year_of_birth = int(input("Enter your year: "))
    month_of_birth = int(input("Enter your month of birth: "))
    day_of_birth = int(input("Enter your day of birth: "))

    age = calculate_age(year_of_birth, month_of_birth, day_of_birth)

    emotional_wave = calculate_emotional_wave(age)
    physical_wave = calculate_physical_wave(age)
    intellectual_wave = calculate_intellectual_wave(age)

    msg = f'''
Hello, {name} {surname}!
Today is your {age} day of life.
Your results are:
Emotional: {emotional_wave} {check_wave(emotional_wave, "emotional", age)}
Physical: {physical_wave} {check_wave(physical_wave, "physical", age)}
Intellectual: {intellectual_wave} {check_wave(intellectual_wave, "intellectual", age)}
    '''
    print(msg)



if __name__ == "__main__":
    main()