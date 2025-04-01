import math

def forwardPass(wiek, waga, wzrost):
    hidden1 = wiek * -0.46122 + waga * 0.97314 + wzrost * -0.39203
    hidden1_po_aktywacji = activation(hidden1 + 0.80109)
    hidden2 = wiek * 0.78548 + waga * 2.10584 + wzrost * -0.57847
    hidden2_po_aktywacji = activation(hidden2 + 0.43529)
    output = hidden1_po_aktywacji * -0.811546 + hidden2_po_aktywacji * 1.03775 - 0.2368
    return output

def activation(x):
    return 1 / (1 + math.exp(-x))

def main():
    test_data = [
        [23, 75, 176],
        [25, 67, 180]

    ]
    
    for data in test_data:
        print(forwardPass(data[0], data[1], data[2]))

if __name__ == "__main__":
    main()