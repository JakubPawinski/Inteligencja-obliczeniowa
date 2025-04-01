import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1) # Save original labels for confusion matrix

# Define model
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
MaxPooling2D((2, 2)),
Flatten(),
Dense(64, activation='relu'),
Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(
    filepath='best_model.h5',  # Nazwa pliku, w którym model będzie zapisywany
    monitor='val_accuracy',   # Monitorujemy dokładność walidacyjną
    save_best_only=True,      # Zapisujemy tylko, gdy wynik jest lepszy niż poprzedni
    mode='max',               # Szukamy maksymalnej wartości (dla dokładności)
    verbose=1                 # Wyświetlamy informacje o zapisie modelu
)

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2,
callbacks=[history, checkpoint])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()

'''
a)
Normalizacja obrazów:
    Obrazy są przekształcane do formatu (28, 28, 1) (dodanie kanału dla skali szarości).
    Piksele są skalowane do zakresu [0, 1] przez podzielenie przez 255.
Kodowanie etykiet:
    to_categorical zamienia etykiety (np. 5) na wektory binarne (np. [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).
Zapisanie oryginalnych etykiet:
    original_test_labels przechowuje oryginalne etykiety (0-9) do późniejszej analizy (np. macierzy pomyłek).

b)
    1. Warstwa konwolucyjna (Conv2D) z 32 filtrami o rozmiarze 3x3 i funkcją aktywacji ReLU.
    2. Warstwa MaxPooling2D zmniejsza rozmiar obrazu przez zastosowanie operacji max pooling (2x2).
    3. Warstwa Flatten przekształca dane do formatu jednowymiarowego.
    4. Warstwa Dense z 64 neuronami i funkcją aktywacji ReLU.
    5. Warstwa Dense z 10 neuronami (dla 10 klas) i funkcją aktywacji softmax. Zwraca prawdopodobieństwa dla każdej klasy.

c)
 9 jako 4 - 15 błędów
 2 jako 7 - 8 błędów

d)
nie widać przeuczania

'''