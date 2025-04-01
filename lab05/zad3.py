import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os

#Constants
DIR_DATA = "./dogs-cats-mini/"
FAST_RUN = True
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
BATCH_SIZE=64

def map_predictions_to_labels(predicted_classes, validation_generator):
    """Mapuje przewidziane klasy numeryczne na nazwy kategorii"""
    label_map = dict((v, k) for k, v in validation_generator.class_indices.items())
    predicted_labels = [label_map[i] for i in predicted_classes]
    return predicted_labels

def visualize_predictions(validate_df, predicted_labels, dir_data, image_size, sample_count=9):
    """Wizualizuje obrazy z porównaniem prawdziwych etykiet i predykcji"""
    validate_df_sample = validate_df.head(sample_count).copy()
    validate_df_sample['predicted_category'] = predicted_labels[:sample_count]
    
    plt.figure(figsize=(12, 9))
    for i, (index, row) in enumerate(validate_df_sample.iterrows()):
        plt.subplot(3, 3, i+1)
        img = load_img(os.path.join(dir_data, row['filename']), target_size=image_size)
        plt.imshow(img)
        color = "green" if row['category'] == row['predicted_category'] else "red"
        plt.title(f"Prawdziwa: {row['category']}\nPredykcja: {row['predicted_category']}", 
                color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return validate_df_sample

def plot_prediction_stats(predicted_labels):
    """Tworzy wykres słupkowy pokazujący liczbę predykcji dla każdej klasy"""
    plt.figure(figsize=(8, 5))
    labels, counts = np.unique(predicted_labels, return_counts=True)
    plt.bar(labels, counts)
    plt.title('Liczba predykcji w każdej kategorii')
    plt.xlabel('Kategoria')
    plt.ylabel('Liczba predykcji')
    plt.show()

def save_results_to_csv(validate_df, predicted_labels, filename='predictions_results.csv'):
    """Zapisuje wyniki predykcji do pliku CSV"""
    results_df = pd.DataFrame({
        'filename': validate_df['filename'],
        'true_category': validate_df['category'],
        'predicted_category': predicted_labels
    })
    results_df.to_csv(filename, index=False)
    print(f"Zapisano wyniki predykcji do pliku '{filename}'")
    return results_df

def show_sample_image():
    filenames = os.listdir(DIR_DATA)
    sample = random.choice(filenames)
    image = load_img(os.path.join(DIR_DATA, sample), target_size=(150, 150))
    plt.imshow(image)
    plt.axis('off')
    plt.title(sample)
    plt.show()

def load_dataset():

    def show_plot(df):
        df['category'].value_counts().plot.bar()
        plt.title("Liczba obrazów w każdej kategorii")
        plt.xlabel("Kategoria")
        plt.ylabel("Liczba obrazów")
        plt.show() 

    filenames = os.listdir(DIR_DATA)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    # show_plot(df)
    return df

def create_generators(train_df, validate_df):
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        DIR_DATA, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        DIR_DATA, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    return train_datagen, train_generator, validation_generator

def visualize_augmentation(train_df, train_datagen):
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        DIR_DATA,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )

    plt.figure(figsize=(12, 12))
    for i in range(0, 15):
        plt.subplot(5, 3, i+1)
        for X_batch, Y_batch in example_generator:
            image = X_batch[0]
            plt.imshow(image)
            break
    plt.tight_layout()
    plt.show()

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    return model

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Przewidziane')
    plt.ylabel('Prawdziwe')
    plt.title('Macierz pomyłek')
    plt.show()


def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Dokładność')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Strata')
    plt.legend()
    plt.show()

def main():
    df = load_dataset()
    print(f"Liczba plików w katalogu: {len(os.listdir(DIR_DATA))}")

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    # print(f"Liczba plików w zbiorze treningowym: {len(train_df)}")
    # print(f"Liczba plików w zbiorze walidacyjnym: {len(validate_df)}")


    train_datagen, train_generator, validation_generator = create_generators(train_df, validate_df)

    model = define_model()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]


    epochs=1 if FAST_RUN else 50
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    model.save("model_dogs_cats.keras")


    plot_history(history)


    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)


    predicted_labels = map_predictions_to_labels(predicted_classes, validation_generator)
    validate_df_sample = visualize_predictions(validate_df, predicted_labels, DIR_DATA, IMAGE_SIZE)
    plot_prediction_stats(predicted_labels)
    results_df = save_results_to_csv(validate_df, predicted_labels)


    cm = confusion_matrix(validate_df['category'], predicted_labels)
    plot_confusion_matrix(cm, ['cat', 'dog'])

if __name__ == "__main__":
    main()