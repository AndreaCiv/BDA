import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

# se serve fare "pip install scipy"


# Funzione per convertire le immagini in bianco e nero
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def train_and_test():
    initializer = tf.keras.initializers.GlorotUniform(seed=5)
    # numero neuroni da avere nello strato hidden per avere rispettivamente
    # 67, 133, 199, 265, 331, 397, 529, 661, 793, 1057, 1321, 1585, 1849, 1981 2047, 2113, 2179, 2245, 2311, 2377, 2641, 2905, 4027, 5281, 10561, 14561, 59401, 264001, 653401 parametri
    n_unit_to_check = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 30, 31, 32, 33, 34, 35, 36, 40, 44, 61, 80, 160,
                       220, 900, 4000, 9900]

    epochs = 5000

    mse_train = []
    mse_test = []
    accuracy_train = []
    accuracy_test = []
    for neurons in n_unit_to_check:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(64,)),
            tf.keras.layers.Dense(units=neurons, initializer=initializer, activation='relu'),
            tf.keras.layers.Dense(units=1, initializer=initializer, activation='sigmoid'),
        ])
        optimizer = tf.keras.optimizers.experimental.SGD(
            learning_rate=0.001,
            momentum=0.95
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse', 'accuracy']
        )

        history = model.fit(x=x_train_downsampled, y=y_train_cut, epochs=epochs, verbose=0)

        evaluation = model.evaluate(x=x_test_downsampled, y=y_test_fully_connected)

        mse_train.append(history.history['mse'][epochs-1])
        mse_test.append(evaluation[0])
        accuracy_train.append(history.history['accuracy'][epochs-1])
        accuracy_test.append(evaluation[2])

    return mse_train, mse_test, accuracy_train, accuracy_test

def print_graphics(mse_train, mse_test, accuracy_train, accuracy_test):
    # Prima immagine
    n_of_params = [67, 133, 199, 265, 331, 397, 529, 661, 793, 1057, 1321, 1585, 1849, 1981, 2047, 2113, 2179, 2245,
                   2311, 2377, 2641, 2905, 4027, 5281, 10561, 14561, 59401, 264001, 653401]

    plt.figure(figsize=(6, 3))

    plt.title("Zero-one Loss (%)")
    plt.plot(n_of_params, (1 - np.array(accuracy_train)) * 100, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, (1 - np.array(accuracy_test)) * 100, linewidth=2.0, color='blue', label="Test")
    plt.xscale('log')
    plt.legend()

    plt.axvline(x=2000)

    plt.show()

    # Seconda immagine
    plt.figure(figsize=(6, 3))

    plt.title("Squared Loss")
    plt.plot(n_of_params, mse_train, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, mse_test, linewidth=2.0, color='blue', label="Test")
    plt.legend()
    plt.xscale('log')
    plt.axvline(x=2000)

    plt.show()


if __name__ == "__main__":

    # Import del dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Trasformazione di tutte le immagini in bianco e nero
    x_train_gray_list = []
    x_test_gray_list = []

    for image in x_train:
        gray_image = rgb2gray(np.array(image.reshape(32, 32, 3)))
        x_train_gray_list.append(gray_image)

    for image in x_test:
        gray_image = rgb2gray(np.array(image.reshape(32, 32, 3)))
        x_test_gray_list.append(gray_image)

    x_train_gray = np.array(x_train_gray_list)
    x_test_gray = np.array(x_test_gray_list)

    print(x_train_gray.shape)
    print(x_test_gray.shape)

    # Normalizzazione delle immagini
    x_train_norm = x_train_gray / 255
    x_test_norm = x_test_gray / 255

    # trovo gli indici dei soli cani e gatti
    indexes_cats_train = np.where(y_train == [3])[0]
    indexes_dogs_train = np.where(y_train == [5])[0]

    indexes_cats_test = np.where(y_test == [3])[0]
    indexes_dogs_test = np.where(y_test == [5])[0]

    # estraggo le immagini di cani e gatti tramite gli indici trovati
    train_cats = x_train_norm[indexes_cats_train]
    train_dogs = x_train_norm[indexes_dogs_train]
    x_train_fully_connected = np.concatenate((train_cats, train_dogs))

    test_cats = x_test_norm[indexes_cats_test]
    test_dogs = x_test_norm[indexes_dogs_test]
    x_test_fully_connected = np.concatenate((test_cats, test_dogs))

    # creo le etichette di train e test
    # 0=gatto, 1=cane
    y_cats = np.zeros(5000)
    y_dogs = np.ones(5000)
    y_train_fully_connected = np.concatenate((y_cats, y_dogs))

    y_cats = np.zeros(1000)
    y_dogs = np.ones(1000)
    y_test_fully_connected = np.concatenate((y_cats, y_dogs))

    # seleziono 960 campioni casuali per l'addestramento
    generator = np.random.default_rng(seed=5)
    idx = generator.integers(low=0, high=10000, size=960)
    y_train_cut = y_train_fully_connected[idx]
    x_train_cut = x_train_fully_connected[idx]

    # downsample delle immagini da 32x32 a 8x8
    x_train_downsampled_list = []
    for image in x_train_cut:
        downsampled = ndimage.zoom(image, .25)
        x_train_downsampled_list.append(downsampled)
    x_train_downsampled = np.array(x_train_downsampled_list).reshape(960, 64)

    x_test_downsampled_list = []
    for image in x_test_fully_connected:
        downsampled = ndimage.zoom(image, .25)
        x_test_downsampled_list.append(downsampled)
    x_test_downsampled = np.array(x_test_downsampled_list).reshape(2000, 64)

    # Train e test delle reti
    mse_train, mse_test, accuracy_train, accuracy_test = train_and_test()
    # Stampa dei grafici
    print_graphics(mse_train, mse_test, accuracy_train, accuracy_test)

