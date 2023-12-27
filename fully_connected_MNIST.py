import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def train_and_test(with_weights_reuse):
    # Inizializzatore Glorot per i pesi
    initializer = tf.keras.initializers.GlorotUniform(seed=5)

    # numero neuroni da avere nello strato hidden per avere rispettivamente
    # 3k, 8k, 10k, 20k, 35k, 38k, 40k, 43k, 50k, 75k, 100k, 200k, 300k, 500k, 800k parametri
    n_unit_to_check = [10, 11, 13, 26, 44, 48, 50, 54, 63, 95, 126, 252, 378, 630, 1007]

    # Arrays per salvare i risultati delle varie reti
    mse_train = []
    mse_test = []
    accuracy_train = []
    accuracy_test = []

    # Numero di epoche di addestramento
    epochs = 6000

    # Ciclo for di addestramento e test per le varie reti
    for neurons in n_unit_to_check:
        # Definizione del modello
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(units=neurons, initializer=initializer, activation='relu'),
            tf.keras.layers.Dense(units=10, initializer=initializer, activation='softmax'),
        ])
        # Definizione dell'ottimizzatore
        optimizer = tf.keras.optimizers.experimental.SGD(
            learning_rate=0.001,
            momentum=0.95
        )
        model.summary()
        # Compilazione del modello
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse', 'accuracy']

        )
        # inizializzazione reti piu grandi con quella più piccola
        if True and neurons != 10:
            index = n_unit_to_check.index(neurons)
            neurons_last_cycle = n_unit_to_check[index - 1]
            file_path = "fully_connected_" + str(neurons_last_cycle) + "_neurons.h5"
            model.load_weights(file_path, skip_mismatch=True, by_name=True)

        history = model.fit(x=x_train_norm, y=y_train_one_hot, epochs=epochs)
        # Salvataggio dei pesi della rete
        if with_weights_reuse:
            model.save_weights("fully_connected_" + str(neurons) + "_neurons.h5")
        evaluation = model.evaluate(x=x_test_norm, y=y_test_one_hot)

        mse_train.append(history.history['mse'][epochs - 1])
        mse_test.append(evaluation[0])
        accuracy_train.append(1 - history.history['accuracy'][epochs - 1])
        accuracy_test.append(1 - evaluation[2])

    return mse_train, mse_test, accuracy_train, accuracy_test


def print_graphics(mse_train, mse_test, accuracy_train, accuracy_test):
    # Lista dei numeri di parametri della rete utilizzati per le etichette dei grafici
    n_of_params = ["3k", "8k", "10k", "20k", "35k", "38k", "40k", "43k", "50k", "75k", "100k", "200k", "300k", "500k",
                   "800k"]

    plt.figure(figsize=(6, 4))
    plt.title("Squared Loss")
    plt.plot(n_of_params, mse_train, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, mse_test, linewidth=2.0, color='blue', label="Test")
    plt.legend()
    plt.axvline(x="40k")

    plt.show()

    plt.figure(figsize=(6, 4))
    plt.title("0-1 Loss")
    plt.plot(n_of_params, accuracy_train, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, accuracy_test, linewidth=2.0, color='blue', label="Test")
    plt.legend()
    plt.axvline(x="40k")

    plt.show()


if __name__ == "__main__":

    # Caricamento dataset MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Generazione di 4000 indici casuali per la selezione delle immagini di train
    generator = np.random.default_rng(seed=5)
    idx = generator.integers(low=0, high=x_train.shape[0], size=4000)

    # Normalizzazione delle immagini in un range tra 0 e 1
    x_train_norm = (x_train / 255).reshape(x_train.shape[0], -1)[idx, :]
    x_test_norm = (x_test / 255).reshape(x_test.shape[0], -1)
    y_train_cut = y_train[idx]

    # Stampa immagine di prova con etichetta
    data = np.array(1 - x_train_norm[1].reshape(28, 28))
    plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    plt.show()
    print(y_train_cut[1])

    # le etichette ora sono numeri da 0 a 9, se utilizziamo la softmax nell'ultimo
    # layer bisogna utilizzare un array one-hot-encoded
    y_train_one_hot = np.eye(10)[y_train_cut]
    y_test_one_hot = np.eye(10)[y_test]

    print(y_train_one_hot[1])

    # Impostare il parametro with_weights_reuse a False se non si volgiono utilizzare i pesi delle reti
    # più piccole per inizializzare quelli delle reti più grandi
    mse_train, mse_test, accuracy_train, accuracy_test = train_and_test(with_weights_reuse=True)
    print_graphics(mse_train, mse_test, accuracy_train, accuracy_test)


