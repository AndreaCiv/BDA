from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot as plt



def train_and_test():
    initializer = tf.keras.initializers.GlorotUniform(seed=5)
    epochs = 6000
    # numero neuroni da avere nello strato hidden per avere rispettivamente
    # [989, 2078, 3046, 4014, 5103, 6071, 7039, 8007, 9096, 10064, 11032, 12121, 14541, 20107, 40072, 60037]
    n_unit_to_check = [8, 17, 25, 33, 42, 50, 58, 66, 75, 83, 91, 100, 120, 166, 331, 496]

    mse_train = []
    mse_test = []
    accuracy_train = []
    accuracy_test = []

    for neurons in n_unit_to_check:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(100,)),
            tf.keras.layers.experimental.RandomFourierFeatures(
                output_dim=neurons,
                kernel_initializer=initializer,
                trainable=False),
            tf.keras.layers.Dense(units=20, activation='softmax'), ])

        optimizer = tf.keras.optimizers.experimental.SGD(
            learning_rate=0.001,
            momentum=0.95)

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse', 'accuracy'])

        history = model.fit(x=x_train_good, y=y_train_one_hot, epochs=epochs, verbose=1)

        evaluation = model.evaluate(x=x_test_good, y=y_test_one_hot)

        mse_train.append(history.history['mse'][epochs - 1])
        mse_test.append(evaluation[0])
        accuracy_train.append(history.history['accuracy'][epochs - 1])
        accuracy_test.append(evaluation[2])

    return mse_train, mse_test, accuracy_train, accuracy_test

def print_graphics(mse_train, mse_test, accuracy_train, accuracy_test):
    # Lista dei numeri di parametri della rete utilizzati per le etichette dei grafici
    n_of_params = [989, 2078, 3046, 4014, 5103, 6071, 7039, 8007, 9096, 10064, 11032, 12121, 14541, 20107, 40072, 60037]

    plt.figure(figsize=(6, 4))
    plt.title("Squared Loss")
    plt.plot(n_of_params, mse_train, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, mse_test, linewidth=2.0, color='blue', label="Test")
    plt.legend()

    plt.show()

    plt.figure(figsize=(6, 4))
    plt.title("0-1 Loss")
    plt.plot(n_of_params, (1 - np.array(accuracy_train)) * 100, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, (1 - np.array(accuracy_test)) * 100, linewidth=2.0, color='blue', label="Test")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    x, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, stratify=y, test_size=0.125
    )
    train_samples, n_features = x_train.shape
    n_classes = np.unique(y).shape[0]
    print(
        "Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i"
        % (train_samples, n_features, n_classes)
    )

    generator = np.random.default_rng(seed=5)
    idx = generator.integers(low=0, high=x_train.shape[0], size=10000)

    x_train_cut = x_train[idx, :]
    y_train_cut = y_train[idx]

    y_train_one_hot = np.eye(20)[y_train_cut]
    y_test_one_hot = np.eye(20)[y_test]

    x_train_good = []
    x_test_good = []

    for vector in x_train_cut.toarray():
        vector = vector[:-7]
        chunks = [vector[x:x + 100] for x in range(0, vector.shape[0], 100)]
        vector_good = np.add.reduce(chunks)
        x_train_good.append(vector_good)

    for vector in x_test.toarray():
        vector = vector[:-7]
        chunks = [vector[x:x + 100] for x in range(0, vector.shape[0], 100)]
        vector_good = np.add.reduce(chunks)
        x_test_good.append(vector_good)

    x_train_good = np.array(x_train_good)
    x_test_good = np.array(x_test_good)

    mse_train, mse_test, accuracy_train, accuracy_test = train_and_test()
    print_graphics(mse_train, mse_test, accuracy_train, accuracy_test)



