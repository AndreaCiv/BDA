from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot as plt



def train_and_test():
    initializer = tf.keras.initializers.GlorotUniform(seed=5)
    epochs = 6000

    n_unit_to_check = [10, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 13000, 15000, 20000, 40000, 60000]

    mse_train =[]
    mse_test =[]
    accuracy_train=[]
    accuracy_test=[]

    for neurons in n_unit_to_check:

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(130108,)),
            tf.keras.layers.experimental.RandomFourierFeatures(
            output_dim=neurons,
            kernel_initializer=initializer,
            trainable = False),
            tf.keras.layers.Dense(units=20, kernel_initializer=initializer, activation='softmax')])

        optimizer = tf.keras.optimizers.experimental.SGD(
            learning_rate=0.001,
            momentum=0.95)

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse', 'accuracy'])

        history = model.fit(x=x_train_cut, y=y_train_one_hot, epochs=epochs, verbose=1)

        evaluation = model.evaluate(x=x_test, y=y_test_one_hot)

        mse_train.append(history.history['mse'][epochs-1])
        mse_test.append(evaluation[0])
        accuracy_train.append(history.history['accuracy'][epochs-1])
        accuracy_test.append(evaluation[2])

    return mse_train, mse_test, accuracy_train, accuracy_test

def print_graphics(mse_train, mse_test, accuracy_train, accuracy_test):
    # Lista dei numeri di parametri della rete utilizzati per le etichette dei grafici
    n_of_params = [10, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 13000, 15000, 20000, 40000, 60000]

    plt.figure(figsize=(6, 4))
    plt.title("Squared Loss")
    plt.plot(n_of_params, mse_train, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, mse_test, linewidth=2.0, color='blue', label="Test")
    plt.legend()
    plt.xscale('log')

    plt.show()

    plt.figure(figsize=(6, 4))
    plt.title("0-1 Loss")
    plt.plot(n_of_params, (1 - np.array(accuracy_train)) * 100, linewidth=2.0, color='orange', label="Train")
    plt.plot(n_of_params, (1 - np.array(accuracy_test)) * 100, linewidth=2.0, color='blue', label="Test")
    plt.legend()
    plt.xscale('log')

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

    mse_train, mse_test, accuracy_train, accuracy_test = train_and_test()
    print_graphics(mse_train, mse_test, accuracy_train, accuracy_test)



