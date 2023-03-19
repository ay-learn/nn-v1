import multiprocessing as mp

import matplotlib.pyplot as plt


def plot(X_train, y_train, X_test, y_test, predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
    plt.scatter(X_test, y_test, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(
            X_test,
            predictions,
            c="r",
            s=4,
            label="Predictions",
        )

    plt.legend(prop={"size": 14})
    # plt.show(block=True)
    plt.show()


def plot_predictions(X_train, y_train, X_test, y_test, predictions=None):
    p = mp.Process(target=plot, args=(X_train, y_train, X_test, y_test, predictions))
    p.start()
    p.join(0.1)  # Wait for 0.1 seconds for the process to start
