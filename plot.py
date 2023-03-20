import matplotlib.pyplot as plt


def plot_predictions(X_train, y_train, X_test, y_test, predictions=None):
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


def plot_model2_loss(epoch_count, loss_train, loss_test):
    plt.figure(figsize=(10, 7))
    plt.plot(epoch_count, loss_train,label="Training loss")
    plt.plot(epoch_count, loss_test,label="Test loss")
    # plt.legend(prop={"size": 14})
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
