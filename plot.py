import matplotlib.pyplot as plt


def plot_predictions(
    train_data=None,
    train_labels=None,
    test_data=None,
    test_labels=None,
    predictions=None,
):
    if train_data is None:
        train_data = X_train
    if train_labels is None:
        train_labels = y_train
    if test_data is None:
        test_data = X_test
    if test_labels is None:
        test_labels = y_test

    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(
            test_data.detach().numpy(),
            predictions.detach().numpy(),
            c="r",
            s=4,
            label="Predictions",
        )

    plt.legend(prop={"size": 14})
    plt.show(block=True)
