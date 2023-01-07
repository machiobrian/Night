from matplotlib import pyplot as plt

def plot_prediction(
    train_data = X_train,
    train_labels= y_train,
    test_data= X_test,
    test_labels = y_test,
    predictions=y_pred
):

    plt.figure(figsize=(5,5))
    #plots the training data
    plt.scatter(train_data, train_labels)
    #plot the models predictions
    plt.scatter(test_data, predictions)
    #plots the testing data
    plt.scatter(test_data, test_labels)
    #show the legend
    plt.legend();