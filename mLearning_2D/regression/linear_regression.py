# Fit linear regression to training set
def train(X_train, y_train):
    from sklearn.linear_model import LinearRegression

    # Train machine
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    return regressor

# Make a prediction using the trained model
def predict(X_test, regressor):

    # Test machine
    y_pred = regressor.predict(X_test)

    return y_pred

# Plot expected vs predicted
def visualize_train(X_train, y_train, regressor):

    import matplotlib.pyplot as plt

    plt.scatter(X_train, y_train, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'green')
    plt.title('Y vs X (Training Set'))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Plot expected vs predicted
def visualize_test(X_test, y_test, regressor):

    import matplotlib.pyplot as plt

    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_test, predict(X_test), color = 'green')
    plt.title('Y vs X (Training Set'))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
