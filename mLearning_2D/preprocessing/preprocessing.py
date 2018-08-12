# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
def import_data(csv):
    dataset = pd.read_csv(csv)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values
    return X,y

# Taking care of missing data
def fix_missing(X):
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    return X

# Encoding categorical data
def categorical_encode(data, independent=True):
    # Encoding the Independent Variable
    # [string1,string2,string3] --> [0,1,2]
    from sklearn.preprocessing import LabelEncoder
    if independent:
        labelencoder_data = LabelEncoder()
        data[:, 0] = labelencoder_data.fit_transform(data[:, 0])

        if equal:
            from sklearn.preprocessing import OneHotEncoder
            # Prevent machine from thinking one category is greater than
            # another
            # [0,1,2] --> [[1,0,0],[0,1,0],[0,0,1]]
            # first column --> France, second column --> Germany, third column --> Spain
            onehotencoder = OneHotEncoder(categorical_features = [0])
            data = onehotencoder.fit_transform(data).toarray()

    else:
        # Encoding the Dependent Variable
        # Dependent variable doesn't need OneHotEncoder
        # ['No','Yes'] --> [0,1]
        labelencoder_data = LabelEncoder()
        data = labelencoder_data.fit_transform(data)

    return data

# Split dataset into training and test sets
def create_sets(X,y):
        from sklearn.model_selection import train_test_split

        return train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
def feature_scale(X_train,X_test):
        # Put columns in same scale so one feature doesn't 
        # dominate another

        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        return X_train, X_test

