import pytest
import mLearning_2D.preprocessing.preprocessing as prep

Data = './Data.csv'

# Test import
X,y = prep.import_data(Data)

# Test fix_missing
X = prep.fix_missing(X)

# Test categorical_encode
X = prep.categorical_encode(X,True,True)
y = prep.categorical_encode(y,False)

# Test create_sets
X_train,X_test,Y_train,Y_test = prep.create_sets(X,y)

# Test feature_scale
X_train,X_test = prep.feature_scale(X_train,X_test)


