import pytest
import mLearning_2D.preprocessing.preprocessing as prep

Data = './Data.csv'

# Test import
X,y = prep.import_data(Data,-1,3)

# Test fix_missing
X = prep.fix_missing(X,1,3)

# Test categorical_encode
X = prep.categorical_encode(X,idx=0)
y = prep.categorical_encode(y,independent=False)

# Test create_sets
X_train,X_test,Y_train,Y_test = prep.create_sets(X,y,size=0.2)

# Test feature_scale
X_train,X_test = prep.feature_scale(X_train,X_test)


