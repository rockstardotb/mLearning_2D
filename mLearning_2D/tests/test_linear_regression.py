import mLearning_2D.preprocessing.preprocessing as prep
import mLearning_2D.regression.linear_regression as linear

Data = './Salary_Data.csv'

# Test import
X,y = prep.import_data(Data,-1,1)

# Test create_sets
X_train,X_test,Y_train,Y_test = prep.create_sets(X,y,size=1/3)

regressor = linear.train(X_train,Y_train)

linear.visualize_train(X_train,Y_train,regressor)
linear.visualize_test(X_test,Y_test,regressor)
