"""
5. Machine learning
    5.1. Build a digit recognition script
    This script adopt XGBoost algorithm.
    step 1: Unzip digits.zip
    step 2: Read training & testing dataset
    step 3: Process data
    step 4: Train model and output score

    Estimated Running Time: 4526s
    Test Accuracy: 97.2%

    Future works:
        1. More conventional machine learning methods can be adopted, such as lightgbm and catboost. Sometime, regression
    methods with the help of bagging and boosting, ending with an optimized rounder function, can lead to better accuracy.
        2. Deep learning methods can be adopted, similar to the code for section 5.2, where CNN is developed.

    Note: Please place this script within the same folder of digits.zip

"""

# importing library
import time
import numpy as np
from zipfile import ZipFile
import os
import cv2
import xgboost as xgb #Import XGBoost package
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def unzip(path='digits.zip'):
    with ZipFile(path, 'r') as zip_parent:
        # extracting all the files
        print('Extracting all the files now...')
        zip_parent.extractall()
        print('Done')


def read_data(path):
    data = []
    label = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Read image in Grayscale
            data.append(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE))
            label.append(float(root.split('\\')[-1]))
    return data, label


def process_data(X, y, X_test):
    """
    This function reshape train + test data; split training + val data: 90% vs 10%; augments data;
    :param X: Training data set
    :param y: Training data labels
    :param X_test: Testing data set
    :return: processed data sets
    """

    # data augmentation
    # datagen = ImageDataGenerator(
    #     rotation_range=15,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    # )
    # datagen.fit(X[:,:,:,np.newaxis])

    # Flatten from 3-D array to 2-D array
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    X_test_new = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_test_new = sc.fit_transform(X_test_new)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=2)



    # Making sure that the values are float so that we can get decimal points after division
    X_train = X_train.astype('float32')
    X_test_new = X_test_new.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    X_train /= 255
    X_test_new /= 255

    return X_test_new, X_train, X_valid, y_train, y_valid


def xgb_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    This function run XGBoost algorithm
    :param X_train: train data
    :param y_train: train label
    :param X_valid: validate data
    :param y_valid: validate label
    :param X_test: test data
    :param y_test: test label
    :return: accuracy score
    """

    # Reading in data to DMatrix, an internal data structure used by XGBoost
    dmatTrain = xgb.DMatrix(X_train, label=y_train)
    dmatVal = xgb.DMatrix(X_valid, label=y_valid)
    dmatTest = xgb.DMatrix(X_test, label=y_test)

    param = {'booster': 'gbtree',  # gbtree is the default tree based model
             'verbosity': 1,  # Specify the amount of information to receive in an error message
             'max_depth': 7,  # Specify maximum tree depth #6
             'eta': 0.08, # Step size shrinkage used in update to prevent overfitting
             'objective': 'multi:softmax',  # Specify multiclass classification
             'num_class': 10,  # Specify number of class labels
             'subsample': 0.8,
             'colsample_bytree': 0.8,
             'alpha': 3,
             'lambda': 2,
             'eval_metric': 'merror'} # Multiclass classification error rate = #(wrong cases)/#(all cases)

    num_round = 550  # Specify the number of rounds for boosting
    early_stopping = 50

    eval_list = [(dmatTrain, "train"), (dmatVal, "validation")]

    print('Start training ...')
    start = time.time()
    bst = xgb.train(param, dmatTrain, num_round, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)
    print('training consumed: ', time.time() - start)
    train_pred = bst.predict(dmatTrain)  # Predict classes in the train data using the trained model

    print('Train Accuracy:', sum(train_pred == y_train) / len(train_pred))  # Calculate train accuracy
    test_pred = bst.predict(dmatTest)  # Predict classes in test data using the trained model
    print("Test Accuracy:", sum(test_pred == y_test) / len(test_pred))



if __name__ == '__main__':
    # 1. Unzip the digits.zip file
    unzip()

    # 2. Read training & testing dataset
    X, y = read_data(path='./data/mnist/training')
    X, y = np.array(X), np.array(y)
    X_test, y_test = read_data(path='./data/mnist/testing')
    X_test, y_test = np.array(X_test), np.array(y_test)

    # 3. process data
    X_test, X_train, X_valid, y_train, y_valid = process_data(X, y, X_test)

    # 4. train model and output score
    xgb_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test)







