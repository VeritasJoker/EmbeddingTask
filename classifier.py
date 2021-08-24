from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

def Scale(data): # Scales x
    scaler = preprocessing.StandardScaler().fit(data) # scaler
    data = scaler.transform(data) # scale training data
    return data

def LogReg(train_x, train_y, test_x, test_y, iter): # LOGISTIC
    train_x = Scale(train_x) # scale training data
    test_x = Scale(test_x) # scale training data
    logreg = LogisticRegression(penalty = "none", solver = "saga", max_iter = iter).fit(train_x, train_y)
    return (logreg.score(test_x, test_y)) # return accuracy

def ElasticNet(train_x, train_y, test_x, test_y, iter): # LOGISTIC ELASTIC PENALTY
    train_x = Scale(train_x) # scale training data
    test_x = Scale(test_x) # scale training data
    logreg = LogisticRegression(penalty = "elasticnet", l1_ratio = 0.5, solver = "saga", max_iter = iter).fit(train_x, train_y)
    return (logreg.score(test_x, test_y)) # return accuracy

def AdaBoost(train_x, train_y, test_x, test_y, n_est): # ADABOOST
    ada = AdaBoostClassifier(n_estimators = n_est).fit(train_x, train_y) # adaboost
    return (ada.score(test_x,test_y)) # return accuracy

def RandForest(train_x, train_y, test_x, test_y): # RANDOM FOREST
    forest = RandomForestClassifier(max_depth = 3).fit(train_x, train_y) # build forest
    return (forest.score(test_x,test_y)) # return accuracy



# Batch Sentence Embedding
print("Loading Data...\n")
train_x1 = np.loadtxt('train_x_batch.txt')
train_y1 = np.loadtxt('train_y.txt')
test_x1 = np.loadtxt('test_x_batch.txt')
test_y1 = np.loadtxt('test_y.txt')
train_x1, train_y1 = SMOTE().fit_resample(train_x1, train_y1) # SMOTE

print("Classifiers for Batch Sentence Embedding:")
print("Logistic Regression Accuracy: ",LogReg(train_x1, train_y1, test_x1, test_y1, 500)) # accuracy: 0.7731
print("Logistic Elastic Net Accuracy: ",ElasticNet(train_x1, train_y1, test_x1, test_y1, 500)) # accuracy: 0.77275
print("Ada Boost Accuracy: ", AdaBoost(train_x1, train_y1, test_x1, test_y1, 500))
print("Random Forest Accuracy:", RandForest(train_x1, train_y1, test_x1, test_y1)) # accuracy: 
print("\n\n")

# Paragraph Embedding
print("Loading Data...\n")
train_x2 = np.loadtxt('train_x_block.txt')
train_y2 = train_y1
test_x2 = np.loadtxt('test_x_block.txt')
test_y2 = test_y1
train_x2, train_y2 = SMOTE().fit_resample(train_x2, train_y2) # SMOTE

print("Classifiers for Paragraph Embedding:")
print("Logistic Regression Accuracy: ",LogReg(train_x2, train_y2, test_x2, test_y2, 500))
print("Logistic Elastic Net Accuracy: ",ElasticNet(train_x2, train_y2, test_x2, test_y2, 500))
print("Ada Boost Accuracy: ", AdaBoost(train_x2, train_y2, test_x2, test_y2, 500))
print("\n\n")

# Sentence Transformer
print("Loading Data...\n")
train_x3 = np.loadtxt('s_train_x_batch.txt')
train_y3 = np.loadtxt('s_train_y.txt')
test_x3 = np.loadtxt('s_test_x_batch.txt')
test_y3 = np.loadtxt('s_test_y.txt')
train_x3, train_y3 = SMOTE().fit_resample(train_x3, train_y3) # SMOTE

print("Classifiers for Sentence Transformers:")
print("Logistic Regression Accuracy: ",LogReg(train_x3, train_y3, test_x3, test_y3, 1000)) # accuracy: 0.784
print("Logistic Elastic Net Accuracy: ",ElasticNet(train_x3, train_y3, test_x3, test_y3, 1000)) # accuracy: 0.799
print("Ada Boost Accuracy: ", AdaBoost(train_x3, train_y3, test_x3, test_y3, 500)) # accuracy: 0.916
print("Random Forest Accuracy:", RandForest(train_x3, train_y3, test_x3, test_y3)) # accuracy: 0.906
print("\n\n")