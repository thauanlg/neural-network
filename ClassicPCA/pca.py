"""
 This script will implement the principal components analysis(PCA) using library for calculating eigenvalues and eigenvectors,
 used by PCA. After that, I've used the Iris dataset to analyse if PCA change significally, that is, if we execute the knn over
 iris dataset, before and after, will there be a difference?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

"""
- Parameters:
    data: numpy matrix where the rows indicates the differents instances and cols the features
    keepRate: the desired rate you want to keep in dataset
- Returns:
    the dataset containing the news features and its values after apply the pca function
"""
def pca(data, keepRate=0.75):
    # normalizing the data
    meanAxis=np.mean(data, axis=0)
    stdAxis=np.std(data, axis=0)
    normData=(data-meanAxis)/stdAxis
    
    # getting the eigenvalues and eigenvector of the covariance matrix and sorting it
    covMatrix=np.cov(normData, rowvar=False)
    eiValue, eiVector = np.linalg.eig(covMatrix)
    decIdx=np.argsort(eiValue)[::-1]
    eiValue, eiVector=eiValue[decIdx], eiVector[:,decIdx]
    
    normConst=np.sum(eiValue)
    rate=0.0
    numAxisKeep=0
    # getting the necessary number os axis to keep the desired rate of data
    while(rate<keepRate):
        rate+=eiValue[numAxisKeep]/normConst
        numAxisKeep+=1
    eiValue, eiVector=eiValue[:numAxisKeep], eiVector[:,:numAxisKeep]
    
    return (normData.dot(eiVector), rate)
"""
- Parameters:
    oldDf: specify the old dataframe used which contain all the features of iris dataset
    newDf: specify the new dataframe used which contain only features returned by pca function
- Returns:
    a dictionary containing the old and new accuracy using only a 5nn(knn) with 25% for testing and 75% for training
"""
def measureAccuracy(oldDf, newDf):
    ret={}
    # split the dataset for training and testing for olds features
    x_train, x_test, y_train, y_test = train_test_split(oldDf.iloc[:,:4].values, oldDf["target"].values, test_size=0.25)
    knn = KNeighborsClassifier()
    knn.fit(X=x_train, y=y_train)
    ret["oldAcc"]=knn.score(X=x_test, y=y_test)
    
    # split the dataset for training and testing for news features
    x_train, x_test, y_train, y_test = train_test_split(newDf.iloc[:,:4].values, newDf[2].values, test_size=0.25)
    knn = KNeighborsClassifier()
    knn.fit(X=x_train, y=y_train)
    knn.score(X=x_test, y=y_test)
    ret["newAcc"]=knn.score(X=x_test, y=y_test)
    
    return ret

# read iris dataset and run pca
sns.set()
iris=load_iris()
newData,_=pca(iris.data)

oldDf=pd.DataFrame(np.c_[iris.data, iris.target], columns=np.r_[np.asarray(iris.feature_names), np.asarray(["target"])])
newDf=pd.DataFrame(np.c_[newData, iris.target])

# ploting old and new dataframes
ax=sns.pairplot(oldDf, vars=oldDf.columns[:4], hue="target", height=2)
ax.fig.suptitle("Todos os atributos")
plt.show()

ax=sns.pairplot(newDf, vars=newDf.columns[:2], hue=2, height=3)
ax.fig.suptitle("Somente dois atributos")
plt.show()

measure=measureAccuracy(oldDf, newDf)

print("Acurácia usando todos os todos atributos e usando apenas os atributos retornados pelo PCA")
print("Todos: {:.2f}\nPCA: {:.2f}".format(measure["oldAcc"], measure["newAcc"]))
