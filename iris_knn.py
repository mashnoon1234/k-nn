from sklearn import datasets
from knn import KNN 

data = datasets.load_iris()
trainingData = data['data']
trainingLabels = data['target']

classifier = KNN(k=5)

classifier.classify(trainingData, trainingLabels)

