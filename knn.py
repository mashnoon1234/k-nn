import numpy as np
import os
import threading
import time
import logging
import PriorityQueue

logging.basicConfig(level=logging.DEBUG,
format='[%(levelname)s] (%(threadName)-9s) %(message)s',)

class KNN:
    def __init__(self, k):
        self.__mean = 0
        self.__standard_deviation = 0
        self.__k = k
        self.__cpu = os.cpu_count()
        self.__lock = threading.Lock()
        self.__best_queue = PriorityQueue(maxsize=k)

    def classify(self, trainingData, trainingLabels):
        self.__mean = np.mean(trainingData, axis=0)
        self.__standard_deviation = np.std(trainingData, axis=0)
        trainingData = self.__normalize(trainingData)
        self.__launch_workforce(trainingData, trainingLabels)
        print(trainingData)
        print(self.__mean)
        print(self.__standard_deviation)

    def __normalize(self, trainingData):
        return (trainingData - self.__mean) / self.__standard_deviation

    def __euclidean_distance(self, record1, record2):
        return np.sqrt(np.sum((record1 - record2) ** 2))
        
    def __launch_workforce(self, trainingData, trainingLabels):
        self.__find_best(trainingData, trainingLabels)

    def __find_best(testData, trainingData, trainingLabels):
        for i in range(self.__k)
            self.__best_queue.put(self.__euclidean_distance(testData, trainingData[i]), trainingLabels[i]])


