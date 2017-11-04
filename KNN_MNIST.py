from mnist import MNIST
from sklearn import neighbors
import numpy as np
import sys
import warnings

def scikit(k=5, trainImages, trainLabels, testImages, testLabels, algo=2):
    warnings.simplefilter("ignore", category=DeprecationWarning)

    #training
    Y = np.array(trainLabels)
    X = np.array(trainImages)
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform', p=algo)
    clf.fit(X, Y)

    #test
    Y2 = np.array(testLabels)
    X2 = np.array(testImages)

    #var
    wrong = 0
    count = 0

    #predict and compare to known label
    for x in X2:
        p = clf.predict(x)
        compare = (p == Y2[count])
        if compare == False: wrong += 1
        count += 1
        printProgressBar(count, len(X2), prefix = ' Progress:', suffix = 'Complete', length = 50)
    errorRate = (wrong/count)*100

    return errorRate

def kNN(k, distAlgorithm):

    mndata = MNIST('samples')

    #training data
    images, labels = mndata.load_training()
    #testing data
    images2, labels2 = mndata.load_testing()

    errorRate = 0.0

    if distAlgorithm == 1:
        errorRate = scikit(k, images, labels, images2, labels2, 1)
    elif distAlgorithm == 2:
        errorRate = scikit(k, images, labels, images2, labels2, 2)
    elif distAlgorithm == 3:
        errorRate = scikit(k, images, labels, images2, labels2, 3)

    return errorRate

def export(distAlgorithm, k, errorRate):
    name = ''
    if distAlgorithm == 1:
        name = 'Manhattan'
    elif distAlgorithm == 2:
        name = 'Euclidian'
    elif distAlgorithm == 3:
        name = 'Minkowski'

    f = open('MnistData.txt', 'a')
    f.write('{:^15}|{:^5d}|{:>8.2f} %\n'.format(name, k, errorRate))

def main():
    k = int(input('Enter k value: '))
    print('Distance algorithms:')
    print(' 1. Manhattan')
    print(' 2. Euclidian')
    print(' 3. Minkowski')
    distAlgorithm = int(input('Enter corresponding # for algorithm: '))
    errorRate = kNN(k, distAlgorithm)
    export(distAlgorithm, k, errorRate)

if __name__ == "__main__":
    main()
