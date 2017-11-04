from mnist import MNIST
from sklearn import neighbors
from sompy.sompy import SOM
import numpy as np
import sys
import warnings

def scikit_KNN(k, trainImages, trainLabels, testImages, testLabels, algo):
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
    errorRate = (wrong/count)*100

    return errorRate

def sompy_SOM(k, trainImages, trainLabels, testImages, testLabels):
    warnings.simplefilter("ignore", category=DeprecationWarning)

    #training
    Y = np.array(trainLabels)
    X = np.array(trainImages)
    clf = sompy.SOM(data=X, neighborhood='Gaussian', mapsize=1000)
    clf.data_labels(Y)

    #test
    Y2 = np.array(testLabels)
    X2 = np.array(testImages)

    #var
    wrong = 0
    count = 0

    #predict and compare to known label
    for x in X2:
        p = clf.predict(x_test=x, k=k)
        compare = (p == Y2[count])
        if compare == False: wrong += 1
        count += 1
    errorRate = (wrong/count)*100

    return errorRate

def loadData(model, k, distAlgorithm):

    mndata = MNIST('samples')

    #training data
    images, labels = mndata.load_training()
    #testing data
    images2, labels2 = mndata.load_testing()

    errorRate = 0.0

    if model == 1:
        errorRate = scikit_KNN(k, images, labels, images2, labels2, distAlgorithm)
    elif model == 2:
        errorRate = sompy_SOM(k, images, labels, images2, labels2)

    return errorRate

def export(model, distAlgorithm, k, errorRate):
    if model == 1:
        name = ''
        if distAlgorithm == 1:
            name = 'Manhattan'
        elif distAlgorithm == 2:
            name = 'Euclidian'
        elif distAlgorithm == 3:
            name = 'Minkowski'
        f = open('MnistData.txt', 'a')
        f.write('{:^15}|{:^5d}|{:>8.2f} %\n'.format(name, k, errorRate))
    elif model == 2:
        name = 'Gaussian'
        f = open('MnistData_SOM.txt', 'a')
        f.write('{:^15}|{:^5d}|{:>8.2f} %\n'.format(name, k, errorRate))

    #    arg1 = model
    #        1 = KNN
    #        2 = SOM
    #    arg2 = Distance algorithm
    #        1 = Manhattan
    #        2 = Euclidian
    #        3 = Minkowski (p=3)
    #    arg3 = k
def main(arg1, arg2=0, arg3=0):
    model = arg1
    algo = arg2
    k = arg3

    errorRate = loadData(model, k, algo)

    export(model, algo, k, errorRate)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
