from mnist import MNIST
from sklearn import neighbors
from time import sleep
import numpy as np
import warnings

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def scikit(k, trainImages, trainLabels, testImages, testLabels, algo):
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

    printProgressBar(0, len(X2), prefix = ' Progress:', suffix = 'Complete', length = 50)
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
    print()
    print('--- MNIST data prediction using k-Nearest Neighbors Classifier ---')

    condition = True
    while condition:
        print()
        k = int(input('Enter k value: '))
        print()
        print('Distance algorithms:')
        print(' 1. Manhattan')
        print(' 2. Euclidian')
        print(' 3. Minkowski')
        distAlgorithm = int(input('Enter corresponding # for algorithm: '))
        print()

        errorRate = kNN(k, distAlgorithm)

        export(distAlgorithm, k, errorRate)

        print()
        print('What do you want to do?')
        print(' 1. Exit')
        print(' 2. Run more tests')
        exit = int(input('Enter corresponding #: '))

        if exit == 1:
            condition = False
            print()

main()
