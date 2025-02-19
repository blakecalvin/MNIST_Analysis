from mnist import MNIST
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import numpy as np
import sys
import warnings
import math

# Chebyshev algorithm to simplify the calculation of different distance algorithms.
# By changing the value of p the Manhattan, Euclidian, and Minkowski algorithms may be calculated.
# p=1 Manhattan
# p=2 Euclidian
# p=3 Minkowski
def Chebyshev(a, b, p):
    summation = 0
    for i in range(0,len(a)):
        summation += pow(abs(b[i]-a[i]), p)
    return pow(summation, float(1/p))

def scikit_MLP(k, trainImages, trainLabels, testImages, testLabels):
    warnings.simplefilter("ignore", category=DeprecationWarning)

    X = np.array(trainImages)
    Y = np.array(trainLabels)

    clf = MLPClassifier(hidden_layer_sizes=(k,))
    clf.fit(X, Y)

    #test
    Y2 = np.array(testLabels)
    X2 = np.array(testImages)

    wrong = 0
    count = 0

    #predict and compare to known label
    for x in X2:
        p = clf.predict(x)
        print(p)
        compare = (p == Y2[count])
        if compare == False: wrong += 1
        count += 1
    errorRate = (wrong/count)*100

    return errorRate

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

def outliers(k, trainImages, trainLabels, testImages, testLabels, distAlgorithm, Percentage=5):
    warnings.simplefilter("ignore", category=DeprecationWarning)

    #training
    Y = np.array(trainLabels)
    X = np.array(trainImages)

    xOptimized = []
    yOptimized = []

    sort = []
    for i in range(0,10): #append 10 empty arrays for the 10 possible digits.
        sort.append([])
    for i in range(0,60000): #sort data in to respected subarrays
        test = Y[i]
        test = {
            0:sort[0].append(X[i]),
            1:sort[1].append(X[i]),
            2:sort[2].append(X[i]),
            3:sort[3].append(X[i]),
            4:sort[4].append(X[i]),
            5:sort[5].append(X[i]),
            6:sort[6].append(X[i]),
            7:sort[7].append(X[i]),
            8:sort[8].append(X[i]),
            9:sort[9].append(X[i]),
        }

    for x in range(0,10):

        print("x:"+str(x))

        k_means = KMeans(n_clusters=1).fit(sort[x])
        centroid = k_means.cluster_centers_
        valFurthest = []
        dataFurthest = []

        for y in range(0,len(sort[x])):

            dist = Chebyshev(sort[x][y], centroid[0], distAlgorithm)

            if len(valFurthest)==0:
                valFurthest.append(dist)
                dataFurthest.append(sort[x][y])

            for z in range(0,len(valFurthest)):
                if dist > valFurthest[z]:
                    valFurthest.insert(z,dist)
                    dataFurthest.insert(z,sort[x][y])
                    break

        for i in range(0,int(len(valFurthest)*(Percentage/100))):
            xOptimized.append(dataFurthest[i])
            yOptimized.append(x)

    xOptimized = np.array(xOptimized)
    yOptimized = np.array(yOptimized) # export as new dataset

    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform', p=distAlgorithm)
    clf.fit(xOptimized,yOptimized)

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
        errorRate = outliers(k, images, labels, images2, labels2, distAlgorithm)
    elif model == 3:
        errorRate = scikit_MLP(k, images, labels, images2, labels2)

    return errorRate

def export(model, distAlgorithm, k, errorRate):
    #KNN
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
    #Outliers
    elif model == 2:
        name = ''
        if distAlgorithm == 1:
            name = 'Manhattan'
        elif distAlgorithm == 2:
            name = 'Euclidian'
        elif distAlgorithm == 3:
            name = 'Minkowski'
        f = open('MnistData_Outliers.txt', 'a')
        f.write('{:^15}|{:^5d}|{:>8.2f} %\n'.format(name, k, errorRate))

    elif model == 3:
        f = open('MnistData_MLP.txt', 'a')
        f.write('{:^16}|{:>8.2f} %\n'.format(k ,errorRate))

    #    arg1 = model
    #        1 = KNN
    #        2 = Outliers
    #    arg2 = Distance algorithm
    #        1 = Manhattan
    #        2 = Euclidian
    #        3 = Minkowski (p=3)
    #    arg3 = k
def main(arg1, arg2, arg3):
    model = arg1
    algo = arg2
    k = arg3

    print(model)

    errorRate = loadData(model, k, algo)

    export(model, algo, k, errorRate)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
