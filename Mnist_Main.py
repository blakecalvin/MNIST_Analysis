import Mnist

'''
    arg1 = model
        1 = KNN
        2 = SOM
    arg2 = Distance algorithm
        1 = Manhattan
        2 = Euclidian
        3 = Minkowski (p=3)
    arg3 = k
'''
for x in range(0, 10):
    Mnist.main(1, 2, 5)
