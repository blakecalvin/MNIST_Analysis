import Mnist

    #    arg1 = model
    #        1 = KNN
    #        2 = Outliers
    #    arg2 = Distance algorithm
    #        1 = Manhattan
    #        2 = Euclidian
    #        3 = Minkowski (p=3)
    #    arg3 = k
for i in range(2,10):
    Mnist.main(arg1=2,arg2=2,arg3=i)
    print(str(i)+": complete.")
