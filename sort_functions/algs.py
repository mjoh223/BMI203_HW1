import numpy as np

def pointless_sort(x):
    """
    This function always returns the same values to show how testing
    works, check out the `test/test_alg.py` file to see.
    """
    return np.array([1,2,3])

def bubblesort(x):
    """
    Describe how you are sorting `x`
    """
    for passnum in range(len(x)-1,0,-1): #count backwards from size of list
        for i in range(passnum):
            if x[i]>x[i+1]:#conditional
                x[i], x[i+1] = x[i+1], x[i] #two assignment
    return x

def quicksort(x):
    quickSortHelper(x,0,len(x)-1)
    return x

def quickSortHelper(x, first, last):
    if first < last:

        splitpoint = partition(x, first, last)

        quickSortHelper(x, first, splitpoint-1)
        quickSortHelper(x, splitpoint+1, last)

def partition(x, first, last):
    pivotvalue = x[first]

    leftmark = first+1
    rightmark = last

    done = False
    while not done:

        while leftmark <= rightmark and x[leftmark] <= pivotvalue:
            leftmark = leftmark + 1

        while x[rightmark] >= pivotvalue and rightmark >= leftmark:
            rightmark = rightmark -1

        if rightmark < leftmark:
            done = True
        else:
            temp = x[leftmark]
            x[leftmark] = x[rightmark]
            x[rightmark] = temp

    temp = x[first]
    x[first] = x[rightmark]
    x[rightmark] = temp
    return rightmark
