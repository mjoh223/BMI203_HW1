import numpy as np

def pointless_sort(x):
    """
    This function always returns the same values to show how testing
    works, check out the `test/test_alg.py` file to see.
    """
    return np.array([1,2,3])

def bubblesort(x): #1 conditional and 2 assignments
    """
    Describe how you are sorting `x`
    """
    c = 0
    a = 0
    for passnum in range(len(x)-1,0,-1): #count backwards from size of list
        for i in range(passnum):
            if x[i]>x[i+1]: #conditional
                c += 1
                a += 2
                x[i], x[i+1] = x[i+1], x[i] #two assignments
    return [x,c,a]

def quicksort(x): #14 assignments and 5 conditionals
    c = 0
    a = 0
    ret = quickSortHelper(x,0,len(x)-1,c,a)
    return ret

def quickSortHelper(x, first, last,c,a):
    if first < last: #conditional
        c += 1
        a += 1
        splitpoint = partition(x, first, last, c, a) #assignment

        sorted_list, c, a = quickSortHelper(x, first, splitpoint-1, c, a)
        sorted_list, c, a = quickSortHelper(x, splitpoint+1, last, c, a)
    return [x,c,a]

def partition(x, first, last, a, c):
    a += 1
    pivotvalue = x[first] #assignment
    a += 1
    leftmark = first+1 #assignment
    a += 1
    rightmark = last #assignment

    done = False #assignment
    c += 1
    while not done:

        while leftmark <= rightmark and x[leftmark] <= pivotvalue: #two conditionals
            c += 2
            a += 1
            leftmark = leftmark + 1 #assignment

        while x[rightmark] >= pivotvalue and rightmark >= leftmark: #two conditionals
            c += 2
            a += 1
            rightmark = rightmark -1 #assignment

        if rightmark < leftmark: #conditional
            c += 1
            a += 1
            done = True #assignment
        else:
            a += 3
            temp = x[leftmark] #assignment
            x[leftmark] = x[rightmark] #assignment
            x[rightmark] = temp #assignment
    a += 3
    temp = x[first] #assignment
    x[first] = x[rightmark] #assignment
    x[rightmark] = temp #assignment
    return rightmark

quicksort(np.random.randint(20, size=20))
quicksort(np.random.randint(20, size=200))

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

timed = []
for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    x = np.random.randint(i, size=i)
    wrappedb = wrapper(bubblesort, x)
    tim = timeit.timeit(wrappedb, number=100)
    timed.append([i,"bubblesort", tim])
for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    x = np.random.randint(i, size=i)
    wrappedq = wrapper(quicksort, x)
    timq = timeit.timeit(wrappedq, number=100)
    timed.append([i,"quicksort", timq])

timed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
timed = pd.DataFrame(timed)
timed.columns = ["time", "algorithm", "speed"]
sns.lmplot(x="time", y="speed", col="algorithm",order=2,data=timed, ci=False)
np.polyfit(x=timed.iloc[:,0], y=timed.iloc[:,2], deg=3)

pd.DataFrame(timed)

df = sns.load_dataset("anscombe")
df
