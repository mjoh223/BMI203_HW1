import numpy as np

def pointless_sort(x):
    """
    This function always returns the same values to show how testing
    works, check out the `test/test_alg.py` file to see.
    """
    return np.array([1,2,3])

def bubblesort(x): #1 conditional and 2 assignments
    c = 0
    a = 0
    for passnum in range(len(x)-1,0,-1): #count backwards from size of list
        for i in range(passnum):
            if x[i]>x[i+1]: #conditional
                c += 1
                a += 2
                x[i], x[i+1] = x[i+1], x[i] #two assignments
    return [x,c,a]
bubblesort(np.random.randint(20, size=20))

def quicksort(x): #14 assignments and 5 conditionals
    c = 0
    a = 0
    ret = quickSortHelper(x,0,len(x)-1,c,a)
    return ret

def quickSortHelper(x, first, last, c, a):
    if first < last: #conditional
        c += 1
        a += 1
        splitpoint = partition(x, first, last, c, a) #assignment

        sorted_list, c, a = quickSortHelper(x, first, splitpoint[0]-1, splitpoint[1], splitpoint[2])
        sorted_list, c, a = quickSortHelper(x, splitpoint[0]+1, last, splitpoint[1], splitpoint[2])
    return [x,c,a]

def partition(x, first, last, c, a):
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
    return [rightmark, c, a]

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
    lt, c, a = bubblesort(np.random.randint(i, size=i))
    timed.append([i,"bubblesort", tim, i**2, c, a])
for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    x = np.random.randint(i, size=i)
    wrappedq = wrapper(quicksort, x)
    timq = timeit.timeit(wrappedq, number=100)
    lt, c, a = quicksort(np.random.randint(i, size=i))
    timed.append([i,"quicksort", timq, i*np.log2(i), c, a])

timed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
timed = pd.DataFrame(timed)
timed.columns = ["n", "algorithm", "time", "expected", "conditionals", "assignments"]
pd.melt(timed)
sns.lmplot(x="n", y="time", col="algorithm",fit_reg=False,data=timed, ci=False)
sns.lineplot(x="n", y="expected", hue="algorithm",data=timed.iloc[10:])
np.polyfit(x=timed.iloc[:,0], y=timed.iloc[:,2], deg=3)

pd.DataFrame(timed)

df = sns.load_dataset("anscombe")
df
sns.set_style('ticks')
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1,2,width_ratios=[1,1])

fig = plt.figure(figsize=(10,5))

ax0 = plt.subplot(gs[:,0])
ax1 = plt.subplot(gs[:,1])
ax0.ticklabel_format(style="sci")
ax0.set(xscale="log", yscale="log")
sns.scatterplot(x="n", y ="time", data = timed,ax=ax0)
