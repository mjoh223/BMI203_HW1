import numpy as np
from sort_functions import algs

def test_pointless_sort():
    # generate random vector of length 10
    x = np.random.rand(10)

    # check that pointless_sort always returns [1,2,3]
    assert np.array_equal(algs.pointless_sort(x), np.array([1,2,3]))

    # generate a new random vector of length 10
    x = np.random.rand(10)

    # check that pointless_sort still returns [1,2,3]
    assert np.array_equal(algs.pointless_sort(x), np.array([1,2,3]))

def test_bubblesort():
    # Actually test bubblesort here. It might be useful to think about
    # some edge cases for your code, where it might fail. Some things to
    # think about: (1) does your code handle 0-element arrays without
    # failing, (2) does your code handle characters?

    x = np.array([1,2,4,0,1])
    empty_array = []
    single_element_vector = np.array([1])
    duplicated_elements = np.array([2,4,0,1,1,1,1,1,1,1,])

    # for now, just attempt to call the bubblesort function, should
    # actually check output
    assert np.array_equal(algs.bubblesort(x), [0, 1, 1, 2, 4])
    assert np.array_equal(algs.bubblesort(empty_array), [])
    assert np.array_equal(algs.bubblesort(single_element_vector), [1])
    assert np.array_equal(algs.bubblesort(duplicated_elements), [0,1,1,1,1,1,1,1,2,4])
def test_quicksort():

    x = np.array([1,2,4,0,1])
    empty_array = []
    single_element_vector = np.array([1])
    duplicated_elements = np.array([2,4,0,1,1,1,1,1,1,1,])

    # for now, just attempt to call the quicksort function, should
    # actually check output
    assert np.array_equal(algs.quicksort(x), [0, 1, 1, 2, 4])
    assert np.array_equal(algs.quicksort(empty_array), [])
    assert np.array_equal(algs.quicksort(single_element_vector), [1])
    assert np.array_equal(algs.quicksort(duplicated_elements), [0,1,1,1,1,1,1,1,2,4])
