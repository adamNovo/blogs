import copy

def insertion_sort(arr):
    for i in range(1, len(arr)):
        pivot = arr[i]
        j = i - 1
        while j >= 0 and pivot < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = pivot
    return arr

if __name__ == "__main__":
    arr = [4,6,87,0,-1]
    assert [-1,0,4,6,87] == insertion_sort(copy.deepcopy(arr))
    arr = [4,6,87,0,-1,11]
    assert [-1,0,4,6,11,87] == insertion_sort(copy.deepcopy(arr))