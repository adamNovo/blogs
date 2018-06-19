import copy

def bubble_sort(arr):
    for i in range(1, len(arr)):
        for j in range(len(arr)-i):
            if arr[j] > arr[j+1]:
                temp = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = temp
    return arr

if __name__ == "__main__":
    arr = [4,6,87,0,-1]
    assert [-1,0,4,6,87] == bubble_sort(copy.deepcopy(arr))
    arr = [4,6,87,0,-1,11]
    assert [-1,0,4,6,11,87] == bubble_sort(copy.deepcopy(arr))