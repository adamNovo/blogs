import copy

def selection_sort(arr):
    for i in range(len(arr)):
        i_min = i
        for j in range(i, len(arr)):
            if arr[j] < arr[i_min]:
                i_min = j
        temp = arr[i_min]
        arr[i_min] = arr[i]
        arr[i] = temp
    return arr

if __name__ == "__main__":
    arr = [4,6,87,0,-1]
    assert [-1,0,4,6,87] == selection_sort(copy.deepcopy(arr))
    arr = [4,6,87,0,-1,11]
    assert [-1,0,4,6,11,87] == selection_sort(copy.deepcopy(arr))
