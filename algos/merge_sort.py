import copy

def merge_sort(arr):
    if len(arr) < 2:
        return arr
    mid = int(len(arr) / 2)
    # divide
    left = arr[:mid]
    right = arr[mid:]
    # recursive
    merge_sort(left)
    merge_sort(right)
    # conquer
    l = 0
    r = 0
    for i in range(len(arr)):
        if l < len(left) and r < len(right):
            if left[l] < right[r]:
                arr[i] = left[l]
                l += 1
            else:
                arr[i] = right[r]
                r += 1
        elif l < len(left):
            arr[i] = left[l]
            l += 1
        elif r < len(right):
            arr[i] = right[r]
            r += 1
    return arr

if __name__ == "__main__":
    arr = [4,6,87,0,-1]
    assert [-1,0,4,6,87] == merge_sort(copy.deepcopy(arr))
    arr = [4,6,87,0,-1,11]
    assert [-1,0,4,6,11,87] == merge_sort(copy.deepcopy(arr))