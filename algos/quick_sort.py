import copy

def quick_sort(arr):
    quick_sort_helper(arr, 0, len(arr))
    return arr

def quick_sort_helper(arr, start, end):
    if start < end:
        p = pivot(arr, start, end)
        quick_sort_helper(arr, start, p)
        quick_sort_helper(arr, p+1, end)

def pivot(arr, start, end):
    pivot_val = arr[start]
    i = start + 1
    for j in range(start + 1, end):
        if arr[j] < pivot_val:
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
            i += 1
    i -= 1
    arr[start] = arr[i]
    arr[i] = pivot_val
    return i

if __name__ == "__main__":
    arr = [4,6,87,0,-1]
    assert [-1,0,4,6,87] == quick_sort(copy.deepcopy(arr))
    arr = [4,6,87,0,-1,11]
    assert [-1,0,4,6,11,87] == quick_sort(copy.deepcopy(arr))