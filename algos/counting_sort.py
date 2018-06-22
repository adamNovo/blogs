import copy

def counting_sort(arr):
    min_val = min(arr)
    max_val = max(arr)
    counts = [0 for i in range((max_val-min_val)+1)]
    for i in arr:
        counts[i-min_val] += 1
    iloc = 0
    for i in range(len(counts)):
        while counts[i] > 0:
            arr[iloc] = i + min_val
            iloc += 1
            counts[i] -= 1
    return arr

if __name__ == "__main__":
    arr = [4,6,87,0,-1]
    assert [-1,0,4,6,87] == counting_sort(copy.deepcopy(arr))
    arr = [4,6,87,0,-1,11]
    assert [-1,0,4,6,11,87] == counting_sort(copy.deepcopy(arr))