import copy
import math

def bucket_sort(arr):
    return bucket_helper(arr, 10)

def bucket_helper(arr, num_buckets):
    buckets = [[] for i in range(num_buckets)]
    min_val = min(arr)
    max_val = max(arr)
    step = math.ceil((max_val - min_val) / num_buckets)
    for i in arr:
        b_idx = int((i - min_val) / step)
        buckets[b_idx].append(i)
    sorted_list = []
    for i in buckets:
        i = sorted(i)
        sorted_list += i
    return sorted_list

if __name__ == "__main__":
    arr = [4,6,87,0,-1]
    assert [-1,0,4,6,87] == bucket_sort(copy.deepcopy(arr))
    arr = [4,6,87,0,-1,11]
    assert [-1,0,4,6,11,87] == bucket_sort(copy.deepcopy(arr))