class Heap(object):
    """
    Represents a binary min heap
    """
    def __init__(self):
        self.heap_list = []
    
    def insert(self, point):
        self.heap_list.append(point)
        self.heapify_up(len(self.heap_list)-1)

    def delete_min(self):
        min_val = self.heap_list.pop(0)
        last_val = self.heap_list.pop(-1)
        self.heap_list = [last_val] + self.heap_list # insert last element to front as root
        self.heapify_down(0)
        return min_val
    
    def heapify_up(self, i):
        if i // 2 >= 0 and self.heap_list[i] < self.heap_list[i // 2]:
            tmp = self.heap_list[i // 2]
            self.heap_list[i // 2] = self.heap_list[i]
            self.heap_list[i] = tmp
            self.heapify_up(i // 2)

    def heapify_down(self, i):
        min_child_i = self.min_child(i)
        if min_child_i != -1 and self.heap_list[i] > self.heap_list[min_child_i]:
            temp = self.heap_list[i]
            self.heap_list[i] = self.heap_list[min_child_i]
            self.heap_list[min_child_i] = temp
            self.heapify_down(min_child_i)

    def min_child(self, i):
        left_child = i * 2 + 1
        right_child = left_child + 1
        if left_child < len(self.heap_list) - 1 and right_child < len(self.heap_list) - 1:
            if self.heap_list[left_child] < self.heap_list[right_child]:
                return left_child
            else:
                return right_child
        elif left_child < len(self.heap_list) - 1:
            return left_child
        else:
            return -1

    def get_list(self):
        return self.heap_list


"""
Example tree
                            5
                10                      14
        24              35      17              41
"""
    
root = Heap()
root.insert(14)
root.insert(35)
root.insert(5)
root.insert(41)
root.insert(17)
root.insert(10)
root.insert(24)
print(root.get_list())

"""
Example delete min
                            10
                24                      14
        41              35      17              
"""
print("Delete root: {}".format(root.delete_min()))
print(root.get_list())