class Node(object):
    """
    Represents a binary search node
    """
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    
    def insert(self, data):
        if not self.data:
            self.data = data
        else:
            if data < self.data:
                if not self.left:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            else:
                if not self.right:
                    self.right = Node(data)
                else:
                    self.right.insert(data)

def traverse_preorder(node):
    print(node.data)
    if node.left:
        traverse_preorder(node.left)
    if node.right:
        traverse_preorder(node.right)

def traverse_inorder(node):
    if node.left:
        traverse_inorder(node.left)
    print(node.data)
    if node.right:
        traverse_inorder(node.right)

def traverse_postorder(node):
    if node.left:
        traverse_postorder(node.left)
    if node.right:
        traverse_postorder(node.right)
    print(node.data)


"""
Example tree
                            27
                14                      35
        10              19      31              42
"""
    
root = Node(27)
root.insert(14)
root.insert(35)
root.insert(10)
root.insert(19)
root.insert(31)
root.insert(42)
print("preorder")
traverse_preorder(root)
print("inorder")
traverse_inorder(root)
print("postorder")
traverse_postorder(root)