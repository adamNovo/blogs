print("Arrays. methods")
print("- append()")
print("- extend()")
print("- insert()")
print("- remove()")
print("- pop()")
print("- clear()")
print("- index()")
print("- count()")
print("- sort()")
print("- reverse()")


print("init")
arr = [1, 2, 5, 7, 3, 4]
print(arr)

arr.append(8)
print("CREATE 8")
print(arr)
print("CREATE from another list [-1, -2]")
arr += [-1, -2]
print(arr)

print("READ at index 3")
print(arr[3])
print("READ at index -2")
print(arr[-2])
print("READ by slice arr[:2]")
print(arr[:2])

print("UPDATE last element to 9")
arr[len(arr)-1] = 9
print(arr)

print("DELETE by value = 3")
arr.remove(3)
print(arr)
print("DELETE by index = 3")
arr.pop(3)
print(arr)