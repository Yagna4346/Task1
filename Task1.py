print("List")
list_op = [1, 2, 3, 4, 5]
print("Initial list:", list_op)
list_op.append(6)
print("List after adding 6:", list_op)
list_op.remove(3)
print("List after removing 3:", list_op)
list_op[3] = 10
print("List after modifying element at index 3:",list_op)
print("\n")
print("Dictionary")
dict_op = {'a': 1, 'b': 2, 'c': 3}
print("Initial dictionary:", dict_op)
dict_op['d'] = 4
print("Dictionary after adding key 'd':", dict_op)
del dict_op['b']
print("Dictionary after removing key 'b':", dict_op)
dict_op['a'] = 10
print("Dictionary after modifying value of key 'a':",dict_op)
print("\n")
print("Set")
set_op = {1, 2, 3, 4, 5}
print("Initial set:", set_op)
set_op.add(6)
print("Set after adding 6:", set_op)
set_op .remove(3)
print("Set after removing 3:", set_op)
set_op.remove(2)
set_op .add(10)
print("Set after removing 2 and adding 10:", set_op)
