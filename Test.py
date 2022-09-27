def addBinary(a, b):
  x, y = int(a, 2), int(b, 2)

  while y:
    x, y = x ^ y, (x & y) << 1

  return bin(x)[2:]

# print(addBinary("11", "1"))

string = "abcdefg"
left = string[1:]
right = string[:-1]
print(left)
print(right)