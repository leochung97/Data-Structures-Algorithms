def addBinary(a, b):
    x, y = int(a, 2), int(b, 2)

    while y:
        x, y = x ^ y, (x & y) << 1
        print(x ^ y)
        print(x & y)

    return bin(x)[2:]


print(addBinary("11", "1"))
