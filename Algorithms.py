from math import sqrt, floor

# is_prime in O(sqrt(n)) time complexity; O(1) space complexity
def is_prime(n):
  if n < 2:
    return False
  
  # O(sqrt(n)) solution - only checks up to integer of sqrt(n); the +1 is to make the range inclusive of n
  for i in range(2, floor(sqrt(n) + 1)):
    if n % i == 0:
      return False

  # Return true here 
  return True

# print(is_prime(5)) # -> True
# print(is_prime(2017)) # ->  True
# print(is_prime(-5)) # -> False
# print(is_prime(713)) # -> False

# uncompress is O(n * m) time complexity and space complexity; n represents the number of "number groups" while m represents the maximum number within a number group; Space complexity is also O(n * m) due to the output variable that is saved dependent on n and m
def uncompress(str):
  numbers = "0123456789"
  # We use a list / array here because in Python, strings are immutable and must create a NEW copy when strings are concatenated! Thus, you should use an array here as it is mutable and therefore more efficient
  result = []
  i = 0
  j = 0
  while j < len(str):
    if str[j] in numbers:
      j += 1
    else:
      # Note: In Python, you can change a string to an integer using int()
      num = int(str[i:j])
      # Note: In Python, you can use .append to add to the end of a list / array
      result.append(str[j] * num)
      j += 1
      i = j

  # This line is outside of the while loop - this is linear (O(n))
  return ''.join(result)

# compress is O(n) time complexity; n represents the length of the input string; Space complexity is O(n) as well as we create a resulting string that is dependent on the input string size; can use two-pointers to track the length of character patterns 
def compress(str):
  # Add a non-alphanumerical character so that the first if statement will count the last character in the string as a new character and thus hit the else statement
  str += "!"
  # We use a list here instead of a string output because if you append to a new string in Python you are technically rebuilding the string and thus having an O(n^2) solution in your while loop
  result = []
  i = 0
  j = 0

  while j < len(str):
    if str[i] == str[j]:
      j += 1
    else:
      num = j - 1
      if num == 1:
        # Don't want to add 1 to the compressed string so we just add the character only
        result.append(str[i])
      else:
        # If the number isn't 1, then we can add both the number and the character
        result.append(num)
        result.append(str[i])

  return ''.join(result)