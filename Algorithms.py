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

# Generic code for setting up a linked list
class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

# Reversing a linked list requires three variables to be set up to keep track of: the previous node, the current node, and the next node
# Time complexity: O(n); you must traverse the entire list to reverse it!
# Space complexity: O(1); we set variables that track the next node's values but it is not dependent on the size of the list
def reverse_list(head):
  # Previous node is set to None because there is no tail to reverse to just yet
  prev = None
  # Track the current node so that our "view" of the previous and next can shift along with the current
  current = head
  # We know that at the end of our original linked list, the tail will be None, so our while loop will end at the tail
  while current is not None:
    # Set next to current.next to keep track of it - once we set the current.next to prev (reverse the linked list), we would otherwise lose track of the next node
    next = current.next
    # Reverses the link to the previous node
    current.next = prev
    # Sets the previous node now to current node (prep for next reverse)
    prev = current
    # Sets the current node to the next node (shifting the "view" to the next set of nodes)
    current = next
  
  # Return previous here because we know that the previous node contains the new links to the reversed nodes
  return prev

# Zipper List requires several variables to be set up in order to be solved ITERATIVELY (more optimal than recursive)
# Time complexity: O(min(n, m)) because the lists can be small and therefore result in a faster solution since we just add the remaining values to the end of the list if one list ends
# Space complexity: O(1); we set variables to track the next node's value but our variables do not depend on the space of either list
def zipper_list(head_1, head_2):
  # Start off by setting a variable equal to the first head of the first linked list
  tail = head_1
  # Move the current view to the next value of the first linked list
  current_1 = head_1.next
  # We haven't added current_2 yet so we should set current_2 to head_2
  current_2 = head_2
  # We use a count variable here to determine when we alternate between the linked lists
  count = 0

  # Check if either list is None -> if it is, end the while loop
  while head_1 is not None and head_2 is not None:
    # If count is even, then we know that we should add current_2
    if count % 2 == 0:
      tail.next = current_2
      current_2 = current_2.next
    # Else we add current_1
    else:
      tail.next = current_1
      current_1 = current_1.next
    
    # Don't forget to move the "view" from tail to its next value and increment the count
    tail = tail.next
    count += 1

  # Since we exited out of the while loop, we know that one of the lists has values remaining and should be added as the next value
  if current_1 is not None:
    tail.next = current_1
  else:
    tail.next = current_2
  
  # Return head_1 here because tail variable has been updating its next values
  return head_1

# Merge List is similar to Zipper List except that you have to set a dummy variable equal to Node(None) so that the function can properly work; otherwise, if you set tail variable to something else, you will face issues defining it's "next" value
# Time complexity: O(min(n, m)); this is because the while loop will only continue until the length of the smaller list and eventually add everything to the tail afterwards
# Space complexity: O(1); this is because we set variables to track the next values of each node but they do not depend on the space of either list input
def merge_lists(head_1, head_2):
  result = tail = Node(None)
  current_1 = head_1
  current_2 = head_2

  while current_1 is not None and current_2 is not None:
    if current_1.val < current_2.val:
      tail.next = current_1
      current_1 = current_1.next
    else:
      tail.next = current_2
      current_2 = current_2.next
    tail = tail.next
  
  if current_1 is not None:
    tail.next = current_1
  else:
    tail.next = current_2
  
  return result.next