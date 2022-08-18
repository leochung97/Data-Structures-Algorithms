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

# Time complexity: O(n); You must iterate through every node in the list input to check for non-unique values
# Space complexity: O(1); No variables are saved that depend on the node's length / size
def is_univalue_list(head):
  unique_val = head.val
  current = head.next
  while current is not None:
    if current.val != unique_val:
      return False
    current = current.next
  return True

# Time complexity: O(n); You must iterate through every node in the list input to check for non-continuous values
# Space complexity: O(1); Variables set are not dependent on input size
def longest_streak(head):
  max_streak = 0
  curr_streak = 0
  prev_val = None
  current = head

  while current is not None:
    if current.val == prev_val:
      curr_streak += 1
    else:
      curr_streak = 1
    
    max_streak = max(max_streak, curr_streak)
    prev_val = current.val
    current = current.next

  return max_streak

# Time complexity: O(n); You must iterate through every node in the list to determine the target value
# Space complexity: O(1); Variables set are not dependent on input size
def remove_node(head, target_val):
  if head.val == target_val:
    return head.next
  
  current = head
  prev = None
  while current is not None:
    if current.val == target_val:
      prev.next = current.next
      break
    prev = current
    current = current.next
  return head

# Time complexity: O(n); At worst, you traverse the entire list to add the new node
# Space complexity: O(1); Variables set are not dependent on input size
def insert_node(head, value, index):
  if index == 0:
    result = Node(value)
    result.next = head
    return result
  
  count = 0
  current = head

  while current is not None:
    if count == index - 1:
      temp = current.next
      current.next = Node(value)
      current.next.next = temp
    
    count += 1
    current = current.next
  
  return head

# Time complexity: O(n); You must go through the entire array to create a linked list of all elements
# Space complexity: O(n); You are creating a linked list of size n, dependent on the size of the array input
def create_linked_list(array):
  result = Node(None)
  tail = result

  for val in array:
    tail.next = Node(val)
    tail = tail.next

  return result.next

# Add_lists is Leetcode 2: Add Two Numbers
# Time complexity: O(max(n, m)); You must go through both linked lists and thus the time complexity is dependent on whichever list is longer in length
# Space complexity: O(max(n, m)); Your output will depend on the greater of both linked lists as your answer is 
def add_lists(head_1, head_2):
  output = tail = Node(None)
  carry = 0
  current_1 = head_1
  current_2 = head_2
  
  while current_1 is not None or current_2 is not None or carry == 1:
    # Makes sure that there are values to add
    val_1 = 0 if current_1 is None else current_1.val
    val_2 = 0 if current_2 is None else current_2.val
    sum = val_1 + val_2 + carry
    carry = 1 if sum > 9 else 0
    # Want to make sure to make a new node pointing to the digit and not the sum! Remember, each node should be ONE DIGIT
    digit = sum % 10
    tail.next = Node(digit)
    tail = tail.next
    
    # Traverse both linked lists so unless they are already None
    if current_1 is not None:
      current_1 = current_1.next
    if current_2 is not None:
      current_2 = current_2.next
    
  return output.next

# Time complexity: O(n); You must go through each node in the binary tree
# Space complexity: O(n); Return output depends on the size of the input binary tree
def depth_first_search(root):
  # Return empty array if the input binary tree is empty
  if not root:
    return []
  
  # Output array will contain all of the elements in DEPTH FIRST SEARCH order
  output = []
  # Begin iterative answer with a stack containing just the root
  stack = [ root ]
  # Check if the stack is empty using falsey if statements
  while stack:
    current = stack.pop()
    output.append(current.val)
    # Check from the right first using falsey if statements
    if current.right:
      stack.append(current.right)
    if current.left:
      stack.append(current.left)
  return output

# Deque (doubly ended queue) in Python is implemented using the module "collections"; Deque is preferred over a list in the cases where we need quicker append and pop operations from both ends of the container, as deque provides an O(1) time complexity for append and pop operations as compared to list whcih provides O(n) time complexity
# deque.append(): Used to insert the value in its argument to the right end of the deque
# deque.appendleft(): Used to insert the value in its argument to the left end of the deque
# deque.pop(): Used to delete an argument from the right end of the deque
# deque.popleft(): Used to delete an argument from the left end of the deque
from collections import deque

# Time complexity: O(n); You must go through each node in the binary tree
# Space complexity: O(n); Return output depends on the size of the input binary tree
def breadth_first_values(root):
  if not root:
    return []
  
  # Set up a special object (deque) that will allow for O(1) operations (append, appendleft, pop, popleft); this is built-in to Python library so fair game for interviews
  # Can't do this problem recursively because it is a queue and not a stack
  queue = deque([ root ])
  output = []
  
  # Checks while queue is still full
  while queue:
    # Pops out the leftmost node and appends it to the outputs
    node = queue.popleft()
    output.append(node.val)
    
    # Checks the node to see if it has any leaves and appends them to the queue
    if node.left:
      queue.append(node.left)
    if node.right:
      queue.append(node.right)
  
  return output

# Recursive solution to solving this problem
# Time complexity: O(n); You must go through each node in the binary treee
# Space complexity: O(n); Implements a call stack (recursive) so it is O(n) space complexity
def max_path_sum(root):
  # You want to make sure that the max(left, right) will never choose the non-existing node
  if root is None:
    return float('-inf')

  # Checks if the node is a leaf node (i.e., has no children); if it is a leaf node, you just need to return its current value
  if root.left is None and root.right is None:
    return root.val
  
  # Return the current value of the node plus the max between its children
  return root.val + max(max_path_sum(root.left), max_path_sum(root.right))