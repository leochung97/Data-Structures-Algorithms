import math
import statistics
from collections import deque

# is_prime in O(sqrt(n)) time complexity; O(1) space complexity
def is_prime(n):
  if n < 2:
    return False
  
  # O(sqrt(n)) solution - only checks up to integer of sqrt(n); the +1 is to make the range inclusive of n
  for i in range(2, math.floor(math.sqrt(n) + 1)):
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
# Time complexity: O(n); you must traverse the entire list
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

# Recursive solution to solving this problem
# Time complexity: O(n); Solution is O(n) as it only goes through each node of the binary tree; however, this is due to the fact that we use the append function and then reversing the result afterwards
# Space complexity: O(n); Could potentially be O(n) size
def path_finder(root, target):
  # Result is going to be the result fo the recursive _path_finder(root, target)
  result = _path_finder(root, target)
  # If the result is None, we know we didn't find the target and thus we should return None
  if result is None:
    return None
  # If the result does exist, then we know that the array is currently set backwards and we need to reverse it using list slicing
  else:
    return result[::-1]

# Note: If you put an underscore in front of a method, it generally sigals that it is a helper function
def _path_finder(root, target):
  # First, check if the root is none - this is our recursive call check
  if root is None:
    return None
  
  # If the root is the target, return an array containing the target
  if root.val == target:
    return [root.val]
  
  # left_path and right_path will return [root.val] but we must append the path nodes - note that the resulting path is reversed but will be reversed in the main function
  left_path = _path_finder(root.left, target)
  if left_path is not None:
    left_path.append(root.val)
    return left_path

  right_path = _path_finder(root.left, target)
  if right_path is not None:
    right_path.append(root.val)
    return right_path
  
  return None

# Time complexity: O(n); We must go through every node to check if the value is equal to the target
# Space complexity: O(n); We are implementing a recursive stack which will hold O(n) space complexity based on the binary tree size
def tree_value_count(root, target):
  # Exits out recursive call if there is no root (i.e., no leaf node)
  if root is None:
    return 0
  
  # Sets match = 1 becase we can add it as a count later on
  match = 1 if root.val == target else 0

  # Recursively adds match and the results from future matches
  return match + tree_value_count(root.left, target) +  tree_value_count(root.right, target)

# Time complexity: O(n); You must go through every node to check the value and add the path
# Space complexity: O(n); At worst you'll have one path containing all node values
def all_tree_paths(root):
  # This will stop our recursive call; this should be placed in the front else we may run into an error if try to call root.left / root.right on a None root
  if root is None:
    return []
  
  # This will return the root.val in a list if it does not have any left or right nodes
  if root.left is None and root.right is None:
    return [ [root.val] ]
  
  # Set up a new list that represents all possible paths
  paths = []
  
  # You assume that the function works and will return all sub-paths here
  left_paths = all_tree_paths(root.left)
  right_paths = all_tree_paths(root.right)

  # For each sub_path in each side of the binary tree, we can append the original root value + the sub_path values as a new list to our paths output
  for sub_paths in left_paths:
    paths.append([root.val, *sub_paths])
  for sub_paths in right_paths:
    paths.append([root.val, *sub_paths])

  # Return the output
  return paths

# Time complexity: O(n); You must go through every node to check the value and add the current level
# Space complexity: O(n); You are returning a variable holding all of the possible levels of the tree
def tree_levels(root):
  if root is None:
    return []
  
  levels = []
  # Implement a queue using python collections deque
  # You want to add the root and the current level of the tree to the queue to indicate what level the tree is currently at
  queue = deque([ (root, 0) ])
  while queue:
    current, level_num = queue.popleft()
    
    # You want to only append to the current level number if the length of the levels is not equivalent - this is becasue we haven't add the next level yet
    if len(levels) == level_num:
      levels.append([ current.val ])
    else:
      levels[level_num].append(current.val)
      
    # Add the left and right nodes to the queue and increment the level; you must increment the level because we are traversing down
    if current.left is not None:
      queue.append((current.left, level_num + 1))
      
    if current.right is not None:
      queue.append((current.right, level_num + 1))
  
  # Return the levels array after appending all of the levels together
  return levels

# This question is basically tree_levels but with an added average - we use the mean function from the statistics library to calculate our average quickly
# Time complexity: O(n); You must go through every node to check the value and add the current level
# Space complexity: O(n); You are returning a variable holding all of the possible levels of the tree
def tree_level_averages(root):
  if root is None:
    return []

  levels = []
  queue = deque([(root, 0)])
  while queue:
    current, level_num = queue.popleft()

    if len(levels) == level_num:
      levels.append([current.val])
    else:
      levels[level_num].append(current.val)
    
    if current.left is not None:
      queue.append((current.left, level_num + 1))
    if current.right is not None:
      queue.append((current.right, level_num + 1))
  
  # This is a Pythonic way of writing a map function; you call mean() function on every level in the levels array returned from your while loop
  return [ statistics.mean(level) for level in levels ]

# This question is deceptive; you need to use depth-first search in order to reach the leaves in order
# Time complexity: O(n); You must go through every node to check if the node is a leaf node
# Space complexity: O(n); At worst, the entire tree will be a leaf
def leaf_list(root):
  # Check if root is None
  if root is None:
    return []

  # Initiate a variable that will contain the leaves
  leaves = []
  # Initiate a stack (depth-first search) that contains your root
  stack = [root]
  
  while stack:
    # Pop off the current node
    current = stack.pop()
    
    # Check if the current node is a leaf node by checking if it has a left or right node; if both are None, you know that this node is a leaf node and therefore can add it to the leaves list
    if current.left is None and current.right is None:
      leaves.append(current.val)
    
    # We want to check the RIGHT first to add to the stack so that when the stack pops another node off, it will pop off the LEFT node first, resulting in the binary tree being checked from left to right
    if current.right is not None:
      stack.append(current.right)
    if current.left is not None:
      stack.append(current.left)
  
  return leaves

# Time complexity: O(e); You must check every edge to see if there is a path from node_A to node_B
# Space complexity: O(n); We use a set to contain the nodes that we have visited so in the worst case we may store every node
def undirected_path(edges, node_A, node_B):
  # First you want to build a graph using the edges provided
  graph = build_graph(edges)
  # You pass the created graph into has_path to determine whether a path is available
  return has_path(graph, node_A, node_B, set())
  
def build_graph(edges):
  # Graph adjacency list can be represented with a hashmap containing key-value pairs of the node and its neighbors
  graph = {}
  
  # For each edge in the provided edges, deconstruct the edge into a, b variable and then set up key-value pairs with the value being an empty list
  for edge in edges:
    a, b = edge
    if a not in graph:
      graph[a] = []
    if b not in graph:
      graph[b] = []
    
    # Append the neighbor to the list (do it both ways as to show that they are two-way neighbors)
    graph[a].append(b)
    graph[b].append(a)
  
  return graph

# Helper function that returns a boolean if the src argument can reach the dst argument
def has_path(graph, src, dst, visited):
  # Will return True and pop off the recursive stack once this conditional is reached
  if src == dst:
    return True
  
  # We want to make sure that we are not accidentally going to have infinite recursive calls in case of a cyclical graph - this is why we have a set() containing our visited nodes
  # We returned False if we already visited so that we can pop this call off of the recursive stack
  if src in visited:
    return False
  
  # Make sure to add the src node to the visited set so that it can be checked later
  visited.add(src)
  
  # Recursively call this function on the neighbor now to check if it is the destination - if it is, then we know we can return True and end the recursion
  for neighbor in graph[src]:
    if has_path(graph, neighbor, dst, visited) == True:
      return True
  
  # If we have not yet returned False, then we know that there is no pathway and we should return False
  return False

# Time complexity: O(e); You must check every edge to find the different connected components
# Space complexity: O(n); You must go through every graph and hold a set of visited nodes
def connected_components_count(graph):
  # Create a visited set that will contain the visited nodes
  # Note that sets have O(1) inclusion and access
  visited = set()
  count = 0
  
  # Recall that graph is provided in a hashmap containg key-value pairs of the node and its adjacency list
  # For each key value, you would explore the graph, the key, and the visited set; check whether it returns True and increment the count of connected components
  for node in graph:
    if explore(graph, node, visited) == True:
      count += 1
    
  return count

# Helper function to determine whether the current node has already been visited and whether its neighbors have already been visited
# If this function returns true, we know that the node hasn't been visited and its component neighbors haven't been visited
def explore(graph, current, visited):
  # Checks if the current node has already been visited
  if current in visited:
    return False
  
  # Adds the current value to the visited set
  visited.add(current)
  
  # Explore each neighbor (DFS)
  for neighbor in graph[current]:
    explore(graph, neighbor, visited)

  # Return True if no falses!
  return True

# Time complexity: O(e); You must check every edge to find the different connected components
# Space complexity: O(n); You must go through every graph and hold a set of visited nodes
def largest_component(graph):
  # Create a visited set that will contain the visited nodes
  visited = set()
  # Set up a variable to return later
  largest = 0
  # Check the size of each component and change largest if the size is largest
  for node in graph:
    size = explore_size(graph, node, visited)
    if size > largest:
      largest = size

  # Return the largest
  return largest

# Helper function to determine the size of the component; you want to return 0 if this node was already visited because this entire component was already considered
def explore_size(graph, node, visited):
  # You want to return 0 because you don't want the size to be taking over the largest
  if node in visited:
    return 0

  visited.add(node)

  # Set size initially to 1 because you don't 
  size = 1
  for neighbor in graph[node]:
    size += explore_size(graph, neighbor, visited)

  return size

# Time complexity: O(e); At worst, we will visit every edge in this breadth first search traversal
# Space complexity: O(e); At worst, we will store every edge into our visited set, thus using O(e) memory
def shortest_path(edges, node_A, node_B):
  # Build a graph using the helper function below
  graph = build_graph(edges)
  # Create a set to keep track of visited nodes - we don't want to accidentally run an infinite loop in the case of cyclical graphs
  visited = set([ node_A ])
  # Create a queue using deque from Python collections library; Include a distance factor so that we can determine how far into the edges we are traversing
  queue = deque([ (node_A, 0) ])

  # While the queue is populated and we haven't returned yet...
  while queue:
    # Deconstruct the first item out of the queue to determine the current node and the current distance
    node, distance = queue.popleft()

    # If the current node is equal to our target node, then we have reached our destination as fast as possible and should return the distance
    if node == node_B:
      return distance

    # If the current node isn't equal to our target node, then we should check the neighbors to see if they will be our target nodes
    for neighbor in graph[node]:
      # Only check the neighbor if we haven't already visited them - this prevents infinite recursive loops
      if neighbor not in visited:
        # Add the neighbor so we prevent infinite loops
        visited.add(neighbor)
        # Make sure to increment the distance because we are moving one edge away to the next node
        queue.append((neighbor, distance + 1))
  
  # If you can't find anything - return -1!
  return -1

# We receive a list of edges and must build a graph out of it - note that the edges only have a connection between two nodes so all we have to do is initiate the two nodes under a hashmap to determine their neighbors
def build_graph(edges):
  graph = {}
  for edge in edges:
    a, b = edge
    if a not in graph:
      graph[a] = []
    if b not in graph:
      graph[b] = []
    
    graph[a].append(b)
    graph[b].append(a)
  
  return graph

# Time complexity: O(xy); We must visit every node in the grid
# Space complexity: O(xy); We store every node in the set
def island_count(grid):
  # Set up a set containing all of the visited nodes
  visited = set()
  # Set up a count that will keep count of the islands that we have explored
  count = 0
  # For each x and y coordinate, we want to check if the explore function returns True
  for x in range(len(grid)):
    for y in range(len(grid[0])):
      # The explore helper function will check if the "island" is viable and then explore all of its possible neighbors to determine whether they are also islands
      # The coordinates of neighbors will be added to the visited set and thus will not result in repeated counts
      if explore(grid, x, y, visited) == True:
        count += 1
  # Return the count of islands afterwards
  return count

def explore(grid, x, y, visited):
  # Check if the row and column coordinates (x, y arguments) are within bounds of the grid
  row_inbounds = 0 <= x < len(grid)
  col_inbounds = 0 <= y < len(grid[0])

  # Return false if the row or column is not in the bounds of the grid
  if not row_inbounds or not col_inbounds:
    return False

  # Return false if the current position isn't land
  if grid[x][y] == "W":
    return False

  # You want to return false if the position is already in visited as well since we probably added this land mass to the count already
  pos = (x, y)
  if pos in visited:
    return False
  # Always make sure to add the node to the visited set!
  visited.add(pos)

  # Explore all of the neighbors in breadth first search; These will return Falses or Trues but it won't matter because we are only exploring to record their coordinates as visited
  explore(grid, x - 1, y, visited)
  explore(grid, x + 1, y, visited)
  explore(grid, x, y - 1, visited)
  explore(grid, x, y + 1, visited)

  # We want to return True here because we know that we have passed all of the other tests and have recorded this position as a piece of land
  # We also already marked all of the land masses as visited so we don't have to worry about double counting in our main function
  return True

# Time complexity: O(rc); At worst, we iterate through every position so it would be O(row * column)
# Space complexity: O(rc); At worst, we iterate through and store every position in our set so it would be O(row * column)
def closest_carrot(grid, starting_row, starting_col):
  # Create a visited set that contains our starting position
  visited = set([ (starting_row, starting_col) ])
  # Create a queue that will track the position that we are checking and the distance traveled
  queue = deque([ (starting_row, starting_col, 0) ])

  # Iterate through the queue as long as it exists or until we reach the carrot "C"
  while queue:
    # Deconstruct the row, column, and distance from the queue's first out element
    row, col, distance = queue.popleft()

    # Return the distance (already deconstructed) if the current position is the carrot
    if grid[row][col] == "C":
      return distance
    
    # If not returned, then we know we must add the current position's neighbors to the queue
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # For each delta, we need to check if the neighbor was already visited and whether it is within bounds - if it is, we can add it to the queue
    for delta in deltas:
      # Deconstruct the delta
      delta_row, delta_col = delta
      # Create the neighbor coordinates
      neighbor_row = row + delta_row
      neighbor_col = col + delta_col
      pos = (neighbor_row, neighbor_col)
      # Check if the neighbor is in the grid boundaries
      row_inbounds = 0 <= neighbor_row < len(grid)
      col_inbounds = 0 <= neighbor_col < len(grid[0])
      
      # If the neighbor is within bounds, not already visited, and not a wall ("X"), then we can proceed to add the position to the visited and append the new position
      if row_inbounds and col_inbounds and pos not in visited and grid[row][col] != "X":
        visited.add(pos)
        queue.append((neighbor_row, neighbor_col, distance + 1))

  # Default case is -1 if we don't find the shortest distance to the carrot
  return -1

# Time complexity: O(e); You must travel through each edge and determine whether it is a part of the longest distance
# Space complexity: O(n); You might store each node in a dictionary to determine whether they are terminal nodes
# Note that the graph in this problem is a directed acyclical graph (DAG), which makes the problem possible; Edges will point one-way and there is no cycle here
def longest_path(graph):
  # Set up a dictionary that will contain "terminal" nodes (nodes with no neighbors that it will point to) and their distances
  distance = {}
  # Check the neighbors of the node - if there are none (length == 0), then add the node to the distance dictionary and initially set their distance to 0
  for node in graph:
    if len(graph[node]) == 0:
      distance[node] = 0
  
  # We want to use a helper function to traverse each node and add to the distance if it reaches a terminal node
  for node in graph:
    traverse_distance(graph, node, distance)

  # Return the highest distance - this is the longest path
  return max(distance.values())

def traverse_distance(graph, node, distance):
  # We know that the node is terminal so we can just end the function early
  if node in distance:
    return distance[node]

  # Create a temp variable that will track the max distance that we have covered thus far
  max_length = 0

  # Check each possible path in neighbors
  for neighbor in graph[node]:
    # Assume that the attempt will return a distance
    attempt = traverse_distance(graph, neighbor, distance)
    # If the attempt's distance is longer than the max length, then we should set max length to the attempt
    if attempt > max_length:
      max_length = attempt
  
  # We know the new max_length through DFS (attempt) but we also need to make sure that we add 1 (current node) to the max_length
  distance[node] = 1 + max_length
  # Retrun the max_distance - this goes back up to our distance dictionary and will eventually be called by max(distance.values())
  return distance[node]

# It helps to draw this problem out as it is not obvious it is supposed to be a DAG
# Time complexity: O(e); We traverse through every edge in our graph so this would just be O(prerequisites)
# Space complexity: O(n); We are using DFS which requires recursion so our space complexity will be the stack of nodes we are visiting
def semesters_required(num_courses, prereqs):
  # We first build a graph using the helper function; Note that the helper function builds a DAG (directed acyclic graph)
  graph = build_graph(num_courses, prereqs)

  # We will track the distance (number of semesters) of each course using a distance hashmap
  distance = {}

  # For each course in the graph, we can say that the distance of the course if 1 if the course has no neighbors that it points to (this is an ending node)
  for course in graph:
    if len(graph[course]) == 0:
      distance[course] = 1

  # For each course in the graph, we want to iterate with a helper function that will take in our graph, check if the node has been given a distance, and add to its distance
  for course in graph:
    traverse_distance(graph, course, distance)

  # Return the max distance as that is the longest semesters required to take all courses and their pre-requisites
  return max(distance.values())


def traverse_distance(graph, node, distance):
  # If the node is already in distance (aka this has already been explored), then just return the distance of that node
  if node in distance:
    return distance[node]

  max_distance = 0

  # For each neighbor, recursively call traverse_distance on its neighbors and assume taht it will return the neighbors distance + 1
  for neighbor in graph[node]:
    neighbor_distance = traverse_distance(graph, neighbor, distance)
    # Set the new max_distance as the max of itself of the neighbor_distance
    max_distance = max(max_distance, neighbor_distance)

  # Since we have updated max_distance for the node, we can update it to max_distance + 1 to include itself
  distance[node] = max_distance + 1
  # Return the distance of this node now so it can recursively add itself to the next iteration
  return distance[node]


def build_graph(num_courses, prereqs):
  # Create the graph variable which will be a hashmap containing the graph nodes and their neighbors
  graph = {}
  # Remember that courses are numbered from 0 to the n-1 number of courses so we should use range to exclude num_courses
  for course in range(0, num_courses):
    # The graph now holds a key of the course (the pre-requisite) and its neighbors (the next course)
    graph[course] = []
  
  # Add only b to a neighbor because this is a directed acyclic graph
  for prereq in prereqs:
    a, b = prereq
    graph[a].append(b)

  # Make sure to return the graph as this is a helper function
  return graph

# This problem is pretty difficult but can be broken down into multiple parts:
# The main idea is to iterate through every grid cell to find one island and to plot that island in our "visited" set so that we can understand where it lies - this should be done through BFS
# After finding a main island, we can use BFS to traverse every grid space in between and count the amount of water spaces in between islands
# Time complexity: O(x * y); You must traverse every node in the grid to check for islands and their respective bridges
# Space complexity: O(x * y); We store the visited nodes in a set so at worst we may store all of the nodes in the grid
def best_bridge(grid):
  # Set up a main island variable that will equate to None for now
  main_island = None
  for x in range(len(grid)):
    for y in range(len(grid[0])):
      # For every x and y coordinate in our grid, we want to use a helper function to determine if there is a potential island
      potential_island = traverse_island(grid, x, y, set())
      # If there is a potential island (determined by a set of coordinates),then we can set it as a main island
      if len(potential_island) > 0:
        main_island = potential_island
        # It is important to break the for loops here otherwise we could potentially continue to iterate through our code and set new islands
        break

  # Now we can set our visited nodes as a set of our main_island coordinates
  visited = set(main_island)
  # We use a queue using Python's deque to optimize the data structure and make our lives easier
  queue = deque([])
  
  # We want to start adding to our queue each position in our island and a distance starting at 0
  for pos in main_island:
    x, y = pos
    queue.append((x, y, 0))

  # For each main island coordinate, we deconstruct the x, y, and distance
  # Then we can check if the coordinate is an island and NOT on the main island
  # If it satisfies that condition, we know we have reached the other island and can return our distance - 1 (we subtract 1 because we don't want to count the second island coordinate)
  while queue:
    x, y, distance = queue.popleft()

    if grid[x][y] == "L" and (x, y) not in main_island:
      return distance - 1

    # If we don't hit the above conditional, then we want to just check our neighbors and add them to our not visited set
    deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for delta in deltas:
      delta_row, delta_col = delta
      neighbor_row = x + delta_row
      neighbor_col = y + delta_col
      neighbor_pos = (neighbor_row, neighbor_col)
      if is_inbounds(grid, neighbor_row, neighbor_col) and neighbor_pos not in visited:
        visited.add(neighbor_pos)
        queue.append((neighbor_row, neighbor_col, distance + 1))

# Helper function to determine whether our coordinates are inbound
def is_inbounds(grid, x, y):
  row_inbounds = 0 <= x < len(grid)
  col_inbounds = 0 <= y < len(grid[0])
  return row_inbounds and col_inbounds

# Helper function to traverse and explore an island; Returns a set of island coordinates
def traverse_island(grid, row, col, visited):
  # Make sure that the coordinates we are iterating through is actually an island
  # We return visited because we use traverse_island to explore a set of island coordinates
  if not is_inbounds(grid, row, col) or grid[row][col] == "W":
    return visited

  pos = (row, col)
  if pos in visited:
    return visited
  visited.add(pos)

  traverse_island(grid, row - 1, col, visited)
  traverse_island(grid, row + 1, col, visited)
  traverse_island(grid, row, col - 1, visited)
  traverse_island(grid, row, col + 1, visited)
  return visited

# Time complexity: O(n^2); At worst, we may traverse all of the nodes in the graph at least twice
# Space complexity: O(n); We are storing the nodes in sets so at worst we may store all of the nodes
def has_cycle(graph):
  # We create two sets here that will contain the information that our current node is visiting and a set of already fully visited and tracked nodes
  visiting, visited = set(), set()
  # For each node in our graph, we want to check if it has a cycle that returns itself - we can do this using a helper function
  for node in graph:
    # If we detect a cycle in the current node (aka we found the node in its current visiting already), then we know we have a cycle
    if cycle_detect(graph, node, visiting, visited):
      return True
  # If we don't find anything, we know we have visited all nodes and have not found any repeating visits
  return False

def cycle_detect(graph, node, visiting, visited):
  # Return false if the node is already in visited because we don't need to check it again for a cycle
  if node in visited:
    return False
  
  # If the node is in visiting however, we know that we have already just checked for this node and found it again and thus found a cycle
  if node in visiting:
    return True
  
  # Make sure to actually add our current node to visiting
  visiting.add(node)
  
  # For each of our neighbors, we can check the node by recursively iterating through the graph and finding if the neighbors are already in visiting - if they are, return True
  for neighbor in graph[node]:
    if cycle_detect(graph, neighbor, visiting, visited):
      return True
    
  # If we don't find the neighbor, we remove the node from our visiting and add it to our visited (to make sure we don't accidentally redo this cycle)
  visiting.remove(node)
  visited.add(node)
  # We return false at the end as well as we have determined no cycles
  return False

# Time complexity: O(a * n); We need to reach a by subtracting from our numbers n, so we have a O(a * n) solution
# Space complexity: O(a); We store up to our amount in our memo so our worst space complexity is O(a)
def sum_possible(amount, numbers):
  # Initialize a memo hashmap that will contain our previously calculated results
  memo = {}
  return _sum_possible(amount, numbers, memo)

def _sum_possible(amount, numbers, memo):
  # If we already know the the result, we don't want to recalculate it and should thus just return memo[amount]
  if amount in memo:
    return memo[amount]
  
  # If amount is less than 0, we know we have gone past the amount and it is not reachable through any of the numbers - thus we return False
  if amount < 0:
    return False
  
  # If amount is equal to 0, then we know that we must have reached it through a combination of numbers and can thus return True
  if amount == 0:
    return True
  
  # For each number in our numbers array, we want to recursively check if there is a possible amount that returns True after subtracting each number
  # This will recursively check every single possible combination of numbers
  for num in numbers:
    if _sum_possible(amount - num, numbers, memo):
      # If true then we need to set memo[amount] and return True
      memo[amount] = True
      return True
    
  # Otherwise, we can just return False and set memo[amount] to False
  memo[amount] = False
  return False

# Time complexity: O(amount * coins); At worst, you will try every combination of coins to reach the specific amount; You can think of this as amount as height and coins as nodes so it would be amount * coins
# Space complexity: O(a); At worst, you will store every value up to the amount value provided
def min_change(amount, coins):
  # Similar to the previous problem, we want to set up a memo to memorize some of our already explored answers
  memo = {}
  # We also set up a variable that will determine whether or not we have found a possible combination of answers that can return our answer
  ans = _min(amount, coins, memo)
  if ans == float('inf'):
    return -1
  else:
    return ans
  
# This helper function will find the smallest combination of coins possible to reach the amount; Otherwise, the minimum number of coins will be left at infinity because no combination was found
def _min(amount, coins, memo):
  # We want to call upon previously calculated results in case there are any - thus we return memo[amount] if it already exists
  if amount in memo:
    return memo[amount]
  
  # If the amount is less than 0, then we know that we have gone too far with this combination of coins and should thus return an arbitrarily large figure to not mess with our minimum coins function
  if amount < 0:
    return float('inf')
  
  # If thje amount is equal to 0, then we know that we have found a possible combination of coins and can return 0 to indicate that we have found a combination
  if amount == 0:
    return 0
  
  # We set this variable to an arbitrarily large figure so that it can be replaced using the minimum function later on
  min_coins = float('inf')
  # Now we want to check for each coin in coins list (each possible combination):
  for coin in coins:
    # We want to recursively determine the paths - if it reaches 0 (possible combination), then we can continuously add a 1 to the result until we find the exact path
    num_coins = 1 + _min(amount - coin, coins, memo)
    # We need to make sure that we set the min_coins
    min_coins = min(min_coins, num_coins)
  
  # Don't forget to set the memo[amount] to the result of the min_coins and return the min_coins
  memo[amount] = min_coins
  return min_coins

# Time complexity: O(x * y); At worst, we may have to explore all of the possible spaces within the grid
# Space complexity: O(x * y); At worst, we may store all of the grid coordinates into our memo
def count_paths(grid):
  # This is a dynamic programming question that we can use memoization to minimize our time complexity; set up a memo that will contain our already explored results
  memo = {}
  # Return a helper function that begins at (0, 0) and will continuously traverse downwards and to the right
  return _count_paths(grid, 0, 0, memo)

def _count_paths(grid, x, y, memo):
  # We want to create a position out of the coordinates we are provided and use that position as our memo keys
  pos = (x, y)
  if pos in memo:
    return memo[pos]
  
  # We write a conditional that determines whether our coordinates are out of the grid and whether we have hit a wall
  if x == len(grid) or y == len(grid[0]) or grid[x][y] == "X":
    return 0
  
  # We also write a conditional that if we have reached the end of the grid, then we know we can return 1 (a path has been found)
  if x == len(grid) - 1 and y == len(grid[0]) - 1:
    return 1
  
  # Recursively set the memo[pos]
  memo[pos] = _count_paths(grid, x + 1, y, memo) + _count_paths(grid, x, y + 1, memo)
  # Return the memo[pos] since it contains the answer
  return memo[pos]

# Time complexity: O(x * y); At worst, we may have to explore all of the possible spaces within the grid
# Space complexity; O(x * y); At worst, we have to store all of the grid coordinates into our memo
def max_path_sum(grid):
  # Again, we want to store our possibly explored options in a memo so we initialize a memo hashmap that will contain previously calculated answers
  memo = {}
  # We assume that our helper function will return the answer
  return _max_path_sum(grid, 0, 0, memo)

def _max_path_sum(grid, x, y, memo):
  # Construct a position coordinate from your x and y arguments so that you can store it into your memo / fetch from your memo
  pos = (x, y)
  if pos in memo:
    return memo[pos]

  # If we are out of bounds, we know that we have reached the end of the grid and should just return 0
  if x == len(grid) or y == len(grid[0]):
    return 0
  
  # However, if we have reached the end of the grid, we should just return its value as the base case
  if x == len(grid) - 1 and y == len(grid[0]) - 1:
    return grid[x][y]
  
  # To make things more neat, we create two variables that represent the downward and rightward movement of our paths
  down = _max_path_sum(grid, x + 1, y, memo)
  right = _max_path_sum(grid, x, y + 1, memo)
  # We then set the memo[pos] to the grid[x][y] + max(down, right); this not only determines which positions have been explored and stores it into our memo, but it also returns the max value that was explored
  memo[pos] = grid[x][y] + max(down, right)
  return memo[pos]

# Time complexity: O(n); We must iterate through every number in the given list
# Space complexity: O(n); We must store every possible result into our memo
def non_adjacent_sum(nums):
  # Create a memo that will store the results of completed calculatios so that they can be pulled later on
  memo = {}
  # We create a helper function that takes in an index argument, i, that will represent where our algorithm is currently checking
  return _non_adjacent_sum(nums, 0, memo)

def _non_adjacent_sum(nums, i, memo):
  # If our i is already in the memo, then we know we can just return its value
  if i in memo:
    return memo[i]

  # If our i has grown past the len of our numbers, then we know that we are out of bounds and should return 0
  if i >= len(nums):
    return 0
  
  # We create two variables; include will include our current number and any non-adjacent numbers as well; exclude will only include the numbers after our initial start
  include = nums[i] + _non_adjacent_sum(nums, i + 2, memo)
  exclude = _non_adjacent_sum(nums, i + 1, memo)
  # Both include and exclude will return the highest sum possible and we should just find the max of that
  # It is better to think of the possible combinations of numbers as a graph for this dynamic programming problem
  memo[i] = max(include, exclude)
  # You want to return memo[i] since it contains the maximum non-adjacent sum
  return memo[i]

# Time complexity: O(n * sqrt(n)); At worst we will go up to sqrt of n (n times)
# Space complexity: O(n); At worst we will store n results in our memo
def summing_squares(n):
  # Create a memo to store already calculated results
  memo = {}
  return _summing_squares(n, memo)

def _summing_squares(n, memo):
  # If we already have the value in our memo, we should just return it instead of calculating it again
  if n in memo:
    return memo[n]

  # If n reaches 0, we can then return 0 as we know we have found a solution that we can stop at
  # Note that we will never reach below n because we are iterating through i > 0 starting from 1
  if n == 0:
    return 0
  
  # Create an arbitrarily large value that will be replaced by a minimum number of summed squares
  min_squares = float('inf')
  # We want to start from 1 (our first perfect square), and go on only to the sqrt(n)
  for i in range(1, math.floor(math.sqrt(n) + 1)):
    # Make a square out of current i
    square = i * i
    # Determine the num_squares for each combination of i
    num_squares = 1 + _summing_squares(n - square, memo)
    # Set min_squares as the newest minimum
    min_squares = min(min_squares, num_squares)
  
  # Set memo[n] as min_squares
  memo[n] = min_squares
  # Return memo[n] or min_squares as it has already been calculated
  return memo[n]

# Time complexity: O(amount * coins); We are building a tree with coins as height and amount as width and will have to explore all of the nodes in the tree
# Space complexity: O(amount * coins); We will store all of the amounts into the memo
def counting_change(amount, coins):
  memo = {}
  return _counting_change(amount, coins, 0, memo)

# Our helper function will take in an index i that will track which coin we are trying to determine the remaining amounts from
def _counting_change(amount, coins, i, memo):
  # We create a Python tuple here that will serve as our memo key - we need this because we want to track which amounts and index of coins amounts to the resulting value
  key = (amount, i)
  if key in memo:
    return memo[key]

  # Our function will reduce amount by the currently tracked coin - this will eventually reach zero and return us one way to create the amount with the change
  if amount == 0:
    return 1

  # This will serve as a base case - end the function when we finish the possible coins
  if i == len(coins):
    return 0

  # This variable will keep track of our total methods to count the change
  total_count = 0
  # We are going to track the current coin using the index i argument
  current_coin = coins[i]
  # We need to track the remaining amounts when coins are duplicated (i.e., 4 = 1 + 1 + 1 + 1) and other possible options
  # We have to make sure that the math is range(0, (amount // current_coin) + 1) because the amount needs divided by the current_coin as much as possible; +1 because we need to reach the end of our range (be inclusive)
  for qty in range(0, (amount // current_coin) + 1):
    # Track the remaining remainders with every possible quantity of coin
    remainder = amount - (qty * current_coin)
    total_count += _counting_change(remainder, coins, i + 1, memo)

  memo[key] = total_count
  return total_count

# Time complexity: O(n^2); We will basically be iterating through each number in numbers squared (from 1 to the number) because we have to check if there are any possible combination of moves to reach the end
# Space complexity: O(n); We only store O(n) results since we are just checking up to the length of numbers
def array_steppers(numbers):
  memo = {}
  # In our helper function, we want to start off at index 0 and find possible routes from the number at index 0
  return _array_steppers(numbers, 0, memo)

def _array_steppers(numbers, i, memo):
  # As always, if we have the result already calculated in our memo, we should just return it
  if i in memo:
    return memo[i]

  # If i has passed the length of numbers, then we know it was possible to reach the end of the array as we can iterate through anything less than the step
  if i >= len(numbers) - 1:
    return True
  
  # We set up a max_step that will track the steps possible for the current number, which should be from 1 to the number
  max_step = numbers[i]
  # For each step, we just want to make sure that _array_steppers can calculate if there is a possible true value and then return it
  for step in range(1, max_step + 1):
    if _array_steppers(numbers, i + step, memo):
      memo[i] = True
      return True
  
  # If we couldn't find any possible values, then we should just return False
  memo[i] = False
  return False

# Time complexity: O(n^2); We must go through each possible substring which will require O(n^2)
# Space complexity: O(n^2); Since we are slicing to create substrings, we need to store all of the possible substrings and their respective results
def max_palin_subsequence(string):
  # Our helper function will take in an i and j (two pointers) along with a memo
  return _max_palin_subsequence(string, 0, len(string) - 1, {})

def _max_palin_subsequence(string, i, j, memo):
  # We store i and j in a tuple that will serve as our memo keys
  key = (i, j)
  if key in memo:
    return memo[key]

  # If the two pointers meet, we know that we can return 1 because there is only one letter left in the string
  if i == j:
    return 1

  # If the two pointers cross over each other, then we know that our string has reached the middle and is no longer a valid string
  if i > j:
    return 0

  # If the two ends of the string are the same character, we know we can add + 2 to our max length as those two letters are palindrome
  # Our next sequence for matching letters should be to increment i and decrement j to check if the next two letters are palindromes
  if string[i] == string[j]:
    memo[key] = 2 + _max_palin_subsequence(string, i + 1, j - 1, memo)
  else:
    # Else, we can just recursively call _max_palin_subsequence on an incremented i and decremented j to check every available substring combination
    # Recall that we are looking for the max so we should only return one result that contains the highest value
    memo[key] = max(
      _max_palin_subsequence(string, i + 1, j, memo),
      _max_palin_subsequence(string, i, j - 1, memo)
    )
  
  # We already set the memo[key] in the previous line s and can just return it
  return memo[key]

# Time complexity: O(n * m); We must iterate through both strings to determine overlapping subsequences
# Space complexity: O(n * m); We are storing results from both strings so the space complexity will be determined by both sizes
# Note for this problem, we purposefully choose to use two pointers instead of slicing our arrays as it is more efficient than recreating new arrays with new slices every time
def overlap_subsequence(string_1, string_2):
  # We call a helper method that will take in a memo and two pointer variables set at the beginning of each string
  return _overlap_subsequence(string_1, string_2, 0, 0, {})

def _overlap_subsequence(string_1, string_2, i, j, memo):
  # For any dynamic programming two pointer questions, you probably want to set up a tuple as a key and then have that tuple serve as the key for your memo
  key = (i, j)
  if key in memo:
    return memo[key]

  # We want to return 0 if we reach the end of either string because we are essentially saying that we do not have any subsequence left to compare to
  if i == len(string_1) - 1 or j == len(string_2) - 1:
    return 0

  # If we have a matching character, we need to add one and return another recursive call on the following characters at i + 1 and j + 1
  if string_1[i] == string_2[j]:
    memo[key] = 1 + _overlap_subsequence(string_1, string_2, i + 1, j + 1, memo)
  # If we don't have a matching character, we need to find the max found from skipping a character on either string
  else:
    memo[key] = max(
      _overlap_subsequence(string_1, string_2, i + 1, j, memo),
      _overlap_subsequence(string_1, string_2, i, j + 1, memo)
    )

  return memo[key]

# Time complexity: O(s * words); You will have to explore the string and the words length together at worst
# Space complexity: O(s); You are only going to store strings that have been "sliced" by the words you iterate through, so the worst space complexity is O(s)
def can_concat(s, words):
  return _can_concat(s, words, {})

def _can_concat(s, words, memo):
  # For string manipulation questions, consider "slicing" the string by the words to see if you can reach the answer; in this case, we must store the sliced strings into our memo
  if s in memo:
    return memo[s]
  
  # If the string is empty, we know we have successfully found a combination of words that have reached the end of the string and we can return True as our base case
  if s == '':
    return True
  
  # For each word in words list, we want to check if the current string starts with it
  for word in words:
    if s.startswith(word):
      # If the current string does start with the word, then we just need to check the back-half of the word (excluding the word at the beginning)
      suffix = s[len(word):]
      # If we can recursively find words in the string that end up causing the string to go to length 0, then we know we have a true case and can return True after setting the memo as True
      if _can_concat(suffix, words, memo):
        memo[s] = True
        return True

  # Otherwise, we want to return false because we could not find words that reduced the strength to length zero
  memo[s] = False
  return False

# Time complexity: O(s * words); You will have to explore the string and all combination of words that may add up to the string
# Space complexity: O(s); You will store all possible combinations of the reduced string so the pace complexity is dependent on only the string
def quickest_concat(s, words):
  # We want to set a result variable that will contain our result; this is due to the fact that we may have an infinite integer returned from our helper function if no possible combination is found
  result = _quick(s, words, {})
  # We return -1 if we do not find a possible combination; otherwise, we just return our result which should hold the minimum moves to concat to the word
  if result != float('inf'):
    return result
  else:
    return -1

def _quick(s, words, memo):
  # Check if we already have calculated for a part of the string before
  if s in memo:
    return memo[s]
  
  # If we have reached the end of the string, then we know that we found a possible combination and can return 0 to start being added onto
  if s == '':
    return 0
  
  # Set up a variable that will track the quickest result
  quickest = float('inf')
  # For each word in the words list, we check if the string begins with the word and then pass along the remaining string into a recursive call
  for word in words:
    if s.startswith(word):
      suffix = s[len(word):]
      # We make sure to add 1 to the result of the recursive call in case it finds a result that fits
      attempt = 1 + _quick(suffix, words, memo)
      # We overwrite quickest with the fastest attempt
      quickest = min(attempt, quickest)
  
  # Don't forget to set memo[s] to quickest and then return quickest
  memo[s] = quickest
  return quickest

# Time complexity: O(n); You run through the entirety of the string once so the solution is at worst O(n)
# Space complexity: O(1); You will only hold a stack / count variable that will track how many valid parentheses there are
def paired_parentheses(string):
  # We use a count variable instead of a stack list in this scenario because we are only tracking parentheses - if we used other brackets then a stack would be better
  count = 0
  
  # Iterate through each character in the string and count to the count if we have an open parentheses
  for char in string:
    if char == '(':
      count += 1
    # Check if we have a closed parentheses that was provided before an open parentheses (i.e., count == 0); if so, then we know we can just return False
    # Otherwise, we can decrement count because we have a valid parentheses
    elif char == ')':
      if count == 0:
        return False
      count -= 1
  
  # We then just return whether count is 0 or not, indicating that all parentheses (if any) were valid
  return count == 0

# Time complexity: O(n); You run through the entire string at least once
# Space complexity; O(n); At worst, you may store the entire string into the stack
def befitting_brackets(string):
  # Create a hashmap that will store all of our brackets - this is most optimal to pull information from as well
  brackets = {
    '(': ')',
    '{': '}',
    '[': ']'
  }

  # Create a stack that we will use to cover our brackets
  stack = []

  for char in string:
    # If the character is a key in brackets, then we just add the value of that key to our stack
    if char in brackets:
      stack.append(brackets[char])
    else:
      # If the stack is truthy and its last character is our character, then we can just pop it off
      if stack and stack[-1] == char:
        stack.pop()
      # Otherwise, we know we have reached either that the stack is falsey or it is not our character and thus can return False early
      else:
        return False
  
  # After we finish our algorithm, want to make sure that the stack is empty - we can do this by checking if the stack is falsey
  return not stack

# Time complexity: O(9^m * n); m = brace groups, n = regular characters; The algorithm is exponential because you can have as many as 9 numbers be determined by the amount of groups * any number of regular characters
# Space compelxity: O(9^m * n); Same reason as above - our stack will continuously hold the characters and will thus be exponential
# It is helpful to draw this problem out as a stack to see what goes into our stack
def decompress_braces(string):
  # We can solve this problem using a stack and a set of defined numbers
  stack = []
  numbers = '123456789'

  for char in string:
    # If our character is a number, we just want to add it to our stack - this will eventually be popped off and used to multiply a segment of characters
    if char in numbers:
      stack.append(char)
    else:
      # If the character is the end of a group of characters, then we need to start tracking a segment that will hold our repeated characters
      if char == '}':
        segment = ''
        # isinstance is a Pythonic way of checking whether the first argument is an instance of the second character
        # We want to make sure that we are continuously popping off elements until we reach an integer that marks our number to multiply the segment by
        while not isinstance(stack[-1], int):
          popped = stack.pop()
          # When we add the element to the segment, we want to make sure we add the popped first because otherwise it will reverse the characters in the sub-group
          segment = popped + segment
        number = stack.pop()
        # We can multiply the segment by the number to repeat it
        # We make sure to add it back to our stack
        stack.append(segment * number)
      # If our character isn't an open bracket then we can just add it to our stack
      elif char != '{':
        stack.append(char)
  
  # We return our now completed stack using a ''.join function
  return ''.join(stack)

# Time complexity: O(n); We must go through all of the characters in the string
# Space complexity: O(n); We may end up storing iterations of each character in the stack
def nesting_score(string):
  # We start off with a stack containing 0 because we want our base case to be zero in case there is no stack provided
  # Note that the [0] will be revised to contain our final answer by our algorithm
  stack = [0]
  for char in string:
    # If our character is an open bracket, we know that we are about to either fill it with more brackets or close it
    if char == '[':
      # If we are opening it, we just add onto our stack another 0 value
      stack.append(0)
    else:
      # We check what to do with the character by first popping it off
      popped = stack.pop()
      # If the value of our stack is currently at zero, we can add +1 to the new last stack element
      if popped == 0:
        stack[-1] += 1
      # Else if we already had a closed bracket before, then we know that we have to multiply the last closed bracket number by 2 and add it back to our stack
      else:
        stack[-1] += 2 * popped
  
  # We return stack[0] which should now have been revised for a new figure
  return stack[0]

# Time complexity: O(2^n); For each element in our input array, we will have two possible subsets that will contain the first element and the rest of the elements
# Space complexity: O(2^n); We are recursively calling subsets on n number of elements so our stack will be 2^n
def subsets(elements):
  # Have a default base case that will return an empty array
  if not elements:
    return [[]]
  
  # The first element and the rest of the elements should be set as variables
  first = elements[0]
  rest = elements[1:]
  # We want to recursively call subsets on the rest of the elements and assume that we receive an array containing all of the subsets of the rest of the elements
  rest_subsets = subsets(rest)

  # We set up an empty first_subset that will continuously append our first element and any subsets afterwards
  first_subset = []
  # For each subset returned from our earlier recursive call, we will append it after our first sub_set result
  for sub in rest_subsets:
    first_subset.append([first, *sub])
  
  # We want to add our first subset and the rest of our subsets because we want to make sure that we have every possible element accounted for
  return first_subset + rest_subsets

# Time complexity: O(n!); For n distinct items, there are n! permutations that are returned as the answer so we have at least O(n!) time complexity
# Space complexity: O(n!); We are returning an array of results containing our permutations of n! results so our space complexity will also be O(n!)
def permutations(items):
  # This is our base case; we want to return an empty array if we have nothing left in our items
  if not items:
    return [[]]

  # Select our first letter of the array -  we will "insert" this into our permutations
  first = items[0]
  remaining = items[1:]
  # Assume that the recursive call will return a list containing all other combinations
  perms = permutations(remaining)
  # We have a result array that will contain all of our possible permutations
  result = []
  # For each permutation, we are going to go through the range of the permutation and insert our first element into each spot
  for perm in perms:
    # Make sure we add + 1 to the range so we can insert the first element into the end of our range as well
    for i in range(len(perm) + 1):
      # Then we append every result of the permutation before and after the first element is inserted
      result.append([*perm[:i], first, *perm[i:]])
  
  # Return the result containing all of the permutations
  return result

# Time complexity: O(n! / k!(n-k)!) or (O("n choose k")); Binomial coefficient
# Space complexity: O(n! / k!(n-k)!) or (O("n choose k")); Binomial coefficient
# We can also solve this greedily using subsets and then finding all results of length k but this solution is more elegant and more efficient
def create_combinations(items, k):
  # If the length of our items is shorter than our k, then we know we cannot make combos of k length and should just return an empty array
  if len(items) < k:
    return []
  
  # If k == 0, we have reached the end of our results and should just return an empty array containing an empty list
  if k == 0:
    return [[]]
  
  # Similarly to other exhaustive recursion problems, we can take out the first item and set up an array that will contain our first element's combinations
  first = items[0]
  first_combos = []
  # We want to begin populating our first_combos array with the results of items after the first element and k - 1
  # We do k - 1 because we will want to return results that will be of length k - 1 since we are adding back the first element (to be size k in total)
  for combo in create_combinations(items[1:], k - 1):
    # This will continuously create combinations of size k
    first_combos.append([first, *combo])
  
  # We then call the function on every item afterwards to see if we can find any other combination of length k
  remaining_combos = create_combinations(items[1:], k)

  # We then return both combinations concatenated - this will return a combined list of combinations that are length k
  return first_combos + remaining_combos

# n = string length
# m = max group size
# Time complexity: O(m ^ n); Your group size can be arbitrarily large and the string length within the group can also be arbitrarily large
# Space complexity: O(m ^ n); You must store all of the results of O(m ^ n) size into an array as a result
def parenthetical_possibilities(s):
  # If our string reaches a length of zero, we cannot find any possiblities and should just return an empty string as our base case output
  if len(s) == 0:
    return ['']

  # We set up an array containing all of our possibilities - we will fill this array and return it with the answer
  all_possibilities = []
  # We deconstruct the choices and the remainder found by the find_choices helper function
  choices, remainder = find_choices(s)
  # Now for each choice that we have within a group of characters, we create branching trees that will look into each choice and add the possibilites of recursive calls onto it
  for choice in choices:
    # This is our recursive leap of faith - we assume that recursively calling parenthetical possibilities will return the rest of the possibilities
    remainder_possibilities = parenthetical_possibilities(remainder)
    # For each possibility, we want to add the choice to the front of it and then append that result
    all_possibilities += [choice + possibility for possibility in remainder_possibilities]
  
  # Don't forget to return all the possibilities we found
  return all_possibilities
  
# We create a helper function that will determine whether we are approaching a group of characters - if we are, we will return a tuple containing the choices in the group and the remaining characters in the string
def find_choices(str):
  if str[0] == '(':
    end = str.index(')')
    choices = str[1:end]
    remainder = str[end + 1:]
    return (choices, remainder)
  else:
    # Otherwise, we will just return the first character as a "choice" and the remainder of the string
    return (str[0], str[1:])

# n = number of words in the sentence
# m = max number of synonyms for each word
# Time complexity: O(m ^ n); We may potentially have a synonym for each word in the sentence and thus we should be O(m ^ n) times at worst
# Space complexity: O(m ^ n); We must store all of the potential results (synonym for each word in the sentence) so our space complexity is also O(m ^ n)
def substitute_synonyms(sentence, synonyms):
  # First we split the words up into a list
  words = sentence.split(' ')
  # We pass the words into our helper function - we assume that the helper function will return subarrays containing all of the possible combinations of the sentence using each synonym for each word in synonyms
  subarrays = generate(words, synonyms)
  # We join all of the results back together for each subarray in the resulting subarrays
  return [' '.join(subarray) for subarray in subarrays]

def generate(words, synonyms):
  # We must start off with our base case which is that words will be of length 0 and we should thus return an empty list
  # Since our base is on the length of words, we should assume that we will decrement words length after every recursive call
  if len(words) == 0:
    return [[]]

  # Set the first and remaining words and variables
  first_word = words[0]
  remaining = words[1:]
  # Assume that we can recursively call and receive the remaining words' subarrays
  subarrays = generate(remaining, synonyms)
  
  # We then check the first word to see if it matches a word in synonyms
  if first_word in synonyms:
    # If there is a word in synonyms, then we need to return a result that contains all of the results using synonym words
    result = []
    # For each synonym, we also recursively call upon subarray in subarrays to make sure that we assuring for other words in the remaining sentence with synonyms
    for synonym in synonyms[first_word]:
      result += [[synonym, *subarray] for subarray in subarrays]
    # All of our results should we stored within the results array so we can return it
    return result
  else:
    # Otherwise, we know that there are no other possibilities for the first word and should just return it along with the rest of the subarray results
    return [[first_word, *subarray] for subarray in subarrays]

# Time complexity: O(n); We are going through each node in the linked list
# Space complexity: O(n); We are storing each node's value in the linked list and then comparing it to the reversed version of itself to check for palindrome
def linked_palindrome(head):
  # Set up a list of values that will be reversed and checked later on
  values = []
  # Set up a current node that will track the node's value
  current = head
  # Use falsey to check whether current is not None
  while current:
    # Append every value into our values array so that we know whether or not the reversed version will be a palindrome
    values.append(current.val)
    # Make sure to always set current to the next one
    current = current.next
  # Use Python to determine whether values is equal to its reversed list (values[::-1] returns another array that starts from the end and goes to the beginning)
  # Note that the syntax for slicing an array / reversing it as follows: [start:end:order] -> you can leave out start and end to return the full array in the order that you want (-1 indicates reversed)
  return values == values[::-1]
  # Note that double equal operator in Python will check if both lists contain the same values and in the same order

# We have two possible solutions here: we can use a list to track all of the values and take the middle index or we can use two pointers to determine the middle value; the two pointers method is more efficient in terms of space complexity
# Time complexity: O(n); We must iterate through the entire linked list to determine the midpoint value
# Space complexity: O(1); We are not storing any variables that are reliant on the input size
def middle_value(head):
  # Set up two pointers, a slow and fast pointer that will track the progress of each node
  slow = head
  fast = head
  # We have to check whether fast is currently not None and also when fast.next is current not None - this is because we are taking two steps at a time and can risk skipping too far ahead into a null node
  while fast is not None and fast.next is not None:
    # The fast node will take two steps at a time while the slow node will take only one step at a time
    # Since fast node is going twice as fast and will eventually reach the end of the linked list faster than the slow one, we know that wherever slow ends up is the middle of the linked list
    slow = slow.next
    fast = fast.next.next
  # We just have to return slow.val since we know that it is currently at the middle of the linked list
  return slow.val

# This is the same as Leetcode #141 - Linked List Cycle; we are using Floyd's Cycle Detection Algorithm in both answers
# Similar to middle_value, we can find two possible solutions; either by using a set to track nodes that we have already seen or by using two pointers
# Time complexity: O(n); Since our fast pointer is traveling at twice the speed of the slow algorithm, at worst our slow pointer will have reached the end of the linked list once before it is "caught" by the fast pointer
# Space complexity: O(1); We are using constant space to keep track of our pointers
def linked_list_cycle(head):
  # Set up a slow and fast pointer
  slow = head
  fast = head
  
  # While fast is not reaching a null node, we can continuously go through the linked list
  while fast and fast.next:
    # We first move the pointers to the next and next next nodes, respectively, for slow and fast
    slow = slow.next
    fast = fast.next.next
    # Then we determine whether fast has caught up to node and is the same value - if it is, then we have a cycle and just return True
    if slow == fast:
      return True
  
  # If we exit out of the while loop, then fast must have hit a null node and thus is not a cycle
  return False

# This problem is similar to Leetcode #236 Lowest Common Ancestor of a Binary Tree; The major difference is that 
# Time complexity: O(n); We must iterate through every node in the binary tree to determine the ancestor
# Space complexity: O(n); We are using recursion to go through each node in our tree, so our implicit stack will be O(n) space complexity
def lowest_common_ancestor(root, val1, val2):
  # The root will have two possible paths - we just need to find paths for both val1 and val2; these paths represent the ancestry of both values
  path1 = find_path(root, val1)
  path2 = find_path(root, val2)
  # We create a set of path2 since we are going to continuously check its elements for a matching element; we use a set for constant time complexity
  # If we used an array, it would still result in the answer but it would be O(n^2) time complexity
  set2 = set(path2)
  for val in path1:
    if val in set2:
      return val
  
def find_path(root, target_val):
  # If the root is None, then we have reached the end of the tree
  if root is None:
    return None
  
  # If we have reached the target value, then we know that this is a possible path and we can just return the root value for now
  if root.val == target_val:
    return [ root.val ]
  
  # We then recursively check if any of paths contain our target value; if we find it, we want to append our current root.val so that its value is noted as a pathway
  left_path = find_path(root.left, target_val)
  if left_path is not None:
    left_path.append(root.val)
    # Then we return the path because we know we found a possible area
    return left_path
  
  right_path = find_path(root.right, target_val)
  if right_path is not None:
    right_path.append(root.val)
    return right_path
  
  # Otherwise, if we found nothing in either path, then we can just return None
  return None

# Time complexity: O(n); We are going through every node to determine if there is a child to flip
# Space complexity: O(n); We are using a recursive call so we have an implicit stack of O(n) space complexity at worst
def flip_tree(root):
  # If we reach a None node, we can just return None as our base case
  if root is None:
    return None

  # Set up a recursive call assuming that the tree will flip
  left, right = flip_tree(root.left), flip_tree(root.right)
  # Then set your root.left and root.right to the right and left, respectively, to flip the nodes
  root.left, root.right = right, left
  # This is all done in place so you should be able to just return your root
  return root

# Time complexity: O(n); We must traverse every node to determine whether it is left or not
# Space complexity: O(n); We are recursively solving the solution so we use an implicit stack of O(n) worst space complexity
def lefty_nodes(root):
  values = []
  # We use a helper function to traverse our root at each level - it will add to values only the leftmost value
  traverse(root, 0, values)
  return values

def traverse(root, level, values):
  # If root is None, we just want to stop the function
  if root is None:
    return
  
  # If the length of values is equal to the level, we should add the root value - this will only add the left-most root value
  if len(values) == level:
    values.append(root.val)

  # We then recursively call upon traverse to go through each possible level
  traverse(root.left, level + 1, values)
  traverse(root.right, level + 1, values)

# Time complexity: O(n^2); At worse, we may have to travel through every node twice
# Space complexity: O(n); We use a stack to recursively check each neighbor and we use a hashmap to contain each node's "color"
def can_color(graph):
  # In this case, we want to use a hashmap to store our node's "colors", which will be represented by True and False
  coloring = {}
  # For each node in the graph, we want to check if the node is NOT already registered in our coloring hashmap
  for node in graph:
    if node not in coloring:
      # If the node is NOT registered, then we can check if it is valid by coloring it False first
      if not valid(graph, node, coloring, False):
        return False
  
  return True
  
# Our helper function will check if the node is in coloring hashmap and return its color
def valid(graph, node, coloring, current_color):
  if node in coloring:
    return current_color == coloring[node]
  
  # If not in coloring, then we can set its current color
  coloring[node] = current_color
  
  # Then we want to recursively check its neighbors 
  for neighbor in graph[node]:
    # We make sure to flip the color by using not current_color
    if not valid(graph, neighbor, coloring, not current_color):
      return False
  
  # If everything works, then we know that we have just set each color to an alternate and can return true at the end
  return True

# Time complexity: O(e); You must travel through every edge provided in the problem so the time complexity will be O(e)
# Space complexity: O(n); You must store every node that has been traveled to with a "color", so the space complexity will be O(n)
# This problem is very similar to our can_color; we just need to build our graph and then check if we have rivalries that will overlap
def tolerant_teams(rivalries):
  # Build the graph first using our traditional helper function that takes in edges (rivalries) and spits out a graph in the form of a hashmap
  graph = build_graph(rivalries)
  
  coloring = {}
  for node in graph:
    if node not in coloring and not is_bipartite(graph, node, coloring, False):
      return False
    
  return True

def is_bipartite(graph, node, coloring, current_color):
  if node in coloring:
    return coloring[node] == current_color
  
  coloring[node] = current_color
  
  for neighbor in graph[node]:
    if not is_bipartite(graph, neighbor, coloring, not current_color):
      return False
    
  return True

def build_graph(rivalries):
  # This is an undirected graph whose neighbors can point to each other
  graph = {}
  for pair in rivalries:
    first, second = pair
    if first not in graph:
      graph[first] = []
    if second not in graph:
      graph[second] = []
    
    graph[first].append(second)
    graph[second].append(first)
    
  return graph

# Time complexity: O(n^2); Since our graph is undirected, we must explore all available routes twice to determine whether there are unique routes
# Space complexity: O(n); We are setting up a visited set that will contain all of the visited nodes, so at worst we will have all nodes or O(n)
def rare_routing(n, roads):
  # We must build our graph first out of the roads that we have
  graph = build_graph(n, roads)
  # We must have a visited set that will contain all of the nodes that we have already explored
  visited = set()
  # We use a helper function starting with the graph we created, node 0, the visited set, and a None to represent the last node that we visited
  traversed = traverse(graph, 0, visited, None)
# Traversed will return a boolean that checks whether there are unique routes - we also check if the length of the visited set is the same as the number of cities; this is because we must have ONE path to each city - if the city is not connected to 0 then we do not have a path!
  return traversed and len(visited) == n

def traverse(graph, node, visited, last_node):
  # If we already have the node, return False
  if node in visited:
    return False
  
  # Make sure to add the node to visited afterwards
  visited.add(node)
  # First part of our if statement prevents us from traveling backwards on a path
  # Second part determines whether we have traveled the path and came across an already explored path, which means we have MORE than one path to that node
  for neighbor in graph[node]:
    if neighbor != last_node and not traverse(graph, neighbor, visited, node):
      return False
  
  # If neither of the above are true, then we can return True
  return True
  
def build_graph(n, roads):
  # Since we know the number of cities in our problem and our nodes are labeled using numbers starting from 0 to n, we can enumerate using the range(n) to set up neighbors for each node
  graph = {}
  
  for i in range(n):
    graph[i] = []
  
  for road in roads:
    a, b = road
    graph[a].append(b)
    graph[b].append(a)
  
  return graph

# Time complexity: O(n^2); Normally the solution would be 2^n because for every numbers we find a subset that includes the current number and a subset that excludes it - since we are memoizing we only use O(n^2) instead
# Space complexity: O(n^2); We are memoizing all of the possible subsets that we are iterating through so we only go through O(n^2) space complexity at worst
def max_increasing_subseq(numbers):
  return _max_increasing_subseq(numbers, 0, float('-inf'), {})

def _max_increasing_subseq(numbers, i, previous, memo):
  # Our i and previous arguments will be changing per iteration so we must use those as our keys (stored in a Python tuple)
  key = (i, previous)
  
  # If we already have the key, we can return it
  if key in memo:
    return memo[key]
  
  # We know that if we reach the length of our input array, we have reached the end and should thus return
  if i == len(numbers):
    return 0
  
  # We want to check our current number
  current = numbers[i]
  # We create an options list to check which subsets will contain the max - we do this later
  options = []
  # We set a variable that will result in the answer of whatever the next subset doesn't take
  dont_take_current = _max_increasing_subseq(numbers, i + 1, previous, memo)
  # We also add that result to our options to determine which is better
  options.append(dont_take_current)
  # If our current is greater than our previous (which starts at float('-inf))
  if current > previous:
    # Then we will set our take_current variable as +1 and use our current as the next "previous" argument
    take_current = 1 + _max_increasing_subseq(numbers, i + 1, current, memo)
    # We also want to add this result to options so we can tell whether it will be the maximum number of not
    options.append(take_current)
  
  # Make sure to store our answer as a memo[key] so that we can return it
  memo[key] = max(options)
  # In Python, you can just use max() on a list to determine the highest value - in this case we want the highest max_increasing_subset resulting from dont_take_current and take_current
  return memo[key]

# Time complexity: O(nm); We must iterate through every position and its cost to check whether it is the best combination
# Space complexity: O(nm); Since we are storing our results in a memo, we may at worst store all of our results in the memo
def positioning_plants(costs):
  # We return a value using a helper function that will take in the original costs argument, a 0 representing the initial position we are determining, a None argument which will represent the last_plant that we looked at, and a hashmap for our memo
  return _positioning_plants(costs, 0, None, {})

def _positioning_plants(costs, pos, last_plant, memo):
  # Since our position and last_plant arguments will change, we use a tuple to store their values in our mmeo
  key = (pos, last_plant)
  # Check first if we already have the result in our memo and return it
  if key in memo:
    return memo[key]

  # If our position has reached the length of costs, then we know we are past our last position and should return 0
  if pos == len(costs):
    return 0

  # Set up a variable that will determine the minimum cost of the plants - we set it initially to float('inf') so that it is arbitrarily large
  min_cost = float('inf')

  # For each plant (position / index) and cost (element value), check first that the plant isn't the same as the last_plant (position) that we already confirmed and then check if it is a possible candidate
  for plant, cost in enumerate(costs[pos]):
    # Since we don't want to have a plant adjacent to its previous spot - we need to check if plant != last_plant (represents the index value within each costs subarray)
    if plant != last_plant:
      # The candidate variable will recursively check all of the possible plants, their costs, and also determine whether the plant is not adjacent to its previous plant
      candidate = cost + _positioning_plants(costs, pos + 1, plant, memo)
      # We keep checking the minimum cost and setting it lower if the candidate is better
      min_cost = min(candidate, min_cost)

  # Make sure to set the result as a memo[key] so that we can return it later if required
  memo[key] = min_cost
  return min_cost

# Time complexity: O(m * n * k); At worst, we may go through the size of the row (m) and column (n) but only up to k moves, so our time complexity is O(mnk)
# Space complexity: O(m * n * k); Since we are using a memo to store our results, our worst space complexity will be O(mnk) as well
def breaking_boundaries(m, n, k, r, c):
  # Use a helper function with a memo
  return _breaking_boundaries(m, n, k, r, c, {})

def _breaking_boundaries(m, n, k, r, c, memo):
  # Our k (number of moves) and our current row / column position will
  key = (k, r, c)
  if key in memo:
    return memo[key]

  # Check if we have reached the outer bounds of m and n - if we have, then we know that we have broken the boundary and can return 1
  row_inbounds = 0 <= r < m
  col_inbounds = 0 <= c < n
  if not row_inbounds or not col_inbounds:
    return 1

  # If k is 0, then we have run out of moves and should just return 0
  if k == 0:
    return 0

  # Set up a count that will track our count
  count = 0
  # Set up a list containing tuples of deltas that we will deconstruct and use for our next moves
  deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  for delta in deltas:
    # Deconstruct the delta
    d_row, d_col = delta
    # Count should add the next position
    count += _breaking_boundaries(m, n, k - 1, r + d_row, c + d_col, memo)

  # Make sure to store the count in our memo in case we go back to the position
  memo[key] = count
  # Return count
  return count

# Time complexity: O(n logn); It is O(n) to run the merge helper function while it is O(logn) to run merge on the left_sorted and right_sorted paths - these are basically cutting the input array down in half each iteration
# Space complexity: O(n); We store our argument in data structures like a deque and a list
def merge_sort(nums):
  # We can just return nums if its only 1 or less in length since there is nothing else to sort it with
  if len(nums) <= 1:
    return nums
  
  # We merge_sort nums[:middle] and nums[middle:] to get the left and right part of the original nums
  # Note that we call merge_sort and not merge! We are only merging at the end!
  middle = (len(nums) // 2)
  left_sorted = merge_sort(nums[:middle])
  right_sorted = merge_sort(nums[middle:])
  return merge(left_sorted, right_sorted)

def merge(list1, list2):
  # We use collections deque to lower our time complexity - we are popping elements off of the front which requires a deque to do so in constant time
  list1 = deque(list1)
  list2 = deque(list2)
  
  # Our merged array will contain the merged results and will be returned in our recursive stack
  merged = []
  # While both lists exist and are truthy
  while list1 and list2:
    # Check the first elements of each
    if list1[0] < list2[0]:
      # Append the lesser one to our merged array
      merged.append(list1.popleft())
    else:
      merged.append(list2.popleft())
  # Then add the rest of list1 and list2 - this ensures that whatever is left is still added
  merged += list1
  merged += list2
  return merged

# This is also problem 56. Merge Intervals on Leetcode
# Time complexity: O(n logn); We are sorting the intervals here so we can go through the rest of the solution in a one-pass
# Space complexity: O(n); We are returning the array in a combined result that contains our array - we thus use O(n) since our variable combined will be the same input size
def combine_intervals(intervals):
  # First we sort our intervals array by the starting times, or x[0] for each interval
  sorted_intervals = sorted(intervals, key=lambda x: x[0])
  # Then we create a combined array that will contain just the first sorted interval - this will essentially act as our "first" array that we will check every other array with
  combined = [ sorted_intervals[0] ]
  
  # For every current_interval in our sorted_intervals starting from the second element onward (remember, we are comparing subsequent elements against the first one for now)
  for current_interval in sorted_intervals[1:]:
    # We deconstruct the last added interval (last_start, last_end from combined[-1]) and we deconstruct the current_interval that we are checking
    last_start, last_end = combined[-1]
    current_start, current_end = current_interval
    # If the current_start is less than or equal to the last_end, then we know we have an intersecting interval and we need to determine when to add that interval to our combined
    if current_start <= last_end:
      # If the current_end is also greater than our last end, then we know that we have ended the interval and can thus combine it / replace our last element of the combined with the new interval
      if current_end > last_end:
        combined[-1] = (last_start, current_end)
    else:
      # If we don't have an intersecting interval, then we can just append the current_interval since it isn't up for consideration
      combined.append(current_interval)
    
  # Then we return combined
  return combined

# Time complexity: O(logn); We effectively split the search in half since we know where to look in a sorted array
# Space complexity: O(1); We will have constant space complexity
def binary_search(numbers, target):
  # Set two pointers that will track our indexes
  low = 0
  high = len(numbers) - 1
  # While low is less than or equal to the highest (we want to make sure it's <= because we need to check if we need to check if it is the middle)
  while low <= high:
    middle = (low + high) // 2
    # If the target is less than the value in the middle, then we know that it is in the left half of the current window that we are checking and should slide high - 1
    if target < numbers[middle]:
      high = middle - 1
    # Else if the target is greater than the value in the middle of our window, we can slide low + 1
    elif target > numbers[middle]:
      low = middle + 1
    # Otherwise if we found our target, we can just return its value
    else:
      return middle
  # However, if we continuously slide our function and don't find anything, we can return -1
  return -1

# Time complexity: O(n); We will visit every node in the BST to discover its value
# Space complexity: O(n); We are storing our results in an array and then checking that array to determine whether the order is correct (i.e., is it a binary tree)
def is_binary_search_tree(root):
  values = []
  # Create a helper function that will populate our values array
  add_values(root, values)

  # Check the values array post-helper function call to determine whether the order of values are not in increasing order - if it isn't, then we know that our tree is not a BST
  for i in range(len(values) - 1):
    if values[i + 1] < values[i]:
      return False

  # Otherwise we can return true
  return True

def add_values(root, values):
  # We want to end our function early if our root is None
  if root is None:
    return

  # We want to recursively add our leftmost values first and then our root value and our right value last
  add_values(root.left, values)
  values.append(root.val)
  add_values(root.right, values)

# Time complexity: O(n); We are going to traverse through every node to check and add their value
# Space complexity: O(n); Since we are returning all of the values in an array, our space complexity must be O(n)
def post_order(root):
  values = []
  # Similar to the last problem, we will use a helper function to add all of our values to the array that we created
  post_the_order(root, values)
  return values

def post_the_order(root, values):
  # AIf the root if None, we can just end the function early
  if root is None:
    return
  
  # Since post order goes from left, right, and then self, we do the recursive of left, right, and then self
  post_the_order(root.left, values)
  post_the_order(root.right, values)
  values.append(root.val)

# Time complexity: O(n^2): Since we are slicing our arrays, we are creating new arrays and iterating through both input arguments
# Space complexity: O(n^2): Since we are solving recursively and creating new arrays through slicing, our worst case space complexity will be O(n^2)
# This is to build and initialize the class Node
class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

# Our function will take in two arguments of the same length and will spit out a tree built off of the in_order and post_order layouts
# We need both arguments because otherwise, we cannot build the tree with 100% certainty (leaves can be in different places than expected if we only use in_order)
def build_tree_in_post(in_order, post_order):
  # Since we know that in_order and post_order are the same length, we just need to check if one of the arguments will be length 0
  if len(in_order) == 0:
    return None

  # Then we find the value of the last element in post_order - we know that this will serve as our root because post_order trees are built around left, right, and then self
  value = post_order[-1]
  # Since post order is built aorund left, right, and then self, we know that its last value must be the root of our tree
  root = Node(value)
  # We need to find the index of our tree's root in the in_order array - we so use .index(value) to find it
  mid = in_order.index(value)
  # Then we split the in_order array into two different arrays so that we can find the post_order of both 
  left_in = in_order[:mid]
  right_in = in_order[mid + 1:]
  # We need to do some careful slicing here to ensure that we have the same values in our left_in and left_post and the same values from our right_in and right_post
  left_post = post_order[:len(left_in)]
  right_post = post_order[len(left_in):-1]
  # Feed both the left sides and right sides into a recursive call - this will automatically build out our tree as it will continue to find the middle val (root) of the sub-trees and build
  root.left = build_tree_in_post(left_in, left_post)
  root.right = build_tree_in_post(right_in, right_post)
  # Then we just return the root as it will contain the full tree
  return root

# Time complexity: O(n^2); Similar to the last problem, we are slicing our arrays and iterating through every node
# Space complexity: O(n^2); Similar to the last problem, we are creating new arrays and solving recursively (stack)
# This is very similar to the last problem except now we are building in pre_order; check the non-slicing solution below for a better time and space complexity solution
def build_tree_in_pre(in_order, pre_order):
  if len(in_order) == 0:
    return None
  
  value = pre_order[0]
  root = Node(value)
  mid = in_order.index(value)
  left_in = in_order[:mid]
  right_in = in_order[mid + 1:]
  left_pre = pre_order[1: 1 + len(left_in)]
  right_pre = pre_order[1 + len(left_in):]
  root.left = build_tree_in_pre(left_in, left_pre)
  root.right = build_tree_in_pre(right_in, right_pre)
  return root

# Time complexity: O(n); We are no longer slicing the arrays but instead tracking indexes, so our time complexity improves to O(n)
# Space complexity: O(n); Since we are no longer slicing arrays and instead only using indexes, our space complexity is only reduced to the implicit recursive stack
def build_tree_in_pre2(in_order, pre_order):
  return _build_tree_in_pre(in_order, pre_order, 0, len(in_order) - 1, 0, len(pre_order) - 1)

def _build_tree_in_pre(in_order, pre_order, in_start, in_end, pre_start, pre_end):
  # If the index of the end is less than the start, then we have gone too far and should return None as the leaf node
  if in_end < in_start:
    return None

  # We begin the same as our slicing version - we create a value at pre_order[0] (or pre_start)
  value = pre_order[pre_start]
  # Set the root as that value
  root = Node(value)
  # Find the index in in_order of that value
  mid = in_order.index(value)
  # Then we need to determine the size of the left and right trees to split upon; We know it will be mid - in_start because we slice from the middle to 0 in the beginning
  left_size = mid - in_start
  # Next we recursively call but change the indexes that we begin from
  # For root.left: We still start from the beginning but not end in-start at mid - 1 to indicate this is the left side
  # We start at pre_start + 1 since we already know pre_start[0] was our root; we also end at pre_start + left_size since we know that is our "end"
  root.left = _build_tree_in_pre(in_order, pre_order, in_start, mid - 1, pre_start + 1, pre_start + left_size)
  # For root.right: in_start should begin at mid + 1 since we found the middle value; in_end should end at the same length
  # We start _pre_start at + 1  + left_size to indicate that we have skipped the first element and skipped the left side that is already factored in root.left; we still end at pre_end
  root.right = _build_tree_in_pre(in_order, pre_order, mid + 1, in_end, pre_start + 1 + left_size, pre_end)
  # Our tree is now built out recursively and we can return root
  return root

# Time complexity: O(min(n, m)); We only need to iterate until the end of the shorter word
# Space complexity: O(1); No extra space taken here so this should just be O(1)
def lexical_order(word_1, word_2, alphabet):
  # Find the greater length - we want to keep iterating up until that length or until one of the words are no longer iterable
  greater_length = max(len(word_1), len(word_2))
  
  # This for loop will continuously check if the letters at i from both words are the same - if they are, we can increment i and check the next
  for i in range(greater_length):
    # First we find and assign a value to the letter - if i < len(word) then we must find an index; if i > len(word) then we know the index is out of bounds and should just assign float('-inf')
    value_1 = alphabet.index(word_1[i]) if i < len(word_1) else float('-inf')
    value_2 = alphabet.index(word_2[i]) if i < len(word_2) else float('-inf')
    
    # We check if value_1 is < value_2 - we need to check this first because we are trying to see if word_1 is in lexical order (comes before word_2)
    # If instead value_2 is < value_1, then we know that word_2 has either ended early or its indexed character is coming before word_1's character and therefore it is not lexical
    # Else, if value_1 == value_2 (implicitly implied here) - then we know we can just increment i and check the next letter
    if value_1 < value_2:
      return True
    elif value_2 < value_1:
      return False
    
  # If we have gone through both words, then we know that they are both the same word and can just return True
  return True

# Time complexity: O(e + n); We must travel through each edge and node so our total time complexity will be O(e + n)
# Space complexity: O(n); We will store the nodes with their respective counts in a hashmap (num_parents) - this will result in a time complexity of at worst O(n)
def topological_order(graph):
  # Since we are going in topological order, there must be at most one node with zero parents
  # To find that node, we will go through each node in the graph and add up the count of their parents
  num_parents = {}
  # Initialize each node in the graph to have zero parents
  for node in graph:
    num_parents[node] = 0
    
  # Then go through each each child node in that graph's node and increment by one to that child's num_parents count
  for node in graph:
    for child in graph[node]:
      num_parents[child] += 1
      
  # Now we know that we will have a node that does not have any parents - we can initialize a "ready" array with the parentless node that will be used to pop off and add to an output array
  ready = [ node for node in graph if num_parents[node] == 0]
  # The order array will contain our actual results
  order = []
  # While the ready array continues to exist...
  while ready:
    # Pop off the current node to review
    node = ready.pop()
    # Append that node to the order
    order.append(node)
    # Then check that node's children
    for child in graph[node]:
      # For each child, we want to decrement its count in num_parents
      num_parents[child] -= 1
      # If the count reaches zero, we know that the node has exhausted all possible parents and can finally be added to the ready list
      # Otherwise, we don't want to add to the ready list because we haven't actually exhausted all parents yet and the order is inaccurate
      if num_parents[child] == 0:
        ready.append(child)

  # Finally, we want to return the order
  return order

# Time complexity: O(e); We are first given a list of edges that we must build out a graph of -> our worst came time complexity will be O(e)
# Space complexity: O(e); We are given a list of edges that we will build a graph (hashmap) out of so our overall space complexitiy will be O(e)
def safe_cracking(hints):
  # Use a helper function to build a directed acyclic graph
  graph = build_graph(hints)
  return topological_order(graph)
  
def build_graph(edges):
  # Build a graph using the helper function
  graph = {}
  # For each edge, deconstruct the edge and create arrays that will contain the nodes of children for both nodes
  for edge in edges:
    a, b = edge
    if a not in graph:
      graph[a] = []
    if b not in graph:
      graph[b] = []
    # Since this is a directed acyclic graph, we know that we can append only b (child) to a (parent)
    graph[a].append(b)
  return graph

# Same algorithm as last time
def topological_order(graph):
  num_parents = {}
  for node in graph:
    num_parents[node] = 0
  
  for node in graph:
    for child in graph[node]:
      num_parents[child] += 1
  
  ready = [ node for node in graph if num_parents[node] == 0 ]
  order = ''
  while ready:
    node = ready.pop()
    order += str(node)
    for child in graph[node]:
      num_parents[child] -= 1
      if num_parents[child] == 0:
        ready.append(child)
        
  return order

# Time complexity: O(3^xy); Each position in the grid that we visit will have at most three potential other positions to check; since we are doing this in a grid it will be O(3^xy) time complexity
# Space complexity: O(xy); We are recursively iterating through each position in the grid so our implicit stack will be O(xy)
def string_search(grid, s):
  # For each position in the grid, we want to check if our helper function will return True
  for x in range(len(grid)):
    for y in range(len(grid[0])):
      if dfs(grid, x, y, s):
        return True
      
  # If not, we can just return False
  return False

def dfs(grid, x, y, s):
  # If our string is empty, it must mean that we have found a possible path and can thus return True
  if s == "":
    return True
  
  # Find out whether we are inbound the grid or not - return False if not else continue
  row_inbounds = 0 <= x < len(grid)
  col_inbounds = 0 <= y < len(grid[0])
  if not row_inbounds or not col_inbounds:
    return False
  
  # Check if the current position's character is what we are looking for (first letter of the string)
  char = grid[x][y]
  # If not, we can just return False
  if char != s[0]:
    return False
  
  # If the current position's character is the first character of our string, then we know we can begin iterating through and checking for other possible letters in the sequence
  # Create a suffix that will track the remaining leters to check through
  suffix = s[1:]
  # Temporarily make our current char a filler character - this will prevent our dfs from returning to this position
  grid[x][y] = "*"
  # Result will go to the top, right, bottom, and left positions to check if their letters will continue to track our suffix
  result = dfs(grid, x + 1, y, suffix) or dfs(grid, x - 1, y, suffix) or dfs(grid, x, y + 1, suffix) or dfs(grid, x, y - 1, suffix)
  # Reset our current grid's character in case result is False, else the graph will contain null characters and potentially ruin results
  grid[x][y] = char
  # Return result assuming that dfs has return True if at least one path was found or False if none were found for all positions
  return result

# Time complexity: O(n); We are doing a single pass-through of the input string
# Space complexity: O(n); We are storing the results in a new output array that will be O(n)
# On a surface level, this problem seems pretty simple but you will need to notice that punctuation will affect how your tokens are recognized
def token_replace(s, tokens):
  # We create an output that will later be joined to return a string containing our replaced words
  output = []
  # We use two pointers with j always being ahead of i
  i = 0
  j = 1
  
  # While i is less than the length of the string...
  while i < len(s):
    # We check first if s[i] is == to "$" - if it is not, we want to just append the current character to output and increment both i and j
    if s[i] != "$":
      output.append(s[i])
      i += 1
      j = i + 1
    # If s[i] == "$" then we need to check where the token ends within the string - we continuously increment j += 1 until we reach the end of the token
    elif s[j] != "$":
      j += 1
    # Since we found the beginning and end of a token, we can take set s[i: j + 1] as our key and find out the value of the token
    else:
      key = s[i: j + 1]
      # We append the value of the token to our output
      output.append(tokens[key])
      # Then we set the tokens to start after the end of our most token
      i = j + 1
      j = i + 1
  
  # Then we return the output as a joined string
  return "".join(output)

# n = length of string
# m = # of tokens
# Time complexity: O(n^m); We must pass through our string but will also pass through the number of tokens possible that contain other tokens - so our best possible case is O(n^m)
# Space complexity: O(n^m); Since we are using an output that will contain our result, we can also expect a space complexity of O(n^m)
def token_transform(s, tokens):
  # Same code as token_replace until we hit our else conditional
  output = []
  i = 0
  j = 1
  while i < len(s):
    if s[i] != "$":
      output.append(s[i])
      i += 1
      j = i + 1
    elif s[j] != "$":
      j += 1
    else:
      key = s[i:j + 1]
      value = tokens[key]
      # We want to first solve for our evaluated value by recursively passing our value into our token_transform -> this will set evaluated_value to the resulting replaced value in the key
      evaluated_value = token_transform(value, tokens)
      # We want to actually memoize this and set the tokens[key] to the evaluated value - failing to do so will cause our program to run factorial
      tokens[key] = evaluated_value
      # Now that we have our evaluated value, we can just add it to our output as before
      output.append(evaluated_value)
      i = j + 1
      j = i + 1
  return "".join(output)