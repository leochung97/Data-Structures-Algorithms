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