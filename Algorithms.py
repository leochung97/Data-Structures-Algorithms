from math import sqrt, floor
from statistics import mean

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
from collections import deque
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
from collections import deque
from statistics import mean
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
  return [ mean(level) for level in levels ]

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