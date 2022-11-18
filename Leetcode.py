from typing import *
from math import *
from collections import Counter

# 1. Two Sum
class Solution:
  # Time complexity: O(n) because you run through the entire List at worst
  # Space complexity: O(n) as you may store the entire List in the hashmap at worst
  def twoSum(self, nums: List[int], target: int) -> List[int]:
    hash = {}
    # Note: Range gives you a range from 0 to the number (not inclusive of the number) if only a single argument is provided
    for i in range(len(nums)):
      complement = target - nums[i]
      if complement in hash:
        return [hash[complement], i]
      hash[nums[i]] = i

# 13. Roman to Integer
class Solution:
  # Time complexity: O(n); You are running through each character of the string to determine the final result
  # Space complexity: O(1); While we are using a hashmap to store some constants, we are actually not using any additional space complexity that is based on input size
  def romanToInt(self, s: str) -> int:
    # Create a numerals hashmap
    numerals = {
      "I": 1,
      "V": 5,
      "X": 10,
      "L": 50,
      "C": 100,
      "D": 500,
      "M": 1000
    }

    total = 0
    i = 0
    while i < len(s):
      # Check if the number ahead is greater than the current number - if it is, then we know we are supposed to subtract the first number and add to the second number
      if i + 1 < len(s) and numerals[s[i]] < numerals[s[i + 1]]:
        total += numerals[s[i + 1]] - numerals[s[i]]
        # Skip by 2 since we know we just added two figures
        i += 2
      else:
        # Otherwise we can just add it
        total += numerals[s[i]]
        i += 1

    return total

# 15. 3Sum
class Solution:
  # Terrible solution - would suggest no-sort solution
  # Time complexity: Asymptotically this is O(n^2) but you should point out that it is actually O(n + nlogn + 2n^2)
  # Space complexity: O(n); We are storing a bunch of lists and sets that cumulate to O(n) space max
  def threeSum(self, nums: List[int]) -> List[List[int]]:
    results = set()

    neg, pos, zeros = [], [], []
    for num in nums:
      if num > 0:
        pos.append(num)
      elif num < 0:
        neg.append(num)
      else:
        zeros.append(num)
    
    # (0, 0, 0)
    if len(zeros) >= 3:
      results.add((0,0,0))
    
    neg_set, pos_set = set(neg), set(pos)

    # (-3, 0, 3)
    if zeros:
      for num in pos_set:
        if -1 * num in neg_set:
          results.add((-1 * num, 0, num))
    
    # (-1, -2, 3)
    for i in range(len(neg)):
      for j in range(i + 1, len(neg)):
        target = -1 * (neg[i] + neg[j])
        if target in pos_set:
          results.add(tuple(sorted([neg[i], neg[j], target])))
    
    # (1, 2, -3)
    for i in range(len(pos)):
      for j in range(i + 1, len(pos)):
        target = -1 * (pos[i] + pos[j])
        if target in neg_set:
          results.add(tuple(sorted([pos[i], pos[j], target])))

    return results

# 20. Valid Parentheses
class Solution:
  # Time complexity: O(n) because you run through the entire string
  # Space complexity: O(n) as the stack will be the size of the input
  def isValid(self, s: str) -> bool:
    # String is not valid if it is an odd length as that means the stack will always have a remaining character
    if len(s) % 2 == 1:
      return False
    
    # Sets up a hashmap containing matching pairs of valid parentheses
    hash = {
      "(":")",
      "{":"}",
      "[":"]"
    }
    
    # Stack allows you to go through the string in order to pop off items on the stack in order; you can use a List or Array for stacks (recall from aA)
    stack = []
    for char in s:
      if char in hash:
        stack.append(char)
      elif len(stack) == 0 or hash[stack.pop()] != char:
        return False

    return len(stack) == 0

# 21. Merge Two Sorted Lists
class ListNode:
  def __init__(self, val = 0, next = None):
    self.val = val
    self.next = next

class Solution:
  # Time complexity: O(n + m) as it depends on the size of both linked lists
  # Space complexity: O(1) because we only set up a few variables that are not determined by input size
  def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    # We set up two variables here because sort_list will continuously be reassigned to its next value whereas output will only keep a receipt of the sort_list nexts
    output = sort_list = ListNode(0)

    # Checks if both lists are populated; if not, we know that we can just assign the sorted_list to the remaining list
    while (list1 and list2):
      if (list1.val < list2.val):
        sort_list.next = list1.val
        # Need to make sure that you reassign list1 to be its next value or the while loop will keep going
        list1 = list1.next
      else:
        sort_list.next = list2.val
        # Need to make sure that you reassign list2 to be its next value or the while loop will keep going
        list2 = list2.next
      # Makes sure to go to the next node of sort_list so that the next tail value can be assigned to another list value
      sort_list = sort_list.next
    
    # Will set sort_list head to whichever linked list has values remaining
    sort_list = list1 or list2
    return output.next

# 53. Maximum Subarray
class Solution:
  # Time complexity: O(n); We iterate through each number in the List once so the time complexity at worse will be O(n)
  # Space complexity: O(1); We do not use any extra space based on input size as we only set two variables
  def maxSubArray(self, nums: List[int]) -> int:
    # This is a dynamic programming solution using Kadane's Algorithm; Essentially, we iterate through each element of the array and determine whether that array is worth keeping
    # Set up two variables equal to the first element of the input List
    current_sub = max_sub = nums[0]

    # For each number starting from the second element of the input List
    for num in nums[1:]:
      # Continuously set current_sub to either the max of the number it is enumerating or the current_sub + the number if higher; this will basically take out any negatives and reset the current_sub to the current number if that's higher
      current_sub = max(num, current_sub + num)
      # You want to also set the max_sub equal to either the max of itself and the current_sub
      max_sub = max(max_sub, current_sub)
    
    # Return max_sub at the end because that's what we are really looking for
    return max_sub

# 56. Merge Intervals
class Solution:
  # Time complexity: O(n logn); We use a sort to then do a one-pass through
  # Space complexity: O(n); We are using variable combined to store our merged intervals so our solution could be at worst O(n) in case nothing gets merged
  def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    # Sort the input intervals by each element's starting time (x[0])
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    # Create a combined interval that will store our results - input the first sorted interval
    combined = [ sorted_intervals[0] ]
    
    # Starting from the second element of sorted_intervals...
    for current_interval in sorted_intervals[1:]:
      # We deconstruct the last added interval
      last_start, last_end = combined[-1]
      # We also deconstruct the current interval that we are looking at
      current_start, current_end = current_interval
      # We check if the current start is less than or equal to the last interval's ending value - if it is, then we know that there is an intersecting interval
      if current_start <= last_end:
        # However, we need to check if this is the only interval that we need to check, so we need to check if the current end is greater than the last end - if it is, then we know that the new interval should just be last_start, current_end
        if current_end > last_end:
          # We replace the last interval
          combined[-1] = [last_start, current_end]
      else:
        # Otherwise, if we don't have an intersecting interval, we can just add it to our combined intervals
        combined.append(current_interval)
    
    return combined

# 57. Insert Interval
class Solution:
  # Time complexity: O(n); We are passing the array once so our worst time complexity is O(n)
  # Space complexity: O(n); Our output is storing the new array which at worst can be O(n) size
  def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    # We set up an i to track the index we are going through and an output to track our results
    n = len(intervals)
    i, output = 0, []
    
    # In our first while loop, we need to figure out which intervals are not even overlapping with our new interval beginning yet; this can be determined by checking whether the end of an interval is less than our new interval start
    while i < n and intervals[i][1] < newInterval[0]:
      output.append(intervals[i])
      i += 1
    
    # Once the first while loop ends, we know that we have found the interval that overlaps with our new interval and must now iterate until we find an interval that does not start before our new interval end
    while i < n and intervals[i][0] <= newInterval[1]:
      # As we continuously increment i, we need to continuously reset the new interval start to the minimum of the intervals in between and the maximum of the interval ends
      newInterval[0] = min(intervals[i][0], newInterval[0])
      newInterval[1] = max(intervals[i][1], newInterval[1])
      i += 1
    # Once we exit out of the previous while loop, we know that we have found our interval and can just append the new one
    output.append(newInterval)
    
    # We are now iterating through the rest of our intervals and just adding whatever we have remaining into our output
    while i < n:
      output.append(intervals[i])
      i += 1
    
    return output

# 70. Climbing Stairs
class Solution:
  # Time complexity: O(n); You must go through n steps to calculate the number of steps
  # Space complexity: O(n); We are storing all of the results from climbing stairs so we use O(n) space
  def climbStairs(self, n: int) -> int:
    # Set up a hashmap that will memoize the results that we have already calculated - this will help reduce the function from O(n^2) to O(n) since we are not running the sequence multiple times for results that we have already calculated
    memo = {}
    # Initialize the results of climbing stairs once or twice
    memo[1] = 1
    memo[2] = 2

    # Helper function to actually climb and store data into the memo
    def climb(n):
      # If we already have the result in our memo, we shouldn't want to run another recursive call on it - we should just pull it from our hashmap
      if n in memo:
        return memo[n]
      # If we don't have the result in our memo already, then we should recursively call climb to get the result - this is basically the Fibonacci sequence
      else:
        # Make sure that we memoize it though
        memo[n] = climb(n - 1) + climb(n - 2)
        return memo[n]

    # Call climb(n) to get the number of steps possible - this is basically just the Fibonacci sequence answer
    # You can technically answer this question in the same manner of the Fibonacci Number (#509)
    return climb(n)

# 98. Valid Binary Search Tree
class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

class Solution:
  # Time complexity: O(n); We go through every node in the tree
  # Space complexity: O(n); We temporarily store all node values in an "order" array that will be used to determine whether the tree is inorder or not
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
      # Binary Search Tree's must be in order of values or it is NOT a binary search tree
      # We set up an order array that will contain all node values -> this will then be iterated through to determine whether values are in order or not
      order = []
      # Use a helper function to build out the inorder values
      self.inOrder(root, order)

      # Simple function to determine whether a node's value is not in order
      for i in range(len(order) - 1):
        if order[i] >= order[i + 1]:
          return False

      # After everything, we can return True
      return True

    # Recursively add the left values first and then the middle and then the right values
    def inOrder(self, root, order):
      # Return early if root is None because we've reached the end of tree
      if root is None:
        return

      # Recurse left first because we want this in order!
      self.inOrder(root.left, order)
      # Then add the current node
      order.append(root.val)
      # Then recurse right
      self.inOrder(root.right, order)

# 110. Balanced Binary Tree
class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

class Solution:
  # Time complexity: O(n); For every subtree, we compute its height in constant time as well as compare the height of its children
  # Space complexity: O(n); The cursion stack may go up to O(n) if the tree is unbalanced
  def _isBalanced(self, root: TreeNode):
    # An empty tree is balanced and has a height of -1
    if not root:
      return True, -1

    # Check the subtrees to see if they are balanced
    leftIsBalanced, leftHeight = self._isBalanced(root.left)
    if not leftIsBalanced:
      return False, 0

    rightIsBalanced, rightHeight = self._isBalanced(root.right)
    if not rightIsBalanced:
      return False, 0

    # If the subtrees are balanced, check if the current tree is balanced using their height
    return (abs(leftHeight - rightHeight) < 2), 1 + max(leftHeight, rightHeight)
    
  def isBalanced(self, root: Optional[TreeNode]) -> bool:
    return self._isBalanced(root)[0]

# 121. Best Time to Buy / Sell Stock
class Solution:
  # Time complexity: O(n) as we only go through the input array of prices once; Space complexity: O(1) because we only set up two variables that aren't determined by input size
  def maxProfit(self, prices: List[int]) -> int:
    # Set up two variables, min_price which acts as our left pointer and is initially set to an arbitrarily large figure (infinity) to ensure that it is replaced by any stock value
    # And max_profit, which is set to 0 to ensure that we return 0 if we do not find any profits at all
    min_price, max_profit = float('inf'), 0

    # price acts as our right pointer and will basically check if the right pointer value gives us a profit against the left pointer -> this will update max_profit if higher
    for price in prices:
      min_price = min(min_price, price)
      max_profit = max(max_profit, price - min_price)
    
    # Returns the max_profit at the end which can be zero if no profits have been found or if the List was empty in the first place
    return max_profit

# 125. Valid Palindrome
class Solution:
  # Time complexity: O(n), we traverse over each character at-most once until the two pointers meet in the middle
  # Space complexity: O(1), we only save variables for the pointers and their size is not determined by the size of the string input
  def isPalindrome(self, s: str) -> bool:
    # Set up two pointers, left and right, determined by the starting and ending index of the string
    left, right = 0, len(s) - 1
    
    # Makes sure that the pointers meet in the middle
    while left < right:
      # Checks if either the left or the right character is not an alphanumeric character; basically removes any spaces
      while left < right and not s[left].isalnum():
        left += 1
      while left < right and not s[right].isalnum():
        right -= 1
      
      # Actually checks if the alphanumeric characters are not the same going forward and backwards, meaning it's not a palindrome if found to be false
      if s[left].lower() != s[right].lower():
        return False
      
      # Want to make sure that you're still going through the string even after checking
      left += 1
      right -= 1
    
    # Return true because the while loop has been exited
    return True

# 141. Linked List Cycle
class Solution:
  # Time complexity: O(n); The fast pointer reaches the end first and the run time depends on the list's length, which is O(n)
  # Space complexity: O(1); We only use two variable nodes (slow and fast) so the space complexity is going to be O(1)
  # Use Floyd's Cycle Finding Algorithm:
  # Set two pointers that act as a slow and fast runner - eventually, the fast runner will catch up to the slow runner if this is cyclical
  # If the fast runner never catches up or the linked list ends, then you know that the list does not cycle
  def hasCycle(self, head: Optional[ListNode])  -> bool:
    # This is for empty lsits
    if head is None:
      return False
    
    # Set up two pointers that act as a slow and fast runner
    slow = head
    fast = head.next

    # Continuously iterate through the linked list to check if the fast runner's nodes are None - if they are, then you know that the linked list ends and is not cyclical
    while slow != fast:
      if fast is None or fast.next is None:
        return False

      slow = slow.next
      fast = fast.next.next

    # You only return true after exiting the while loop (after determining that slow is indeed equal to fast)
    return True

# 152. Maximum Product Subarray
class Solution:
  # Time complexity: O(n); We will go through the entire array so we are size O(n)
  # Space complexity: O(1); We are using constant space to track our variables
  def maxProduct(self, nums: List[int]) -> int:
    # Base case if nums list is empty
    if not nums:
      return 0

    # Set up two variables - one will track the highest and the other will track the lowest
    result = max_so_far = min_so_far = nums[0]
    # Result will be returned but should be continuously set to max_so_far

    for i in range(1, len(nums)):
      curr = nums[i]
      # We need to track both the lowest min (negative) and highest max (positive) and then determine whether that is our temp_max
      temp_max = max(curr, max_so_far * curr, min_so_far * curr)
      min_so_far = min(curr, max_so_far * curr, min_so_far * curr)

      max_so_far = temp_max

      result = max(max_so_far, result)

    return result

# 167. Two Sum II
class Solution:
  # This is a twist on the classic Two Sum except the input array is now sorted, which indicates that you should probably use a binary search or two pointers; note that binary search is slightly slower O(nlog n)
  # Time complexity: O(n); We are using two pointers method to check the elements at each pointer but at worst we may still go through every element in the array
  # Space complexity: O(1); We are only using two variables so this is constant space
  def twoSum(self, numbers: List[int], target: int) -> List[int]:
    # Set up the left and right pointers to be equal to the start of the index and the end of the index, respectively
    left, right = 0, len(numbers) - 1

    # While left is less than right...
    while left < right:
      # Check if the left number + right number is equal to the target and return the answer (in this case we add +1 because the solution specifies so)
      if numbers[left] + numbers[right] == target:
        return [left + 1, right + 1]
      # Else if the added number is less than the target, then we know we have to increment the left pointer to increase the sum
      elif numbers[left] + numbers[right] < target:
        left += 1
      # Else if the added number is greater than the target, then we know we have to decrement the right pointer to decrease the sum
      else:
        right -= 1
    
    # Return [-1, -1] if we cannot find a pair that sums up to the target
    return [-1, -1]

# 169. Majority Element
class Solution:
  # First solution is by solving using a hashmap
  # Time complexity: O(n); You must go through the list at least once to store all of the integers
  # Space complexity: O(n); You must store all of the integers in the list into the hashmap so the worst space complexity is O(n)
  def majorityElement(self, nums: List[int]) -> int:
    # Create a hashmap and set it as a variable
    hashmap = {}
    # For each number, we want to increment the count and then check whether it is the majority number
    for num in nums:
      hashmap[num] = hashmap.get(num, 0) + 1
      if hashmap[num] > len(nums) / 2:
        return num

  # Second solution is by solving using Boyer-Moore majority voting algorithm
  # Time complexity: O(n); Boyer-Moore performs constant work n times, so the algorithm runs in linear time
  # Space complexity: O(1); Allocates only constant memory
  def majorityElement2(self, nums: List[int]) -> int:
    # Set up a majority and count variable to track the majority number and a count that will keep track of the majority count
    majority =  0
    count = 0
    for num in nums:
      # If the count is zero, then you know that the majority number has been reset and should thus be set to the new num
      if count == 0:
        majority = num
      # If the majority isn't already the number, then just decrement the count of it
      if majority != num:
        count -= 1
      # However, if the majority is already the number, then we need to increment the count of it
      else:
        count += 1
    # Return the tracked majority number
    return majority

# 226. Invert Binary Tree
class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

class Solution:
  # Time complexity: O(n); You must traverse the entire node tree so the best possible solution is O(n)
  # Space complexity: O(n); You must return the reversed tree so it will be O(n) space complexity (dependent on input)
  def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    # Checks if the root node exists
    if root:
      # Recursively reverses the root node's branches
      root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
      # Make sure to return root here so that the recursion continues unless if statement is no longer completed
      return root
    # No need to explicitly return root here but invertTree would return [] if no root
    return root

# 235. Lowest Common Ancestor of a Binary Search Tree
# Note that this is a Binary Search Tree, which requires the left and right node values to be less than and greater than, respectively, the root node values
# Time Complexity: O(n); This is the worst case scenario as you might have to traverse every node in the tree if the tree is a singly-linked list; Note that the average time complexity would be O(log(n)) as you split the tree nodes in half in this iterative solution
# Space Complexity: O(1); This is not iterated using recursion so there is no stack and the only variable that is saved is not reliant on the input size

class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # Set a current node to continuously check through
    # Note that this assumes that there is a p and q
    node = root
    # While this node exists (is not None)
    while node:
      parent_val = node.val
      # Since we know this is a binary search tree, we can check if p and q are both less than or greater than. If one is less and the other is greater, than it means that the current node must be our lowest common ancestor because we cannot traverse down left or right and find another common ancestor
      # Else if p and q are both greater than or less than the current node value, then we can just traverse into one side (left or right) to get closer to the LCA
      if p.val > parent_val and q.val > parent_val:
        node = node.right
      elif p.val < parent_val and q.val < parent_val:
        node = node.left
      else:
        return node

# 236. Lowest Common Ancestor of a Binary Tree
# Time complexity: O(n); We must iterate through every node in the tree to determine the lowest common ancestor
# Space complexity: O(n); We are using a recursive call which will have us use a stack of at worst O(n)
class Solution:
  # This is a very simple recursive function that will solve for the lowest common ancestor of both p and q
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # if root is None, p, or q, we should just return it
    if root in (None, p, q ):
      return root

    # Otherwise, we set up a left and right pathway that looks for p and q in its left / right nodes
    left, right = (self.lowestCommonAncestor(kid, p, q) for kid in (root.left, root.right))
    # If left and right both return a root value (not a None), then we know that our path has been found and we can return it
    # Otherwise, we must go into either left or right to determine which has both ancestors
    return root if left and right else left or right

# 242. Valid Anagram
class Solution:
  # Time complexity: O(n); You must traverse both strings whose length is n each
  # Space complexity: O(1); Size of the strings doesn't matter as only a hashmap is stored containing the count of the strings
  def isAnagram(self, s: str, t: str) -> bool:
    # Checks if the length of the strings are the same; if not, return False
    if len(s) != len(t):
      return False
    
    # Counter is a built-in Python function that comes from collections library; it iterates through every character in the string and creates a hashmap counter
    return Counter(s) == Counter(t)

    # Otherwise, you can use the following answer to be more explicit:
    # counter = collections.defaultdict(int)
    # for char in s: counter[char] += 1
    # for char in t: counter[char] -= 1
    # return all(char == 0 for char in counter.values())

# 252. Meeting Rooms
class Solution:
  # Time complexity: O(nlog n); You sort the intervals so the best time complexity is O(nlog n)
  # Space complexity: O(1); No additional space needed to solve this
  def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
    # Sort all of the intervals by the starting time
    intervals.sort(key=lambda x: x[0])

    # For each interval starting from the second one, check if the starting interval is less than the previous ending interval
    for i in range(1, len(intervals) - 1):
      # If it is less, then we know that we have intersecting meeting times and thus the person cannot attend all meetings -> return False
      if intervals[i][0] < intervals[i - 1][-1]:
        return False

    # If no falses, then we should be good
    return True

# 392. Is Subsequence
class Solution:
  # Time complexity: O(t); At worst, we will go through all of string t
  # Space complexity: O(1); We use constant space to track our pointers
  def isSubsequence(self, s: str, t: str) -> bool:
    # If string s is shorter than string t, then there is no way that string s will be a subsequence of string t
    if len(s) > len(t):
      return False
    
    # If string s is length 0 or non-existent, then it is automatically a subsequence of string t (can just delete all characters)
    if len(s) == 0:
      return True
    
    subsequence = 0
    # We set our first pointer, i, to track the characters at index i of string t
    for i in range(len(t)):
      # If the subsequence is currently less than the length of string s and the current character we are looking at matches, then we can add 1 to subsequence
      # Subsequence tracks the order and characters of string s - if we go in order and find that we contain all of the characters from string t, we know that string s is a valid subsequence
      if subsequence <= len(s) - 1 and s[subsequence] == t[i]:
        subsequence += 1
    
    # We just check at the end if we have reached the length of string s, aka checked every character
    return subsequence == len(s)

# 409. Longest Palindrome
class Solution:
  # Time complexity: O(n); You must go through each character in the input string to store into the set
  # Space complexity: O(n); You may store every character in the input string into a set in the worst case (every character is different)
  def longestPalindrome(self, s: str) -> int:
    # Set up a new set using Python set
    count = set()
    # For every character in the input string (whether it is capitalized or not), input the character into the set if not already there and remove it if it is
    for char in s:
      if char not in count:
        count.add(char)
      else:
        count.remove(char)
    
    # Check if the length of the set is 0; if it is zero, then you know that every character in the input string has a pair and is therefore the longest palindrome
    # Otherwise, you know that the length of the remaining set contains all of the non-paired characters that can be removed from the "longest" calculation
    # You add one to the end of the answer because you want to make sure that one of the non-paired characters is included into the longest palindrome
    if len(count) != 0:
      return len(s) - len(count) + 1
    else:
      return len(s)

# 509. Fibonacci Number
class Solution:
  # This is a two-pointer solution to the classic Fibonacci problem
  # Time complexity: O(n); You must iterate n times to return the final fibonacci number
  # Space complexity: O(1); You use constant space for the variables assigned
  def fib(self, n: int) -> int:
    # Assign two variables to 0 and 1
    a, b = 0, 1
    # For each number from 0 to n, you will reassign a and b variables to b and a + b, respectively, essentially getting to the next fibonnaci number by using two pointers
    for i in range(n):
      a, b = b, a + b
    # You want to return a instead of b here because b will always be reassigned to the next fibonacci number and not the current one
    return a
  # Note: There is a more optimal solution here using the golden ratio forumla but it would require you to know the golden ratio (1 + (5 ** 0.5)) / 2
  # The time complexity is more efficient using the golden ratio (O(log n)) but it requires knowledge of the golden ratio 

# 542. 01 Matrix
from collections import deque

class Solution:
  # Time complexity: O(x * y); We must go through every position in the grid
  # Space complexity: O(x * y); At worst, we may have to temporarily store every number in the grid for our queue
  # We are going to implement this solution using BFS because we must determine the distance away from a 0
  def updateMatrix(self, grid: List[List[int]]) -> List[List[int]]:
    # We are using BFS so set up a queue
    queue = deque()
    # We don't want to revisit any cells that we've already calculated values for, so set up a set
    visited = set()
    for x in range(len(grid)):
      for y in range(len(grid[0])):
        # For every cell, we want to see the distance from 0's - we can calculate this by starting from every 0 and working towards a non-zero figure
        if grid[x][y] == 0:
          visited.add((x, y))
          queue.append((x, y))

    while queue:
      # Deconstruct x and y from the queue's topmost position
      x, y = queue.popleft()
      directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
      # For each direction, we want to check if the neighbors have already been visited and whether they are valid
      # If they have not been visited and are valid, we know that they must be NON-ZERO figures (otherwise they would have been in visited already) - we can increment their values based on that
      for dirr in directions:
        dx, dy = dirr
        newX, newY = x + dx, y + dy
        x_inbounds = 0 <= newX < len(grid)
        y_inbounds = 0 <= newY < len(grid[0])

        if x_inbounds and y_inbounds and (newX, newY) not in visited:
          # Increment by one using the current position's value (could be > 0)
          grid[newX][newY] = grid[x][y] + 1
          # Make sure to add our new neighbor to the visited since we have technically changed its value and visited it
          visited.add((newX, newY))
          # Then make sure to add it to our queue so that we can check its neighbors as well
          queue.append((newX, newY))
    
    return grid

# 543. Diameter of a Binary Tree
class Solution:
  # Time complexity: O(n); You must iterate through all nodes to determine the diameter of the binary tree
  # Space complexity: O(n); You recursively call upon longest_path so there is an stack containing each node
  def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    # Set up a diameter variable that will track the diamater and be returned later
    diameter = 0

    # Set up a helper function that will actually count the nodes for us
    def longest_path(node):
      # We use Python's nonlocal here to bring the previously issued variable into the scope of this function
      nonlocal diameter

      # We know that if the node is None then we can just return 0 instead of adding to the current distance
      if not node:
        return 0

      # Recursively call longest_path on the left and right paths of the node assuming - assuming they are not None, they will continue calling and adding to their path
      left_path = longest_path(node.left)
      right_path = longest_path(node.right)

      # We know both paths now so we can set diameter equal to the max of either itself or the paths
      diameter = max(diameter, left_path + right_path)

      # You want to return the max of both paths + 1 because you are recursively returning this value to be set as your new diameter; 
      return max(left_path, right_path) + 1

    # Don't forget to call the actual function on the root so that it updates the diamater
    longest_path(root)
    # Return the diameter because that's what we are looking for here
    return diameter

# 680. Valid Palindrome II
class Solution:
  # Time complexity: O(n); We are using two pointers to iterate through every character in the string
  # Space complexity: O(1); We are using a few variables but none depend on the input size of the string
  def validPalindrome(self, s: str) -> bool:
    # Create a helper function that just determines if the string is a valid palindrome
    def check_palindrome(s, i, j):
      while i < j:
        if s[i] != s[j]:
          return False
        i += 1
        j -= 1
      return True
    
    # Set up the two pointers to be at the start and end of the string
    i = 0
    j = len(s) - 1
    # In our while loop, we want to check if any of the characters are NOT equal (i.e., not a palindrome) and then check if we can still create a palindrome by excluding a letter
    while i < j:
      if s[i] != s[j]:
        # The result of this should be true if either side is a palindrome (i.e., one letter was removed and still palindrome)
        return check_palindrome(s, i + 1, j) or check_palindrome(s, i, j - 1)
      i += 1
      j -= 1
    # If we have checked all our letters and they are equal, then we can just return True
    return True

# 704. Binary Search
class Solution:
  # Time complexity: O(log N); You are splitting the input size of the list (nums) in half in each iteration of your while loop
  # Space complexity: O(1); You are only creating variables with constant space complexity as none of them are dependent on the input size
  def search(self, nums: List[int], target: int) -> int:
    # Set up two variables (two pointers) to track left and right indexes
    left, right = 0, len(nums) - 1

    # should conclude if left ever goes over the right
    while (left <= right):
      # Sets the new mid up at the beginning of every while loop as to account for a new "view" of the array
      # // operator returns floor division, rounding down to the nearest whole integer
      mid = (left + right) // 2
      # Checks if the middle index is the target; if it is, returns it
      if nums[mid] == target:
        return mid
      # Checks if the middle index is to the left of the target, in which case it makes left = mid + 1 so that the "view" is only on the right part of the array
      elif nums[mid] < target:
        left = mid + 1
      # Else, we know that the target is less than the nums[mid] so set right = mid - 1 to "view" only the left portion of the array
      else:
        right = mid - 1
    # If all else fails, return -1 to indicate that the number does not exist in the solution
    return -1

# 733. Flood Fill
class Solution:
  # Time complexity: O(n); You are going to visiting every "pixel" of the image so you will iterate through N pixels
  # Space complexity: O(n); The size of the implicit call stack using DFS
  def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
    # Set up two variables that will hold the max dimensions of the image (X, Y)
    X, Y = len(image), len(image[0])
    # Set up a variable to track the color at the starting pixel (sr, sc)
    color = image[sr][sc]
    # You want to return the image here if the color is already equal to the newColor because you don't want to flood fill the same color again
    if color == newColor: return image

    def dfs(x, y):
      if image[x][y] == color:
        # Set the current pixel to the new color if it isn't already
        image[x][y] = newColor
        # DFS the pixel left and right if they aren't out of bounds
        if x >= 1:
          dfs(x - 1, y)
        if x + 1 < X:
          dfs(x + 1, y)
        # DFS the pixel above and below if they aren't out of bounds
        if y >= 1:
          dfs(x, y - 1)
        if y + 1 < Y:
          dfs(x, y + 1)
    
    # Don't forget to actually run the DFS on the image starting at the starting pixel
    dfs(sr, sc)
    # Remember that the image will be altered and thus should be returned as is
    return image

# 1029. Two City Scheduling
class Solution:
  # Time complexity: O(nlogn); We are sorting our costs to determine the cheapest cities to go to City A and the cheapest to go to City B
  # Space complexity: O(n); In Python, the sort function uses the Timsort algorithm which is a combination of Merge Sort and Insertion Sort -> this takes O(n) additional space
  def twoCitySchedCost(self, costs: List[List[int]]) -> int:
    # Sort the costs in-place by determining which cities are cheaper to go to City A first; all of the cities that are greater (x[0] - x[1]) will end up being cheaper for City B
    costs.sort(key = lambda x: x[0] - x[1])

    # Set up the total costs variable since this is what we want to return
    total = 0
    # N represents the length of the costs divided by 2 because we need to 
    n = len(costs) // 2
    # The beginning half of the sorted costs show cheaper costs to go to City A - we add those; the ending half of the sorted costs show cheaper costs to go to City B - we add those
    for i in range(n):
      total += costs[i][0] + costs[i + n][1]

    # Don't forget to return the total!
    return total

# 1064. Fixed Point
class Solution:
  # Time complexity: O(log n); Since we're using a binary search, we halve the size of the array in each loop and thus reduce our time complexity to O(log n) time
  # Space complexity: O(1); We don't use any additional space to perform the binary search so the space complexity is constant
  def fixedPoint(self, arr: List[int]) -> int:
    # Set up two variables that will serve as our binary search pointers
    left, right = 0, len(arr) - 1
    # Initiate the answer as -1 in case we never find our fixedPoint
    answer = -1

    # Binary search starts with the left pointer being less than or equal to the right pointer
    while left <= right:
      # Set up a middle variable that will track the current mid of the current arr[left::right]
      mid = (left + right) // 2

      # If the index is equal to itself at the middle of the arr, then we can say that the answer is mid
      # We decrement the right to mid - 1 because we still want to check if there is a SMALLER index that satisfies arr[i] == 1
      if arr[mid] == mid:
        answer = mid
        right = mid - 1
      # Else we check if the arr[mid] is less than the current mid - if it is, we must increment left
      elif arr[mid] < mid:
        left = mid + 1
      # Finally if all else fails, we know that arr[mid] is greater than the midpoint so we should decrement right
      else:
        right = mid - 1

    # Make sure to return the answer - this should be a new value or -1 if nothing is found
    return answer

# 1396. Design Underground System
import collections
class UndergroundSystem:
  def __init__(self):
    self.checked_in = {}
    self.journey_data = collections.defaultdict(lambda: [0, 0])

  # Time complexity: O(n) worst case; O(1) average to insert
  # Space complexity: O(n) worst case
  def checkIn(self, id: int, stationName: str, t: int) -> None:
    self.checked_in[id] = [stationName, t]

  # Time complexity: O(n) worst case; O(1) average to delete
  # Space complexity: O(n) worst case
  def checkOut(self, id: int, stationName: str, t: int) -> None:
    startStation, startTime = self.checked_in.pop(id)
    self.journey_data[(startStation, stationName)][0] += t - startTime
    self.journey_data[(startStation, stationName)][1] += 1

  # Time complexity: O(n) worst case; O(1) average to search
  # Space complexity: O(n) worst case
  def getAverageTime(self, startStation: str, endStation: str) -> float:
    total_time, total_trips = self.journey_data[(startStation, endStation)]
    return total_time / total_trips
