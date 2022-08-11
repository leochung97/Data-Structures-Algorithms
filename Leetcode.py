from typing import *
from math import *

# 1. Two Sum
class Solution:
  # Time complexity: O(n) because you run through the entire List at worst; Space complexity: O(n) as you may store the entire List in the hashmap at worst
  def twoSum(self, nums: List[int], target: int) -> List[int]:
    hash = {}
    # Note: Range gives you a range from 0 to the number (not inclusive of the number) if only a single argument is provided
    for i in range(len(nums)):
      complement = target - nums[i]
      if complement in hash:
        return [hash[complement], i]
      hash[nums[i]] = i

# 20. Valid Parentheses
class Solution:
  # Time complexity: O(n) because you run through the entire string; Space complexity: O(n) as the stack will be the size of the input
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
  # Time complexity: O(n + m) as it depends on the size of both linked lists; Space complexity: O(1) because we only set up a few variables that are not determined by input size
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

#121. Best Time to Buy / Sell Stock
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

# 226. Invert Binary Tree

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

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