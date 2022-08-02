from typing import Optional

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

#121 Best Time to Buy / Sell Stock
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