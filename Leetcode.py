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
