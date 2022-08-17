# Notes

Python Notes:
- Truthy values are values that evaluate to True in a boolean context
- Falsey values are values that evaluate to False in a boolean context
- Falsey values include empty sequences (lists, tuples, strings, dictionaries, sets), zero in every numeric type, None, and False
- Truthy values include non-empty sequences, numbers (except 0 in every numeric type), and basically every value that is not falsey

## Deque (doubly ended queue) 
Implemented using the module "collections"; Deque is preferred over a list in the cases where we need quicker append and pop operations from both ends of the container, as deque provides an O(1) time complexity for append and pop operations as compared to list which provides O(n) time complexity for the same operations
- deque.append(): Used to insert the value in its argument to the right end of the deque
- deque.appendleft(): Used to insert the value in its argument to the left end of the deque
- deque.pop(): Used to delete an argument from the right end of the deque
- deque.popleft(): Used to delete an argument from the left end of the deque