class Solution:
  def lastRemaining(self, n: int) -> int:
    head = 1
    step = 1
    remaining = n
    left_to_right = True
    
    while remaining > 1:
        if left_to_right:
            head += step
        elif remaining % 2 == 1:
            head += step
            
        step *= 2
        remaining //= 2
        left_to_right = not left_to_right
        
    return head