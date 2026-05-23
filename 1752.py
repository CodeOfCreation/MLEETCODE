class Solution:
 def check(self,nums: list[int]) -> bool:
    descent_count = 0
    n = len(nums)
    
    for i in range(n):
        if nums[i] > nums[(i + 1) % n]:
            descent_count += 1
            if descent_count > 1:
                return False
                
    return True