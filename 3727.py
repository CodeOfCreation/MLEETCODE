from typing import List
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        nums.sort(key=abs)

        n = len(nums)
        negative = sum(x * x for x in nums[:n // 2])
        positive = sum(x * x for x in nums[n // 2:])

        return positive - negative