#2540. Minimum Common Value
from typing import List
class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        for i in nums1:
            for j in nums2:
                if i==j:
                    return i
        
