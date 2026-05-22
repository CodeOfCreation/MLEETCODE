from typing import List

class Solution:
    def plusOne(self, digits: List[int]):
        num_str = "".join(map(str, digits))
        num = int(num_str) + 1
        return [int(d) for d in str(num)]