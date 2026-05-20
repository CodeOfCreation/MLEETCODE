#2657. Find the Prefix Common Array of Two Arrays

from typing import List

class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        n = len(A)
        frequency = [0] * (n + 1)
        common_count = 0
        C = []

        for i in range(n):
            frequency[A[i]] += 1
            if frequency[A[i]] == 2:
                common_count += 1

            frequency[B[i]] += 1
            if frequency[B[i]] == 2:
                common_count += 1

            C.append(common_count)

        return C