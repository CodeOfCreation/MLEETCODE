from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        cnt = [0] * 31

        for num in nums:
            for bit in range(31):
                if num & (1 << bit):
                    cnt[bit] += 1

        ans = 0

        for _ in range(k):
            x = 0

            for bit in range(31):
                if cnt[bit] > 0:
                    x |= (1 << bit)
                    cnt[bit] -= 1

            ans = (ans + x * x) % MOD

        return ans