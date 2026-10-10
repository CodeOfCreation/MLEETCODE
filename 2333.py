class Solution:
    def minSumSquareDiff(self, nums1: list[int], nums2: list[int],
                         k1: int, k2: int) -> int:
        diff = [abs(a - b) for a, b in zip(nums1, nums2)]
        k = k1 + k2

        if sum(diff) <= k:
            return 0

        left, right = 0, max(diff)

        while left < right:
            mid = (left + right) // 2
            ops = sum(max(0, d - mid) for d in diff)

            if ops <= k:
                right = mid
            else:
                left = mid + 1

        level = left
        ans = 0
        remaining = k

        for d in diff:
            if d > level:
                remaining -= d - level
                ans += level * level
            else:
                ans += d * d
        ans -= remaining * (2 * level - 1)

        return ans