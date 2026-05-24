class Solution:
 import sys

# Increase recursion depth to safely accommodate worst-case linear chains
 sys.setrecursionlimit(2000)

 def maxJumps(self ,arr: list[int], d: int) -> int:
    n = len(arr)
    memo = [0] * n  # 0 indicates an uncomputed state

    def dfs(i: int) -> int:
        if memo[i] != 0:
            return memo[i]
            
        max_steps = 1  # Count the current index itself
        
        # Evaluate rightward jumps
        for j in range(i + 1, min(i + d, n - 1) + 1):
            if arr[j] >= arr[i]:
                break  # Blocked by a value >= arr[i]
            max_steps = max(max_steps, 1 + dfs(j))
            
        # Evaluate leftward jumps
        for j in range(i - 1, max(i - d, 0) - 1, -1):
            if arr[j] >= arr[i]:
                break  # Blocked by a value >= arr[i]
            max_steps = max(max_steps, 1 + dfs(j))
            
        memo[i] = max_steps
        return max_steps

    return max(dfs(i) for i in range(n))