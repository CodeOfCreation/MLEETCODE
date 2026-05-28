class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        N = m + n - 2
        k = min(m - 1, n - 1)
        result = 1
        
        # Multiplicative formula: C(N, k) = Π(N - i) / (i + 1)
        for i in range(k):
            result = result * (N - i) // (i + 1)
            
        return result