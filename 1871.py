class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        n = len(s)
        
        # Early termination: destination must be '0'
        if s[-1] == '1':
            return False
            
        dp = [False] * n
        dp[0] = True
        
        # pre[i] stores the count of reachable indices in s[0...i]
        pre = [0] * n
        pre[0] = 1
        
        for i in range(1, n):
            if s[i] == '0':
                left = i - maxJump
                right = i - minJump
                
                if right >= 0:
                    count_in_range = pre[right] - (pre[left - 1] if left > 0 else 0)
                    if count_in_range > 0:
                        dp[i] = True
            pre[i] = pre[i - 1] + (1 if dp[i] else 0)
            
        return dp[-1]