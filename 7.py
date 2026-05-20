#7. Reverse Integer

class Solution:
    def reverse(self, x: int) -> int:
        INT_MAX = 2**31 - 1  # 2147483647
        sign = -1 if x < 0 else 1
        x = abs(x)
        reversed_val = 0
        
        while x > 0:
            digit = x % 10
            
            # Pre-overflow verification against positive boundary
            if (reversed_val > INT_MAX // 10) or \
               (reversed_val == INT_MAX // 10 and digit > 7):
                return 0
                
            reversed_val = reversed_val * 10 + digit
            x //= 10
            
        return reversed_val * sign