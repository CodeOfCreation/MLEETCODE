class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        first_upper = [-1] * 26
        last_lower = [-1] * 26
        
        for i, ch in enumerate(word):
            if ch.isupper():
                idx = ord(ch) - ord('A')
                if first_upper[idx] == -1:
                    first_upper[idx] = i
            else:
                idx = ord(ch) - ord('a')
                last_lower[idx] = i
                
        count = 0
        for j in range(26):
            if first_upper[j] != -1 and last_lower[j] != -1 and last_lower[j] < first_upper[j]:
                count += 1
                
        return count