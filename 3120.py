class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        present_chars = set(word)
        special_count = 0
        
        for char in present_chars:
            if char.islower() and char.upper() in present_chars:
                special_count += 1
                
        return special_count