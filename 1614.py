class Solution:
    def maxDepth(self, s: str) -> int:
        max=0
        stack=[]

        for i in s:
            if i == '(':
                stack.append(i)
                if max < len(stack):
                    max = len(stack)
            elif i ==')':
                stack.pop()
        return max