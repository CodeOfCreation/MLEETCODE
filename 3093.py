class Solution:
    def stringIndices(self, wordsContainer: list[str], wordsQuery: list[str]) -> list[int]:
        # Parallel arrays for Trie nodes to minimize memory overhead
        children = [[-1] * 26]
        best_len = [float('inf')]
        best_idx = [-1]

        # Build Trie with reversed container words
        for idx, word in enumerate(wordsContainer):
            L = len(word)
            node = 0
            
            # Update root (represents empty suffix)
            if L < best_len[node]:
                best_len[node] = L
                best_idx[node] = idx

            for ch in reversed(word):
                c = ord(ch) - 97  # ord('a') == 97
                if children[node][c] == -1:
                    children.append([-1] * 26)
                    best_len.append(float('inf'))
                    best_idx.append(-1)
                    children[node][c] = len(children) - 1
                    
                node = children[node][c]
                # Update best candidate: strictly shorter length guarantees precedence
                if L < best_len[node]:
                    best_len[node] = L
                    best_idx[node] = idx

        # Resolve queries
        ans = []
        for q in wordsQuery:
            node = 0
            res = best_idx[0]  # Fallback for empty suffix match
            
            for ch in reversed(q):
                c = ord(ch) - 97
                if children[node][c] != -1:
                    node = children[node][c]
                    res = best_idx[node]
                else:
                    break
            ans.append(res)
            
        return ans