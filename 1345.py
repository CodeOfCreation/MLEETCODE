from collections import defaultdict, deque
from typing import List

class Solution:
    def minJumps(self, arr: List[int]) -> int:
        n = len(arr)

        if n == 1:
            return 0

        # Store indices for each value
        graph = defaultdict(list)

        for i, val in enumerate(arr):
            graph[val].append(i)

        q = deque([0])
        visited = {0}
        steps = 0

        while q:
            for _ in range(len(q)):
                i = q.popleft()

                # Reached end
                if i == n - 1:
                    return steps

                # Possible next jumps
                neighbors = []

                # same value jumps
                neighbors.extend(graph[arr[i]])

                # i - 1
                if i - 1 >= 0:
                    neighbors.append(i - 1)

                # i + 1
                if i + 1 < n:
                    neighbors.append(i + 1)

                for nei in neighbors:
                    if nei not in visited:
                        visited.add(nei)
                        q.append(nei)

                # Important optimization:
                # clear list so we don't process same-value indices again
                graph[arr[i]].clear()

            steps += 1

        return -1