from typing import List
from collections import deque
import heapq


class Solution:
    # Four possible movement directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)

        # -------------------------------
        # Step 1: Multi-source BFS
        # Compute distance of every cell
        # from the nearest thief.
        # -------------------------------
        queue = deque()

        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    queue.append((r, c))
                    grid[r][c] = 0          # Distance of thief to itself
                else:
                    grid[r][c] = -1         # Unvisited

        while queue:
            r, c = queue.popleft()

            for dr, dc in self.directions:
                nr, nc = r + dr, c + dc

                if (
                    self.inBounds(nr, nc, n)
                    and grid[nr][nc] == -1
                ):
                    grid[nr][nc] = grid[r][c] + 1
                    queue.append((nr, nc))

        # ---------------------------------------
        # Step 2: Dijkstra (Maximum Bottleneck)
        # ---------------------------------------
        pq = [(-grid[0][0], 0, 0)]

        # Mark start as visited
        grid[0][0] = -1

        while pq:
            safe, r, c = heapq.heappop(pq)

            safe = -safe

            if r == n - 1 and c == n - 1:
                return safe

            for dr, dc in self.directions:
                nr, nc = r + dr, c + dc

                if (
                    self.inBounds(nr, nc, n)
                    and grid[nr][nc] != -1
                ):
                    new_safe = min(safe, grid[nr][nc])

                    heapq.heappush(
                        pq,
                        (-new_safe, nr, nc)
                    )

                    # Mark visited
                    grid[nr][nc] = -1

        return -1

    def inBounds(self, r: int, c: int, n: int) -> bool:
        return 0 <= r < n and 0 <= c < n