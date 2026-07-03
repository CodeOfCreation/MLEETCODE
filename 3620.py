from collections import deque
from typing import List


class Solution:
    def findMaxPathScore(self, edges: List[List[int]], online: List[bool], k: int) -> int:
        n = len(online)

        if n == 1:
            return 0
        if not edges:
            return -1

        # Build original graph and topological order
        graph = [[] for _ in range(n)]
        indegree = [0] * n
        mx = 0

        for u, v, w in edges:
            graph[u].append((v, w))
            indegree[v] += 1
            mx = max(mx, w)

        q = deque(i for i in range(n) if indegree[i] == 0)
        topo = []

        while q:
            u = q.popleft()
            topo.append(u)
            for v, _ in graph[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)

        def check(limit):
            INF = 10**30
            dist = [INF] * n
            dist[0] = 0

            for u in topo:
                if dist[u] == INF:
                    continue
                if u != 0 and u != n - 1 and not online[u]:
                    continue

                for v, w in graph[u]:
                    if w < limit:
                        continue
                    if v != 0 and v != n - 1 and not online[v]:
                        continue
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w

            return dist[n - 1] <= k

        lo, hi = 0, mx
        ans = -1

        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return ans