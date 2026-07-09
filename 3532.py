class Solution:
    def pathExistenceQueries(self, n, nums, maxDiff, queries):

        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            pa = find(a)
            pb = find(b)

            if pa == pb:
                return

            if rank[pa] < rank[pb]:
                parent[pa] = pb
            elif rank[pa] > rank[pb]:
                parent[pb] = pa
            else:
                parent[pb] = pa
                rank[pa] += 1

        # connect adjacent nodes
        for i in range(n - 1):
            if nums[i+1] - nums[i] <= maxDiff:
                union(i, i+1)

        ans = []

        for u, v in queries:
            ans.append(find(u) == find(v))

        return ans