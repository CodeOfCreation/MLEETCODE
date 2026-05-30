import bisect

class Solution:
    def getResults(self, queries: list[list[int]]) -> list[bool]:
        MAX_X = 50001
        size = 1
        while size <= MAX_X:
            size *= 2
        tree = [0] * (2 * size)
        
        def update(pos: int, val: int) -> None:
            i = pos + size
            tree[i] = val
            while i > 1:
                tree[i >> 1] = max(tree[i], tree[i ^ 1])
                i >>= 1
                
        def query(l: int, r: int) -> int:
            res = 0
            l += size
            r += size
            while l <= r:
                if l & 1:
                    res = max(res, tree[l])
                    l += 1
                if not (r & 1):
                    res = max(res, tree[r])
                    r -= 1
                l >>= 1
                r >>= 1
            return res
            
        # Initialize with implicit boundaries at 0 and MAX_X
        update(MAX_X, MAX_X)
        obs = [0, MAX_X]
        
        results = []
        for q in queries:
            if q[0] == 1:
                x = q[1]
                idx = bisect.bisect_left(obs, x)
                prev_obs = obs[idx - 1]
                next_obs = obs[idx]
                
                # Split the gap [prev_obs, next_obs] into [prev_obs, x] and [x, next_obs]
                update(next_obs, next_obs - x)
                update(x, x - prev_obs)
                obs.insert(idx, x)
            else:
                x, sz = q[1], q[2]
                idx = bisect.bisect_right(obs, x) - 1
                last_obs = obs[idx]
                
                # Placement is possible if max internal gap or tail gap >= sz
                max_gap = max(query(0, x), x - last_obs)
                results.append(max_gap >= sz)
                
        return results