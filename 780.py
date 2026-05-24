class Solution:
   def reachingPoints(self ,sx: int, sy: int, tx: int, ty: int) -> bool:
    # Reverse reduction using modulo arithmetic
    while tx > sx and ty > sy and tx != ty:
        if tx > ty:
            tx %= ty
        else:
            ty %= tx
            
    # Verify if the reduced state can align with (sx, sy)
    if tx == sx and ty == sy:
        return True
    elif tx == sx:
        return ty >= sy and (ty - sy) % tx == 0
    elif ty == sy:
        return tx >= sx and (tx - sx) % ty == 0
    else:
        return False