def count(m,n):
    if n == 0:
        return 1
    a = m
    s = m
    sn = n
    while n >= 2:
        a = a - 1
        n = n - 1
        s = s * a
        sn = sn * n
    return s / sn


def taijie(m):
    all_count = 0
    for x in range(0, m+2, 2):
        y = (m - x)/2
        all_count = all_count + count(m-y,y)
    return all_count

print taijie(91)
