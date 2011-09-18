from numpy import asarray


def neighborhood_cube(data, *pos):
    a = asarray(data)

    if len(pos) > a.ndim:
        raise TypeError("Too many indices.")
    elif len(pos) < a.ndim:
        raise TypeError("Not enough indices.")
    elif not all(0 < v < a.shape[i] - 1 for i, v in enumerate(pos)):
        raise IndexError("Indices out of range")
    else:
        return a[tuple(slice(v-1,v+2) for i, v in enumerate(pos))]


def multi_enumerate(data):
    f = lambda t, n: tuple((i+(j,),w)  for i, v in t for j, w in enumerate(v))
    return reduce(f, range(asarray(data).ndim), (((), data),))


def ranking(data):
    t = sorted(tuple((v, i) for i, v in multi_enumerate(data)))
    return dict((k, i) for i, (v, k) in enumerate(t))


def reverse_dict(d):
    return dict((v, k) for k, v in d.items())


def star_index_lists(n):
    if n == 0:
        return (((), ((),)),)
    else:
        f = lambda s, t: tuple(a + (b,) for a in s for b in t)
        return tuple((k + (m,), f(s, t))
                     for k, s in star_index_lists(n-1)
                     for m, t in enumerate(((0, 1), (1,), (1, 2))))
