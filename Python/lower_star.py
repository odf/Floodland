from numpy import asarray


def reverse_dict(d):
    return dict((v, k) for k, v in d.items())


def multi_enumerate(data):
    f = lambda t, n: tuple((i+(j,),w)  for i, v in t for j, w in enumerate(v))
    return reduce(f, range(asarray(data).ndim), (((), data),))


def ranking(enum):
    t = sorted(tuple((v, i) for i, v in enum))
    return dict((k, i) for i, (v, k) in enumerate(t))


def induced_lower_ranking(cells, node_ranking):
    f = lambda pos: tuple(reversed(sorted(map(lambda i: node_ranking[i], pos))))
    enum = ((i, f(p)) for i, p in cells)
    return ranking((i, v) for i, v in enum if v[0] == node_ranking[1,1])


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


def cubic_star_enumerate(n):
    if n == 0:
        return (((), ((),)),)
    else:
        f = lambda s, t: tuple(a + (b,) for a in s for b in t)
        return ((k + (m,), f(s, t))
                for k, s in cubic_star_enumerate(n-1)
                for m, t in enumerate(((0, 1), (1,), (1, 2))))


def lower_star_ranking(data, *pos):
    ranks = ranking(multi_enumerate(neighborhood_cube(data, *pos)))
    star = cubic_star_enumerate(len(pos))
    return induced_lower_ranking(star, ranks)
