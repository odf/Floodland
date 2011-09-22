from numpy import asarray


def count(gen):
    return reduce(lambda n, x: n + 1, gen, 0)


def first(gen):
    return next(gen, None)


def ranking(enum):
    t = sorted(tuple((v, i) for i, v in enum))
    return dict((k, i) for i, (v, k) in enumerate(t))


def reverse_dict(d):
    return dict((v, k) for k, v in d.items())


def values_in_order(ranking):
    ranked_to_value = reverse_dict(ranking)
    return tuple(ranked_to_value[k] for k in range(len(ranked_to_value)))


def flow(cells, faces):
    used = dict((cell, False) for cell in cells)

    def free_faces(cell):
        return (f for f in faces(cell) if not used[f])

    def first_with_free_neighbor_count(n):
        return first(c for c in cells
                     if not used[c] and count(free_faces(c)) == n)

    result = []

    while True:
        paired_cell = first_with_free_neighbor_count(1)
        if paired_cell:
            partner = first(free_faces(paired_cell))
            used[paired_cell] = used[partner] = True
            result.append((paired_cell, partner))
        else:
            singular_cell = first_with_free_neighbor_count(0)
            if singular_cell:
                used[singular_cell] = True
                result.append((singular_cell,))
            else:
                return result


def multi_enumerate(data):
    f = lambda t, n: tuple((i+(j,),w)  for i, v in t for j, w in enumerate(v))
    return reduce(f, range(asarray(data).ndim), (((), data),))


def neighborhood_cube(data, pos):
    a = asarray(data)

    if len(pos) > a.ndim:
        raise TypeError("Too many indices.")
    elif len(pos) < a.ndim:
        raise TypeError("Not enough indices.")
    elif not all(0 < v < a.shape[i] - 1 for i, v in enumerate(pos)):
        raise IndexError("Indices out of range")
    else:
        return a[tuple(slice(v-1,v+2) for i, v in enumerate(pos))]


def memoize(f, cache = {}):
    def g(*args, **kwargs):
        key = ( f, tuple(args), frozenset(kwargs.items()) )
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]
    return g


@memoize
def cubic_star_enumerate(n):
    if n == 0:
        return (((), ((),)),)
    else:
        f = lambda s, t: tuple(a + (b,) for a in s for b in t)
        return tuple((k + (m,), f(s, t))
                     for k, s in cubic_star_enumerate(n-1)
                     for m, t in enumerate(((0, 1), (1,), (1, 2))))


@memoize
def cubic_star_face(cell, i):
    return tuple((1 if j == i else v) for j, v in enumerate(cell))


@memoize
def cubic_star_faces(cell):
    return tuple(cubic_star_face(cell, i) for i, v in enumerate(cell) if v != 1)


def lower_star_ranking(data, pos):
    ranks = ranking(multi_enumerate(neighborhood_cube(data, pos)))
    cell_key = lambda cell: tuple(reversed(sorted(ranks[i] for i in cell)))
    enum = ((i, cell_key(p)) for i, p in cubic_star_enumerate(len(pos)))
    return ranking((i, v) for i, v in enum if v[0] == ranks[1,1])


def lower_star_flow(data, pos):
    cells = values_in_order(lower_star_ranking(data, pos))
    return flow(cells, cubic_star_faces)
