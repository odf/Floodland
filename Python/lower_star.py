from numpy import asarray


def count(gen):
    return reduce(lambda n, x: n + 1, gen, 0)


def first(gen):
    return next(gen, None)


def reverse_dict(d):
    return dict((v, k) for k, v in d.items())


def multi_enumerate(data):
    f = lambda t, n: tuple((i+(j,),w)  for i, v in t for j, w in enumerate(v))
    return reduce(f, range(asarray(data).ndim), (((), data),))


def ranking(enum):
    t = sorted(tuple((v, i) for i, v in enum))
    return dict((k, i) for i, (v, k) in enumerate(t))


def lower_induced_ranking(cells, node_ranking):
    f = lambda cell: tuple(reversed(sorted(node_ranking[i] for i in cell)))
    enum = ((i, f(p)) for i, p in cells)
    return ranking((i, v) for i, v in enum if v[0] == node_ranking[1,1])


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


def cubic_star_enumerate(n):
    if n == 0:
        return (((), ((),)),)
    else:
        f = lambda s, t: tuple(a + (b,) for a in s for b in t)
        return ((k + (m,), f(s, t))
                for k, s in cubic_star_enumerate(n-1)
                for m, t in enumerate(((0, 1), (1,), (1, 2))))


def lower_star_ranking(data, pos):
    ranks = ranking(multi_enumerate(neighborhood_cube(data, pos)))
    star = cubic_star_enumerate(len(pos))
    return lower_induced_ranking(star, ranks)


def face(cell, i):
    return tuple((1 if j == i else v) for j, v in enumerate(cell))


def faces(cell):
    return tuple(face(cell, i) for i, v in enumerate(cell) if v != 1)


def lower_star_pairings(data, pos):
    ranking = lower_star_ranking(data, pos)
    ranked = reverse_dict(ranking)
    cells = tuple(ranked[k] for k in range(len(ranked)))

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
