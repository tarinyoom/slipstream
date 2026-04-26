#pragma once

/* MAC grid indexing, staggered layout helpers, and interpolation.
   Shared by all backends — keep this header free of backend-specific types. */

namespace slipstream {

// Flat index into a cell-centered field of shape [shape[0], ..., shape[ndim-1]].
inline int cell_idx(const int* shape, int ndim, const int* ijk) {
    int idx = 0;
    for (int d = 0; d < ndim; ++d)
        idx = idx * shape[d] + ijk[d];
    return idx;
}

// Flat index into velocity component `dim`. Shape is the same as cell-centered except
// shape[dim] is shape[dim]+1 (one extra face in the staggered dimension).
inline int face_idx(const int* shape, int ndim, int dim, const int* ijk) {
    int idx = 0;
    for (int d = 0; d < ndim; ++d) {
        int s = (d == dim) ? shape[d] + 1 : shape[d];
        idx = idx * s + ijk[d];
    }
    return idx;
}

} // namespace slipstream
