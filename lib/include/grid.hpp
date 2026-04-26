#pragma once

namespace slipstream {

inline int cell_idx(const int* shape, int ndim, const int* ijk) {
    int idx = 0;
    for (int d = 0; d < ndim; ++d)
        idx = idx * shape[d] + ijk[d];
    return idx;
}

inline int face_idx(const int* shape, int ndim, int dim, const int* ijk) {
    int idx = 0;
    for (int d = 0; d < ndim; ++d) {
        int s = (d == dim) ? shape[d] + 1 : shape[d];
        idx = idx * s + ijk[d];
    }
    return idx;
}

} // namespace slipstream
