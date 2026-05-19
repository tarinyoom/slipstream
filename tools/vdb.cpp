#include "vdb.hpp"

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

namespace slipstream {

void write_vdb(const char*            path,
               std::span<const float> density,
               int                    nx,
               int                    ny,
               int                    nz)
{
    openvdb::initialize();

    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
    grid->setName("density");
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);

    auto acc = grid->getAccessor();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                float v = density[(i * ny + j) * nz + k];
                if (v != 0.0f)
                    acc.setValue(openvdb::Coord(j, k, i), v);
            }

    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    openvdb::io::File(path).write(grids);
}

} // namespace slipstream
