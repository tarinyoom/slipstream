#pragma once

#include <span>

namespace slipstream {

void write_vdb(const char*            path,
               std::span<const float> density,
               int                    nx,
               int                    ny);

void write_vdb(const char*            path,
               std::span<const float> density,
               int                    nx,
               int                    ny,
               int                    nz);

} // namespace slipstream
