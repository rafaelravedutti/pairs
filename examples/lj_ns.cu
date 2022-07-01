#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
//---
#include "runtime/pairs.hpp"
#include "runtime/read_from_file.hpp"
#include "runtime/vtk.hpp"
#include "runtime/devices/cuda.hpp"

using namespace pairs;

__constant__ int d_dim_cells[3];

__global__ void enforce_pbc_kernel0(int nlocal, double grid0_d0_min, double grid0_d0_max, double grid0_d1_min, double grid0_d1_max, double grid0_d2_min, double grid0_d2_max, double *position, double e97, double e104, double e111, double e118, double e125, double e132) {
    const int i5 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i5 < nlocal)) {
        const int e92 = i5 * 3;
        const double p6_0 = position[e92];
        const bool e94 = p6_0 < grid0_d0_min;
        if(e94) {
            const int e95 = i5 * 3;
            const double p7_0 = position[e95];
            const double e98 = p7_0 + e97;
            position[e95] = e98;
        }
        const int e99 = i5 * 3;
        const double p8_0 = position[e99];
        const bool e101 = p8_0 > grid0_d0_max;
        if(e101) {
            const int e102 = i5 * 3;
            const double p9_0 = position[e102];
            const double e105 = p9_0 - e104;
            position[e102] = e105;
        }
        const int e106 = i5 * 3;
        const int e107 = e106 + 1;
        const double p10_1 = position[e107];
        const bool e108 = p10_1 < grid0_d1_min;
        if(e108) {
            const int e109 = i5 * 3;
            const int e110 = e109 + 1;
            const double p11_1 = position[e110];
            const double e112 = p11_1 + e111;
            position[e110] = e112;
        }
        const int e113 = i5 * 3;
        const int e114 = e113 + 1;
        const double p12_1 = position[e114];
        const bool e115 = p12_1 > grid0_d1_max;
        if(e115) {
            const int e116 = i5 * 3;
            const int e117 = e116 + 1;
            const double p13_1 = position[e117];
            const double e119 = p13_1 - e118;
            position[e117] = e119;
        }
        const int e120 = i5 * 3;
        const int e121 = e120 + 2;
        const double p14_2 = position[e121];
        const bool e122 = p14_2 < grid0_d2_min;
        if(e122) {
            const int e123 = i5 * 3;
            const int e124 = e123 + 2;
            const double p15_2 = position[e124];
            const double e126 = p15_2 + e125;
            position[e124] = e126;
        }
        const int e127 = i5 * 3;
        const int e128 = e127 + 2;
        const double p16_2 = position[e128];
        const bool e129 = p16_2 > grid0_d2_max;
        if(e129) {
            const int e130 = i5 * 3;
            const int e131 = e130 + 2;
            const double p17_2 = position[e131];
            const double e133 = p17_2 - e132;
            position[e131] = e133;
        }
    }
}
__global__ void update_pbc_kernel0(int npbc, int nlocal, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *pbc_map, int *pbc_mult, double *position, double e297, double e307, double e317) {
    const int i9 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i9 < npbc)) {
        const int e290 = nlocal + i9;
        const int e291 = e290 * 3;
        const int a39 = pbc_map[i9];
        const int e293 = a39 * 3;
        const double p28_0 = position[e293];
        const int e295 = i9 * 3;
        const int a40 = pbc_mult[e295];
        const double e298 = a40 * e297;
        const double e299 = p28_0 + e298;
        position[e291] = e299;
        const int e300 = nlocal + i9;
        const int e301 = e300 * 3;
        const int e302 = e301 + 1;
        const int a41 = pbc_map[i9];
        const int e303 = a41 * 3;
        const int e304 = e303 + 1;
        const double p30_1 = position[e304];
        const int e305 = i9 * 3;
        const int e306 = e305 + 1;
        const int a42 = pbc_mult[e306];
        const double e308 = a42 * e307;
        const double e309 = p30_1 + e308;
        position[e302] = e309;
        const int e310 = nlocal + i9;
        const int e311 = e310 * 3;
        const int e312 = e311 + 2;
        const int a43 = pbc_map[i9];
        const int e313 = a43 * 3;
        const int e314 = e313 + 2;
        const double p32_2 = position[e314];
        const int e315 = i9 * 3;
        const int e316 = e315 + 2;
        const int a44 = pbc_mult[e316];
        const double e318 = a44 * e317;
        const double e319 = p32_2 + e318;
        position[e312] = e319;
    }
}
__global__ void build_cell_lists_kernel0(int ncells, int *cell_sizes) {
    const int i10 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i10 < ncells)) {
        cell_sizes[i10] = 0;
    }
}
__global__ void build_cell_lists_kernel1(int nlocal, int npbc, double grid0_d0_min, double grid0_d1_min, double grid0_d2_min, int ncells, int cell_capacity, int *dim_cells, int *particle_cell, int *cell_particles, int *cell_sizes, int *resizes, double *position, int a47, int a46) {
    const int i11 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i11 < (nlocal + npbc))) {
        const int e321 = i11 * 3;
        const double p33_0 = position[e321];
        const double e323 = p33_0 - grid0_d0_min;
        const double e324 = e323 / 2.8;
        const int e333 = (int)(e324) * a46;
        const int e325 = i11 * 3;
        const int e326 = e325 + 1;
        const double p34_1 = position[e326];
        const double e327 = p34_1 - grid0_d1_min;
        const double e328 = e327 / 2.8;
        const int e334 = e333 + (int)(e328);
        const int e335 = e334 * a47;
        const int e329 = i11 * 3;
        const int e330 = e329 + 2;
        const double p35_2 = position[e330];
        const double e331 = p35_2 - grid0_d2_min;
        const double e332 = e331 / 2.8;
        const int e336 = e335 + (int)(e332);
        const bool e337 = e336 >= 0;
        const bool e338 = e336 <= ncells;
        const bool e339 = e337 && e338;
        if(e339) {
            particle_cell[i11] = e336;
            const int e340 = e336 * cell_capacity;
            const int e341 = e340 + pairs::atomic_add_resize_check(&(cell_sizes[e336]), 1, &(resizes[0]), cell_capacity);
            cell_particles[e341] = i11;
        }
    }
}
__global__ void neighbor_lists_build_kernel0(int nlocal, int *numneighs) {
    const int i12 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i12 < nlocal)) {
        numneighs[i12] = 0;
    }
}
__global__ void neighbor_lists_build_kernel1(int nlocal, int ncells, int cell_capacity, int neighborlist_capacity, int nstencil, int *particle_cell, int *stencil, int *cell_particles, int *neighborlists, int *numneighs, int *resizes, int *cell_sizes, double *position) {
    const int i16 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i16 < nlocal)) {
        const int a57 = particle_cell[i16];
        for(int i17 = 0; i17 < nstencil; i17++) {
            const int a58 = stencil[i17];
            const int e385 = a57 + a58;
            const bool e386 = e385 >= 0;
            const bool e387 = e385 <= ncells;
            const bool e388 = e386 && e387;
            if(e388) {
                const int a59 = cell_sizes[e385];
                const int e389 = e385 * cell_capacity;
                const int e397 = i16 * 3;
                const int e406 = i16 * 3;
                const int e407 = e406 + 1;
                const int e416 = i16 * 3;
                const int e417 = e416 + 2;
                const double p39_0 = position[e397];
                const double p39_1 = position[e407];
                const double p39_2 = position[e417];
                const int e342 = i16 * neighborlist_capacity;
                for(int i18 = 0; i18 < a59; i18++) {
                    const int e390 = e389 + i18;
                    const int a60 = cell_particles[e390];
                    const bool e391 = a60 != i16;
                    if(e391) {
                        const int e399 = a60 * 3;
                        const int e408 = a60 * 3;
                        const int e409 = e408 + 1;
                        const int e418 = a60 * 3;
                        const int e419 = e418 + 2;
                        const double p40_0 = position[e399];
                        const double p40_1 = position[e409];
                        const double p40_2 = position[e419];
                        const double e392_0 = p39_0 - p40_0;
                        const double e392_1 = p39_1 - p40_1;
                        const double e392_2 = p39_2 - p40_2;
                        const double e401 = e392_0 * e392_0;
                        const double e410 = e392_1 * e392_1;
                        const double e411 = e401 + e410;
                        const double e420 = e392_2 * e392_2;
                        const double e421 = e411 + e420;
                        const bool e422 = e421 < 2.8;
                        if(e422) {
                            const int a52 = numneighs[i16];
                            const int e343 = e342 + a52;
                            neighborlists[e343] = a60;
                            const int e344 = a52 + 1;
                            const int e441 = e344 + 1;
                            const bool e442 = e441 >= neighborlist_capacity;
                            if(e442) {
                                resizes[0] = e344;
                            } else {
                                numneighs[i16] = e344;
                            }
                        }
                    }
                }
            }
        }
    }
}
__global__ void reset_volatile_properties_kernel0(int nlocal, double *force) {
    const int i13 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i13 < nlocal)) {
        const int e345 = i13 * 3;
        const int e347 = i13 * 3;
        const int e348 = e347 + 1;
        const int e349 = i13 * 3;
        const int e350 = e349 + 2;
        force[e345] = 0.0;
        force[e348] = 0.0;
        force[e350] = 0.0;
    }
}
__global__ void module0_kernel0(int nlocal, int neighborlist_capacity, int *neighborlists, int *numneighs, double *position, double *force) {
    const int i14 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i14 < nlocal)) {
        const int a55 = numneighs[i14];
        const int e358 = i14 * 3;
        const int e367 = i14 * 3;
        const int e368 = e367 + 1;
        const int e377 = i14 * 3;
        const int e378 = e377 + 2;
        const double p37_0 = position[e358];
        const double p37_1 = position[e368];
        const double p37_2 = position[e378];
        const int e351 = i14 * neighborlist_capacity;
        const int e14 = i14 * 3;
        const int e18 = i14 * 3;
        const int e19 = e18 + 1;
        const int e22 = i14 * 3;
        const int e23 = e22 + 2;
        for(int i15 = 0; i15 < a55; i15++) {
            const int e352 = e351 + i15;
            const int a56 = neighborlists[e352];
            const int e360 = a56 * 3;
            const int e369 = a56 * 3;
            const int e370 = e369 + 1;
            const int e379 = a56 * 3;
            const int e380 = e379 + 2;
            const double p38_0 = position[e360];
            const double p38_1 = position[e370];
            const double p38_2 = position[e380];
            const double e353_0 = p37_0 - p38_0;
            const double e353_1 = p37_1 - p38_1;
            const double e353_2 = p37_2 - p38_2;
            const double e362 = e353_0 * e353_0;
            const double e371 = e353_1 * e353_1;
            const double e372 = e362 + e371;
            const double e381 = e353_2 * e353_2;
            const double e382 = e372 + e381;
            const bool e383 = e382 < 2.5;
            if(e383) {
                const double p0_0 = force[e14];
                const double p0_1 = force[e19];
                const double p0_2 = force[e23];
                const double e1 = 1.0 / e382;
                const double e2 = e1 * e1;
                const double e3 = e2 * e1;
                const double e423 = 48.0 * e3;
                const double e7 = e3 - 0.5;
                const double e424 = e423 * e7;
                const double e425 = e424 * e1;
                const double e9_0 = e353_0 * e425;
                const double e9_1 = e353_1 * e425;
                const double e9_2 = e353_2 * e425;
                const double e11_0 = p0_0 + e9_0;
                const double e11_1 = p0_1 + e9_1;
                const double e11_2 = p0_2 + e9_2;
                force[e14] = e11_0;
                force[e19] = e11_1;
                force[e23] = e11_2;
            }
        }
    }
}
__global__ void module1_kernel0(int nlocal, double *velocity, double *force, double *mass, double *position) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if((i0 < nlocal)) {
        const int e31 = i0 * 3;
        const int e37 = i0 * 3;
        const int e38 = e37 + 1;
        const int e43 = i0 * 3;
        const int e44 = e43 + 2;
        const double p1_0 = velocity[e31];
        const double p1_1 = velocity[e38];
        const double p1_2 = velocity[e44];
        const int e29 = i0 * 3;
        const int e35 = i0 * 3;
        const int e36 = e35 + 1;
        const int e41 = i0 * 3;
        const int e42 = e41 + 2;
        const double p2_0 = force[e29];
        const double p2_1 = force[e36];
        const double p2_2 = force[e42];
        const double e24_0 = 0.005 * p2_0;
        const double e24_1 = 0.005 * p2_1;
        const double e24_2 = 0.005 * p2_2;
        const double p3 = mass[i0];
        const double e25_0 = e24_0 / p3;
        const double e25_1 = e24_1 / p3;
        const double e25_2 = e24_2 / p3;
        const double e26_0 = p1_0 + e25_0;
        const double e26_1 = p1_1 + e25_1;
        const double e26_2 = p1_2 + e25_2;
        velocity[e31] = e26_0;
        velocity[e38] = e26_1;
        velocity[e44] = e26_2;
        const int e51 = i0 * 3;
        const int e57 = i0 * 3;
        const int e58 = e57 + 1;
        const int e63 = i0 * 3;
        const int e64 = e63 + 2;
        const double p4_0 = position[e51];
        const double p4_1 = position[e58];
        const double p4_2 = position[e64];
        const int e49 = i0 * 3;
        const int e55 = i0 * 3;
        const int e56 = e55 + 1;
        const int e61 = i0 * 3;
        const int e62 = e61 + 2;
        const double p5_0 = velocity[e49];
        const double p5_1 = velocity[e56];
        const double p5_2 = velocity[e62];
        const double e45_0 = 0.005 * p5_0;
        const double e45_1 = 0.005 * p5_1;
        const double e45_2 = 0.005 * p5_2;
        const double e46_0 = p4_0 + e45_0;
        const double e46_1 = p4_1 + e45_1;
        const double e46_2 = p4_2 + e45_2;
        position[e51] = e46_0;
        position[e58] = e46_1;
        position[e64] = e46_2;
    }
}
void module0(int neighborlist_capacity, int nlocal, int *numneighs, int *neighborlists, double *position, double *force) {
    const int e577 = nlocal - 0;
    const int e578 = e577 + 32;
    const int e579 = e578 - 1;
    const int e580 = e579 / 32;
    module0_kernel0<<<e580, 32>>>(nlocal, neighborlist_capacity, neighborlists, numneighs, position, force);
}
void module1(int nlocal, double *velocity, double *force, double *mass, double *position) {
    const int e582 = nlocal - 0;
    const int e583 = e582 + 32;
    const int e584 = e583 - 1;
    const int e585 = e584 / 32;
    module1_kernel0<<<e585, 32>>>(nlocal, velocity, force, mass, position);
}
void build_cell_lists_stencil(double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int ncells_capacity, int *ncells, int *nstencil, int *dim_cells, int *resizes, int *stencil) {
    const double e74 = grid0_d0_max - grid0_d0_min;
    const double e75 = e74 / 2.8;
    const int e76 = ceil(e75) + 2;
    dim_cells[0] = e76;
    const double e78 = grid0_d1_max - grid0_d1_min;
    const double e79 = e78 / 2.8;
    const int e80 = ceil(e79) + 2;
    dim_cells[1] = e80;
    const double e82 = grid0_d2_max - grid0_d2_min;
    const double e83 = e82 / 2.8;
    const int e84 = ceil(e83) + 2;
    dim_cells[2] = e84;
    const int a7 = dim_cells[0];
    const int a9 = dim_cells[1];
    const int e81 = a7 * a9;
    const int a11 = dim_cells[2];
    const int e85 = e81 * a11;
    const int e426 = e85 + 1;
    const bool e427 = e426 >= ncells_capacity;
    if(e427) {
        resizes[0] = e85;
    } else {
        (*ncells) = e85;
    }
    (*nstencil) = 0;
    for(int i2 = -1; i2 < 2; i2++) {
        for(int i3 = -1; i3 < 2; i3++) {
            const int a12 = dim_cells[0];
            const int e86 = i2 * a12;
            const int e87 = e86 + i3;
            const int a13 = dim_cells[1];
            const int e88 = e87 * a13;
            for(int i4 = -1; i4 < 2; i4++) {
                const int e89 = e88 + i4;
                stencil[(*nstencil)] = e89;
                const int e90 = (*nstencil) + 1;
                (*nstencil) = e90;
            }
        }
    }
}
void enforce_pbc(double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int nlocal, double *position) {
    const double e97 = grid0_d0_max - grid0_d0_min;
    const double e104 = grid0_d0_max - grid0_d0_min;
    const double e111 = grid0_d1_max - grid0_d1_min;
    const double e118 = grid0_d1_max - grid0_d1_min;
    const double e125 = grid0_d2_max - grid0_d2_min;
    const double e132 = grid0_d2_max - grid0_d2_min;
    const int e542 = nlocal - 0;
    const int e543 = e542 + 32;
    const int e544 = e543 - 1;
    const int e545 = e544 / 32;
    enforce_pbc_kernel0<<<e545, 32>>>(nlocal, grid0_d0_min, grid0_d0_max, grid0_d1_min, grid0_d1_max, grid0_d2_min, grid0_d2_max, position, e97, e104, e111, e118, e125, e132);
}
void setup_pbc(int nlocal, double grid0_d0_max, double grid0_d0_min, int pbc_capacity, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *npbc, int *pbc_map, int *pbc_mult, int *resizes, double *position) {
    (*npbc) = 0;
    const int e134 = nlocal + (*npbc);
    const double e135 = grid0_d0_max - grid0_d0_min;
    const double e138 = grid0_d0_min + 2.8;
    const double e163 = grid0_d0_max - 2.8;
    for(int i6 = 0; i6 < e134; i6++) {
        const int e170 = i6 * 3;
        const int e177 = i6 * 3;
        const int e178 = e177 + 1;
        const int e183 = i6 * 3;
        const int e184 = e183 + 2;
        const double p18_0 = position[e170];
        const double p18_1 = position[e178];
        const double p18_2 = position[e184];
        const bool e139 = p18_0 < e138;
        if(e139) {
            pbc_map[(*npbc)] = i6;
            const int e141 = (*npbc) * 3;
            pbc_mult[e141] = 1;
            const int e140 = nlocal + (*npbc);
            const int e143 = e140 * 3;
            const int e150 = e140 * 3;
            const int e151 = e150 + 1;
            const int e156 = e140 * 3;
            const int e157 = e156 + 2;
            const double e147 = p18_0 + e135;
            position[e143] = e147;
            const int e148 = (*npbc) * 3;
            const int e149 = e148 + 1;
            pbc_mult[e149] = 0;
            position[e151] = p18_1;
            const int e154 = (*npbc) * 3;
            const int e155 = e154 + 2;
            pbc_mult[e155] = 0;
            position[e157] = p18_2;
            const int e160 = (*npbc) + 1;
            const int e428 = e160 + 1;
            const bool e429 = e428 >= pbc_capacity;
            if(e429) {
                resizes[0] = e160;
            } else {
                (*npbc) = e160;
            }
        }
        const bool e164 = p18_0 > e163;
        if(e164) {
            pbc_map[(*npbc)] = i6;
            const int e166 = (*npbc) * 3;
            pbc_mult[e166] = -1;
            const int e165 = nlocal + (*npbc);
            const int e168 = e165 * 3;
            const int e175 = e165 * 3;
            const int e176 = e175 + 1;
            const int e181 = e165 * 3;
            const int e182 = e181 + 2;
            const double e172 = p18_0 - e135;
            position[e168] = e172;
            const int e173 = (*npbc) * 3;
            const int e174 = e173 + 1;
            pbc_mult[e174] = 0;
            position[e176] = p18_1;
            const int e179 = (*npbc) * 3;
            const int e180 = e179 + 2;
            pbc_mult[e180] = 0;
            position[e182] = p18_2;
            const int e185 = (*npbc) + 1;
            const int e430 = e185 + 1;
            const bool e431 = e430 >= pbc_capacity;
            if(e431) {
                resizes[0] = e185;
            } else {
                (*npbc) = e185;
            }
        }
    }
    const int e186 = nlocal + (*npbc);
    const double e187 = grid0_d1_max - grid0_d1_min;
    const double e190 = grid0_d1_min + 2.8;
    const double e215 = grid0_d1_max - 2.8;
    for(int i7 = 0; i7 < e186; i7++) {
        const int e222 = i7 * 3;
        const int e223 = e222 + 1;
        const int e229 = i7 * 3;
        const int e235 = i7 * 3;
        const int e236 = e235 + 2;
        const double p21_0 = position[e229];
        const double p21_1 = position[e223];
        const double p21_2 = position[e236];
        const bool e191 = p21_1 < e190;
        if(e191) {
            pbc_map[(*npbc)] = i7;
            const int e193 = (*npbc) * 3;
            const int e194 = e193 + 1;
            pbc_mult[e194] = 1;
            const int e192 = nlocal + (*npbc);
            const int e195 = e192 * 3;
            const int e196 = e195 + 1;
            const int e202 = e192 * 3;
            const int e208 = e192 * 3;
            const int e209 = e208 + 2;
            const double e199 = p21_1 + e187;
            position[e196] = e199;
            const int e200 = (*npbc) * 3;
            pbc_mult[e200] = 0;
            position[e202] = p21_0;
            const int e206 = (*npbc) * 3;
            const int e207 = e206 + 2;
            pbc_mult[e207] = 0;
            position[e209] = p21_2;
            const int e212 = (*npbc) + 1;
            const int e432 = e212 + 1;
            const bool e433 = e432 >= pbc_capacity;
            if(e433) {
                resizes[0] = e212;
            } else {
                (*npbc) = e212;
            }
        }
        const bool e216 = p21_1 > e215;
        if(e216) {
            pbc_map[(*npbc)] = i7;
            const int e218 = (*npbc) * 3;
            const int e219 = e218 + 1;
            pbc_mult[e219] = -1;
            const int e217 = nlocal + (*npbc);
            const int e220 = e217 * 3;
            const int e221 = e220 + 1;
            const int e227 = e217 * 3;
            const int e233 = e217 * 3;
            const int e234 = e233 + 2;
            const double e224 = p21_1 - e187;
            position[e221] = e224;
            const int e225 = (*npbc) * 3;
            pbc_mult[e225] = 0;
            position[e227] = p21_0;
            const int e231 = (*npbc) * 3;
            const int e232 = e231 + 2;
            pbc_mult[e232] = 0;
            position[e234] = p21_2;
            const int e237 = (*npbc) + 1;
            const int e434 = e237 + 1;
            const bool e435 = e434 >= pbc_capacity;
            if(e435) {
                resizes[0] = e237;
            } else {
                (*npbc) = e237;
            }
        }
    }
    const int e238 = nlocal + (*npbc);
    const double e239 = grid0_d2_max - grid0_d2_min;
    const double e242 = grid0_d2_min + 2.8;
    const double e267 = grid0_d2_max - 2.8;
    for(int i8 = 0; i8 < e238; i8++) {
        const int e274 = i8 * 3;
        const int e275 = e274 + 2;
        const int e281 = i8 * 3;
        const int e287 = i8 * 3;
        const int e288 = e287 + 1;
        const double p24_0 = position[e281];
        const double p24_1 = position[e288];
        const double p24_2 = position[e275];
        const bool e243 = p24_2 < e242;
        if(e243) {
            pbc_map[(*npbc)] = i8;
            const int e245 = (*npbc) * 3;
            const int e246 = e245 + 2;
            pbc_mult[e246] = 1;
            const int e244 = nlocal + (*npbc);
            const int e247 = e244 * 3;
            const int e248 = e247 + 2;
            const int e254 = e244 * 3;
            const int e260 = e244 * 3;
            const int e261 = e260 + 1;
            const double e251 = p24_2 + e239;
            position[e248] = e251;
            const int e252 = (*npbc) * 3;
            pbc_mult[e252] = 0;
            position[e254] = p24_0;
            const int e258 = (*npbc) * 3;
            const int e259 = e258 + 1;
            pbc_mult[e259] = 0;
            position[e261] = p24_1;
            const int e264 = (*npbc) + 1;
            const int e436 = e264 + 1;
            const bool e437 = e436 >= pbc_capacity;
            if(e437) {
                resizes[0] = e264;
            } else {
                (*npbc) = e264;
            }
        }
        const bool e268 = p24_2 > e267;
        if(e268) {
            pbc_map[(*npbc)] = i8;
            const int e270 = (*npbc) * 3;
            const int e271 = e270 + 2;
            pbc_mult[e271] = -1;
            const int e269 = nlocal + (*npbc);
            const int e272 = e269 * 3;
            const int e273 = e272 + 2;
            const int e279 = e269 * 3;
            const int e285 = e269 * 3;
            const int e286 = e285 + 1;
            const double e276 = p24_2 - e239;
            position[e273] = e276;
            const int e277 = (*npbc) * 3;
            pbc_mult[e277] = 0;
            position[e279] = p24_0;
            const int e283 = (*npbc) * 3;
            const int e284 = e283 + 1;
            pbc_mult[e284] = 0;
            position[e286] = p24_1;
            const int e289 = (*npbc) + 1;
            const int e438 = e289 + 1;
            const bool e439 = e438 >= pbc_capacity;
            if(e439) {
                resizes[0] = e289;
            } else {
                (*npbc) = e289;
            }
        }
    }
}
void update_pbc(double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int nlocal, int npbc, int *pbc_map, int *pbc_mult, double *position) {
    const double e297 = grid0_d0_max - grid0_d0_min;
    const double e307 = grid0_d1_max - grid0_d1_min;
    const double e317 = grid0_d2_max - grid0_d2_min;
    const int e547 = npbc - 0;
    const int e548 = e547 + 32;
    const int e549 = e548 - 1;
    const int e550 = e549 / 32;
    update_pbc_kernel0<<<e550, 32>>>(npbc, nlocal, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, pbc_map, pbc_mult, position, e297, e307, e317);
}
void build_cell_lists(int ncells, int nlocal, int npbc, double grid0_d0_min, double grid0_d1_min, double grid0_d2_min, int cell_capacity, int *cell_sizes, int *dim_cells, int *h_dim_cells, int *particle_cell, int *resizes, int *cell_particles, double *position) {
    const int e552 = ncells - 0;
    const int e553 = e552 + 32;
    const int e554 = e553 - 1;
    const int e555 = e554 / 32;
    build_cell_lists_kernel0<<<e555, 32>>>(ncells, cell_sizes);
    const int a46 = h_dim_cells[1];
    const int a47 = h_dim_cells[2];
    const int e557 = (nlocal + npbc) - 0;
    const int e558 = e557 + 32;
    const int e559 = e558 - 1;
    const int e560 = e559 / 32;
    build_cell_lists_kernel1<<<e560, 32>>>(nlocal, npbc, grid0_d0_min, grid0_d1_min, grid0_d2_min, ncells, cell_capacity, dim_cells, particle_cell, cell_particles, cell_sizes, resizes, position, a47, a46);
}
void neighbor_lists_build(int nlocal, int ncells, int cell_capacity, int neighborlist_capacity, int nstencil, int *numneighs, int *particle_cell, int *stencil, int *cell_sizes, int *cell_particles, int *neighborlists, int *resizes, double *position) {
    const int e562 = nlocal - 0;
    const int e563 = e562 + 32;
    const int e564 = e563 - 1;
    const int e565 = e564 / 32;
    neighbor_lists_build_kernel0<<<e565, 32>>>(nlocal, numneighs);
    const int e567 = nlocal - 0;
    const int e568 = e567 + 32;
    const int e569 = e568 - 1;
    const int e570 = e569 / 32;
    neighbor_lists_build_kernel1<<<e570, 32>>>(nlocal, ncells, cell_capacity, neighborlist_capacity, nstencil, particle_cell, stencil, cell_particles, neighborlists, numneighs, resizes, cell_sizes, position);
}
void reset_volatile_properties(int nlocal, double *force) {
    const int e572 = nlocal - 0;
    const int e573 = e572 + 32;
    const int e574 = e573 - 1;
    const int e575 = e574 / 32;
    reset_volatile_properties_kernel0<<<e575, 32>>>(nlocal, force);
}
int main() {
    PairsSim *ps = new PairsSim();
    int particle_capacity = 10000;
    int nlocal = 0;
    int nghost = 0;
    double grid0_d0_min = 0;
    double grid0_d0_max = 0;
    double grid0_d1_min = 0;
    double grid0_d1_max = 0;
    double grid0_d2_min = 0;
    double grid0_d2_max = 0;
    int nstencil = 0;
    int ncells = 1;
    int ncells_capacity = 100;
    int cell_capacity = 20;
    int neighborlist_capacity = 32;
    int npbc = 0;
    int pbc_capacity = 100;
    int *resizes = (int *) malloc((sizeof(int) * 3));
    int *d_resizes = (int *) pairs::device_alloc((sizeof(int) * 3));
    double grid_buffer[6];
    int dim_cells[3];
    int *cell_particles = (int *) malloc((sizeof(int) * (ncells_capacity * cell_capacity)));
    int *d_cell_particles = (int *) pairs::device_alloc((sizeof(int) * (ncells_capacity * cell_capacity)));
    int *cell_sizes = (int *) malloc((sizeof(int) * ncells_capacity));
    int *d_cell_sizes = (int *) pairs::device_alloc((sizeof(int) * ncells_capacity));
    int *stencil = (int *) malloc((sizeof(int) * 27));
    int *d_stencil = (int *) pairs::device_alloc((sizeof(int) * 27));
    int *particle_cell = (int *) malloc((sizeof(int) * particle_capacity));
    int *d_particle_cell = (int *) pairs::device_alloc((sizeof(int) * particle_capacity));
    int *neighborlists = (int *) malloc((sizeof(int) * (particle_capacity * neighborlist_capacity)));
    int *d_neighborlists = (int *) pairs::device_alloc((sizeof(int) * (particle_capacity * neighborlist_capacity)));
    int *numneighs = (int *) malloc((sizeof(int) * particle_capacity));
    int *d_numneighs = (int *) pairs::device_alloc((sizeof(int) * particle_capacity));
    int *pbc_map = (int *) malloc((sizeof(int) * pbc_capacity));
    int *d_pbc_map = (int *) pairs::device_alloc((sizeof(int) * pbc_capacity));
    int *pbc_mult = (int *) malloc((sizeof(int) * (pbc_capacity * 3)));
    int *d_pbc_mult = (int *) pairs::device_alloc((sizeof(int) * (pbc_capacity * 3)));
    unsigned long long int prop_hflags[1] = {18446744073709551615ULL};
    unsigned long long int prop_dflags[1] = {0ULL};
    double *mass = (double *) malloc((sizeof(double) * ((0 + particle_capacity) + pbc_capacity)));
    double *d_mass = (double *) pairs::device_alloc((sizeof(double) * ((0 + particle_capacity) + pbc_capacity)));
    ps->addProperty(Property(0, "mass", mass, Prop_Float));
    double *position = (double *) malloc((sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
    double *d_position = (double *) pairs::device_alloc((sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
    ps->addProperty(Property(1, "position", position, Prop_Vector, AoS, ((0 + particle_capacity) + pbc_capacity), 3));
    double *velocity = (double *) malloc((sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
    double *d_velocity = (double *) pairs::device_alloc((sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
    ps->addProperty(Property(2, "velocity", velocity, Prop_Vector, AoS, ((0 + particle_capacity) + pbc_capacity), 3));
    double *force = (double *) malloc((sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
    double *d_force = (double *) pairs::device_alloc((sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
    ps->addProperty(Property(3, "force", force, Prop_Vector, AoS, ((0 + particle_capacity) + pbc_capacity), 3));
    const int prop_list_0[] = {0, 1, 2};
    nlocal = pairs::read_particle_data(ps, "data/minimd_setup_4x4x4.input", grid_buffer, prop_list_0, 3);
    const double a0 = grid_buffer[0];
    grid0_d0_min = a0;
    const double a1 = grid_buffer[1];
    grid0_d0_max = a1;
    const double a2 = grid_buffer[2];
    grid0_d1_min = a2;
    const double a3 = grid_buffer[3];
    grid0_d1_max = a3;
    const double a4 = grid_buffer[4];
    grid0_d2_min = a4;
    const double a5 = grid_buffer[5];
    grid0_d2_max = a5;
    resizes[0] = 1;
    while((resizes[0] > 0)) {
        resizes[0] = 0;
        const unsigned long long int a91 = prop_hflags[0];
        const unsigned long long int e480 = a91 | 0;
        prop_hflags[0] = e480;
        const unsigned long long int a93 = prop_dflags[0];
        const unsigned long long int e481 = a93 & (unsigned long long int)(-1);
        prop_dflags[0] = e481;
        build_cell_lists_stencil(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, ncells_capacity, &ncells, &nstencil, dim_cells, resizes, stencil);
        const int a73 = resizes[0];
        const bool e444 = a73 > 0;
        if(e444) {
            const int a74 = resizes[0];
            const int e445 = a74 * 2;
            ncells_capacity = e445;
            cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
            d_cell_particles = (int *) pairs::device_realloc(d_cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
            cell_sizes = (int *) realloc(cell_sizes, (sizeof(int) * ncells_capacity));
            d_cell_sizes = (int *) pairs::device_realloc(d_cell_sizes, (sizeof(int) * ncells_capacity));
        }
    }
    pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, 0);
    const int e91 = nlocal + npbc;
    pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e91, 0);
    for(int i1 = 0; i1 < 101; i1++) {
        if(((i1 % 20) == 0)) {
            const unsigned long long int a94 = prop_dflags[0];
            const unsigned long long int e482 = a94 & 2;
            const bool e483 = e482 == 0;
            if(e483) {
                pairs::copy_to_device(position, d_position, (sizeof(double) * (3 * particle_capacity)));
            }
            const unsigned long long int a96 = prop_dflags[0];
            const unsigned long long int e486 = a96 | 2;
            prop_dflags[0] = e486;
            const unsigned long long int a98 = prop_hflags[0];
            const unsigned long long int e487 = a98 & (unsigned long long int)(-3);
            prop_hflags[0] = e487;
            enforce_pbc(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, nlocal, d_position);
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                const unsigned long long int a99 = prop_hflags[0];
                const unsigned long long int e488 = a99 & 2;
                const bool e489 = e488 == 0;
                if(e489) {
                    pairs::copy_to_host(d_position, position, (sizeof(double) * (3 * particle_capacity)));
                }
                const unsigned long long int a101 = prop_hflags[0];
                const unsigned long long int e492 = a101 | 2;
                prop_hflags[0] = e492;
                const unsigned long long int a103 = prop_dflags[0];
                const unsigned long long int e493 = a103 & (unsigned long long int)(-3);
                prop_dflags[0] = e493;
                setup_pbc(nlocal, grid0_d0_max, grid0_d0_min, pbc_capacity, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, &npbc, pbc_map, pbc_mult, resizes, position);
                const int a78 = resizes[0];
                const bool e465 = a78 > 0;
                if(e465) {
                    const int a79 = resizes[0];
                    const int e466 = a79 * 2;
                    pbc_capacity = e466;
                    pbc_map = (int *) realloc(pbc_map, (sizeof(int) * pbc_capacity));
                    d_pbc_map = (int *) pairs::device_realloc(d_pbc_map, (sizeof(int) * pbc_capacity));
                    pbc_mult = (int *) realloc(pbc_mult, (sizeof(int) * (pbc_capacity * 3)));
                    d_pbc_mult = (int *) pairs::device_realloc(d_pbc_mult, (sizeof(int) * (pbc_capacity * 3)));
                    mass = (double *) realloc(mass, (sizeof(double) * ((0 + particle_capacity) + pbc_capacity)));
                    d_mass = (double *) pairs::device_realloc(d_mass, (sizeof(double) * ((0 + particle_capacity) + pbc_capacity)));
                    ps->updateProperty(0, mass);
                    position = (double *) realloc(position, (sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
                    d_position = (double *) pairs::device_realloc(d_position, (sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
                    ps->updateProperty(1, position, ((0 + particle_capacity) + pbc_capacity), 3);
                    velocity = (double *) realloc(velocity, (sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
                    d_velocity = (double *) pairs::device_realloc(d_velocity, (sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
                    ps->updateProperty(2, velocity, ((0 + particle_capacity) + pbc_capacity), 3);
                    force = (double *) realloc(force, (sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
                    d_force = (double *) pairs::device_realloc(d_force, (sizeof(double) * (((0 + particle_capacity) + pbc_capacity) * 3)));
                    ps->updateProperty(3, force, ((0 + particle_capacity) + pbc_capacity), 3);
                }
            }
        } else {
            const unsigned long long int a104 = prop_dflags[0];
            const unsigned long long int e494 = a104 & 2;
            const bool e495 = e494 == 0;
            if(e495) {
                pairs::copy_to_device(position, d_position, (sizeof(double) * (3 * particle_capacity)));
            }
            const unsigned long long int a106 = prop_dflags[0];
            const unsigned long long int e498 = a106 | 2;
            prop_dflags[0] = e498;
            const unsigned long long int a108 = prop_hflags[0];
            const unsigned long long int e499 = a108 & (unsigned long long int)(-3);
            prop_hflags[0] = e499;
            update_pbc(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, nlocal, npbc, d_pbc_map, d_pbc_mult, d_position);
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                const unsigned long long int a109 = prop_dflags[0];
                const unsigned long long int e500 = a109 & 2;
                const bool e501 = e500 == 0;
                if(e501) {
                    pairs::copy_to_device(position, d_position, (sizeof(double) * (3 * particle_capacity)));
                }
                const unsigned long long int a111 = prop_dflags[0];
                const unsigned long long int e504 = a111 | 2;
                prop_dflags[0] = e504;
                const unsigned long long int a113 = prop_hflags[0];
                const unsigned long long int e505 = a113 & (unsigned long long int)(-1);
                prop_hflags[0] = e505;
                build_cell_lists(ncells, nlocal, npbc, grid0_d0_min, grid0_d1_min, grid0_d2_min, cell_capacity, d_cell_sizes, d_dim_cells, dim_cells, d_particle_cell, d_resizes, d_cell_particles, d_position);
                const int a83 = resizes[0];
                const bool e471 = a83 > 0;
                if(e471) {
                    const int a84 = resizes[0];
                    const int e472 = a84 * 2;
                    cell_capacity = e472;
                    cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
                    d_cell_particles = (int *) pairs::device_realloc(d_cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
                }
            }
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                const unsigned long long int a114 = prop_dflags[0];
                const unsigned long long int e506 = a114 & 2;
                const bool e507 = e506 == 0;
                if(e507) {
                    pairs::copy_to_device(position, d_position, (sizeof(double) * (3 * particle_capacity)));
                }
                const unsigned long long int a116 = prop_dflags[0];
                const unsigned long long int e510 = a116 | 2;
                prop_dflags[0] = e510;
                const unsigned long long int a118 = prop_hflags[0];
                const unsigned long long int e511 = a118 & (unsigned long long int)(-1);
                prop_hflags[0] = e511;
                neighbor_lists_build(nlocal, ncells, cell_capacity, neighborlist_capacity, nstencil, d_numneighs, d_particle_cell, d_stencil, d_cell_sizes, d_cell_particles, d_neighborlists, d_resizes, d_position);
                const int a88 = resizes[0];
                const bool e476 = a88 > 0;
                if(e476) {
                    const int a89 = resizes[0];
                    const int e477 = a89 * 2;
                    neighborlist_capacity = e477;
                    neighborlists = (int *) realloc(neighborlists, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                    d_neighborlists = (int *) pairs::device_realloc(d_neighborlists, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                }
            }
        }
        const unsigned long long int a120 = prop_dflags[0];
        const unsigned long long int e512 = a120 | 0;
        prop_dflags[0] = e512;
        const unsigned long long int a122 = prop_hflags[0];
        const unsigned long long int e513 = a122 & (unsigned long long int)(-9);
        prop_hflags[0] = e513;
        reset_volatile_properties(nlocal, d_force);
        const unsigned long long int a123 = prop_dflags[0];
        const unsigned long long int e514 = a123 & 2;
        const bool e515 = e514 == 0;
        if(e515) {
            pairs::copy_to_device(position, d_position, (sizeof(double) * (3 * particle_capacity)));
        }
        const unsigned long long int a124 = prop_dflags[0];
        const unsigned long long int e518 = a124 & 8;
        const bool e519 = e518 == 0;
        if(e519) {
            pairs::copy_to_device(force, d_force, (sizeof(double) * (3 * particle_capacity)));
        }
        const unsigned long long int a126 = prop_dflags[0];
        const unsigned long long int e522 = a126 | 10;
        prop_dflags[0] = e522;
        const unsigned long long int a128 = prop_hflags[0];
        const unsigned long long int e523 = a128 & (unsigned long long int)(-9);
        prop_hflags[0] = e523;
        module0(neighborlist_capacity, nlocal, d_numneighs, d_neighborlists, d_position, d_force);
        const unsigned long long int a129 = prop_dflags[0];
        const unsigned long long int e524 = a129 & 2;
        const bool e525 = e524 == 0;
        if(e525) {
            pairs::copy_to_device(position, d_position, (sizeof(double) * (3 * particle_capacity)));
        }
        const unsigned long long int a130 = prop_dflags[0];
        const unsigned long long int e528 = a130 & 1;
        const bool e529 = e528 == 0;
        if(e529) {
            pairs::copy_to_device(mass, d_mass, (sizeof(double) * particle_capacity));
        }
        const unsigned long long int a131 = prop_dflags[0];
        const unsigned long long int e531 = a131 & 4;
        const bool e532 = e531 == 0;
        if(e532) {
            pairs::copy_to_device(velocity, d_velocity, (sizeof(double) * (3 * particle_capacity)));
        }
        const unsigned long long int a132 = prop_dflags[0];
        const unsigned long long int e535 = a132 & 8;
        const bool e536 = e535 == 0;
        if(e536) {
            pairs::copy_to_device(force, d_force, (sizeof(double) * (3 * particle_capacity)));
        }
        const unsigned long long int a134 = prop_dflags[0];
        const unsigned long long int e539 = a134 | 15;
        prop_dflags[0] = e539;
        const unsigned long long int a136 = prop_hflags[0];
        const unsigned long long int e540 = a136 & (unsigned long long int)(-7);
        prop_hflags[0] = e540;
        module1(nlocal, d_velocity, d_force, d_mass, d_position);
        const int e73 = i1 + 1;
        pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, e73);
        const int e384 = nlocal + npbc;
        pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e384, e73);
    }
    return 0;
}
