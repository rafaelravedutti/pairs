#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
//---
#include "runtime/pairs.hpp"
#include "runtime/read_from_file.hpp"
#include "runtime/vtk.hpp"
#include "runtime/devices/dummy.hpp"

using namespace pairs;


void lj(int neighborlist_capacity, int nlocal, int *numneighs, int *neighborlists, double *position, double *force) {
    PAIRS_DEBUG("lj\n");
    for(int i14 = 0; i14 < nlocal; i14++) {
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
void euler(int nlocal, double *velocity, double *force, double *mass, double *position) {
    PAIRS_DEBUG("euler\n");
    for(int i0 = 0; i0 < nlocal; i0++) {
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
void build_cell_lists_stencil(double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int ncells_capacity, int *ncells, int *nstencil, int *dim_cells, int *resizes, int *stencil) {
    PAIRS_DEBUG("build_cell_lists_stencil\n");
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
    PAIRS_DEBUG("enforce_pbc\n");
    const double e97 = grid0_d0_max - grid0_d0_min;
    const double e104 = grid0_d0_max - grid0_d0_min;
    const double e111 = grid0_d1_max - grid0_d1_min;
    const double e118 = grid0_d1_max - grid0_d1_min;
    const double e125 = grid0_d2_max - grid0_d2_min;
    const double e132 = grid0_d2_max - grid0_d2_min;
    for(int i5 = 0; i5 < nlocal; i5++) {
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
void setup_comm(int nlocal, double grid0_d0_max, double grid0_d0_min, int ghost_capacity, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *nghost, int *ghost_map, int *ghost_mult, int *resizes, double *position) {
    PAIRS_DEBUG("setup_comm\n");
    (*nghost) = 0;
    const int e134 = nlocal + (*nghost);
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
            ghost_map[(*nghost)] = i6;
            const int e141 = (*nghost) * 3;
            ghost_mult[e141] = 1;
            const int e140 = nlocal + (*nghost);
            const int e143 = e140 * 3;
            const int e150 = e140 * 3;
            const int e151 = e150 + 1;
            const int e156 = e140 * 3;
            const int e157 = e156 + 2;
            const double e147 = p18_0 + e135;
            position[e143] = e147;
            const int e148 = (*nghost) * 3;
            const int e149 = e148 + 1;
            ghost_mult[e149] = 0;
            position[e151] = p18_1;
            const int e154 = (*nghost) * 3;
            const int e155 = e154 + 2;
            ghost_mult[e155] = 0;
            position[e157] = p18_2;
            const int e160 = (*nghost) + 1;
            const int e428 = e160 + 1;
            const bool e429 = e428 >= ghost_capacity;
            if(e429) {
                resizes[0] = e160;
            } else {
                (*nghost) = e160;
            }
        }
        const bool e164 = p18_0 > e163;
        if(e164) {
            ghost_map[(*nghost)] = i6;
            const int e166 = (*nghost) * 3;
            ghost_mult[e166] = -1;
            const int e165 = nlocal + (*nghost);
            const int e168 = e165 * 3;
            const int e175 = e165 * 3;
            const int e176 = e175 + 1;
            const int e181 = e165 * 3;
            const int e182 = e181 + 2;
            const double e172 = p18_0 - e135;
            position[e168] = e172;
            const int e173 = (*nghost) * 3;
            const int e174 = e173 + 1;
            ghost_mult[e174] = 0;
            position[e176] = p18_1;
            const int e179 = (*nghost) * 3;
            const int e180 = e179 + 2;
            ghost_mult[e180] = 0;
            position[e182] = p18_2;
            const int e185 = (*nghost) + 1;
            const int e430 = e185 + 1;
            const bool e431 = e430 >= ghost_capacity;
            if(e431) {
                resizes[0] = e185;
            } else {
                (*nghost) = e185;
            }
        }
    }
    const int e186 = nlocal + (*nghost);
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
            ghost_map[(*nghost)] = i7;
            const int e193 = (*nghost) * 3;
            const int e194 = e193 + 1;
            ghost_mult[e194] = 1;
            const int e192 = nlocal + (*nghost);
            const int e195 = e192 * 3;
            const int e196 = e195 + 1;
            const int e202 = e192 * 3;
            const int e208 = e192 * 3;
            const int e209 = e208 + 2;
            const double e199 = p21_1 + e187;
            position[e196] = e199;
            const int e200 = (*nghost) * 3;
            ghost_mult[e200] = 0;
            position[e202] = p21_0;
            const int e206 = (*nghost) * 3;
            const int e207 = e206 + 2;
            ghost_mult[e207] = 0;
            position[e209] = p21_2;
            const int e212 = (*nghost) + 1;
            const int e432 = e212 + 1;
            const bool e433 = e432 >= ghost_capacity;
            if(e433) {
                resizes[0] = e212;
            } else {
                (*nghost) = e212;
            }
        }
        const bool e216 = p21_1 > e215;
        if(e216) {
            ghost_map[(*nghost)] = i7;
            const int e218 = (*nghost) * 3;
            const int e219 = e218 + 1;
            ghost_mult[e219] = -1;
            const int e217 = nlocal + (*nghost);
            const int e220 = e217 * 3;
            const int e221 = e220 + 1;
            const int e227 = e217 * 3;
            const int e233 = e217 * 3;
            const int e234 = e233 + 2;
            const double e224 = p21_1 - e187;
            position[e221] = e224;
            const int e225 = (*nghost) * 3;
            ghost_mult[e225] = 0;
            position[e227] = p21_0;
            const int e231 = (*nghost) * 3;
            const int e232 = e231 + 2;
            ghost_mult[e232] = 0;
            position[e234] = p21_2;
            const int e237 = (*nghost) + 1;
            const int e434 = e237 + 1;
            const bool e435 = e434 >= ghost_capacity;
            if(e435) {
                resizes[0] = e237;
            } else {
                (*nghost) = e237;
            }
        }
    }
    const int e238 = nlocal + (*nghost);
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
            ghost_map[(*nghost)] = i8;
            const int e245 = (*nghost) * 3;
            const int e246 = e245 + 2;
            ghost_mult[e246] = 1;
            const int e244 = nlocal + (*nghost);
            const int e247 = e244 * 3;
            const int e248 = e247 + 2;
            const int e254 = e244 * 3;
            const int e260 = e244 * 3;
            const int e261 = e260 + 1;
            const double e251 = p24_2 + e239;
            position[e248] = e251;
            const int e252 = (*nghost) * 3;
            ghost_mult[e252] = 0;
            position[e254] = p24_0;
            const int e258 = (*nghost) * 3;
            const int e259 = e258 + 1;
            ghost_mult[e259] = 0;
            position[e261] = p24_1;
            const int e264 = (*nghost) + 1;
            const int e436 = e264 + 1;
            const bool e437 = e436 >= ghost_capacity;
            if(e437) {
                resizes[0] = e264;
            } else {
                (*nghost) = e264;
            }
        }
        const bool e268 = p24_2 > e267;
        if(e268) {
            ghost_map[(*nghost)] = i8;
            const int e270 = (*nghost) * 3;
            const int e271 = e270 + 2;
            ghost_mult[e271] = -1;
            const int e269 = nlocal + (*nghost);
            const int e272 = e269 * 3;
            const int e273 = e272 + 2;
            const int e279 = e269 * 3;
            const int e285 = e269 * 3;
            const int e286 = e285 + 1;
            const double e276 = p24_2 - e239;
            position[e273] = e276;
            const int e277 = (*nghost) * 3;
            ghost_mult[e277] = 0;
            position[e279] = p24_0;
            const int e283 = (*nghost) * 3;
            const int e284 = e283 + 1;
            ghost_mult[e284] = 0;
            position[e286] = p24_1;
            const int e289 = (*nghost) + 1;
            const int e438 = e289 + 1;
            const bool e439 = e438 >= ghost_capacity;
            if(e439) {
                resizes[0] = e289;
            } else {
                (*nghost) = e289;
            }
        }
    }
}
void update_comm(double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int nlocal, int nghost, int *ghost_map, int *ghost_mult, double *position) {
    PAIRS_DEBUG("update_comm\n");
    const double e297 = grid0_d0_max - grid0_d0_min;
    const double e307 = grid0_d1_max - grid0_d1_min;
    const double e317 = grid0_d2_max - grid0_d2_min;
    for(int i9 = 0; i9 < nghost; i9++) {
        const int e290 = nlocal + i9;
        const int e291 = e290 * 3;
        const int a39 = ghost_map[i9];
        const int e293 = a39 * 3;
        const double p28_0 = position[e293];
        const int e295 = i9 * 3;
        const int a40 = ghost_mult[e295];
        const double e298 = a40 * e297;
        const double e299 = p28_0 + e298;
        position[e291] = e299;
        const int e300 = nlocal + i9;
        const int e301 = e300 * 3;
        const int e302 = e301 + 1;
        const int a41 = ghost_map[i9];
        const int e303 = a41 * 3;
        const int e304 = e303 + 1;
        const double p30_1 = position[e304];
        const int e305 = i9 * 3;
        const int e306 = e305 + 1;
        const int a42 = ghost_mult[e306];
        const double e308 = a42 * e307;
        const double e309 = p30_1 + e308;
        position[e302] = e309;
        const int e310 = nlocal + i9;
        const int e311 = e310 * 3;
        const int e312 = e311 + 2;
        const int a43 = ghost_map[i9];
        const int e313 = a43 * 3;
        const int e314 = e313 + 2;
        const double p32_2 = position[e314];
        const int e315 = i9 * 3;
        const int e316 = e315 + 2;
        const int a44 = ghost_mult[e316];
        const double e318 = a44 * e317;
        const double e319 = p32_2 + e318;
        position[e312] = e319;
    }
}
void build_cell_lists(int ncells, int nlocal, int nghost, double grid0_d0_min, double grid0_d1_min, double grid0_d2_min, int cell_capacity, int *cell_sizes, int *dim_cells, int *particle_cell, int *resizes, int *cell_particles, double *position) {
    PAIRS_DEBUG("build_cell_lists\n");
    for(int i10 = 0; i10 < ncells; i10++) {
        cell_sizes[i10] = 0;
    }
    const int e320 = nlocal + nghost;
    const int a46 = dim_cells[1];
    const int a47 = dim_cells[2];
    for(int i11 = 0; i11 < e320; i11++) {
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
void neighbor_lists_build(int nlocal, int ncells, int cell_capacity, int neighborlist_capacity, int nstencil, int *numneighs, int *particle_cell, int *stencil, int *cell_sizes, int *cell_particles, int *neighborlists, int *resizes, double *position) {
    PAIRS_DEBUG("neighbor_lists_build\n");
    for(int i12 = 0; i12 < nlocal; i12++) {
        numneighs[i12] = 0;
    }
    for(int i16 = 0; i16 < nlocal; i16++) {
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
void reset_volatile_properties(int nlocal, double *force) {
    PAIRS_DEBUG("reset_volatile_properties\n");
    for(int i13 = 0; i13 < nlocal; i13++) {
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
int main() {
    PairsSimulation *pairs = new PairsSimulation(4, 11);
    int particle_capacity = 10000;
    int nlocal = 0;
    int nghost = 0;
    int ghost_capacity = 100;
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
    int *resizes;
    pairs->addArray(0, "resizes", &resizes, nullptr, (sizeof(int) * 3));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    int *ghost_map;
    pairs->addArray(1, "ghost_map", &ghost_map, nullptr, (sizeof(int) * ghost_capacity));
    int *ghost_mult;
    pairs->addArray(2, "ghost_mult", &ghost_mult, nullptr, (sizeof(int) * (ghost_capacity * 3)));
    double grid_buffer[6];
    pairs->addStaticArray(3, "grid_buffer", grid_buffer, nullptr, (sizeof(double) * 6));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    int dim_cells[3];
    pairs->addStaticArray(4, "dim_cells", dim_cells, nullptr, (sizeof(int) * 3));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    int *cell_particles;
    pairs->addArray(5, "cell_particles", &cell_particles, nullptr, (sizeof(int) * (ncells_capacity * cell_capacity)));
    int *cell_sizes;
    pairs->addArray(6, "cell_sizes", &cell_sizes, nullptr, (sizeof(int) * ncells_capacity));
    int *stencil;
    pairs->addArray(7, "stencil", &stencil, nullptr, (sizeof(int) * 27));
    int *particle_cell;
    pairs->addArray(8, "particle_cell", &particle_cell, nullptr, (sizeof(int) * particle_capacity));
    int *neighborlists;
    pairs->addArray(9, "neighborlists", &neighborlists, nullptr, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
    int *numneighs;
    pairs->addArray(10, "numneighs", &numneighs, nullptr, (sizeof(int) * particle_capacity));
    double *mass;
    pairs->addProperty(0, "mass", &mass, nullptr, Prop_Float, AoS, (0 + particle_capacity));
    double *position;
    pairs->addProperty(1, "position", &position, nullptr, Prop_Vector, AoS, (0 + particle_capacity), 3);
    double *velocity;
    pairs->addProperty(2, "velocity", &velocity, nullptr, Prop_Vector, AoS, (0 + particle_capacity), 3);
    double *force;
    pairs->addProperty(3, "force", &force, nullptr, Prop_Vector, AoS, (0 + particle_capacity), 3);
    const int prop_list_0[] = {0, 1, 2};
    nlocal = pairs::read_particle_data(pairs, "data/minimd_setup_4x4x4.input", grid_buffer, prop_list_0, 3);
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
        build_cell_lists_stencil(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, ncells_capacity, &ncells, &nstencil, dim_cells, resizes, stencil);
        const int a73 = resizes[0];
        const bool e444 = a73 > 0;
        if(e444) {
            PAIRS_DEBUG("resizes[0] -> ncells_capacity\n");
            const int a74 = resizes[0];
            const int e445 = a74 * 2;
            ncells_capacity = e445;
            pairs->reallocArray(5, &cell_particles, nullptr, (sizeof(int) * (ncells_capacity * cell_capacity)));
            pairs->reallocArray(6, &cell_sizes, nullptr, (sizeof(int) * ncells_capacity));
        }
    }
    pairs::vtk_write_data(pairs, "output/test_cpu_local", 0, nlocal, 0);
    const int e91 = nlocal + nghost;
    pairs::vtk_write_data(pairs, "output/test_cpu_ghost", nlocal, e91, 0);
    for(int i1 = 0; i1 < 101; i1++) {
        if(((i1 % 20) == 0)) {
            enforce_pbc(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, nlocal, position);
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                setup_comm(nlocal, grid0_d0_max, grid0_d0_min, ghost_capacity, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, &nghost, ghost_map, ghost_mult, resizes, position);
                const int a78 = resizes[0];
                const bool e450 = a78 > 0;
                if(e450) {
                    PAIRS_DEBUG("resizes[0] -> ghost_capacity\n");
                    const int a79 = resizes[0];
                    const int e451 = a79 * 2;
                    ghost_capacity = e451;
                    pairs->reallocArray(1, &ghost_map, nullptr, (sizeof(int) * ghost_capacity));
                    pairs->reallocArray(2, &ghost_mult, nullptr, (sizeof(int) * (ghost_capacity * 3)));
                }
            }
        } else {
            update_comm(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, nlocal, nghost, ghost_map, ghost_mult, position);
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                build_cell_lists(ncells, nlocal, nghost, grid0_d0_min, grid0_d1_min, grid0_d2_min, cell_capacity, cell_sizes, dim_cells, particle_cell, resizes, cell_particles, position);
                const int a83 = resizes[0];
                const bool e456 = a83 > 0;
                if(e456) {
                    PAIRS_DEBUG("resizes[0] -> cell_capacity\n");
                    const int a84 = resizes[0];
                    const int e457 = a84 * 2;
                    cell_capacity = e457;
                    pairs->reallocArray(5, &cell_particles, nullptr, (sizeof(int) * (ncells_capacity * cell_capacity)));
                }
            }
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                neighbor_lists_build(nlocal, ncells, cell_capacity, neighborlist_capacity, nstencil, numneighs, particle_cell, stencil, cell_sizes, cell_particles, neighborlists, resizes, position);
                const int a88 = resizes[0];
                const bool e461 = a88 > 0;
                if(e461) {
                    PAIRS_DEBUG("resizes[0] -> neighborlist_capacity\n");
                    const int a89 = resizes[0];
                    const int e462 = a89 * 2;
                    neighborlist_capacity = e462;
                    pairs->reallocArray(9, &neighborlists, nullptr, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                }
            }
        }
        reset_volatile_properties(nlocal, force);
        lj(neighborlist_capacity, nlocal, numneighs, neighborlists, position, force);
        euler(nlocal, velocity, force, mass, position);
        const int e73 = i1 + 1;
        pairs::vtk_write_data(pairs, "output/test_cpu_local", 0, nlocal, e73);
        const int e384 = nlocal + nghost;
        pairs::vtk_write_data(pairs, "output/test_cpu_ghost", nlocal, e384, e73);
    }
    return 0;
}
