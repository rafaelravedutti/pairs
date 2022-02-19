#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
//---
#include "runtime/pairs.hpp"
#include "runtime/read_from_file.hpp"
#include "runtime/vtk.hpp"

using namespace pairs;

__constant__ int d_dim_cells[3];

void module_0(int neighborlist_capacity, int nlocal, int *neighborlists, int *numneighs, double *position, double *force) {
    for(int i14 = 0; i14 < nlocal; i14++) {
        const int e369 = i14 * neighborlist_capacity;
        const int e376 = i14 * 3;
        const double p23_0 = position[e376];
        const int e385 = i14 * 3;
        const int e386 = e385 + 1;
        const double p23_1 = position[e386];
        const int e395 = i14 * 3;
        const int e396 = e395 + 2;
        const double p23_2 = position[e396];
        const int e14 = i14 * 3;
        const int e18 = i14 * 3;
        const int e19 = e18 + 1;
        const int e22 = i14 * 3;
        const int e23 = e22 + 2;
        const int a56 = numneighs[i14];
        for(int i15 = 0; i15 < a56; i15++) {
            const int e370 = e369 + i15;
            const int a57 = neighborlists[e370];
            const int e378 = a57 * 3;
            const double p24_0 = position[e378];
            const int e387 = a57 * 3;
            const int e388 = e387 + 1;
            const double p24_1 = position[e388];
            const int e397 = a57 * 3;
            const int e398 = e397 + 2;
            const double p24_2 = position[e398];
            const double e371_0 = p23_0 - p24_0;
            const double e371_1 = p23_1 - p24_1;
            const double e371_2 = p23_2 - p24_2;
            const double e380 = e371_0 * e371_0;
            const double e389 = e371_1 * e371_1;
            const double e390 = e380 + e389;
            const double e399 = e371_2 * e371_2;
            const double e400 = e390 + e399;
            const bool e401 = e400 < 2.5;
            if(e401) {
                const double e1 = 1.0 / e400;
                const double e2 = e1 * e1;
                const double e3 = e2 * e1;
                const double p0_0 = force[e14];
                const double p0_1 = force[e19];
                const double p0_2 = force[e23];
                const double e7 = e3 - 0.5;
                const double e441 = 48.0 * e3;
                const double e442 = e441 * e7;
                const double e443 = e442 * e1;
                const double e9_0 = e371_0 * e443;
                const double e9_1 = e371_1 * e443;
                const double e9_2 = e371_2 * e443;
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
void module_1(int nlocal, double *velocity, double *force, double *mass, double *position) {
    for(int i0 = 0; i0 < nlocal; i0++) {
        const int e31 = i0 * 3;
        const double p1_0 = velocity[e31];
        const int e37 = i0 * 3;
        const int e38 = e37 + 1;
        const double p1_1 = velocity[e38];
        const int e43 = i0 * 3;
        const int e44 = e43 + 2;
        const double p1_2 = velocity[e44];
        const int e29 = i0 * 3;
        const double p2_0 = force[e29];
        const int e35 = i0 * 3;
        const int e36 = e35 + 1;
        const double p2_1 = force[e36];
        const int e41 = i0 * 3;
        const int e42 = e41 + 2;
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
        const double p4_0 = position[e51];
        const int e57 = i0 * 3;
        const int e58 = e57 + 1;
        const double p4_1 = position[e58];
        const int e63 = i0 * 3;
        const int e64 = e63 + 2;
        const double p4_2 = position[e64];
        const int e49 = i0 * 3;
        const double p5_0 = velocity[e49];
        const int e55 = i0 * 3;
        const int e56 = e55 + 1;
        const double p5_1 = velocity[e56];
        const int e61 = i0 * 3;
        const int e62 = e61 + 2;
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
    const double e95 = grid0_d0_max - grid0_d0_min;
    const double e96 = e95 / 2.8;
    const int e97 = ceil(e96) + 2;
    dim_cells[0] = e97;
    const double e99 = grid0_d1_max - grid0_d1_min;
    const double e100 = e99 / 2.8;
    const int e101 = ceil(e100) + 2;
    dim_cells[1] = e101;
    const int a7 = dim_cells[0];
    const int a9 = dim_cells[1];
    const int e102 = a7 * a9;
    const double e103 = grid0_d2_max - grid0_d2_min;
    const double e104 = e103 / 2.8;
    const int e105 = ceil(e104) + 2;
    dim_cells[2] = e105;
    const int a11 = dim_cells[2];
    const int e106 = e102 * a11;
    const int e444 = e106 + 1;
    const bool e445 = e444 >= ncells_capacity;
    if(e445) {
        resizes[0] = e106;
    } else {
        (*ncells) = e106;
    }
    (*nstencil) = 0;
    for(int i2 = -1; i2 < 2; i2++) {
        const int a12 = dim_cells[0];
        const int e107 = i2 * a12;
        for(int i3 = -1; i3 < 2; i3++) {
            const int e108 = e107 + i3;
            const int a13 = dim_cells[1];
            const int e109 = e108 * a13;
            for(int i4 = -1; i4 < 2; i4++) {
                const int e110 = e109 + i4;
                stencil[(*nstencil)] = e110;
                const int e111 = (*nstencil) + 1;
                (*nstencil) = e111;
            }
        }
    }
}
void enforce_pbc(double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int nlocal, double *position) {
    const double e118 = grid0_d0_max - grid0_d0_min;
    const double e125 = grid0_d0_max - grid0_d0_min;
    const double e132 = grid0_d1_max - grid0_d1_min;
    const double e139 = grid0_d1_max - grid0_d1_min;
    const double e146 = grid0_d2_max - grid0_d2_min;
    const double e153 = grid0_d2_max - grid0_d2_min;
    for(int i5 = 0; i5 < nlocal; i5++) {
        const int e123 = i5 * 3;
        const double p6_0 = position[e123];
        const int e137 = i5 * 3;
        const int e138 = e137 + 1;
        const double p6_1 = position[e138];
        const int e151 = i5 * 3;
        const int e152 = e151 + 2;
        const double p6_2 = position[e152];
        const bool e115 = p6_0 < grid0_d0_min;
        if(e115) {
            const double e119 = p6_0 + e118;
            position[e123] = e119;
        }
        const bool e122 = p6_0 > grid0_d0_max;
        if(e122) {
            const double e126 = p6_0 - e125;
            position[e123] = e126;
        }
        const bool e129 = p6_1 < grid0_d1_min;
        if(e129) {
            const double e133 = p6_1 + e132;
            position[e138] = e133;
        }
        const bool e136 = p6_1 > grid0_d1_max;
        if(e136) {
            const double e140 = p6_1 - e139;
            position[e138] = e140;
        }
        const bool e143 = p6_2 < grid0_d2_min;
        if(e143) {
            const double e147 = p6_2 + e146;
            position[e152] = e147;
        }
        const bool e150 = p6_2 > grid0_d2_max;
        if(e150) {
            const double e154 = p6_2 - e153;
            position[e152] = e154;
        }
    }
}
void setup_pbc(int nlocal, double grid0_d0_max, double grid0_d0_min, int pbc_capacity, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *npbc, int *pbc_map, int *pbc_mult, int *resizes, double *position) {
    (*npbc) = 0;
    const int e155 = nlocal + (*npbc);
    const double e157 = grid0_d0_max - grid0_d0_min;
    const double e160 = grid0_d0_min + 2.8;
    const double e184 = grid0_d0_max - 2.8;
    for(int i6 = 0; i6 < e155; i6++) {
        const int e190 = i6 * 3;
        const double p7_0 = position[e190];
        const int e197 = i6 * 3;
        const int e198 = e197 + 1;
        const double p7_1 = position[e198];
        const int e203 = i6 * 3;
        const int e204 = e203 + 2;
        const double p7_2 = position[e204];
        const int e156 = nlocal + (*npbc);
        const int e188 = e156 * 3;
        const double p8_0 = position[e188];
        const int e195 = e156 * 3;
        const int e196 = e195 + 1;
        const double p8_1 = position[e196];
        const int e201 = e156 * 3;
        const int e202 = e201 + 2;
        const double p8_2 = position[e202];
        const bool e161 = p7_0 < e160;
        if(e161) {
            pbc_map[(*npbc)] = i6;
            const int e162 = (*npbc) * 3;
            pbc_mult[e162] = 1;
            const double e168 = p7_0 + e157;
            position[e188] = e168;
            const int e169 = (*npbc) * 3;
            const int e170 = e169 + 1;
            pbc_mult[e170] = 0;
            position[e196] = p7_1;
            const int e175 = (*npbc) * 3;
            const int e176 = e175 + 2;
            pbc_mult[e176] = 0;
            position[e202] = p7_2;
            const int e181 = (*npbc) + 1;
            const int e446 = e181 + 1;
            const bool e447 = e446 >= pbc_capacity;
            if(e447) {
                resizes[0] = e181;
            } else {
                (*npbc) = e181;
            }
        }
        const bool e185 = p7_0 > e184;
        if(e185) {
            pbc_map[(*npbc)] = i6;
            const int e186 = (*npbc) * 3;
            pbc_mult[e186] = -1;
            const double e192 = p7_0 - e157;
            position[e188] = e192;
            const int e193 = (*npbc) * 3;
            const int e194 = e193 + 1;
            pbc_mult[e194] = 0;
            position[e196] = p7_1;
            const int e199 = (*npbc) * 3;
            const int e200 = e199 + 2;
            pbc_mult[e200] = 0;
            position[e202] = p7_2;
            const int e205 = (*npbc) + 1;
            const int e448 = e205 + 1;
            const bool e449 = e448 >= pbc_capacity;
            if(e449) {
                resizes[0] = e205;
            } else {
                (*npbc) = e205;
            }
        }
    }
    const int e206 = nlocal + (*npbc);
    const double e208 = grid0_d1_max - grid0_d1_min;
    const double e211 = grid0_d1_min + 2.8;
    const double e235 = grid0_d1_max - 2.8;
    for(int i7 = 0; i7 < e206; i7++) {
        const int e248 = i7 * 3;
        const double p9_0 = position[e248];
        const int e241 = i7 * 3;
        const int e242 = e241 + 1;
        const double p9_1 = position[e242];
        const int e254 = i7 * 3;
        const int e255 = e254 + 2;
        const double p9_2 = position[e255];
        const int e207 = nlocal + (*npbc);
        const int e246 = e207 * 3;
        const double p10_0 = position[e246];
        const int e239 = e207 * 3;
        const int e240 = e239 + 1;
        const double p10_1 = position[e240];
        const int e252 = e207 * 3;
        const int e253 = e252 + 2;
        const double p10_2 = position[e253];
        const bool e212 = p9_1 < e211;
        if(e212) {
            pbc_map[(*npbc)] = i7;
            const int e213 = (*npbc) * 3;
            const int e214 = e213 + 1;
            pbc_mult[e214] = 1;
            const double e219 = p9_1 + e208;
            position[e240] = e219;
            const int e220 = (*npbc) * 3;
            pbc_mult[e220] = 0;
            position[e246] = p9_0;
            const int e226 = (*npbc) * 3;
            const int e227 = e226 + 2;
            pbc_mult[e227] = 0;
            position[e253] = p9_2;
            const int e232 = (*npbc) + 1;
            const int e450 = e232 + 1;
            const bool e451 = e450 >= pbc_capacity;
            if(e451) {
                resizes[0] = e232;
            } else {
                (*npbc) = e232;
            }
        }
        const bool e236 = p9_1 > e235;
        if(e236) {
            pbc_map[(*npbc)] = i7;
            const int e237 = (*npbc) * 3;
            const int e238 = e237 + 1;
            pbc_mult[e238] = -1;
            const double e243 = p9_1 - e208;
            position[e240] = e243;
            const int e244 = (*npbc) * 3;
            pbc_mult[e244] = 0;
            position[e246] = p9_0;
            const int e250 = (*npbc) * 3;
            const int e251 = e250 + 2;
            pbc_mult[e251] = 0;
            position[e253] = p9_2;
            const int e256 = (*npbc) + 1;
            const int e452 = e256 + 1;
            const bool e453 = e452 >= pbc_capacity;
            if(e453) {
                resizes[0] = e256;
            } else {
                (*npbc) = e256;
            }
        }
    }
    const int e257 = nlocal + (*npbc);
    const double e259 = grid0_d2_max - grid0_d2_min;
    const double e262 = grid0_d2_min + 2.8;
    const double e286 = grid0_d2_max - 2.8;
    for(int i8 = 0; i8 < e257; i8++) {
        const int e299 = i8 * 3;
        const double p11_0 = position[e299];
        const int e305 = i8 * 3;
        const int e306 = e305 + 1;
        const double p11_1 = position[e306];
        const int e292 = i8 * 3;
        const int e293 = e292 + 2;
        const double p11_2 = position[e293];
        const int e258 = nlocal + (*npbc);
        const int e297 = e258 * 3;
        const double p12_0 = position[e297];
        const int e303 = e258 * 3;
        const int e304 = e303 + 1;
        const double p12_1 = position[e304];
        const int e290 = e258 * 3;
        const int e291 = e290 + 2;
        const double p12_2 = position[e291];
        const bool e263 = p11_2 < e262;
        if(e263) {
            pbc_map[(*npbc)] = i8;
            const int e264 = (*npbc) * 3;
            const int e265 = e264 + 2;
            pbc_mult[e265] = 1;
            const double e270 = p11_2 + e259;
            position[e291] = e270;
            const int e271 = (*npbc) * 3;
            pbc_mult[e271] = 0;
            position[e297] = p11_0;
            const int e277 = (*npbc) * 3;
            const int e278 = e277 + 1;
            pbc_mult[e278] = 0;
            position[e304] = p11_1;
            const int e283 = (*npbc) + 1;
            const int e454 = e283 + 1;
            const bool e455 = e454 >= pbc_capacity;
            if(e455) {
                resizes[0] = e283;
            } else {
                (*npbc) = e283;
            }
        }
        const bool e287 = p11_2 > e286;
        if(e287) {
            pbc_map[(*npbc)] = i8;
            const int e288 = (*npbc) * 3;
            const int e289 = e288 + 2;
            pbc_mult[e289] = -1;
            const double e294 = p11_2 - e259;
            position[e291] = e294;
            const int e295 = (*npbc) * 3;
            pbc_mult[e295] = 0;
            position[e297] = p11_0;
            const int e301 = (*npbc) * 3;
            const int e302 = e301 + 1;
            pbc_mult[e302] = 0;
            position[e304] = p11_1;
            const int e307 = (*npbc) + 1;
            const int e456 = e307 + 1;
            const bool e457 = e456 >= pbc_capacity;
            if(e457) {
                resizes[0] = e307;
            } else {
                (*npbc) = e307;
            }
        }
    }
}
void update_pbc(double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int nlocal, int npbc, int *pbc_map, int *pbc_mult, double *position) {
    const double e315 = grid0_d0_max - grid0_d0_min;
    const double e325 = grid0_d1_max - grid0_d1_min;
    const double e335 = grid0_d2_max - grid0_d2_min;
    for(int i9 = 0; i9 < npbc; i9++) {
        const int e308 = nlocal + i9;
        const int e309 = e308 * 3;
        const double p13_0 = position[e309];
        const int a39 = pbc_map[i9];
        const int e311 = a39 * 3;
        const double p14_0 = position[e311];
        const int e313 = i9 * 3;
        const int a40 = pbc_mult[e313];
        const double e316 = a40 * e315;
        const double e317 = p14_0 + e316;
        position[e309] = e317;
        const int e318 = nlocal + i9;
        const int e319 = e318 * 3;
        const int e320 = e319 + 1;
        const double p15_1 = position[e320];
        const int a41 = pbc_map[i9];
        const int e321 = a41 * 3;
        const int e322 = e321 + 1;
        const double p16_1 = position[e322];
        const int e323 = i9 * 3;
        const int e324 = e323 + 1;
        const int a42 = pbc_mult[e324];
        const double e326 = a42 * e325;
        const double e327 = p16_1 + e326;
        position[e320] = e327;
        const int e328 = nlocal + i9;
        const int e329 = e328 * 3;
        const int e330 = e329 + 2;
        const double p17_2 = position[e330];
        const int a43 = pbc_map[i9];
        const int e331 = a43 * 3;
        const int e332 = e331 + 2;
        const double p18_2 = position[e332];
        const int e333 = i9 * 3;
        const int e334 = e333 + 2;
        const int a44 = pbc_mult[e334];
        const double e336 = a44 * e335;
        const double e337 = p18_2 + e336;
        position[e330] = e337;
    }
}
void build_cell_lists(int ncells, double grid0_d0_min, double grid0_d1_min, double grid0_d2_min, int cell_capacity, int nlocal, int npbc, int *cell_sizes, int *dim_cells, int *particle_cell, int *cell_particles, int *resizes, double *position) {
    for(int i10 = 0; i10 < ncells; i10++) {
        cell_sizes[i10] = 0;
    }
    const int e541 = nlocal + npbc;
    for(int i11 = 0; i11 < e541; i11++) {
        const int e338 = i11 * 3;
        const double p19_0 = position[e338];
        const double e340 = p19_0 - grid0_d0_min;
        const double e341 = e340 / 2.8;
        const int e342 = i11 * 3;
        const int e343 = e342 + 1;
        const double p20_1 = position[e343];
        const double e344 = p20_1 - grid0_d1_min;
        const double e345 = e344 / 2.8;
        const int e346 = i11 * 3;
        const int e347 = e346 + 2;
        const double p21_2 = position[e347];
        const double e348 = p21_2 - grid0_d2_min;
        const double e349 = e348 / 2.8;
        const int a46 = dim_cells[1];
        const int e350 = (int)(e341) * a46;
        const int e351 = e350 + (int)(e345);
        const int a47 = dim_cells[2];
        const int e352 = e351 * a47;
        const int e353 = e352 + (int)(e349);
        const bool e354 = e353 >= 0;
        const bool e355 = e353 <= ncells;
        const bool e356 = e354 && e355;
        if(e356) {
            particle_cell[i11] = e353;
            const int e357 = e353 * cell_capacity;
            const int a48 = cell_sizes[e353];
            const int e358 = e357 + a48;
            cell_particles[e358] = i11;
            const int e359 = a48 + 1;
            const int e458 = e359 + 1;
            const bool e459 = e458 >= cell_capacity;
            if(e459) {
                resizes[0] = e359;
            } else {
                cell_sizes[e353] = e359;
            }
        }
    }
}
void neighbor_lists_build(int nlocal, int ncells, int cell_capacity, int neighborlist_capacity, int nstencil, int *numneighs, int *particle_cell, int *stencil, int *cell_particles, int *neighborlists, int *resizes, int *cell_sizes, double *position) {
    for(int i12 = 0; i12 < nlocal; i12++) {
        numneighs[i12] = 0;
    }
    for(int i16 = 0; i16 < nlocal; i16++) {
        for(int i17 = 0; i17 < nstencil; i17++) {
            const int a58 = particle_cell[i16];
            const int a59 = stencil[i17];
            const int e403 = a58 + a59;
            const bool e404 = e403 >= 0;
            const bool e405 = e403 <= ncells;
            const bool e406 = e404 && e405;
            if(e406) {
                const int e407 = e403 * cell_capacity;
                const int e415 = i16 * 3;
                const double p25_0 = position[e415];
                const int e424 = i16 * 3;
                const int e425 = e424 + 1;
                const double p25_1 = position[e425];
                const int e434 = i16 * 3;
                const int e435 = e434 + 2;
                const double p25_2 = position[e435];
                const int e360 = i16 * neighborlist_capacity;
                const int a60 = cell_sizes[e403];
                for(int i18 = 0; i18 < a60; i18++) {
                    const int e408 = e407 + i18;
                    const int a61 = cell_particles[e408];
                    const bool e409 = a61 != i16;
                    if(e409) {
                        const int e417 = a61 * 3;
                        const double p26_0 = position[e417];
                        const int e426 = a61 * 3;
                        const int e427 = e426 + 1;
                        const double p26_1 = position[e427];
                        const int e436 = a61 * 3;
                        const int e437 = e436 + 2;
                        const double p26_2 = position[e437];
                        const double e410_0 = p25_0 - p26_0;
                        const double e410_1 = p25_1 - p26_1;
                        const double e410_2 = p25_2 - p26_2;
                        const double e419 = e410_0 * e410_0;
                        const double e428 = e410_1 * e410_1;
                        const double e429 = e419 + e428;
                        const double e438 = e410_2 * e410_2;
                        const double e439 = e429 + e438;
                        const bool e440 = e439 < 2.8;
                        if(e440) {
                            const int a53 = numneighs[i16];
                            const int e361 = e360 + a53;
                            neighborlists[e361] = a61;
                            const int e362 = a53 + 1;
                            const int e460 = e362 + 1;
                            const bool e461 = e460 >= neighborlist_capacity;
                            if(e461) {
                                resizes[0] = e362;
                            } else {
                                numneighs[i16] = e362;
                            }
                        }
                    }
                }
            }
        }
    }
}
void reset_volatile_properties(int nlocal, double *force) {
    for(int i13 = 0; i13 < nlocal; i13++) {
        const int e363 = i13 * 3;
        const double p22_0 = force[e363];
        const int e365 = i13 * 3;
        const int e366 = e365 + 1;
        const double p22_1 = force[e366];
        const int e367 = i13 * 3;
        const int e368 = e367 + 2;
        const double p22_2 = force[e368];
        force[e363] = 0.0;
        force[e366] = 0.0;
        force[e368] = 0.0;
    }
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
    double *mass = (double *) malloc((sizeof(double) * (particle_capacity + pbc_capacity)));
    double *d_mass = (double *) pairs::device_alloc((sizeof(double) * (particle_capacity + pbc_capacity)));
    ps->addProperty(Property(0, "mass", mass, Prop_Float));
    double *position = (double *) malloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    double *d_position = (double *) pairs::device_alloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    ps->addProperty(Property(1, "position", position, Prop_Vector, AoS, (particle_capacity + pbc_capacity), 3));
    double *velocity = (double *) malloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    double *d_velocity = (double *) pairs::device_alloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    ps->addProperty(Property(2, "velocity", velocity, Prop_Vector, AoS, (particle_capacity + pbc_capacity), 3));
    double *force = (double *) malloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    double *d_force = (double *) pairs::device_alloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    ps->addProperty(Property(3, "force", force, Prop_Vector, AoS, (particle_capacity + pbc_capacity), 3));
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
        const unsigned long long int a92 = prop_hflags[0];
        const unsigned long long int e499 = a92 | 0;
        prop_hflags[0] = e499;
        const unsigned long long int a94 = prop_dflags[0];
        const unsigned long long int e500 = a94 & -1;
        prop_dflags[0] = e500;
        build_cell_lists_stencil(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, ncells_capacity, &ncells, &nstencil, dim_cells, resizes, stencil);
        const int a74 = resizes[0];
        const bool e463 = a74 > 0;
        if(e463) {
            const int a75 = resizes[0];
            const int e464 = a75 * 2;
            ncells_capacity = e464;
            cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
            d_cell_particles = (int *) pairs::device_realloc(d_cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
            cell_sizes = (int *) realloc(cell_sizes, (sizeof(int) * ncells_capacity));
            d_cell_sizes = (int *) pairs::device_realloc(d_cell_sizes, (sizeof(int) * ncells_capacity));
        }
    }
    const int e112 = nlocal + npbc;
    pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, 0);
    pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e112, 0);
    for(int i1 = 0; i1 < 101; i1++) {
        if(((i1 % 20) == 0)) {
            const unsigned long long int a95 = prop_dflags[0];
            const unsigned long long int e501 = a95 & 2;
            const bool e502 = e501 == 0;
            if(e502) {
                pairs::copy_to_device(position, d_position)
            }
            const unsigned long long int a97 = prop_dflags[0];
            const unsigned long long int e503 = a97 | 2;
            prop_dflags[0] = e503;
            const unsigned long long int a99 = prop_hflags[0];
            const unsigned long long int e504 = a99 & -3;
            prop_hflags[0] = e504;
            enforce_pbc(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, nlocal, d_position);
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                const unsigned long long int a100 = prop_hflags[0];
                const unsigned long long int e505 = a100 & 2;
                const bool e506 = e505 == 0;
                if(e506) {
                    pairs::copy_to_host(d_position, position)
                }
                const unsigned long long int a102 = prop_hflags[0];
                const unsigned long long int e507 = a102 | 2;
                prop_hflags[0] = e507;
                const unsigned long long int a104 = prop_dflags[0];
                const unsigned long long int e508 = a104 & -3;
                prop_dflags[0] = e508;
                setup_pbc(nlocal, grid0_d0_max, grid0_d0_min, pbc_capacity, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, &npbc, pbc_map, pbc_mult, resizes, position);
                const int a79 = resizes[0];
                const bool e484 = a79 > 0;
                if(e484) {
                    const int a80 = resizes[0];
                    const int e485 = a80 * 2;
                    pbc_capacity = e485;
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
            const unsigned long long int a105 = prop_dflags[0];
            const unsigned long long int e509 = a105 & 2;
            const bool e510 = e509 == 0;
            if(e510) {
                pairs::copy_to_device(position, d_position)
            }
            const unsigned long long int a107 = prop_dflags[0];
            const unsigned long long int e511 = a107 | 2;
            prop_dflags[0] = e511;
            const unsigned long long int a109 = prop_hflags[0];
            const unsigned long long int e512 = a109 & -3;
            prop_hflags[0] = e512;
            update_pbc(grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, nlocal, npbc, d_pbc_map, d_pbc_mult, d_position);
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                const unsigned long long int a110 = prop_dflags[0];
                const unsigned long long int e513 = a110 & 2;
                const bool e514 = e513 == 0;
                if(e514) {
                    pairs::copy_to_device(position, d_position)
                }
                const unsigned long long int a112 = prop_dflags[0];
                const unsigned long long int e515 = a112 | 2;
                prop_dflags[0] = e515;
                const unsigned long long int a114 = prop_hflags[0];
                const unsigned long long int e516 = a114 & -1;
                prop_hflags[0] = e516;
                build_cell_lists(ncells, grid0_d0_min, grid0_d1_min, grid0_d2_min, cell_capacity, nlocal, npbc, d_cell_sizes, d_dim_cells, d_particle_cell, d_cell_particles, d_resizes, d_position);
                const int a84 = resizes[0];
                const bool e490 = a84 > 0;
                if(e490) {
                    const int a85 = resizes[0];
                    const int e491 = a85 * 2;
                    cell_capacity = e491;
                    cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
                    d_cell_particles = (int *) pairs::device_realloc(d_cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
                }
            }
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                const unsigned long long int a115 = prop_dflags[0];
                const unsigned long long int e517 = a115 & 2;
                const bool e518 = e517 == 0;
                if(e518) {
                    pairs::copy_to_device(position, d_position)
                }
                const unsigned long long int a117 = prop_dflags[0];
                const unsigned long long int e519 = a117 | 2;
                prop_dflags[0] = e519;
                const unsigned long long int a119 = prop_hflags[0];
                const unsigned long long int e520 = a119 & -1;
                prop_hflags[0] = e520;
                neighbor_lists_build(nlocal, ncells, cell_capacity, neighborlist_capacity, nstencil, d_numneighs, d_particle_cell, d_stencil, d_cell_particles, d_neighborlists, d_resizes, d_cell_sizes, d_position);
                const int a89 = resizes[0];
                const bool e495 = a89 > 0;
                if(e495) {
                    const int a90 = resizes[0];
                    const int e496 = a90 * 2;
                    neighborlist_capacity = e496;
                    neighborlists = (int *) realloc(neighborlists, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                    d_neighborlists = (int *) pairs::device_realloc(d_neighborlists, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                }
            }
        }
        const unsigned long long int a120 = prop_dflags[0];
        const unsigned long long int e521 = a120 & 8;
        const bool e522 = e521 == 0;
        if(e522) {
            pairs::copy_to_device(force, d_force)
        }
        const unsigned long long int a122 = prop_dflags[0];
        const unsigned long long int e523 = a122 | 8;
        prop_dflags[0] = e523;
        const unsigned long long int a124 = prop_hflags[0];
        const unsigned long long int e524 = a124 & -9;
        prop_hflags[0] = e524;
        reset_volatile_properties(nlocal, d_force);
        const unsigned long long int a125 = prop_dflags[0];
        const unsigned long long int e525 = a125 & 8;
        const bool e526 = e525 == 0;
        if(e526) {
            pairs::copy_to_device(force, d_force)
        }
        const unsigned long long int a126 = prop_dflags[0];
        const unsigned long long int e527 = a126 & 2;
        const bool e528 = e527 == 0;
        if(e528) {
            pairs::copy_to_device(position, d_position)
        }
        const unsigned long long int a128 = prop_dflags[0];
        const unsigned long long int e529 = a128 | 10;
        prop_dflags[0] = e529;
        const unsigned long long int a130 = prop_hflags[0];
        const unsigned long long int e530 = a130 & -9;
        prop_hflags[0] = e530;
        module_0(neighborlist_capacity, nlocal, d_neighborlists, d_numneighs, d_position, d_force);
        const unsigned long long int a131 = prop_dflags[0];
        const unsigned long long int e531 = a131 & 4;
        const bool e532 = e531 == 0;
        if(e532) {
            pairs::copy_to_device(velocity, d_velocity)
        }
        const unsigned long long int a132 = prop_dflags[0];
        const unsigned long long int e533 = a132 & 1;
        const bool e534 = e533 == 0;
        if(e534) {
            pairs::copy_to_device(mass, d_mass)
        }
        const unsigned long long int a133 = prop_dflags[0];
        const unsigned long long int e535 = a133 & 8;
        const bool e536 = e535 == 0;
        if(e536) {
            pairs::copy_to_device(force, d_force)
        }
        const unsigned long long int a134 = prop_dflags[0];
        const unsigned long long int e537 = a134 & 2;
        const bool e538 = e537 == 0;
        if(e538) {
            pairs::copy_to_device(position, d_position)
        }
        const unsigned long long int a136 = prop_dflags[0];
        const unsigned long long int e539 = a136 | 15;
        prop_dflags[0] = e539;
        const unsigned long long int a138 = prop_hflags[0];
        const unsigned long long int e540 = a138 & -7;
        prop_hflags[0] = e540;
        module_1(nlocal, d_velocity, d_force, d_mass, d_position);
        const int e73 = i1 + 1;
        const int e402 = nlocal + npbc;
        pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, e73);
        pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e402, e73);
    }
    return 0;
}
