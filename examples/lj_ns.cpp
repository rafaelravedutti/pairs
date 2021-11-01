#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
//---
#include "../src/pairs/runtime/pairs.hpp"
#include "../src/pairs/runtime/read_from_file.hpp"
#include "../src/pairs/runtime/vtk.hpp"

using namespace pairs;

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
    double grid_buffer[6];
    int dim_cells[3];
    int *cell_particles = (int *) malloc((sizeof(int) * (ncells_capacity * cell_capacity)));
    int *cell_sizes = (int *) malloc((sizeof(int) * ncells_capacity));
    int *stencil = (int *) malloc((sizeof(int) * 27));
    int *particle_cell = (int *) malloc((sizeof(int) * particle_capacity));
    int *neighborlists = (int *) malloc((sizeof(int) * (particle_capacity * neighborlist_capacity)));
    int *numneighs = (int *) malloc((sizeof(int) * particle_capacity));
    int *pbc_map = (int *) malloc((sizeof(int) * pbc_capacity));
    int *pbc_mult = (int *) malloc((sizeof(int) * (pbc_capacity * 3)));
    double *mass = (double *) malloc((sizeof(double) * (particle_capacity + pbc_capacity)));
    ps->addProperty(Property(0, "mass", mass, Prop_Float));
    double *position = (double *) malloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    ps->addProperty(Property(1, "position", position, Prop_Vector, AoS, (particle_capacity + pbc_capacity), 3));
    double *velocity = (double *) malloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
    ps->addProperty(Property(2, "velocity", velocity, Prop_Vector, AoS, (particle_capacity + pbc_capacity), 3));
    double *force = (double *) malloc((sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
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
    const double e94 = grid0_d0_max - grid0_d0_min;
    const double e95 = e94 / 2.8;
    const int e96 = ceil(e95) + 2;
    dim_cells[0] = e96;
    const double e98 = grid0_d1_max - grid0_d1_min;
    const double e99 = e98 / 2.8;
    const int e100 = ceil(e99) + 2;
    dim_cells[1] = e100;
    const int a7 = dim_cells[0];
    const int a9 = dim_cells[1];
    const int e101 = a7 * a9;
    const double e102 = grid0_d2_max - grid0_d2_min;
    const double e103 = e102 / 2.8;
    const int e104 = ceil(e103) + 2;
    dim_cells[2] = e104;
    const int a11 = dim_cells[2];
    const int e105 = e101 * a11;
    ncells = e105;
    int resize = 1;
    while((resize > 0)) {
        resize = 0;
        const bool e107 = ncells >= ncells_capacity;
        if(e107) {
            resize = ncells;
        }
        const bool e108 = resize > 0;
        if(e108) {
            fprintf(stdout, "Resize ncells_capacity\n");
            fflush(stdout);
            const int e109 = resize * 2;
            ncells_capacity = e109;
            cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
            cell_sizes = (int *) realloc(cell_sizes, (sizeof(int) * ncells_capacity));
        }
    }
    nstencil = 0;
    for(int i2 = -1; i2 < 2; i2++) {
        const int a12 = dim_cells[0];
        const int e113 = i2 * a12;
        for(int i3 = -1; i3 < 2; i3++) {
            const int e114 = e113 + i3;
            const int a13 = dim_cells[1];
            const int e115 = e114 * a13;
            for(int i4 = -1; i4 < 2; i4++) {
                const int e116 = e115 + i4;
                stencil[nstencil] = e116;
                const int e117 = nstencil + 1;
                nstencil = e117;
            }
        }
    }
    const int e118 = nlocal + npbc;
    pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, 0);
    pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e118, 0);
    for(int i1 = 0; i1 < 101; i1++) {
        const int e65 = i1 % 20;
        const bool e66 = e65 == 0;
        if(e66) {
            //pairs::copy_to_device(position)
            const double e124 = grid0_d0_max - grid0_d0_min;
            const double e131 = grid0_d0_max - grid0_d0_min;
            const double e138 = grid0_d1_max - grid0_d1_min;
            const double e145 = grid0_d1_max - grid0_d1_min;
            const double e152 = grid0_d2_max - grid0_d2_min;
            const double e159 = grid0_d2_max - grid0_d2_min;
            for(int i5 = 0; i5 < nlocal; i5++) {
                const int e119 = i5 * 3;
                const double p6_0 = position[e119];
                const bool e121 = p6_0 < grid0_d0_min;
                if(e121) {
                    const int e122 = i5 * 3;
                    const double p7_0 = position[e122];
                    const double e125 = p7_0 + e124;
                    position[e122] = e125;
                }
                const int e126 = i5 * 3;
                const double p8_0 = position[e126];
                const bool e128 = p8_0 > grid0_d0_max;
                if(e128) {
                    const int e129 = i5 * 3;
                    const double p9_0 = position[e129];
                    const double e132 = p9_0 - e131;
                    position[e129] = e132;
                }
                const int e133 = i5 * 3;
                const int e134 = e133 + 1;
                const double p10_1 = position[e134];
                const bool e135 = p10_1 < grid0_d1_min;
                if(e135) {
                    const int e136 = i5 * 3;
                    const int e137 = e136 + 1;
                    const double p11_1 = position[e137];
                    const double e139 = p11_1 + e138;
                    position[e137] = e139;
                }
                const int e140 = i5 * 3;
                const int e141 = e140 + 1;
                const double p12_1 = position[e141];
                const bool e142 = p12_1 > grid0_d1_max;
                if(e142) {
                    const int e143 = i5 * 3;
                    const int e144 = e143 + 1;
                    const double p13_1 = position[e144];
                    const double e146 = p13_1 - e145;
                    position[e144] = e146;
                }
                const int e147 = i5 * 3;
                const int e148 = e147 + 2;
                const double p14_2 = position[e148];
                const bool e149 = p14_2 < grid0_d2_min;
                if(e149) {
                    const int e150 = i5 * 3;
                    const int e151 = e150 + 2;
                    const double p15_2 = position[e151];
                    const double e153 = p15_2 + e152;
                    position[e151] = e153;
                }
                const int e154 = i5 * 3;
                const int e155 = e154 + 2;
                const double p16_2 = position[e155];
                const bool e156 = p16_2 > grid0_d2_max;
                if(e156) {
                    const int e157 = i5 * 3;
                    const int e158 = e157 + 2;
                    const double p17_2 = position[e158];
                    const double e160 = p17_2 - e159;
                    position[e158] = e160;
                }
            }
        }
        const int e67 = i1 % 20;
        const bool e68 = e67 == 0;
        if(e68) {
            resize = 1;
            while((resize > 0)) {
                resize = 0;
                npbc = 0;
                const int e162 = nlocal + npbc;
                const double e166 = grid0_d0_min + 2.8;
                const double e177 = grid0_d0_max - grid0_d0_min;
                const double e194 = grid0_d0_max - 2.8;
                const double e205 = grid0_d0_max - grid0_d0_min;
                for(int i6 = 0; i6 < e162; i6++) {
                    const int e163 = nlocal + npbc;
                    const int e164 = i6 * 3;
                    const double p18_0 = position[e164];
                    const bool e167 = p18_0 < e166;
                    if(e167) {
                        const bool e168 = npbc >= pbc_capacity;
                        if(e168) {
                            const bool e169 = resize > npbc;
                            resize = (e169) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i6;
                            const int e171 = npbc * 3;
                            pbc_mult[e171] = 1;
                            const int e173 = e163 * 3;
                            const double p19_0 = position[e173];
                            const int e175 = i6 * 3;
                            const double p20_0 = position[e175];
                            const double e178 = p20_0 + e177;
                            position[e173] = e178;
                            const int e179 = npbc * 3;
                            const int e180 = e179 + 1;
                            pbc_mult[e180] = 0;
                            const int e181 = e163 * 3;
                            const int e182 = e181 + 1;
                            const double p21_1 = position[e182];
                            const int e183 = i6 * 3;
                            const int e184 = e183 + 1;
                            const double p22_1 = position[e184];
                            position[e182] = p22_1;
                            const int e185 = npbc * 3;
                            const int e186 = e185 + 2;
                            pbc_mult[e186] = 0;
                            const int e187 = e163 * 3;
                            const int e188 = e187 + 2;
                            const double p23_2 = position[e188];
                            const int e189 = i6 * 3;
                            const int e190 = e189 + 2;
                            const double p24_2 = position[e190];
                            position[e188] = p24_2;
                            const int e191 = npbc + 1;
                            npbc = e191;
                        }
                    }
                    const int e192 = i6 * 3;
                    const double p25_0 = position[e192];
                    const bool e195 = p25_0 > e194;
                    if(e195) {
                        const bool e196 = npbc >= pbc_capacity;
                        if(e196) {
                            const bool e197 = resize > npbc;
                            resize = (e197) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i6;
                            const int e199 = npbc * 3;
                            pbc_mult[e199] = -1;
                            const int e201 = e163 * 3;
                            const double p26_0 = position[e201];
                            const int e203 = i6 * 3;
                            const double p27_0 = position[e203];
                            const double e206 = p27_0 - e205;
                            position[e201] = e206;
                            const int e207 = npbc * 3;
                            const int e208 = e207 + 1;
                            pbc_mult[e208] = 0;
                            const int e209 = e163 * 3;
                            const int e210 = e209 + 1;
                            const double p28_1 = position[e210];
                            const int e211 = i6 * 3;
                            const int e212 = e211 + 1;
                            const double p29_1 = position[e212];
                            position[e210] = p29_1;
                            const int e213 = npbc * 3;
                            const int e214 = e213 + 2;
                            pbc_mult[e214] = 0;
                            const int e215 = e163 * 3;
                            const int e216 = e215 + 2;
                            const double p30_2 = position[e216];
                            const int e217 = i6 * 3;
                            const int e218 = e217 + 2;
                            const double p31_2 = position[e218];
                            position[e216] = p31_2;
                            const int e219 = npbc + 1;
                            npbc = e219;
                        }
                    }
                }
                const int e220 = nlocal + npbc;
                const double e224 = grid0_d1_min + 2.8;
                const double e235 = grid0_d1_max - grid0_d1_min;
                const double e252 = grid0_d1_max - 2.8;
                const double e263 = grid0_d1_max - grid0_d1_min;
                for(int i7 = 0; i7 < e220; i7++) {
                    const int e221 = nlocal + npbc;
                    const int e222 = i7 * 3;
                    const int e223 = e222 + 1;
                    const double p32_1 = position[e223];
                    const bool e225 = p32_1 < e224;
                    if(e225) {
                        const bool e226 = npbc >= pbc_capacity;
                        if(e226) {
                            const bool e227 = resize > npbc;
                            resize = (e227) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i7;
                            const int e229 = npbc * 3;
                            const int e230 = e229 + 1;
                            pbc_mult[e230] = 1;
                            const int e231 = e221 * 3;
                            const int e232 = e231 + 1;
                            const double p33_1 = position[e232];
                            const int e233 = i7 * 3;
                            const int e234 = e233 + 1;
                            const double p34_1 = position[e234];
                            const double e236 = p34_1 + e235;
                            position[e232] = e236;
                            const int e237 = npbc * 3;
                            pbc_mult[e237] = 0;
                            const int e239 = e221 * 3;
                            const double p35_0 = position[e239];
                            const int e241 = i7 * 3;
                            const double p36_0 = position[e241];
                            position[e239] = p36_0;
                            const int e243 = npbc * 3;
                            const int e244 = e243 + 2;
                            pbc_mult[e244] = 0;
                            const int e245 = e221 * 3;
                            const int e246 = e245 + 2;
                            const double p37_2 = position[e246];
                            const int e247 = i7 * 3;
                            const int e248 = e247 + 2;
                            const double p38_2 = position[e248];
                            position[e246] = p38_2;
                            const int e249 = npbc + 1;
                            npbc = e249;
                        }
                    }
                    const int e250 = i7 * 3;
                    const int e251 = e250 + 1;
                    const double p39_1 = position[e251];
                    const bool e253 = p39_1 > e252;
                    if(e253) {
                        const bool e254 = npbc >= pbc_capacity;
                        if(e254) {
                            const bool e255 = resize > npbc;
                            resize = (e255) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i7;
                            const int e257 = npbc * 3;
                            const int e258 = e257 + 1;
                            pbc_mult[e258] = -1;
                            const int e259 = e221 * 3;
                            const int e260 = e259 + 1;
                            const double p40_1 = position[e260];
                            const int e261 = i7 * 3;
                            const int e262 = e261 + 1;
                            const double p41_1 = position[e262];
                            const double e264 = p41_1 - e263;
                            position[e260] = e264;
                            const int e265 = npbc * 3;
                            pbc_mult[e265] = 0;
                            const int e267 = e221 * 3;
                            const double p42_0 = position[e267];
                            const int e269 = i7 * 3;
                            const double p43_0 = position[e269];
                            position[e267] = p43_0;
                            const int e271 = npbc * 3;
                            const int e272 = e271 + 2;
                            pbc_mult[e272] = 0;
                            const int e273 = e221 * 3;
                            const int e274 = e273 + 2;
                            const double p44_2 = position[e274];
                            const int e275 = i7 * 3;
                            const int e276 = e275 + 2;
                            const double p45_2 = position[e276];
                            position[e274] = p45_2;
                            const int e277 = npbc + 1;
                            npbc = e277;
                        }
                    }
                }
                const int e278 = nlocal + npbc;
                const double e282 = grid0_d2_min + 2.8;
                const double e293 = grid0_d2_max - grid0_d2_min;
                const double e310 = grid0_d2_max - 2.8;
                const double e321 = grid0_d2_max - grid0_d2_min;
                for(int i8 = 0; i8 < e278; i8++) {
                    const int e279 = nlocal + npbc;
                    const int e280 = i8 * 3;
                    const int e281 = e280 + 2;
                    const double p46_2 = position[e281];
                    const bool e283 = p46_2 < e282;
                    if(e283) {
                        const bool e284 = npbc >= pbc_capacity;
                        if(e284) {
                            const bool e285 = resize > npbc;
                            resize = (e285) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i8;
                            const int e287 = npbc * 3;
                            const int e288 = e287 + 2;
                            pbc_mult[e288] = 1;
                            const int e289 = e279 * 3;
                            const int e290 = e289 + 2;
                            const double p47_2 = position[e290];
                            const int e291 = i8 * 3;
                            const int e292 = e291 + 2;
                            const double p48_2 = position[e292];
                            const double e294 = p48_2 + e293;
                            position[e290] = e294;
                            const int e295 = npbc * 3;
                            pbc_mult[e295] = 0;
                            const int e297 = e279 * 3;
                            const double p49_0 = position[e297];
                            const int e299 = i8 * 3;
                            const double p50_0 = position[e299];
                            position[e297] = p50_0;
                            const int e301 = npbc * 3;
                            const int e302 = e301 + 1;
                            pbc_mult[e302] = 0;
                            const int e303 = e279 * 3;
                            const int e304 = e303 + 1;
                            const double p51_1 = position[e304];
                            const int e305 = i8 * 3;
                            const int e306 = e305 + 1;
                            const double p52_1 = position[e306];
                            position[e304] = p52_1;
                            const int e307 = npbc + 1;
                            npbc = e307;
                        }
                    }
                    const int e308 = i8 * 3;
                    const int e309 = e308 + 2;
                    const double p53_2 = position[e309];
                    const bool e311 = p53_2 > e310;
                    if(e311) {
                        const bool e312 = npbc >= pbc_capacity;
                        if(e312) {
                            const bool e313 = resize > npbc;
                            resize = (e313) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i8;
                            const int e315 = npbc * 3;
                            const int e316 = e315 + 2;
                            pbc_mult[e316] = -1;
                            const int e317 = e279 * 3;
                            const int e318 = e317 + 2;
                            const double p54_2 = position[e318];
                            const int e319 = i8 * 3;
                            const int e320 = e319 + 2;
                            const double p55_2 = position[e320];
                            const double e322 = p55_2 - e321;
                            position[e318] = e322;
                            const int e323 = npbc * 3;
                            pbc_mult[e323] = 0;
                            const int e325 = e279 * 3;
                            const double p56_0 = position[e325];
                            const int e327 = i8 * 3;
                            const double p57_0 = position[e327];
                            position[e325] = p57_0;
                            const int e329 = npbc * 3;
                            const int e330 = e329 + 1;
                            pbc_mult[e330] = 0;
                            const int e331 = e279 * 3;
                            const int e332 = e331 + 1;
                            const double p58_1 = position[e332];
                            const int e333 = i8 * 3;
                            const int e334 = e333 + 1;
                            const double p59_1 = position[e334];
                            position[e332] = p59_1;
                            const int e335 = npbc + 1;
                            npbc = e335;
                        }
                    }
                }
                const bool e336 = resize > 0;
                if(e336) {
                    fprintf(stdout, "Resize pbc_capacity\n");
                    fflush(stdout);
                    const int e337 = resize * 2;
                    pbc_capacity = e337;
                    pbc_map = (int *) realloc(pbc_map, (sizeof(int) * pbc_capacity));
                    pbc_mult = (int *) realloc(pbc_mult, (sizeof(int) * (pbc_capacity * 3)));
                    mass = (double *) realloc(mass, (sizeof(double) * (particle_capacity + pbc_capacity)));
                    ps->updateProperty(0, mass);
                    position = (double *) realloc(position, (sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
                    ps->updateProperty(1, position, (particle_capacity + pbc_capacity), 3);
                    velocity = (double *) realloc(velocity, (sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
                    ps->updateProperty(2, velocity, (particle_capacity + pbc_capacity), 3);
                    force = (double *) realloc(force, (sizeof(double) * ((particle_capacity + pbc_capacity) * 3)));
                    ps->updateProperty(3, force, (particle_capacity + pbc_capacity), 3);
                }
            }
        } else {
            const double e357 = grid0_d0_max - grid0_d0_min;
            const double e367 = grid0_d1_max - grid0_d1_min;
            const double e377 = grid0_d2_max - grid0_d2_min;
            for(int i9 = 0; i9 < npbc; i9++) {
                const int e350 = nlocal + i9;
                const int e351 = e350 * 3;
                const double p60_0 = position[e351];
                const int a39 = pbc_map[i9];
                const int e353 = a39 * 3;
                const double p61_0 = position[e353];
                const int e355 = i9 * 3;
                const int a40 = pbc_mult[e355];
                const double e358 = a40 * e357;
                const double e359 = p61_0 + e358;
                position[e351] = e359;
                const int e360 = nlocal + i9;
                const int e361 = e360 * 3;
                const int e362 = e361 + 1;
                const double p62_1 = position[e362];
                const int a41 = pbc_map[i9];
                const int e363 = a41 * 3;
                const int e364 = e363 + 1;
                const double p63_1 = position[e364];
                const int e365 = i9 * 3;
                const int e366 = e365 + 1;
                const int a42 = pbc_mult[e366];
                const double e368 = a42 * e367;
                const double e369 = p63_1 + e368;
                position[e362] = e369;
                const int e370 = nlocal + i9;
                const int e371 = e370 * 3;
                const int e372 = e371 + 2;
                const double p64_2 = position[e372];
                const int a43 = pbc_map[i9];
                const int e373 = a43 * 3;
                const int e374 = e373 + 2;
                const double p65_2 = position[e374];
                const int e375 = i9 * 3;
                const int e376 = e375 + 2;
                const int a44 = pbc_mult[e376];
                const double e378 = a44 * e377;
                const double e379 = p65_2 + e378;
                position[e372] = e379;
            }
        }
        const int e69 = i1 % 20;
        const bool e70 = e69 == 0;
        if(e70) {
            resize = 1;
            while((resize > 0)) {
                resize = 0;
                for(int i10 = 0; i10 < ncells; i10++) {
                    cell_sizes[i10] = 0;
                }
                const int e499 = nlocal + npbc;
                for(int i11 = 0; i11 < e499; i11++) {
                    const int e381 = i11 * 3;
                    const double p66_0 = position[e381];
                    const double e383 = p66_0 - grid0_d0_min;
                    const double e384 = e383 / 2.8;
                    const int e385 = i11 * 3;
                    const int e386 = e385 + 1;
                    const double p67_1 = position[e386];
                    const double e387 = p67_1 - grid0_d1_min;
                    const double e388 = e387 / 2.8;
                    const int e389 = i11 * 3;
                    const int e390 = e389 + 2;
                    const double p68_2 = position[e390];
                    const double e391 = p68_2 - grid0_d2_min;
                    const double e392 = e391 / 2.8;
                    const int a46 = dim_cells[1];
                    const int e393 = (int)(e384) * a46;
                    const int e394 = e393 + (int)(e388);
                    const int a47 = dim_cells[2];
                    const int e395 = e394 * a47;
                    const int e396 = e395 + (int)(e392);
                    const bool e397 = e396 >= 0;
                    const bool e398 = e396 <= ncells;
                    const bool e399 = e397 && e398;
                    if(e399) {
                        const int a48 = cell_sizes[e396];
                        const bool e400 = a48 >= cell_capacity;
                        if(e400) {
                            resize = a48;
                        } else {
                            const int e401 = e396 * cell_capacity;
                            const int e402 = e401 + a48;
                            cell_particles[e402] = i11;
                            particle_cell[i11] = e396;
                        }
                        const int e403 = a48 + 1;
                        cell_sizes[e396] = e403;
                    }
                }
                const bool e404 = resize > 0;
                if(e404) {
                    fprintf(stdout, "Resize cell_capacity\n");
                    fflush(stdout);
                    const int e405 = resize * 2;
                    cell_capacity = e405;
                    cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
                }
            }
        }
        const int e71 = i1 % 20;
        const bool e72 = e71 == 0;
        if(e72) {
            resize = 1;
            while((resize > 0)) {
                resize = 0;
                for(int i12 = 0; i12 < nlocal; i12++) {
                    numneighs[i12] = 0;
                }
                for(int i16 = 0; i16 < nlocal; i16++) {
                    for(int i17 = 0; i17 < nstencil; i17++) {
                        const int a58 = particle_cell[i16];
                        const int a59 = stencil[i17];
                        const int e457 = a58 + a59;
                        const bool e458 = e457 >= 0;
                        const bool e459 = e457 <= ncells;
                        const bool e460 = e458 && e459;
                        if(e460) {
                            const int e461 = e457 * cell_capacity;
                            const int e469 = i16 * 3;
                            const double p72_0 = position[e469];
                            const int e478 = i16 * 3;
                            const int e479 = e478 + 1;
                            const double p72_1 = position[e479];
                            const int e488 = i16 * 3;
                            const int e489 = e488 + 2;
                            const double p72_2 = position[e489];
                            const int e410 = i16 * neighborlist_capacity;
                            const int a60 = cell_sizes[e457];
                            for(int i18 = 0; i18 < a60; i18++) {
                                const int e462 = e461 + i18;
                                const int a61 = cell_particles[e462];
                                const bool e463 = a61 != i16;
                                if(e463) {
                                    const int e471 = a61 * 3;
                                    const double p73_0 = position[e471];
                                    const int e480 = a61 * 3;
                                    const int e481 = e480 + 1;
                                    const double p73_1 = position[e481];
                                    const int e490 = a61 * 3;
                                    const int e491 = e490 + 2;
                                    const double p73_2 = position[e491];
                                    const double e464_0 = p72_0 - p73_0;
                                    const double e464_1 = p72_1 - p73_1;
                                    const double e464_2 = p72_2 - p73_2;
                                    const double e473 = e464_0 * e464_0;
                                    const double e482 = e464_1 * e464_1;
                                    const double e483 = e473 + e482;
                                    const double e492 = e464_2 * e464_2;
                                    const double e493 = e483 + e492;
                                    const bool e494 = e493 < 2.8;
                                    if(e494) {
                                        const int a53 = numneighs[i16];
                                        const bool e409 = a53 >= neighborlist_capacity;
                                        if(e409) {
                                            resize = a53;
                                        } else {
                                            const int e411 = e410 + a53;
                                            neighborlists[e411] = a61;
                                            const int e412 = a53 + 1;
                                            numneighs[i16] = e412;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                const bool e413 = resize > 0;
                if(e413) {
                    fprintf(stdout, "Resize neighborlist_capacity\n");
                    fflush(stdout);
                    const int e414 = resize * 2;
                    neighborlist_capacity = e414;
                    neighborlists = (int *) realloc(neighborlists, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                }
            }
        }
        //pairs::copy_to_device(force)
        for(int i13 = 0; i13 < nlocal; i13++) {
            const int e417 = i13 * 3;
            const double p69_0 = force[e417];
            const int e419 = i13 * 3;
            const int e420 = e419 + 1;
            const double p69_1 = force[e420];
            const int e421 = i13 * 3;
            const int e422 = e421 + 2;
            const double p69_2 = force[e422];
            force[e417] = 0.0;
            force[e420] = 0.0;
            force[e422] = 0.0;
        }
        for(int i14 = 0; i14 < nlocal; i14++) {
            const int e423 = i14 * neighborlist_capacity;
            const int e430 = i14 * 3;
            const double p70_0 = position[e430];
            const int e439 = i14 * 3;
            const int e440 = e439 + 1;
            const double p70_1 = position[e440];
            const int e449 = i14 * 3;
            const int e450 = e449 + 2;
            const double p70_2 = position[e450];
            const int e14 = i14 * 3;
            const int e18 = i14 * 3;
            const int e19 = e18 + 1;
            const int e22 = i14 * 3;
            const int e23 = e22 + 2;
            const int a56 = numneighs[i14];
            for(int i15 = 0; i15 < a56; i15++) {
                const int e424 = e423 + i15;
                const int a57 = neighborlists[e424];
                const int e432 = a57 * 3;
                const double p71_0 = position[e432];
                const int e441 = a57 * 3;
                const int e442 = e441 + 1;
                const double p71_1 = position[e442];
                const int e451 = a57 * 3;
                const int e452 = e451 + 2;
                const double p71_2 = position[e452];
                const double e425_0 = p70_0 - p71_0;
                const double e425_1 = p70_1 - p71_1;
                const double e425_2 = p70_2 - p71_2;
                const double e434 = e425_0 * e425_0;
                const double e443 = e425_1 * e425_1;
                const double e444 = e434 + e443;
                const double e453 = e425_2 * e425_2;
                const double e454 = e444 + e453;
                const bool e455 = e454 < 2.5;
                if(e455) {
                    const double e1 = 1.0 / e454;
                    const double e2 = e1 * e1;
                    const double e3 = e2 * e1;
                    const double p0_0 = force[e14];
                    const double p0_1 = force[e19];
                    const double p0_2 = force[e23];
                    const double e7 = e3 - 0.5;
                    const double e495 = 48.0 * e3;
                    const double e496 = e495 * e7;
                    const double e497 = e496 * e1;
                    const double e10_0 = e425_0 * e497;
                    const double e10_1 = e425_1 * e497;
                    const double e10_2 = e425_2 * e497;
                    const double e11_0 = p0_0 + e10_0;
                    const double e11_1 = p0_1 + e10_1;
                    const double e11_2 = p0_2 + e10_2;
                    force[e14] = e11_0;
                    force[e19] = e11_1;
                    force[e23] = e11_2;
                }
            }
        }
        //pairs::copy_to_device(velocity)
        //pairs::copy_to_device(mass)
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
        const int e73 = i1 + 1;
        const int e456 = nlocal + npbc;
        pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, e73);
        pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e456, e73);
    }
}
