#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
//---
#include "runtime/pairs.hpp"
#include "runtime/read_from_file.hpp"
#include "runtime/vtk.hpp"

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
    int resize = 0;
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
    fprintf(stdout, "CellListsStencilBuild\n");
    fflush(stdout);
    const double e462 = grid0_d0_max - grid0_d0_min;
    const double e463 = e462 / 2.8;
    const int e464 = ceil(e463) + 2;
    dim_cells[0] = e464;
    const double e466 = grid0_d1_max - grid0_d1_min;
    const double e467 = e466 / 2.8;
    const int e468 = ceil(e467) + 2;
    dim_cells[1] = e468;
    const int a54 = dim_cells[0];
    const int a56 = dim_cells[1];
    const int e469 = a54 * a56;
    const double e470 = grid0_d2_max - grid0_d2_min;
    const double e471 = e470 / 2.8;
    const int e472 = ceil(e471) + 2;
    dim_cells[2] = e472;
    const int a58 = dim_cells[2];
    const int e473 = e469 * a58;
    ncells = e473;
    resize = 1;
    while((resize > 0)) {
        resize = 0;
        const bool e475 = ncells >= ncells_capacity;
        if(e475) {
            resize = ncells;
        }
        const bool e476 = resize > 0;
        if(e476) {
            fprintf(stdout, "Resize ncells_capacity\n");
            fflush(stdout);
            const int e477 = resize * 2;
            ncells_capacity = e477;
            cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
            cell_sizes = (int *) realloc(cell_sizes, (sizeof(int) * ncells_capacity));
        }
    }
    nstencil = 0;
    for(int i15 = -1; i15 < 2; i15++) {
        const int a59 = dim_cells[0];
        const int e481 = i15 * a59;
        for(int i16 = -1; i16 < 2; i16++) {
            const int e482 = e481 + i16;
            const int a60 = dim_cells[1];
            const int e483 = e482 * a60;
            for(int i17 = -1; i17 < 2; i17++) {
                const int e484 = e483 + i17;
                stencil[nstencil] = e484;
                const int e485 = nstencil + 1;
                nstencil = e485;
            }
        }
    }
    const int e486 = nlocal + npbc;
    pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, 0);
    pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e486, 0);
    for(int i14 = 0; i14 < 101; i14++) {
        const int e452 = i14 % 20;
        const bool e453 = e452 == 0;
        if(e453) {
            fprintf(stdout, "EnforcePBC\n");
            fflush(stdout);
            const double e115 = grid0_d0_max - grid0_d0_min;
            const double e122 = grid0_d0_max - grid0_d0_min;
            const double e129 = grid0_d1_max - grid0_d1_min;
            const double e136 = grid0_d1_max - grid0_d1_min;
            const double e143 = grid0_d2_max - grid0_d2_min;
            const double e150 = grid0_d2_max - grid0_d2_min;
            for(int i3 = 0; i3 < nlocal; i3++) {
                const int e110 = i3 * 3;
                const double p8_0 = position[e110];
                const bool e112 = p8_0 < grid0_d0_min;
                if(e112) {
                    const int e113 = i3 * 3;
                    const double p9_0 = position[e113];
                    const double e116 = p9_0 + e115;
                    position[e113] = e116;
                }
                const int e117 = i3 * 3;
                const double p10_0 = position[e117];
                const bool e119 = p10_0 > grid0_d0_max;
                if(e119) {
                    const int e120 = i3 * 3;
                    const double p11_0 = position[e120];
                    const double e123 = p11_0 - e122;
                    position[e120] = e123;
                }
                const int e124 = i3 * 3;
                const int e125 = e124 + 1;
                const double p12_1 = position[e125];
                const bool e126 = p12_1 < grid0_d1_min;
                if(e126) {
                    const int e127 = i3 * 3;
                    const int e128 = e127 + 1;
                    const double p13_1 = position[e128];
                    const double e130 = p13_1 + e129;
                    position[e128] = e130;
                }
                const int e131 = i3 * 3;
                const int e132 = e131 + 1;
                const double p14_1 = position[e132];
                const bool e133 = p14_1 > grid0_d1_max;
                if(e133) {
                    const int e134 = i3 * 3;
                    const int e135 = e134 + 1;
                    const double p15_1 = position[e135];
                    const double e137 = p15_1 - e136;
                    position[e135] = e137;
                }
                const int e138 = i3 * 3;
                const int e139 = e138 + 2;
                const double p16_2 = position[e139];
                const bool e140 = p16_2 < grid0_d2_min;
                if(e140) {
                    const int e141 = i3 * 3;
                    const int e142 = e141 + 2;
                    const double p17_2 = position[e142];
                    const double e144 = p17_2 + e143;
                    position[e142] = e144;
                }
                const int e145 = i3 * 3;
                const int e146 = e145 + 2;
                const double p18_2 = position[e146];
                const bool e147 = p18_2 > grid0_d2_max;
                if(e147) {
                    const int e148 = i3 * 3;
                    const int e149 = e148 + 2;
                    const double p19_2 = position[e149];
                    const double e151 = p19_2 - e150;
                    position[e149] = e151;
                }
            }
        }
        const int e454 = i14 % 20;
        const bool e455 = e454 == 0;
        if(e455) {
            fprintf(stdout, "SetupPBC\n");
            fflush(stdout);
            resize = 1;
            while((resize > 0)) {
                resize = 0;
                npbc = 0;
                const int e153 = nlocal + npbc;
                const double e157 = grid0_d0_min + 2.8;
                const double e168 = grid0_d0_max - grid0_d0_min;
                const double e185 = grid0_d0_max - 2.8;
                const double e196 = grid0_d0_max - grid0_d0_min;
                for(int i4 = 0; i4 < e153; i4++) {
                    const int e154 = nlocal + npbc;
                    const int e155 = i4 * 3;
                    const double p20_0 = position[e155];
                    const bool e158 = p20_0 < e157;
                    if(e158) {
                        const bool e159 = npbc >= pbc_capacity;
                        if(e159) {
                            const bool e160 = resize > npbc;
                            resize = (e160) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i4;
                            const int e162 = npbc * 3;
                            pbc_mult[e162] = 1;
                            const int e164 = e154 * 3;
                            const double p21_0 = position[e164];
                            const int e166 = i4 * 3;
                            const double p22_0 = position[e166];
                            const double e169 = p22_0 + e168;
                            position[e164] = e169;
                            const int e170 = npbc * 3;
                            const int e171 = e170 + 1;
                            pbc_mult[e171] = 0;
                            const int e172 = e154 * 3;
                            const int e173 = e172 + 1;
                            const double p23_1 = position[e173];
                            const int e174 = i4 * 3;
                            const int e175 = e174 + 1;
                            const double p24_1 = position[e175];
                            position[e173] = p24_1;
                            const int e176 = npbc * 3;
                            const int e177 = e176 + 2;
                            pbc_mult[e177] = 0;
                            const int e178 = e154 * 3;
                            const int e179 = e178 + 2;
                            const double p25_2 = position[e179];
                            const int e180 = i4 * 3;
                            const int e181 = e180 + 2;
                            const double p26_2 = position[e181];
                            position[e179] = p26_2;
                            const int e182 = npbc + 1;
                            npbc = e182;
                        }
                    }
                    const int e183 = i4 * 3;
                    const double p27_0 = position[e183];
                    const bool e186 = p27_0 > e185;
                    if(e186) {
                        const bool e187 = npbc >= pbc_capacity;
                        if(e187) {
                            const bool e188 = resize > npbc;
                            resize = (e188) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i4;
                            const int e190 = npbc * 3;
                            pbc_mult[e190] = -1;
                            const int e192 = e154 * 3;
                            const double p28_0 = position[e192];
                            const int e194 = i4 * 3;
                            const double p29_0 = position[e194];
                            const double e197 = p29_0 - e196;
                            position[e192] = e197;
                            const int e198 = npbc * 3;
                            const int e199 = e198 + 1;
                            pbc_mult[e199] = 0;
                            const int e200 = e154 * 3;
                            const int e201 = e200 + 1;
                            const double p30_1 = position[e201];
                            const int e202 = i4 * 3;
                            const int e203 = e202 + 1;
                            const double p31_1 = position[e203];
                            position[e201] = p31_1;
                            const int e204 = npbc * 3;
                            const int e205 = e204 + 2;
                            pbc_mult[e205] = 0;
                            const int e206 = e154 * 3;
                            const int e207 = e206 + 2;
                            const double p32_2 = position[e207];
                            const int e208 = i4 * 3;
                            const int e209 = e208 + 2;
                            const double p33_2 = position[e209];
                            position[e207] = p33_2;
                            const int e210 = npbc + 1;
                            npbc = e210;
                        }
                    }
                }
                const int e211 = nlocal + npbc;
                const double e215 = grid0_d1_min + 2.8;
                const double e226 = grid0_d1_max - grid0_d1_min;
                const double e243 = grid0_d1_max - 2.8;
                const double e254 = grid0_d1_max - grid0_d1_min;
                for(int i5 = 0; i5 < e211; i5++) {
                    const int e212 = nlocal + npbc;
                    const int e213 = i5 * 3;
                    const int e214 = e213 + 1;
                    const double p34_1 = position[e214];
                    const bool e216 = p34_1 < e215;
                    if(e216) {
                        const bool e217 = npbc >= pbc_capacity;
                        if(e217) {
                            const bool e218 = resize > npbc;
                            resize = (e218) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i5;
                            const int e220 = npbc * 3;
                            const int e221 = e220 + 1;
                            pbc_mult[e221] = 1;
                            const int e222 = e212 * 3;
                            const int e223 = e222 + 1;
                            const double p35_1 = position[e223];
                            const int e224 = i5 * 3;
                            const int e225 = e224 + 1;
                            const double p36_1 = position[e225];
                            const double e227 = p36_1 + e226;
                            position[e223] = e227;
                            const int e228 = npbc * 3;
                            pbc_mult[e228] = 0;
                            const int e230 = e212 * 3;
                            const double p37_0 = position[e230];
                            const int e232 = i5 * 3;
                            const double p38_0 = position[e232];
                            position[e230] = p38_0;
                            const int e234 = npbc * 3;
                            const int e235 = e234 + 2;
                            pbc_mult[e235] = 0;
                            const int e236 = e212 * 3;
                            const int e237 = e236 + 2;
                            const double p39_2 = position[e237];
                            const int e238 = i5 * 3;
                            const int e239 = e238 + 2;
                            const double p40_2 = position[e239];
                            position[e237] = p40_2;
                            const int e240 = npbc + 1;
                            npbc = e240;
                        }
                    }
                    const int e241 = i5 * 3;
                    const int e242 = e241 + 1;
                    const double p41_1 = position[e242];
                    const bool e244 = p41_1 > e243;
                    if(e244) {
                        const bool e245 = npbc >= pbc_capacity;
                        if(e245) {
                            const bool e246 = resize > npbc;
                            resize = (e246) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i5;
                            const int e248 = npbc * 3;
                            const int e249 = e248 + 1;
                            pbc_mult[e249] = -1;
                            const int e250 = e212 * 3;
                            const int e251 = e250 + 1;
                            const double p42_1 = position[e251];
                            const int e252 = i5 * 3;
                            const int e253 = e252 + 1;
                            const double p43_1 = position[e253];
                            const double e255 = p43_1 - e254;
                            position[e251] = e255;
                            const int e256 = npbc * 3;
                            pbc_mult[e256] = 0;
                            const int e258 = e212 * 3;
                            const double p44_0 = position[e258];
                            const int e260 = i5 * 3;
                            const double p45_0 = position[e260];
                            position[e258] = p45_0;
                            const int e262 = npbc * 3;
                            const int e263 = e262 + 2;
                            pbc_mult[e263] = 0;
                            const int e264 = e212 * 3;
                            const int e265 = e264 + 2;
                            const double p46_2 = position[e265];
                            const int e266 = i5 * 3;
                            const int e267 = e266 + 2;
                            const double p47_2 = position[e267];
                            position[e265] = p47_2;
                            const int e268 = npbc + 1;
                            npbc = e268;
                        }
                    }
                }
                const int e269 = nlocal + npbc;
                const double e273 = grid0_d2_min + 2.8;
                const double e284 = grid0_d2_max - grid0_d2_min;
                const double e301 = grid0_d2_max - 2.8;
                const double e312 = grid0_d2_max - grid0_d2_min;
                for(int i6 = 0; i6 < e269; i6++) {
                    const int e270 = nlocal + npbc;
                    const int e271 = i6 * 3;
                    const int e272 = e271 + 2;
                    const double p48_2 = position[e272];
                    const bool e274 = p48_2 < e273;
                    if(e274) {
                        const bool e275 = npbc >= pbc_capacity;
                        if(e275) {
                            const bool e276 = resize > npbc;
                            resize = (e276) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i6;
                            const int e278 = npbc * 3;
                            const int e279 = e278 + 2;
                            pbc_mult[e279] = 1;
                            const int e280 = e270 * 3;
                            const int e281 = e280 + 2;
                            const double p49_2 = position[e281];
                            const int e282 = i6 * 3;
                            const int e283 = e282 + 2;
                            const double p50_2 = position[e283];
                            const double e285 = p50_2 + e284;
                            position[e281] = e285;
                            const int e286 = npbc * 3;
                            pbc_mult[e286] = 0;
                            const int e288 = e270 * 3;
                            const double p51_0 = position[e288];
                            const int e290 = i6 * 3;
                            const double p52_0 = position[e290];
                            position[e288] = p52_0;
                            const int e292 = npbc * 3;
                            const int e293 = e292 + 1;
                            pbc_mult[e293] = 0;
                            const int e294 = e270 * 3;
                            const int e295 = e294 + 1;
                            const double p53_1 = position[e295];
                            const int e296 = i6 * 3;
                            const int e297 = e296 + 1;
                            const double p54_1 = position[e297];
                            position[e295] = p54_1;
                            const int e298 = npbc + 1;
                            npbc = e298;
                        }
                    }
                    const int e299 = i6 * 3;
                    const int e300 = e299 + 2;
                    const double p55_2 = position[e300];
                    const bool e302 = p55_2 > e301;
                    if(e302) {
                        const bool e303 = npbc >= pbc_capacity;
                        if(e303) {
                            const bool e304 = resize > npbc;
                            resize = (e304) ? ((resize + 1)) : (npbc);
                        } else {
                            pbc_map[npbc] = i6;
                            const int e306 = npbc * 3;
                            const int e307 = e306 + 2;
                            pbc_mult[e307] = -1;
                            const int e308 = e270 * 3;
                            const int e309 = e308 + 2;
                            const double p56_2 = position[e309];
                            const int e310 = i6 * 3;
                            const int e311 = e310 + 2;
                            const double p57_2 = position[e311];
                            const double e313 = p57_2 - e312;
                            position[e309] = e313;
                            const int e314 = npbc * 3;
                            pbc_mult[e314] = 0;
                            const int e316 = e270 * 3;
                            const double p58_0 = position[e316];
                            const int e318 = i6 * 3;
                            const double p59_0 = position[e318];
                            position[e316] = p59_0;
                            const int e320 = npbc * 3;
                            const int e321 = e320 + 1;
                            pbc_mult[e321] = 0;
                            const int e322 = e270 * 3;
                            const int e323 = e322 + 1;
                            const double p60_1 = position[e323];
                            const int e324 = i6 * 3;
                            const int e325 = e324 + 1;
                            const double p61_1 = position[e325];
                            position[e323] = p61_1;
                            const int e326 = npbc + 1;
                            npbc = e326;
                        }
                    }
                }
                const bool e327 = resize > 0;
                if(e327) {
                    fprintf(stdout, "Resize pbc_capacity\n");
                    fflush(stdout);
                    const int e328 = resize * 2;
                    pbc_capacity = e328;
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
            fprintf(stdout, "UpdatePBC\n");
            fflush(stdout);
            const double e348 = grid0_d0_max - grid0_d0_min;
            const double e358 = grid0_d1_max - grid0_d1_min;
            const double e368 = grid0_d2_max - grid0_d2_min;
            for(int i7 = 0; i7 < npbc; i7++) {
                const int e341 = nlocal + i7;
                const int e342 = e341 * 3;
                const double p62_0 = position[e342];
                const int a32 = pbc_map[i7];
                const int e344 = a32 * 3;
                const double p63_0 = position[e344];
                const int e346 = i7 * 3;
                const int a33 = pbc_mult[e346];
                const double e349 = a33 * e348;
                const double e350 = p63_0 + e349;
                position[e342] = e350;
                const int e351 = nlocal + i7;
                const int e352 = e351 * 3;
                const int e353 = e352 + 1;
                const double p64_1 = position[e353];
                const int a34 = pbc_map[i7];
                const int e354 = a34 * 3;
                const int e355 = e354 + 1;
                const double p65_1 = position[e355];
                const int e356 = i7 * 3;
                const int e357 = e356 + 1;
                const int a35 = pbc_mult[e357];
                const double e359 = a35 * e358;
                const double e360 = p65_1 + e359;
                position[e353] = e360;
                const int e361 = nlocal + i7;
                const int e362 = e361 * 3;
                const int e363 = e362 + 2;
                const double p66_2 = position[e363];
                const int a36 = pbc_map[i7];
                const int e364 = a36 * 3;
                const int e365 = e364 + 2;
                const double p67_2 = position[e365];
                const int e366 = i7 * 3;
                const int e367 = e366 + 2;
                const int a37 = pbc_mult[e367];
                const double e369 = a37 * e368;
                const double e370 = p67_2 + e369;
                position[e363] = e370;
            }
        }
        const int e456 = i14 % 20;
        const bool e457 = e456 == 0;
        if(e457) {
            fprintf(stdout, "CellListsBuild\n");
            fflush(stdout);
            resize = 1;
            while((resize > 0)) {
                resize = 0;
                for(int i8 = 0; i8 < ncells; i8++) {
                    cell_sizes[i8] = 0;
                }
                const int e511 = nlocal + npbc;
                for(int i9 = 0; i9 < e511; i9++) {
                    const int e372 = i9 * 3;
                    const double p68_0 = position[e372];
                    const double e374 = p68_0 - grid0_d0_min;
                    const double e375 = e374 / 2.8;
                    const int e376 = i9 * 3;
                    const int e377 = e376 + 1;
                    const double p69_1 = position[e377];
                    const double e378 = p69_1 - grid0_d1_min;
                    const double e379 = e378 / 2.8;
                    const int e380 = i9 * 3;
                    const int e381 = e380 + 2;
                    const double p70_2 = position[e381];
                    const double e382 = p70_2 - grid0_d2_min;
                    const double e383 = e382 / 2.8;
                    const int a39 = dim_cells[1];
                    const int e384 = (int)(e375) * a39;
                    const int e385 = e384 + (int)(e379);
                    const int a40 = dim_cells[2];
                    const int e386 = e385 * a40;
                    const int e387 = e386 + (int)(e383);
                    const bool e388 = e387 >= 0;
                    const bool e389 = e387 <= ncells;
                    const bool e390 = e388 && e389;
                    if(e390) {
                        const int a41 = cell_sizes[e387];
                        const bool e391 = a41 >= cell_capacity;
                        if(e391) {
                            resize = a41;
                        } else {
                            const int e392 = e387 * cell_capacity;
                            const int e393 = e392 + a41;
                            cell_particles[e393] = i9;
                            particle_cell[i9] = e387;
                        }
                        const int e394 = a41 + 1;
                        cell_sizes[e387] = e394;
                    }
                }
                const bool e395 = resize > 0;
                if(e395) {
                    fprintf(stdout, "Resize cell_capacity\n");
                    fflush(stdout);
                    const int e396 = resize * 2;
                    cell_capacity = e396;
                    cell_particles = (int *) realloc(cell_particles, (sizeof(int) * (ncells_capacity * cell_capacity)));
                }
            }
        }
        const int e458 = i14 % 20;
        const bool e459 = e458 == 0;
        if(e459) {
            fprintf(stdout, "NeighborListsBuild\n");
            fflush(stdout);
            resize = 1;
            while((resize > 0)) {
                resize = 0;
                for(int i10 = 0; i10 < nlocal; i10++) {
                    numneighs[i10] = 0;
                    for(int i11 = 0; i11 < nstencil; i11++) {
                        const int a46 = particle_cell[i10];
                        const int a47 = stencil[i11];
                        const int e400 = a46 + a47;
                        const bool e401 = e400 >= 0;
                        const bool e402 = e400 <= ncells;
                        const bool e403 = e401 && e402;
                        if(e403) {
                            const int e404 = e400 * cell_capacity;
                            const int e412 = i10 * 3;
                            const double p71_0 = position[e412];
                            const int e421 = i10 * 3;
                            const int e422 = e421 + 1;
                            const double p71_1 = position[e422];
                            const int e431 = i10 * 3;
                            const int e432 = e431 + 2;
                            const double p71_2 = position[e432];
                            const int e439 = i10 * neighborlist_capacity;
                            const int a48 = cell_sizes[e400];
                            for(int i12 = 0; i12 < a48; i12++) {
                                const int e405 = e404 + i12;
                                const int a49 = cell_particles[e405];
                                const bool e406 = a49 != i10;
                                if(e406) {
                                    const int e414 = a49 * 3;
                                    const double p72_0 = position[e414];
                                    const int e423 = a49 * 3;
                                    const int e424 = e423 + 1;
                                    const double p72_1 = position[e424];
                                    const int e433 = a49 * 3;
                                    const int e434 = e433 + 2;
                                    const double p72_2 = position[e434];
                                    const double e407_0 = p71_0 - p72_0;
                                    const double e407_1 = p71_1 - p72_1;
                                    const double e407_2 = p71_2 - p72_2;
                                    const double e416 = e407_0 * e407_0;
                                    const double e425 = e407_1 * e407_1;
                                    const double e426 = e416 + e425;
                                    const double e435 = e407_2 * e407_2;
                                    const double e436 = e426 + e435;
                                    const bool e437 = e436 < 2.8;
                                    if(e437) {
                                        const int a50 = numneighs[i10];
                                        const bool e438 = a50 >= neighborlist_capacity;
                                        if(e438) {
                                            resize = a50;
                                        } else {
                                            const int e440 = e439 + a50;
                                            neighborlists[e440] = a49;
                                            const int e441 = a50 + 1;
                                            numneighs[i10] = e441;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                const bool e442 = resize > 0;
                if(e442) {
                    fprintf(stdout, "Resize neighborlist_capacity\n");
                    fflush(stdout);
                    const int e443 = resize * 2;
                    neighborlist_capacity = e443;
                    neighborlists = (int *) realloc(neighborlists, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                }
            }
        }
        fprintf(stdout, "PropertiesResetVolatile\n");
        fflush(stdout);
        for(int i13 = 0; i13 < nlocal; i13++) {
            const int e446 = i13 * 3;
            const double p73_0 = force[e446];
            const int e448 = i13 * 3;
            const int e449 = e448 + 1;
            const double p73_1 = force[e449];
            const int e450 = i13 * 3;
            const int e451 = e450 + 2;
            const double p73_2 = force[e451];
            force[e446] = 0.0;
            force[e449] = 0.0;
            force[e451] = 0.0;
        }
        pairs::copy_to_device(position)
        pairs::copy_to_device(force)
        for(int i0 = 0; i0 < nlocal; i0++) {
            const int e1 = i0 * neighborlist_capacity;
            const int e47 = i0 * 3;
            const double p0_0 = position[e47];
            const int e55 = i0 * 3;
            const int e56 = e55 + 1;
            const double p0_1 = position[e56];
            const int e63 = i0 * 3;
            const int e64 = e63 + 2;
            const double p0_2 = position[e64];
            const int e51 = i0 * 3;
            const int e59 = i0 * 3;
            const int e60 = e59 + 1;
            const int e67 = i0 * 3;
            const int e68 = e67 + 2;
            const int a6 = numneighs[i0];
            for(int i1 = 0; i1 < a6; i1++) {
                const int e2 = e1 + i1;
                const int a7 = neighborlists[e2];
                const int e49 = a7 * 3;
                const double p1_0 = position[e49];
                const int e57 = a7 * 3;
                const int e58 = e57 + 1;
                const double p1_1 = position[e58];
                const int e65 = a7 * 3;
                const int e66 = e65 + 2;
                const double p1_2 = position[e66];
                const double e3_0 = p0_0 - p1_0;
                const double e3_1 = p0_1 - p1_1;
                const double e3_2 = p0_2 - p1_2;
                const double e12 = e3_0 * e3_0;
                const double e21 = e3_1 * e3_1;
                const double e22 = e12 + e21;
                const double e31 = e3_2 * e3_2;
                const double e32 = e22 + e31;
                const bool e33 = e32 < 2.5;
                if(e33) {
                    const double e34 = 1.0 / e32;
                    const double e35 = e34 * e34;
                    const double e36 = e35 * e34;
                    const double p2_0 = force[e51];
                    const double p2_1 = force[e60];
                    const double p2_2 = force[e68];
                    const double e40 = e36 - 0.5;
                    const double e507 = 48.0 * e36;
                    const double e508 = e507 * e40;
                    const double e509 = e508 * e34;
                    const double e43_0 = e3_0 * e509;
                    const double e43_1 = e3_1 * e509;
                    const double e43_2 = e3_2 * e509;
                    const double e44_0 = p2_0 + e43_0;
                    const double e44_1 = p2_1 + e43_1;
                    const double e44_2 = p2_2 + e43_2;
                    force[e51] = e44_0;
                    force[e60] = e44_1;
                    force[e68] = e44_2;
                }
            }
        }
        pairs::copy_to_device(velocity)
        pairs::copy_to_device(mass)
        for(int i2 = 0; i2 < nlocal; i2++) {
            const int e76 = i2 * 3;
            const double p3_0 = velocity[e76];
            const int e82 = i2 * 3;
            const int e83 = e82 + 1;
            const double p3_1 = velocity[e83];
            const int e88 = i2 * 3;
            const int e89 = e88 + 2;
            const double p3_2 = velocity[e89];
            const int e74 = i2 * 3;
            const double p4_0 = force[e74];
            const int e80 = i2 * 3;
            const int e81 = e80 + 1;
            const double p4_1 = force[e81];
            const int e86 = i2 * 3;
            const int e87 = e86 + 2;
            const double p4_2 = force[e87];
            const double e69_0 = 0.005 * p4_0;
            const double e69_1 = 0.005 * p4_1;
            const double e69_2 = 0.005 * p4_2;
            const double p5 = mass[i2];
            const double e70_0 = e69_0 / p5;
            const double e70_1 = e69_1 / p5;
            const double e70_2 = e69_2 / p5;
            const double e71_0 = p3_0 + e70_0;
            const double e71_1 = p3_1 + e70_1;
            const double e71_2 = p3_2 + e70_2;
            velocity[e76] = e71_0;
            velocity[e83] = e71_1;
            velocity[e89] = e71_2;
            const int e96 = i2 * 3;
            const double p6_0 = position[e96];
            const int e102 = i2 * 3;
            const int e103 = e102 + 1;
            const double p6_1 = position[e103];
            const int e108 = i2 * 3;
            const int e109 = e108 + 2;
            const double p6_2 = position[e109];
            const int e94 = i2 * 3;
            const double p7_0 = velocity[e94];
            const int e100 = i2 * 3;
            const int e101 = e100 + 1;
            const double p7_1 = velocity[e101];
            const int e106 = i2 * 3;
            const int e107 = e106 + 2;
            const double p7_2 = velocity[e107];
            const double e90_0 = 0.005 * p7_0;
            const double e90_1 = 0.005 * p7_1;
            const double e90_2 = 0.005 * p7_2;
            const double e91_0 = p6_0 + e90_0;
            const double e91_1 = p6_1 + e90_1;
            const double e91_2 = p6_2 + e90_2;
            position[e96] = e91_0;
            position[e103] = e91_1;
            position[e109] = e91_2;
        }
        const int e461 = nlocal + npbc;
        const int e460 = i14 + 1;
        pairs::vtk_write_data(ps, "output/test_local", 0, nlocal, e460);
        pairs::vtk_write_data(ps, "output/test_pbc", nlocal, e461, e460);
    }
}
