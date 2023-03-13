#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
//---
#include "runtime/pairs.hpp"
#include "runtime/read_from_file.hpp"
#include "runtime/vtk.hpp"

using namespace pairs;


void lj(PairsSimulation *pairs, int neighborlist_capacity, int nlocal, int *numneighs, int *neighborlists, double *position, double *force) {
    PAIRS_DEBUG("lj\n");
    for(int i48 = 0; i48 < nlocal; i48++) {
        const int a454 = numneighs[i48];
        const int e979 = i48 * 3;
        const int e988 = i48 * 3;
        const int e989 = e988 + 1;
        const int e998 = i48 * 3;
        const int e999 = e998 + 2;
        const double p148_0 = position[e979];
        const double p148_1 = position[e989];
        const double p148_2 = position[e999];
        const int e972 = i48 * neighborlist_capacity;
        const int e14 = i48 * 3;
        const int e18 = i48 * 3;
        const int e19 = e18 + 1;
        const int e22 = i48 * 3;
        const int e23 = e22 + 2;
        for(int i49 = 0; i49 < a454; i49++) {
            const int e973 = e972 + i49;
            const int a455 = neighborlists[e973];
            const int e981 = a455 * 3;
            const int e990 = a455 * 3;
            const int e991 = e990 + 1;
            const int e1000 = a455 * 3;
            const int e1001 = e1000 + 2;
            const double p149_0 = position[e981];
            const double p149_1 = position[e991];
            const double p149_2 = position[e1001];
            const double e974_0 = p148_0 - p149_0;
            const double e974_1 = p148_1 - p149_1;
            const double e974_2 = p148_2 - p149_2;
            const double e983 = e974_0 * e974_0;
            const double e992 = e974_1 * e974_1;
            const double e993 = e983 + e992;
            const double e1002 = e974_2 * e974_2;
            const double e1003 = e993 + e1002;
            const bool e1004 = e1003 < 2.5;
            if(e1004) {
                const double p0_0 = force[e14];
                const double p0_1 = force[e19];
                const double p0_2 = force[e23];
                const double e1 = 1.0 / e1003;
                const double e2 = e1 * e1;
                const double e3 = e2 * e1;
                const double e1044 = 48.0 * e3;
                const double e7 = e3 - 0.5;
                const double e1045 = e1044 * e7;
                const double e1046 = e1045 * e1;
                const double e9_0 = e974_0 * e1046;
                const double e9_1 = e974_1 * e1046;
                const double e9_2 = e974_2 * e1046;
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
void euler(PairsSimulation *pairs, int nlocal, double *velocity, double *force, double *mass, double *position) {
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
void build_cell_lists_stencil(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int ncells_capacity, int *ncells, int *nstencil, int *dim_cells, int *resizes, int *stencil) {
    PAIRS_DEBUG("build_cell_lists_stencil\n");
    const double e83 = grid0_d0_max - grid0_d0_min;
    const double e84 = e83 / 2.8;
    const int e85 = ceil(e84) + 2;
    dim_cells[0] = e85;
    const double e87 = grid0_d1_max - grid0_d1_min;
    const double e88 = e87 / 2.8;
    const int e89 = ceil(e88) + 2;
    dim_cells[1] = e89;
    const double e91 = grid0_d2_max - grid0_d2_min;
    const double e92 = e91 / 2.8;
    const int e93 = ceil(e92) + 2;
    dim_cells[2] = e93;
    const int a37 = dim_cells[0];
    const int a39 = dim_cells[1];
    const int e90 = a37 * a39;
    const int a41 = dim_cells[2];
    const int e94 = e90 * a41;
    const int e1047 = e94 + 1;
    const bool e1048 = e1047 >= ncells_capacity;
    if(e1048) {
        resizes[0] = e94;
    } else {
        (*ncells) = e94;
    }
    (*nstencil) = 0;
    for(int i2 = -1; i2 < 2; i2++) {
        for(int i3 = -1; i3 < 2; i3++) {
            const int a42 = dim_cells[0];
            const int e95 = i2 * a42;
            const int e96 = e95 + i3;
            const int a43 = dim_cells[1];
            const int e97 = e96 * a43;
            for(int i4 = -1; i4 < 2; i4++) {
                const int e98 = e97 + i4;
                stencil[(*nstencil)] = e98;
                const int e99 = (*nstencil) + 1;
                (*nstencil) = e99;
            }
        }
    }
}
void determine_exchange_particles0(PairsSimulation *pairs, int nlocal, int nghost, int send_capacity, int *nsend_all, int *nsend, int *nrecv, int *exchg_flag, double *subdom, int *pbc, int *send_map, int *send_mult, int *resizes, double *position) {
    PAIRS_DEBUG("determine_exchange_particles0\n");
    nsend[0] = 0;
    nrecv[0] = 0;
    nsend[1] = 0;
    nrecv[1] = 0;
    for(int i5 = 0; i5 < nlocal; i5++) {
        exchg_flag[i5] = 0;
    }
    const int e101 = nlocal + nghost;
    const double a50 = subdom[0];
    const int a52 = pbc[0];
    for(int i6 = 0; i6 < e101; i6++) {
        const int e102 = i6 * 3;
        const double p6_0 = position[e102];
        const bool e105 = p6_0 < a50;
        if(e105) {
            const int atm_add0 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add0] = i6;
            exchg_flag[i6] = 1;
            const int e106 = atm_add0 * 3;
            send_mult[e106] = a52;
            const int e108 = atm_add0 * 3;
            const int e109 = e108 + 1;
            send_mult[e109] = 0;
            const int e110 = atm_add0 * 3;
            const int e111 = e110 + 2;
            send_mult[e111] = 0;
            const int a58 = nsend[0];
            const int e112 = a58 + 1;
            const int e1055 = e112 + 1;
            const bool e1056 = e1055 >= send_capacity;
            if(e1056) {
                resizes[0] = e112;
            } else {
                nsend[0] = e112;
            }
        }
    }
    const int e113 = nlocal + nghost;
    const double a59 = subdom[1];
    const int a61 = pbc[1];
    for(int i7 = 0; i7 < e113; i7++) {
        const int e114 = i7 * 3;
        const double p7_0 = position[e114];
        const bool e117 = p7_0 > a59;
        if(e117) {
            const int atm_add1 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add1] = i7;
            exchg_flag[i7] = 1;
            const int e118 = atm_add1 * 3;
            send_mult[e118] = a61;
            const int e120 = atm_add1 * 3;
            const int e121 = e120 + 1;
            send_mult[e121] = 0;
            const int e122 = atm_add1 * 3;
            const int e123 = e122 + 2;
            send_mult[e123] = 0;
            const int a67 = nsend[1];
            const int e124 = a67 + 1;
            const int e1063 = e124 + 1;
            const bool e1064 = e1063 >= send_capacity;
            if(e1064) {
                resizes[0] = e124;
            } else {
                nsend[1] = e124;
            }
        }
    }
}
void set_communication_offsets0(PairsSimulation *pairs, int *send_offsets, int *recv_offsets, int *nsend, int *nrecv) {
    PAIRS_DEBUG("set_communication_offsets0\n");
    send_offsets[0] = 0;
    recv_offsets[0] = 0;
    const int a70 = nsend[0];
    send_offsets[1] = a70;
    const int a71 = nrecv[0];
    recv_offsets[1] = a71;
}
void pack_ghost_particles0_0_1_2(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, double *send_buffer, int *send_mult, int *send_offsets, int *nsend, double *mass, double *position, double *velocity) {
    PAIRS_DEBUG("pack_ghost_particles0_0_1_2\n");
    const double e138 = grid0_d0_max - grid0_d0_min;
    const double e147 = grid0_d1_max - grid0_d1_min;
    const double e156 = grid0_d2_max - grid0_d2_min;
    for(int i8 = send_offsets[0]; i8 < (send_offsets[0] + (nsend[0] + nsend[1])); i8++) {
        const int e132 = i8 * 7;
        const int a79 = send_map[i8];
        const double p8 = mass[a79];
        send_buffer[e132] = p8;
        const int e141 = i8 * 7;
        const int e142 = e141 + 1;
        const int e134 = a79 * 3;
        const double p9_0 = position[e134];
        const int e136 = i8 * 3;
        const int a81 = send_mult[e136];
        const double e139 = a81 * e138;
        const double e140 = p9_0 + e139;
        send_buffer[e142] = e140;
        const int e150 = i8 * 7;
        const int e151 = e150 + 2;
        const int e143 = a79 * 3;
        const int e144 = e143 + 1;
        const double p10_1 = position[e144];
        const int e145 = i8 * 3;
        const int e146 = e145 + 1;
        const int a83 = send_mult[e146];
        const double e148 = a83 * e147;
        const double e149 = p10_1 + e148;
        send_buffer[e151] = e149;
        const int e159 = i8 * 7;
        const int e160 = e159 + 3;
        const int e152 = a79 * 3;
        const int e153 = e152 + 2;
        const double p11_2 = position[e153];
        const int e154 = i8 * 3;
        const int e155 = e154 + 2;
        const int a85 = send_mult[e155];
        const double e157 = a85 * e156;
        const double e158 = p11_2 + e157;
        send_buffer[e160] = e158;
        const int e163 = i8 * 7;
        const int e164 = e163 + 4;
        const int e161 = a79 * 3;
        const double p12_0 = velocity[e161];
        send_buffer[e164] = p12_0;
        const int e167 = i8 * 7;
        const int e168 = e167 + 5;
        const int e165 = a79 * 3;
        const int e166 = e165 + 1;
        const double p13_1 = velocity[e166];
        send_buffer[e168] = p13_1;
        const int e171 = i8 * 7;
        const int e172 = e171 + 6;
        const int e169 = a79 * 3;
        const int e170 = e169 + 2;
        const double p14_2 = velocity[e170];
        send_buffer[e172] = p14_2;
    }
}
void remove_exchanged_particles_pt1(PairsSimulation *pairs, int nlocal, int nsend_all, int *send_map, int *exchg_flag, int *exchg_copy_to) {
    PAIRS_DEBUG("remove_exchanged_particles_pt1\n");
    int tmp0 = 0;
    const int e173 = nlocal - 1;
    tmp0 = e173;
    const int e174 = nlocal - nsend_all;
    for(int i9 = 0; i9 < nsend_all; i9++) {
        const int a90 = send_map[i9];
        const bool e175 = a90 < e174;
        if(e175) {
            while((exchg_flag[tmp0] == 1)) {
                const int e177 = tmp0 - 1;
                tmp0 = e177;
            }
            exchg_copy_to[i9] = tmp0;
            const int e178 = tmp0 - 1;
            tmp0 = e178;
        } else {
            exchg_copy_to[i9] = -1;
        }
    }
}
void remove_exchanged_particles_pt2(PairsSimulation *pairs, int nsend_all, int *nlocal, int *exchg_copy_to, int *send_map, double *mass, double *position, double *velocity) {
    PAIRS_DEBUG("remove_exchanged_particles_pt2\n");
    for(int i10 = 0; i10 < nsend_all; i10++) {
        const int a94 = exchg_copy_to[i10];
        const bool e179 = a94 > 0;
        if(e179) {
            const int a95 = send_map[i10];
            const double p16 = mass[a94];
            mass[a95] = p16;
            const int e180 = a95 * 3;
            const int e182 = a94 * 3;
            const double p18_0 = position[e182];
            position[e180] = p18_0;
            const int e184 = a95 * 3;
            const int e185 = e184 + 1;
            const int e186 = a94 * 3;
            const int e187 = e186 + 1;
            const double p20_1 = position[e187];
            position[e185] = p20_1;
            const int e188 = a95 * 3;
            const int e189 = e188 + 2;
            const int e190 = a94 * 3;
            const int e191 = e190 + 2;
            const double p22_2 = position[e191];
            position[e189] = p22_2;
            const int e192 = a95 * 3;
            const int e194 = a94 * 3;
            const double p24_0 = velocity[e194];
            velocity[e192] = p24_0;
            const int e196 = a95 * 3;
            const int e197 = e196 + 1;
            const int e198 = a94 * 3;
            const int e199 = e198 + 1;
            const double p26_1 = velocity[e199];
            velocity[e197] = p26_1;
            const int e200 = a95 * 3;
            const int e201 = e200 + 2;
            const int e202 = a94 * 3;
            const int e203 = e202 + 2;
            const double p28_2 = velocity[e203];
            velocity[e201] = p28_2;
        }
    }
    const int e204 = (*nlocal) - nsend_all;
    (*nlocal) = e204;
}
void unpack_ghost_particles0_0_1_2(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *mass, double *position, double *velocity) {
    PAIRS_DEBUG("unpack_ghost_particles0_0_1_2\n");
    for(int i11 = recv_offsets[0]; i11 < (recv_offsets[0] + (nrecv[0] + nrecv[1])); i11++) {
        const int e208 = nlocal + i11;
        const int e209 = i11 * 7;
        const double a99 = recv_buffer[e209];
        mass[e208] = a99;
        const int e211 = nlocal + i11;
        const int e212 = e211 * 3;
        const int e214 = i11 * 7;
        const int e215 = e214 + 1;
        const double a100 = recv_buffer[e215];
        position[e212] = a100;
        const int e216 = nlocal + i11;
        const int e217 = e216 * 3;
        const int e218 = e217 + 1;
        const int e219 = i11 * 7;
        const int e220 = e219 + 2;
        const double a101 = recv_buffer[e220];
        position[e218] = a101;
        const int e221 = nlocal + i11;
        const int e222 = e221 * 3;
        const int e223 = e222 + 2;
        const int e224 = i11 * 7;
        const int e225 = e224 + 3;
        const double a102 = recv_buffer[e225];
        position[e223] = a102;
        const int e226 = nlocal + i11;
        const int e227 = e226 * 3;
        const int e229 = i11 * 7;
        const int e230 = e229 + 4;
        const double a103 = recv_buffer[e230];
        velocity[e227] = a103;
        const int e231 = nlocal + i11;
        const int e232 = e231 * 3;
        const int e233 = e232 + 1;
        const int e234 = i11 * 7;
        const int e235 = e234 + 5;
        const double a104 = recv_buffer[e235];
        velocity[e233] = a104;
        const int e236 = nlocal + i11;
        const int e237 = e236 * 3;
        const int e238 = e237 + 2;
        const int e239 = i11 * 7;
        const int e240 = e239 + 6;
        const double a105 = recv_buffer[e240];
        velocity[e238] = a105;
    }
}
void change_size_after_exchange0(PairsSimulation *pairs, int particle_capacity, int *nlocal, int *nrecv, int *resizes) {
    PAIRS_DEBUG("change_size_after_exchange0\n");
    const int a106 = nrecv[0];
    const int a107 = nrecv[1];
    const int e242 = a106 + a107;
    const int e243 = (*nlocal) + e242;
    const int e1065 = e243 + 1;
    const bool e1066 = e1065 >= particle_capacity;
    if(e1066) {
        resizes[0] = e243;
    } else {
        (*nlocal) = e243;
    }
}
void determine_exchange_particles1(PairsSimulation *pairs, int nlocal, int nghost, int send_capacity, int *nsend_all, int *nsend, int *nrecv, int *exchg_flag, double *subdom, int *pbc, int *send_map, int *send_mult, int *resizes, double *position) {
    PAIRS_DEBUG("determine_exchange_particles1\n");
    nsend[2] = 0;
    nrecv[2] = 0;
    nsend[3] = 0;
    nrecv[3] = 0;
    for(int i12 = 0; i12 < nlocal; i12++) {
        exchg_flag[i12] = 0;
    }
    const int e244 = nlocal + nghost;
    const double a113 = subdom[2];
    const int a115 = pbc[2];
    for(int i13 = 0; i13 < e244; i13++) {
        const int e245 = i13 * 3;
        const int e246 = e245 + 1;
        const double p36_1 = position[e246];
        const bool e248 = p36_1 < a113;
        if(e248) {
            const int atm_add2 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add2] = i13;
            exchg_flag[i13] = 1;
            const int e249 = atm_add2 * 3;
            send_mult[e249] = 0;
            const int e251 = atm_add2 * 3;
            const int e252 = e251 + 1;
            send_mult[e252] = a115;
            const int e253 = atm_add2 * 3;
            const int e254 = e253 + 2;
            send_mult[e254] = 0;
            const int a121 = nsend[2];
            const int e255 = a121 + 1;
            const int e1073 = e255 + 1;
            const bool e1074 = e1073 >= send_capacity;
            if(e1074) {
                resizes[0] = e255;
            } else {
                nsend[2] = e255;
            }
        }
    }
    const int e256 = nlocal + nghost;
    const double a122 = subdom[3];
    const int a124 = pbc[3];
    for(int i14 = 0; i14 < e256; i14++) {
        const int e257 = i14 * 3;
        const int e258 = e257 + 1;
        const double p37_1 = position[e258];
        const bool e260 = p37_1 > a122;
        if(e260) {
            const int atm_add3 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add3] = i14;
            exchg_flag[i14] = 1;
            const int e261 = atm_add3 * 3;
            send_mult[e261] = 0;
            const int e263 = atm_add3 * 3;
            const int e264 = e263 + 1;
            send_mult[e264] = a124;
            const int e265 = atm_add3 * 3;
            const int e266 = e265 + 2;
            send_mult[e266] = 0;
            const int a130 = nsend[3];
            const int e267 = a130 + 1;
            const int e1081 = e267 + 1;
            const bool e1082 = e1081 >= send_capacity;
            if(e1082) {
                resizes[0] = e267;
            } else {
                nsend[3] = e267;
            }
        }
    }
}
void set_communication_offsets1(PairsSimulation *pairs, int *nsend, int *send_offsets, int *nrecv, int *recv_offsets) {
    PAIRS_DEBUG("set_communication_offsets1\n");
    const int a131 = nsend[0];
    const int a133 = nsend[1];
    const int e270 = a131 + a133;
    send_offsets[2] = e270;
    const int a132 = nrecv[0];
    const int a134 = nrecv[1];
    const int e271 = a132 + a134;
    recv_offsets[2] = e271;
    const int a137 = nsend[2];
    const int e272 = e270 + a137;
    send_offsets[3] = e272;
    const int a138 = nrecv[2];
    const int e273 = e271 + a138;
    recv_offsets[3] = e273;
}
void pack_ghost_particles1_0_1_2(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, double *send_buffer, int *send_mult, int *send_offsets, int *nsend, double *mass, double *position, double *velocity) {
    PAIRS_DEBUG("pack_ghost_particles1_0_1_2\n");
    const double e285 = grid0_d0_max - grid0_d0_min;
    const double e294 = grid0_d1_max - grid0_d1_min;
    const double e303 = grid0_d2_max - grid0_d2_min;
    for(int i15 = send_offsets[2]; i15 < (send_offsets[2] + (nsend[2] + nsend[3])); i15++) {
        const int e279 = i15 * 7;
        const int a146 = send_map[i15];
        const double p38 = mass[a146];
        send_buffer[e279] = p38;
        const int e288 = i15 * 7;
        const int e289 = e288 + 1;
        const int e281 = a146 * 3;
        const double p39_0 = position[e281];
        const int e283 = i15 * 3;
        const int a148 = send_mult[e283];
        const double e286 = a148 * e285;
        const double e287 = p39_0 + e286;
        send_buffer[e289] = e287;
        const int e297 = i15 * 7;
        const int e298 = e297 + 2;
        const int e290 = a146 * 3;
        const int e291 = e290 + 1;
        const double p40_1 = position[e291];
        const int e292 = i15 * 3;
        const int e293 = e292 + 1;
        const int a150 = send_mult[e293];
        const double e295 = a150 * e294;
        const double e296 = p40_1 + e295;
        send_buffer[e298] = e296;
        const int e306 = i15 * 7;
        const int e307 = e306 + 3;
        const int e299 = a146 * 3;
        const int e300 = e299 + 2;
        const double p41_2 = position[e300];
        const int e301 = i15 * 3;
        const int e302 = e301 + 2;
        const int a152 = send_mult[e302];
        const double e304 = a152 * e303;
        const double e305 = p41_2 + e304;
        send_buffer[e307] = e305;
        const int e310 = i15 * 7;
        const int e311 = e310 + 4;
        const int e308 = a146 * 3;
        const double p42_0 = velocity[e308];
        send_buffer[e311] = p42_0;
        const int e314 = i15 * 7;
        const int e315 = e314 + 5;
        const int e312 = a146 * 3;
        const int e313 = e312 + 1;
        const double p43_1 = velocity[e313];
        send_buffer[e315] = p43_1;
        const int e318 = i15 * 7;
        const int e319 = e318 + 6;
        const int e316 = a146 * 3;
        const int e317 = e316 + 2;
        const double p44_2 = velocity[e317];
        send_buffer[e319] = p44_2;
    }
}
void unpack_ghost_particles1_0_1_2(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *mass, double *position, double *velocity) {
    PAIRS_DEBUG("unpack_ghost_particles1_0_1_2\n");
    for(int i18 = recv_offsets[2]; i18 < (recv_offsets[2] + (nrecv[2] + nrecv[3])); i18++) {
        const int e355 = nlocal + i18;
        const int e356 = i18 * 7;
        const double a166 = recv_buffer[e356];
        mass[e355] = a166;
        const int e358 = nlocal + i18;
        const int e359 = e358 * 3;
        const int e361 = i18 * 7;
        const int e362 = e361 + 1;
        const double a167 = recv_buffer[e362];
        position[e359] = a167;
        const int e363 = nlocal + i18;
        const int e364 = e363 * 3;
        const int e365 = e364 + 1;
        const int e366 = i18 * 7;
        const int e367 = e366 + 2;
        const double a168 = recv_buffer[e367];
        position[e365] = a168;
        const int e368 = nlocal + i18;
        const int e369 = e368 * 3;
        const int e370 = e369 + 2;
        const int e371 = i18 * 7;
        const int e372 = e371 + 3;
        const double a169 = recv_buffer[e372];
        position[e370] = a169;
        const int e373 = nlocal + i18;
        const int e374 = e373 * 3;
        const int e376 = i18 * 7;
        const int e377 = e376 + 4;
        const double a170 = recv_buffer[e377];
        velocity[e374] = a170;
        const int e378 = nlocal + i18;
        const int e379 = e378 * 3;
        const int e380 = e379 + 1;
        const int e381 = i18 * 7;
        const int e382 = e381 + 5;
        const double a171 = recv_buffer[e382];
        velocity[e380] = a171;
        const int e383 = nlocal + i18;
        const int e384 = e383 * 3;
        const int e385 = e384 + 2;
        const int e386 = i18 * 7;
        const int e387 = e386 + 6;
        const double a172 = recv_buffer[e387];
        velocity[e385] = a172;
    }
}
void change_size_after_exchange1(PairsSimulation *pairs, int particle_capacity, int *nlocal, int *nrecv, int *resizes) {
    PAIRS_DEBUG("change_size_after_exchange1\n");
    const int a173 = nrecv[2];
    const int a174 = nrecv[3];
    const int e389 = a173 + a174;
    const int e390 = (*nlocal) + e389;
    const int e1083 = e390 + 1;
    const bool e1084 = e1083 >= particle_capacity;
    if(e1084) {
        resizes[0] = e390;
    } else {
        (*nlocal) = e390;
    }
}
void determine_exchange_particles2(PairsSimulation *pairs, int nlocal, int nghost, int send_capacity, int *nsend_all, int *nsend, int *nrecv, int *exchg_flag, double *subdom, int *pbc, int *send_map, int *send_mult, int *resizes, double *position) {
    PAIRS_DEBUG("determine_exchange_particles2\n");
    nsend[4] = 0;
    nrecv[4] = 0;
    nsend[5] = 0;
    nrecv[5] = 0;
    for(int i19 = 0; i19 < nlocal; i19++) {
        exchg_flag[i19] = 0;
    }
    const int e391 = nlocal + nghost;
    const double a180 = subdom[4];
    const int a182 = pbc[4];
    for(int i20 = 0; i20 < e391; i20++) {
        const int e392 = i20 * 3;
        const int e393 = e392 + 2;
        const double p66_2 = position[e393];
        const bool e395 = p66_2 < a180;
        if(e395) {
            const int atm_add4 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add4] = i20;
            exchg_flag[i20] = 1;
            const int e396 = atm_add4 * 3;
            send_mult[e396] = 0;
            const int e398 = atm_add4 * 3;
            const int e399 = e398 + 1;
            send_mult[e399] = 0;
            const int e400 = atm_add4 * 3;
            const int e401 = e400 + 2;
            send_mult[e401] = a182;
            const int a188 = nsend[4];
            const int e402 = a188 + 1;
            const int e1091 = e402 + 1;
            const bool e1092 = e1091 >= send_capacity;
            if(e1092) {
                resizes[0] = e402;
            } else {
                nsend[4] = e402;
            }
        }
    }
    const int e403 = nlocal + nghost;
    const double a189 = subdom[5];
    const int a191 = pbc[5];
    for(int i21 = 0; i21 < e403; i21++) {
        const int e404 = i21 * 3;
        const int e405 = e404 + 2;
        const double p67_2 = position[e405];
        const bool e407 = p67_2 > a189;
        if(e407) {
            const int atm_add5 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add5] = i21;
            exchg_flag[i21] = 1;
            const int e408 = atm_add5 * 3;
            send_mult[e408] = 0;
            const int e410 = atm_add5 * 3;
            const int e411 = e410 + 1;
            send_mult[e411] = 0;
            const int e412 = atm_add5 * 3;
            const int e413 = e412 + 2;
            send_mult[e413] = a191;
            const int a197 = nsend[5];
            const int e414 = a197 + 1;
            const int e1099 = e414 + 1;
            const bool e1100 = e1099 >= send_capacity;
            if(e1100) {
                resizes[0] = e414;
            } else {
                nsend[5] = e414;
            }
        }
    }
}
void set_communication_offsets2(PairsSimulation *pairs, int *nsend, int *send_offsets, int *nrecv, int *recv_offsets) {
    PAIRS_DEBUG("set_communication_offsets2\n");
    const int a198 = nsend[0];
    const int a200 = nsend[1];
    const int e417 = a198 + a200;
    const int a202 = nsend[2];
    const int e419 = e417 + a202;
    const int a204 = nsend[3];
    const int e421 = e419 + a204;
    send_offsets[4] = e421;
    const int a199 = nrecv[0];
    const int a201 = nrecv[1];
    const int e418 = a199 + a201;
    const int a203 = nrecv[2];
    const int e420 = e418 + a203;
    const int a205 = nrecv[3];
    const int e422 = e420 + a205;
    recv_offsets[4] = e422;
    const int a208 = nsend[4];
    const int e423 = e421 + a208;
    send_offsets[5] = e423;
    const int a209 = nrecv[4];
    const int e424 = e422 + a209;
    recv_offsets[5] = e424;
}
void pack_ghost_particles2_0_1_2(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, double *send_buffer, int *send_mult, int *send_offsets, int *nsend, double *mass, double *position, double *velocity) {
    PAIRS_DEBUG("pack_ghost_particles2_0_1_2\n");
    const double e436 = grid0_d0_max - grid0_d0_min;
    const double e445 = grid0_d1_max - grid0_d1_min;
    const double e454 = grid0_d2_max - grid0_d2_min;
    for(int i22 = send_offsets[4]; i22 < (send_offsets[4] + (nsend[4] + nsend[5])); i22++) {
        const int e430 = i22 * 7;
        const int a217 = send_map[i22];
        const double p68 = mass[a217];
        send_buffer[e430] = p68;
        const int e439 = i22 * 7;
        const int e440 = e439 + 1;
        const int e432 = a217 * 3;
        const double p69_0 = position[e432];
        const int e434 = i22 * 3;
        const int a219 = send_mult[e434];
        const double e437 = a219 * e436;
        const double e438 = p69_0 + e437;
        send_buffer[e440] = e438;
        const int e448 = i22 * 7;
        const int e449 = e448 + 2;
        const int e441 = a217 * 3;
        const int e442 = e441 + 1;
        const double p70_1 = position[e442];
        const int e443 = i22 * 3;
        const int e444 = e443 + 1;
        const int a221 = send_mult[e444];
        const double e446 = a221 * e445;
        const double e447 = p70_1 + e446;
        send_buffer[e449] = e447;
        const int e457 = i22 * 7;
        const int e458 = e457 + 3;
        const int e450 = a217 * 3;
        const int e451 = e450 + 2;
        const double p71_2 = position[e451];
        const int e452 = i22 * 3;
        const int e453 = e452 + 2;
        const int a223 = send_mult[e453];
        const double e455 = a223 * e454;
        const double e456 = p71_2 + e455;
        send_buffer[e458] = e456;
        const int e461 = i22 * 7;
        const int e462 = e461 + 4;
        const int e459 = a217 * 3;
        const double p72_0 = velocity[e459];
        send_buffer[e462] = p72_0;
        const int e465 = i22 * 7;
        const int e466 = e465 + 5;
        const int e463 = a217 * 3;
        const int e464 = e463 + 1;
        const double p73_1 = velocity[e464];
        send_buffer[e466] = p73_1;
        const int e469 = i22 * 7;
        const int e470 = e469 + 6;
        const int e467 = a217 * 3;
        const int e468 = e467 + 2;
        const double p74_2 = velocity[e468];
        send_buffer[e470] = p74_2;
    }
}
void unpack_ghost_particles2_0_1_2(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *mass, double *position, double *velocity) {
    PAIRS_DEBUG("unpack_ghost_particles2_0_1_2\n");
    for(int i25 = recv_offsets[4]; i25 < (recv_offsets[4] + (nrecv[4] + nrecv[5])); i25++) {
        const int e506 = nlocal + i25;
        const int e507 = i25 * 7;
        const double a237 = recv_buffer[e507];
        mass[e506] = a237;
        const int e509 = nlocal + i25;
        const int e510 = e509 * 3;
        const int e512 = i25 * 7;
        const int e513 = e512 + 1;
        const double a238 = recv_buffer[e513];
        position[e510] = a238;
        const int e514 = nlocal + i25;
        const int e515 = e514 * 3;
        const int e516 = e515 + 1;
        const int e517 = i25 * 7;
        const int e518 = e517 + 2;
        const double a239 = recv_buffer[e518];
        position[e516] = a239;
        const int e519 = nlocal + i25;
        const int e520 = e519 * 3;
        const int e521 = e520 + 2;
        const int e522 = i25 * 7;
        const int e523 = e522 + 3;
        const double a240 = recv_buffer[e523];
        position[e521] = a240;
        const int e524 = nlocal + i25;
        const int e525 = e524 * 3;
        const int e527 = i25 * 7;
        const int e528 = e527 + 4;
        const double a241 = recv_buffer[e528];
        velocity[e525] = a241;
        const int e529 = nlocal + i25;
        const int e530 = e529 * 3;
        const int e531 = e530 + 1;
        const int e532 = i25 * 7;
        const int e533 = e532 + 5;
        const double a242 = recv_buffer[e533];
        velocity[e531] = a242;
        const int e534 = nlocal + i25;
        const int e535 = e534 * 3;
        const int e536 = e535 + 2;
        const int e537 = i25 * 7;
        const int e538 = e537 + 6;
        const double a243 = recv_buffer[e538];
        velocity[e536] = a243;
    }
}
void change_size_after_exchange2(PairsSimulation *pairs, int particle_capacity, int *nlocal, int *nrecv, int *resizes) {
    PAIRS_DEBUG("change_size_after_exchange2\n");
    const int a244 = nrecv[4];
    const int a245 = nrecv[5];
    const int e540 = a244 + a245;
    const int e541 = (*nlocal) + e540;
    const int e1101 = e541 + 1;
    const bool e1102 = e1101 >= particle_capacity;
    if(e1102) {
        resizes[0] = e541;
    } else {
        (*nlocal) = e541;
    }
}
void determine_ghost_particles0(PairsSimulation *pairs, int nlocal, int nghost, int send_capacity, int *nsend_all, int *nsend, int *nrecv, double *subdom, int *pbc, int *send_map, int *send_mult, int *resizes, double *position) {
    PAIRS_DEBUG("determine_ghost_particles0\n");
    nsend[0] = 0;
    nrecv[0] = 0;
    nsend[1] = 0;
    nrecv[1] = 0;
    const int e542 = nlocal + nghost;
    const double a250 = subdom[0];
    const double e545 = a250 + 2.8;
    const int a252 = pbc[0];
    for(int i26 = 0; i26 < e542; i26++) {
        const int e543 = i26 * 3;
        const double p96_0 = position[e543];
        const bool e546 = p96_0 < e545;
        if(e546) {
            const int atm_add6 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add6] = i26;
            const int e547 = atm_add6 * 3;
            send_mult[e547] = a252;
            const int e549 = atm_add6 * 3;
            const int e550 = e549 + 1;
            send_mult[e550] = 0;
            const int e551 = atm_add6 * 3;
            const int e552 = e551 + 2;
            send_mult[e552] = 0;
            const int a257 = nsend[0];
            const int e553 = a257 + 1;
            const int e1109 = e553 + 1;
            const bool e1110 = e1109 >= send_capacity;
            if(e1110) {
                resizes[0] = e553;
            } else {
                nsend[0] = e553;
            }
        }
    }
    const int e554 = nlocal + nghost;
    const double a258 = subdom[1];
    const double e557 = a258 - 2.8;
    const int a260 = pbc[1];
    for(int i27 = 0; i27 < e554; i27++) {
        const int e555 = i27 * 3;
        const double p97_0 = position[e555];
        const bool e558 = p97_0 > e557;
        if(e558) {
            const int atm_add7 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add7] = i27;
            const int e559 = atm_add7 * 3;
            send_mult[e559] = a260;
            const int e561 = atm_add7 * 3;
            const int e562 = e561 + 1;
            send_mult[e562] = 0;
            const int e563 = atm_add7 * 3;
            const int e564 = e563 + 2;
            send_mult[e564] = 0;
            const int a265 = nsend[1];
            const int e565 = a265 + 1;
            const int e1117 = e565 + 1;
            const bool e1118 = e1117 >= send_capacity;
            if(e1118) {
                resizes[0] = e565;
            } else {
                nsend[1] = e565;
            }
        }
    }
}
void pack_ghost_particles0_0_1(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, double *send_buffer, int *send_mult, int *send_offsets, int *nsend, double *mass, double *position) {
    PAIRS_DEBUG("pack_ghost_particles0_0_1\n");
    const double e579 = grid0_d0_max - grid0_d0_min;
    const double e588 = grid0_d1_max - grid0_d1_min;
    const double e597 = grid0_d2_max - grid0_d2_min;
    for(int i28 = send_offsets[0]; i28 < (send_offsets[0] + (nsend[0] + nsend[1])); i28++) {
        const int e573 = i28 * 4;
        const int a277 = send_map[i28];
        const double p98 = mass[a277];
        send_buffer[e573] = p98;
        const int e582 = i28 * 4;
        const int e583 = e582 + 1;
        const int e575 = a277 * 3;
        const double p99_0 = position[e575];
        const int e577 = i28 * 3;
        const int a279 = send_mult[e577];
        const double e580 = a279 * e579;
        const double e581 = p99_0 + e580;
        send_buffer[e583] = e581;
        const int e591 = i28 * 4;
        const int e592 = e591 + 2;
        const int e584 = a277 * 3;
        const int e585 = e584 + 1;
        const double p100_1 = position[e585];
        const int e586 = i28 * 3;
        const int e587 = e586 + 1;
        const int a281 = send_mult[e587];
        const double e589 = a281 * e588;
        const double e590 = p100_1 + e589;
        send_buffer[e592] = e590;
        const int e600 = i28 * 4;
        const int e601 = e600 + 3;
        const int e593 = a277 * 3;
        const int e594 = e593 + 2;
        const double p101_2 = position[e594];
        const int e595 = i28 * 3;
        const int e596 = e595 + 2;
        const int a283 = send_mult[e596];
        const double e598 = a283 * e597;
        const double e599 = p101_2 + e598;
        send_buffer[e601] = e599;
    }
}
void unpack_ghost_particles0_0_1(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *mass, double *position) {
    PAIRS_DEBUG("unpack_ghost_particles0_0_1\n");
    for(int i29 = recv_offsets[0]; i29 < (recv_offsets[0] + (nrecv[0] + nrecv[1])); i29++) {
        const int e605 = nlocal + i29;
        const int e606 = i29 * 4;
        const double a288 = recv_buffer[e606];
        mass[e605] = a288;
        const int e608 = nlocal + i29;
        const int e609 = e608 * 3;
        const int e611 = i29 * 4;
        const int e612 = e611 + 1;
        const double a289 = recv_buffer[e612];
        position[e609] = a289;
        const int e613 = nlocal + i29;
        const int e614 = e613 * 3;
        const int e615 = e614 + 1;
        const int e616 = i29 * 4;
        const int e617 = e616 + 2;
        const double a290 = recv_buffer[e617];
        position[e615] = a290;
        const int e618 = nlocal + i29;
        const int e619 = e618 * 3;
        const int e620 = e619 + 2;
        const int e621 = i29 * 4;
        const int e622 = e621 + 3;
        const double a291 = recv_buffer[e622];
        position[e620] = a291;
    }
}
void determine_ghost_particles1(PairsSimulation *pairs, int nlocal, int nghost, int send_capacity, int *nsend_all, int *nsend, int *nrecv, double *subdom, int *pbc, int *send_map, int *send_mult, int *resizes, double *position) {
    PAIRS_DEBUG("determine_ghost_particles1\n");
    nsend[2] = 0;
    nrecv[2] = 0;
    nsend[3] = 0;
    nrecv[3] = 0;
    const int e623 = nlocal + nghost;
    const double a296 = subdom[2];
    const double e626 = a296 + 2.8;
    const int a298 = pbc[2];
    for(int i30 = 0; i30 < e623; i30++) {
        const int e624 = i30 * 3;
        const int e625 = e624 + 1;
        const double p106_1 = position[e625];
        const bool e627 = p106_1 < e626;
        if(e627) {
            const int atm_add8 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add8] = i30;
            const int e628 = atm_add8 * 3;
            send_mult[e628] = 0;
            const int e630 = atm_add8 * 3;
            const int e631 = e630 + 1;
            send_mult[e631] = a298;
            const int e632 = atm_add8 * 3;
            const int e633 = e632 + 2;
            send_mult[e633] = 0;
            const int a303 = nsend[2];
            const int e634 = a303 + 1;
            const int e1125 = e634 + 1;
            const bool e1126 = e1125 >= send_capacity;
            if(e1126) {
                resizes[0] = e634;
            } else {
                nsend[2] = e634;
            }
        }
    }
    const int e635 = nlocal + nghost;
    const double a304 = subdom[3];
    const double e638 = a304 - 2.8;
    const int a306 = pbc[3];
    for(int i31 = 0; i31 < e635; i31++) {
        const int e636 = i31 * 3;
        const int e637 = e636 + 1;
        const double p107_1 = position[e637];
        const bool e639 = p107_1 > e638;
        if(e639) {
            const int atm_add9 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add9] = i31;
            const int e640 = atm_add9 * 3;
            send_mult[e640] = 0;
            const int e642 = atm_add9 * 3;
            const int e643 = e642 + 1;
            send_mult[e643] = a306;
            const int e644 = atm_add9 * 3;
            const int e645 = e644 + 2;
            send_mult[e645] = 0;
            const int a311 = nsend[3];
            const int e646 = a311 + 1;
            const int e1133 = e646 + 1;
            const bool e1134 = e1133 >= send_capacity;
            if(e1134) {
                resizes[0] = e646;
            } else {
                nsend[3] = e646;
            }
        }
    }
}
void pack_ghost_particles1_0_1(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, double *send_buffer, int *send_mult, int *send_offsets, int *nsend, double *mass, double *position) {
    PAIRS_DEBUG("pack_ghost_particles1_0_1\n");
    const double e664 = grid0_d0_max - grid0_d0_min;
    const double e673 = grid0_d1_max - grid0_d1_min;
    const double e682 = grid0_d2_max - grid0_d2_min;
    for(int i32 = send_offsets[2]; i32 < (send_offsets[2] + (nsend[2] + nsend[3])); i32++) {
        const int e658 = i32 * 4;
        const int a327 = send_map[i32];
        const double p108 = mass[a327];
        send_buffer[e658] = p108;
        const int e667 = i32 * 4;
        const int e668 = e667 + 1;
        const int e660 = a327 * 3;
        const double p109_0 = position[e660];
        const int e662 = i32 * 3;
        const int a329 = send_mult[e662];
        const double e665 = a329 * e664;
        const double e666 = p109_0 + e665;
        send_buffer[e668] = e666;
        const int e676 = i32 * 4;
        const int e677 = e676 + 2;
        const int e669 = a327 * 3;
        const int e670 = e669 + 1;
        const double p110_1 = position[e670];
        const int e671 = i32 * 3;
        const int e672 = e671 + 1;
        const int a331 = send_mult[e672];
        const double e674 = a331 * e673;
        const double e675 = p110_1 + e674;
        send_buffer[e677] = e675;
        const int e685 = i32 * 4;
        const int e686 = e685 + 3;
        const int e678 = a327 * 3;
        const int e679 = e678 + 2;
        const double p111_2 = position[e679];
        const int e680 = i32 * 3;
        const int e681 = e680 + 2;
        const int a333 = send_mult[e681];
        const double e683 = a333 * e682;
        const double e684 = p111_2 + e683;
        send_buffer[e686] = e684;
    }
}
void unpack_ghost_particles1_0_1(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *mass, double *position) {
    PAIRS_DEBUG("unpack_ghost_particles1_0_1\n");
    for(int i33 = recv_offsets[2]; i33 < (recv_offsets[2] + (nrecv[2] + nrecv[3])); i33++) {
        const int e690 = nlocal + i33;
        const int e691 = i33 * 4;
        const double a338 = recv_buffer[e691];
        mass[e690] = a338;
        const int e693 = nlocal + i33;
        const int e694 = e693 * 3;
        const int e696 = i33 * 4;
        const int e697 = e696 + 1;
        const double a339 = recv_buffer[e697];
        position[e694] = a339;
        const int e698 = nlocal + i33;
        const int e699 = e698 * 3;
        const int e700 = e699 + 1;
        const int e701 = i33 * 4;
        const int e702 = e701 + 2;
        const double a340 = recv_buffer[e702];
        position[e700] = a340;
        const int e703 = nlocal + i33;
        const int e704 = e703 * 3;
        const int e705 = e704 + 2;
        const int e706 = i33 * 4;
        const int e707 = e706 + 3;
        const double a341 = recv_buffer[e707];
        position[e705] = a341;
    }
}
void determine_ghost_particles2(PairsSimulation *pairs, int nlocal, int nghost, int send_capacity, int *nsend_all, int *nsend, int *nrecv, double *subdom, int *pbc, int *send_map, int *send_mult, int *resizes, double *position) {
    PAIRS_DEBUG("determine_ghost_particles2\n");
    nsend[4] = 0;
    nrecv[4] = 0;
    nsend[5] = 0;
    nrecv[5] = 0;
    const int e708 = nlocal + nghost;
    const double a346 = subdom[4];
    const double e711 = a346 + 2.8;
    const int a348 = pbc[4];
    for(int i34 = 0; i34 < e708; i34++) {
        const int e709 = i34 * 3;
        const int e710 = e709 + 2;
        const double p116_2 = position[e710];
        const bool e712 = p116_2 < e711;
        if(e712) {
            const int atm_add10 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add10] = i34;
            const int e713 = atm_add10 * 3;
            send_mult[e713] = 0;
            const int e715 = atm_add10 * 3;
            const int e716 = e715 + 1;
            send_mult[e716] = 0;
            const int e717 = atm_add10 * 3;
            const int e718 = e717 + 2;
            send_mult[e718] = a348;
            const int a353 = nsend[4];
            const int e719 = a353 + 1;
            const int e1141 = e719 + 1;
            const bool e1142 = e1141 >= send_capacity;
            if(e1142) {
                resizes[0] = e719;
            } else {
                nsend[4] = e719;
            }
        }
    }
    const int e720 = nlocal + nghost;
    const double a354 = subdom[5];
    const double e723 = a354 - 2.8;
    const int a356 = pbc[5];
    for(int i35 = 0; i35 < e720; i35++) {
        const int e721 = i35 * 3;
        const int e722 = e721 + 2;
        const double p117_2 = position[e722];
        const bool e724 = p117_2 > e723;
        if(e724) {
            const int atm_add11 = pairs::host_atomic_add(&((*nsend_all)), 1);
            send_map[atm_add11] = i35;
            const int e725 = atm_add11 * 3;
            send_mult[e725] = 0;
            const int e727 = atm_add11 * 3;
            const int e728 = e727 + 1;
            send_mult[e728] = 0;
            const int e729 = atm_add11 * 3;
            const int e730 = e729 + 2;
            send_mult[e730] = a356;
            const int a361 = nsend[5];
            const int e731 = a361 + 1;
            const int e1149 = e731 + 1;
            const bool e1150 = e1149 >= send_capacity;
            if(e1150) {
                resizes[0] = e731;
            } else {
                nsend[5] = e731;
            }
        }
    }
}
void pack_ghost_particles2_0_1(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, double *send_buffer, int *send_mult, int *send_offsets, int *nsend, double *mass, double *position) {
    PAIRS_DEBUG("pack_ghost_particles2_0_1\n");
    const double e753 = grid0_d0_max - grid0_d0_min;
    const double e762 = grid0_d1_max - grid0_d1_min;
    const double e771 = grid0_d2_max - grid0_d2_min;
    for(int i36 = send_offsets[4]; i36 < (send_offsets[4] + (nsend[4] + nsend[5])); i36++) {
        const int e747 = i36 * 4;
        const int a381 = send_map[i36];
        const double p118 = mass[a381];
        send_buffer[e747] = p118;
        const int e756 = i36 * 4;
        const int e757 = e756 + 1;
        const int e749 = a381 * 3;
        const double p119_0 = position[e749];
        const int e751 = i36 * 3;
        const int a383 = send_mult[e751];
        const double e754 = a383 * e753;
        const double e755 = p119_0 + e754;
        send_buffer[e757] = e755;
        const int e765 = i36 * 4;
        const int e766 = e765 + 2;
        const int e758 = a381 * 3;
        const int e759 = e758 + 1;
        const double p120_1 = position[e759];
        const int e760 = i36 * 3;
        const int e761 = e760 + 1;
        const int a385 = send_mult[e761];
        const double e763 = a385 * e762;
        const double e764 = p120_1 + e763;
        send_buffer[e766] = e764;
        const int e774 = i36 * 4;
        const int e775 = e774 + 3;
        const int e767 = a381 * 3;
        const int e768 = e767 + 2;
        const double p121_2 = position[e768];
        const int e769 = i36 * 3;
        const int e770 = e769 + 2;
        const int a387 = send_mult[e770];
        const double e772 = a387 * e771;
        const double e773 = p121_2 + e772;
        send_buffer[e775] = e773;
    }
}
void unpack_ghost_particles2_0_1(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *mass, double *position) {
    PAIRS_DEBUG("unpack_ghost_particles2_0_1\n");
    for(int i37 = recv_offsets[4]; i37 < (recv_offsets[4] + (nrecv[4] + nrecv[5])); i37++) {
        const int e779 = nlocal + i37;
        const int e780 = i37 * 4;
        const double a392 = recv_buffer[e780];
        mass[e779] = a392;
        const int e782 = nlocal + i37;
        const int e783 = e782 * 3;
        const int e785 = i37 * 4;
        const int e786 = e785 + 1;
        const double a393 = recv_buffer[e786];
        position[e783] = a393;
        const int e787 = nlocal + i37;
        const int e788 = e787 * 3;
        const int e789 = e788 + 1;
        const int e790 = i37 * 4;
        const int e791 = e790 + 2;
        const double a394 = recv_buffer[e791];
        position[e789] = a394;
        const int e792 = nlocal + i37;
        const int e793 = e792 * 3;
        const int e794 = e793 + 2;
        const int e795 = i37 * 4;
        const int e796 = e795 + 3;
        const double a395 = recv_buffer[e796];
        position[e794] = a395;
    }
}
void pack_ghost_particles0_1(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, int *send_mult, double *send_buffer, int *send_offsets, int *nsend, double *position) {
    PAIRS_DEBUG("pack_ghost_particles0_1\n");
    const double e804 = grid0_d0_max - grid0_d0_min;
    const double e813 = grid0_d1_max - grid0_d1_min;
    const double e822 = grid0_d2_max - grid0_d2_min;
    for(int i38 = send_offsets[0]; i38 < (send_offsets[0] + (nsend[0] + nsend[1])); i38++) {
        const int e807 = i38 * 3;
        const int a399 = send_map[i38];
        const int e800 = a399 * 3;
        const double p126_0 = position[e800];
        const int e802 = i38 * 3;
        const int a400 = send_mult[e802];
        const double e805 = a400 * e804;
        const double e806 = p126_0 + e805;
        send_buffer[e807] = e806;
        const int e816 = i38 * 3;
        const int e817 = e816 + 1;
        const int e809 = a399 * 3;
        const int e810 = e809 + 1;
        const double p127_1 = position[e810];
        const int e811 = i38 * 3;
        const int e812 = e811 + 1;
        const int a402 = send_mult[e812];
        const double e814 = a402 * e813;
        const double e815 = p127_1 + e814;
        send_buffer[e817] = e815;
        const int e825 = i38 * 3;
        const int e826 = e825 + 2;
        const int e818 = a399 * 3;
        const int e819 = e818 + 2;
        const double p128_2 = position[e819];
        const int e820 = i38 * 3;
        const int e821 = e820 + 2;
        const int a404 = send_mult[e821];
        const double e823 = a404 * e822;
        const double e824 = p128_2 + e823;
        send_buffer[e826] = e824;
    }
}
void unpack_ghost_particles0_1(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *position) {
    PAIRS_DEBUG("unpack_ghost_particles0_1\n");
    for(int i39 = recv_offsets[0]; i39 < (recv_offsets[0] + (nrecv[0] + nrecv[1])); i39++) {
        const int e830 = nlocal + i39;
        const int e831 = e830 * 3;
        const int e833 = i39 * 3;
        const double a409 = recv_buffer[e833];
        position[e831] = a409;
        const int e835 = nlocal + i39;
        const int e836 = e835 * 3;
        const int e837 = e836 + 1;
        const int e838 = i39 * 3;
        const int e839 = e838 + 1;
        const double a410 = recv_buffer[e839];
        position[e837] = a410;
        const int e840 = nlocal + i39;
        const int e841 = e840 * 3;
        const int e842 = e841 + 2;
        const int e843 = i39 * 3;
        const int e844 = e843 + 2;
        const double a411 = recv_buffer[e844];
        position[e842] = a411;
    }
}
void pack_ghost_particles1_1(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, int *send_mult, double *send_buffer, int *send_offsets, int *nsend, double *position) {
    PAIRS_DEBUG("pack_ghost_particles1_1\n");
    const double e852 = grid0_d0_max - grid0_d0_min;
    const double e861 = grid0_d1_max - grid0_d1_min;
    const double e870 = grid0_d2_max - grid0_d2_min;
    for(int i40 = send_offsets[2]; i40 < (send_offsets[2] + (nsend[2] + nsend[3])); i40++) {
        const int e855 = i40 * 3;
        const int a415 = send_map[i40];
        const int e848 = a415 * 3;
        const double p132_0 = position[e848];
        const int e850 = i40 * 3;
        const int a416 = send_mult[e850];
        const double e853 = a416 * e852;
        const double e854 = p132_0 + e853;
        send_buffer[e855] = e854;
        const int e864 = i40 * 3;
        const int e865 = e864 + 1;
        const int e857 = a415 * 3;
        const int e858 = e857 + 1;
        const double p133_1 = position[e858];
        const int e859 = i40 * 3;
        const int e860 = e859 + 1;
        const int a418 = send_mult[e860];
        const double e862 = a418 * e861;
        const double e863 = p133_1 + e862;
        send_buffer[e865] = e863;
        const int e873 = i40 * 3;
        const int e874 = e873 + 2;
        const int e866 = a415 * 3;
        const int e867 = e866 + 2;
        const double p134_2 = position[e867];
        const int e868 = i40 * 3;
        const int e869 = e868 + 2;
        const int a420 = send_mult[e869];
        const double e871 = a420 * e870;
        const double e872 = p134_2 + e871;
        send_buffer[e874] = e872;
    }
}
void unpack_ghost_particles1_1(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *position) {
    PAIRS_DEBUG("unpack_ghost_particles1_1\n");
    for(int i41 = recv_offsets[2]; i41 < (recv_offsets[2] + (nrecv[2] + nrecv[3])); i41++) {
        const int e878 = nlocal + i41;
        const int e879 = e878 * 3;
        const int e881 = i41 * 3;
        const double a425 = recv_buffer[e881];
        position[e879] = a425;
        const int e883 = nlocal + i41;
        const int e884 = e883 * 3;
        const int e885 = e884 + 1;
        const int e886 = i41 * 3;
        const int e887 = e886 + 1;
        const double a426 = recv_buffer[e887];
        position[e885] = a426;
        const int e888 = nlocal + i41;
        const int e889 = e888 * 3;
        const int e890 = e889 + 2;
        const int e891 = i41 * 3;
        const int e892 = e891 + 2;
        const double a427 = recv_buffer[e892];
        position[e890] = a427;
    }
}
void pack_ghost_particles2_1(PairsSimulation *pairs, double grid0_d0_max, double grid0_d0_min, double grid0_d1_max, double grid0_d1_min, double grid0_d2_max, double grid0_d2_min, int *send_map, int *send_mult, double *send_buffer, int *send_offsets, int *nsend, double *position) {
    PAIRS_DEBUG("pack_ghost_particles2_1\n");
    const double e900 = grid0_d0_max - grid0_d0_min;
    const double e909 = grid0_d1_max - grid0_d1_min;
    const double e918 = grid0_d2_max - grid0_d2_min;
    for(int i42 = send_offsets[4]; i42 < (send_offsets[4] + (nsend[4] + nsend[5])); i42++) {
        const int e903 = i42 * 3;
        const int a431 = send_map[i42];
        const int e896 = a431 * 3;
        const double p138_0 = position[e896];
        const int e898 = i42 * 3;
        const int a432 = send_mult[e898];
        const double e901 = a432 * e900;
        const double e902 = p138_0 + e901;
        send_buffer[e903] = e902;
        const int e912 = i42 * 3;
        const int e913 = e912 + 1;
        const int e905 = a431 * 3;
        const int e906 = e905 + 1;
        const double p139_1 = position[e906];
        const int e907 = i42 * 3;
        const int e908 = e907 + 1;
        const int a434 = send_mult[e908];
        const double e910 = a434 * e909;
        const double e911 = p139_1 + e910;
        send_buffer[e913] = e911;
        const int e921 = i42 * 3;
        const int e922 = e921 + 2;
        const int e914 = a431 * 3;
        const int e915 = e914 + 2;
        const double p140_2 = position[e915];
        const int e916 = i42 * 3;
        const int e917 = e916 + 2;
        const int a436 = send_mult[e917];
        const double e919 = a436 * e918;
        const double e920 = p140_2 + e919;
        send_buffer[e922] = e920;
    }
}
void unpack_ghost_particles2_1(PairsSimulation *pairs, int nlocal, double *recv_buffer, int *recv_offsets, int *nrecv, double *position) {
    PAIRS_DEBUG("unpack_ghost_particles2_1\n");
    for(int i43 = recv_offsets[4]; i43 < (recv_offsets[4] + (nrecv[4] + nrecv[5])); i43++) {
        const int e926 = nlocal + i43;
        const int e927 = e926 * 3;
        const int e929 = i43 * 3;
        const double a441 = recv_buffer[e929];
        position[e927] = a441;
        const int e931 = nlocal + i43;
        const int e932 = e931 * 3;
        const int e933 = e932 + 1;
        const int e934 = i43 * 3;
        const int e935 = e934 + 1;
        const double a442 = recv_buffer[e935];
        position[e933] = a442;
        const int e936 = nlocal + i43;
        const int e937 = e936 * 3;
        const int e938 = e937 + 2;
        const int e939 = i43 * 3;
        const int e940 = e939 + 2;
        const double a443 = recv_buffer[e940];
        position[e938] = a443;
    }
}
void build_cell_lists(PairsSimulation *pairs, int ncells, int nlocal, int nghost, double grid0_d0_min, double grid0_d1_min, double grid0_d2_min, int cell_capacity, int *cell_sizes, int *dim_cells, int *particle_cell, int *resizes, int *cell_particles, double *position) {
    PAIRS_DEBUG("build_cell_lists\n");
    for(int i44 = 0; i44 < ncells; i44++) {
        cell_sizes[i44] = 0;
    }
    const int e941 = nlocal + nghost;
    const int a445 = dim_cells[1];
    const int a446 = dim_cells[2];
    for(int i45 = 0; i45 < e941; i45++) {
        const int e942 = i45 * 3;
        const double p144_0 = position[e942];
        const double e944 = p144_0 - grid0_d0_min;
        const double e945 = e944 / 2.8;
        const int e954 = (int)(e945) * a445;
        const int e946 = i45 * 3;
        const int e947 = e946 + 1;
        const double p145_1 = position[e947];
        const double e948 = p145_1 - grid0_d1_min;
        const double e949 = e948 / 2.8;
        const int e955 = e954 + (int)(e949);
        const int e956 = e955 * a446;
        const int e950 = i45 * 3;
        const int e951 = e950 + 2;
        const double p146_2 = position[e951];
        const double e952 = p146_2 - grid0_d2_min;
        const double e953 = e952 / 2.8;
        const int e957 = e956 + (int)(e953);
        const bool e958 = e957 >= 0;
        const bool e959 = e957 <= ncells;
        const bool e960 = e958 && e959;
        if(e960) {
            particle_cell[i45] = e957;
            const int atm_add12 = pairs::atomic_add_resize_check(&(cell_sizes[e957]), 1, &(resizes[0]), cell_capacity);
            const int e961 = e957 * cell_capacity;
            const int e962 = e961 + atm_add12;
            cell_particles[e962] = i45;
        }
    }
}
void neighbor_lists_build(PairsSimulation *pairs, int nlocal, int ncells, int cell_capacity, int neighborlist_capacity, int nstencil, int *numneighs, int *particle_cell, int *stencil, int *cell_sizes, int *cell_particles, int *neighborlists, int *resizes, double *position) {
    PAIRS_DEBUG("neighbor_lists_build\n");
    for(int i46 = 0; i46 < nlocal; i46++) {
        numneighs[i46] = 0;
    }
    for(int i50 = 0; i50 < nlocal; i50++) {
        const int a456 = particle_cell[i50];
        for(int i51 = 0; i51 < nstencil; i51++) {
            const int a457 = stencil[i51];
            const int e1006 = a456 + a457;
            const bool e1007 = e1006 >= 0;
            const bool e1008 = e1006 <= ncells;
            const bool e1009 = e1007 && e1008;
            if(e1009) {
                const int a458 = cell_sizes[e1006];
                const int e1010 = e1006 * cell_capacity;
                const int e1018 = i50 * 3;
                const int e1027 = i50 * 3;
                const int e1028 = e1027 + 1;
                const int e1037 = i50 * 3;
                const int e1038 = e1037 + 2;
                const double p150_0 = position[e1018];
                const double p150_1 = position[e1028];
                const double p150_2 = position[e1038];
                const int e963 = i50 * neighborlist_capacity;
                for(int i52 = 0; i52 < a458; i52++) {
                    const int e1011 = e1010 + i52;
                    const int a459 = cell_particles[e1011];
                    const bool e1012 = a459 != i50;
                    if(e1012) {
                        const int e1020 = a459 * 3;
                        const int e1029 = a459 * 3;
                        const int e1030 = e1029 + 1;
                        const int e1039 = a459 * 3;
                        const int e1040 = e1039 + 2;
                        const double p151_0 = position[e1020];
                        const double p151_1 = position[e1030];
                        const double p151_2 = position[e1040];
                        const double e1013_0 = p150_0 - p151_0;
                        const double e1013_1 = p150_1 - p151_1;
                        const double e1013_2 = p150_2 - p151_2;
                        const double e1022 = e1013_0 * e1013_0;
                        const double e1031 = e1013_1 * e1013_1;
                        const double e1032 = e1022 + e1031;
                        const double e1041 = e1013_2 * e1013_2;
                        const double e1042 = e1032 + e1041;
                        const bool e1043 = e1042 < 2.8;
                        if(e1043) {
                            const int a451 = numneighs[i50];
                            const int e964 = e963 + a451;
                            neighborlists[e964] = a459;
                            const int e965 = a451 + 1;
                            const int e1153 = e965 + 1;
                            const bool e1154 = e1153 >= neighborlist_capacity;
                            if(e1154) {
                                resizes[0] = e965;
                            } else {
                                numneighs[i50] = e965;
                            }
                        }
                    }
                }
            }
        }
    }
}
void reset_volatile_properties(PairsSimulation *pairs, int nlocal, double *force) {
    PAIRS_DEBUG("reset_volatile_properties\n");
    for(int i47 = 0; i47 < nlocal; i47++) {
        const int e966 = i47 * 3;
        const int e968 = i47 * 3;
        const int e969 = e968 + 1;
        const int e970 = i47 * 3;
        const int e971 = e970 + 2;
        force[e966] = 0.0;
        force[e969] = 0.0;
        force[e971] = 0.0;
    }
}
int main(int argc, char **argv) {
    PairsSimulation *pairs = new PairsSimulation(4, 24, DimRanges);
    int particle_capacity = 1000000;
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
    int nsend_all = 0;
    int send_capacity = 100000;
    int recv_capacity = 100000;
    int elem_capacity = 10;
    int neigh_capacity = 6;
    int *resizes;
    pairs->addArray(0, "resizes", &resizes, nullptr, (sizeof(int) * 3));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    int neighbor_ranks[6];
    pairs->addStaticArray(1, "neighbor_ranks", neighbor_ranks, nullptr, (sizeof(int) * 6));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    int pbc[6];
    pairs->addStaticArray(2, "pbc", pbc, nullptr, (sizeof(int) * 6));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    double subdom[6];
    pairs->addStaticArray(3, "subdom", subdom, nullptr, (sizeof(double) * 6));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    double grid_buffer[6];
    pairs->addStaticArray(4, "grid_buffer", grid_buffer, nullptr, (sizeof(double) * 6));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    int dim_cells[3];
    pairs->addStaticArray(5, "dim_cells", dim_cells, nullptr, (sizeof(int) * 3));
    pairs->clearArrayHostFlag(0); // resizes
    pairs->clearArrayDeviceFlag(0); // resizes
    int *cell_particles;
    pairs->addArray(6, "cell_particles", &cell_particles, nullptr, (sizeof(int) * (ncells_capacity * cell_capacity)));
    int *cell_sizes;
    pairs->addArray(7, "cell_sizes", &cell_sizes, nullptr, (sizeof(int) * ncells_capacity));
    int *stencil;
    pairs->addArray(8, "stencil", &stencil, nullptr, (sizeof(int) * 27));
    int *particle_cell;
    pairs->addArray(9, "particle_cell", &particle_cell, nullptr, (sizeof(int) * particle_capacity));
    int *neighborlists;
    pairs->addArray(10, "neighborlists", &neighborlists, nullptr, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
    int *numneighs;
    pairs->addArray(11, "numneighs", &numneighs, nullptr, (sizeof(int) * particle_capacity));
    int *nsend;
    pairs->addArray(12, "nsend", &nsend, nullptr, (sizeof(int) * neigh_capacity));
    int *send_offsets;
    pairs->addArray(13, "send_offsets", &send_offsets, nullptr, (sizeof(int) * neigh_capacity));
    double *send_buffer;
    pairs->addArray(14, "send_buffer", &send_buffer, nullptr, (sizeof(double) * (send_capacity * elem_capacity)));
    int *send_map;
    pairs->addArray(15, "send_map", &send_map, nullptr, (sizeof(int) * send_capacity));
    int *exchg_flag;
    pairs->addArray(16, "exchg_flag", &exchg_flag, nullptr, (sizeof(int) * particle_capacity));
    int *exchg_copy_to;
    pairs->addArray(17, "exchg_copy_to", &exchg_copy_to, nullptr, (sizeof(int) * send_capacity));
    int *send_mult;
    pairs->addArray(18, "send_mult", &send_mult, nullptr, (sizeof(int) * (send_capacity * 3)));
    int *nrecv;
    pairs->addArray(19, "nrecv", &nrecv, nullptr, (sizeof(int) * neigh_capacity));
    int *recv_offsets;
    pairs->addArray(20, "recv_offsets", &recv_offsets, nullptr, (sizeof(int) * neigh_capacity));
    double *recv_buffer;
    pairs->addArray(21, "recv_buffer", &recv_buffer, nullptr, (sizeof(double) * (recv_capacity * elem_capacity)));
    int *recv_map;
    pairs->addArray(22, "recv_map", &recv_map, nullptr, (sizeof(int) * recv_capacity));
    int *recv_mult;
    pairs->addArray(23, "recv_mult", &recv_mult, nullptr, (sizeof(int) * (recv_capacity * 3)));
    double *mass;
    pairs->addProperty(0, "mass", &mass, nullptr, Prop_Float, AoS, (0 + particle_capacity));
    double *position;
    pairs->addProperty(1, "position", &position, nullptr, Prop_Vector, AoS, (0 + particle_capacity), 3);
    double *velocity;
    pairs->addProperty(2, "velocity", &velocity, nullptr, Prop_Vector, AoS, (0 + particle_capacity), 3);
    double *force;
    pairs->addProperty(3, "force", &force, nullptr, Prop_Vector, AoS, (0 + particle_capacity), 3);
    pairs::read_grid_data(pairs, "data/minimd_setup_32x32x32.input", grid_buffer);
    const double a30 = grid_buffer[0];
    grid0_d0_min = a30;
    const double a31 = grid_buffer[1];
    grid0_d0_max = a31;
    const double a32 = grid_buffer[2];
    grid0_d1_min = a32;
    const double a33 = grid_buffer[3];
    grid0_d1_max = a33;
    const double a34 = grid_buffer[4];
    grid0_d2_min = a34;
    const double a35 = grid_buffer[5];
    grid0_d2_max = a35;
    pairs->initDomain(&argc, &argv, grid0_d0_min, grid0_d0_max, grid0_d1_min, grid0_d1_max, grid0_d2_min, grid0_d2_max);
    pairs->fillCommunicationArrays(neighbor_ranks, pbc, subdom);
    const int prop_list_0[] = {0, 1, 2};
    nlocal = pairs::read_particle_data(pairs, "data/minimd_setup_32x32x32.input", prop_list_0, 3);
    resizes[0] = 1;
    while((resizes[0] > 0)) {
        resizes[0] = 0;
        build_cell_lists_stencil(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, ncells_capacity, &ncells, &nstencil, dim_cells, resizes, stencil);
        const int a482 = resizes[0];
        const bool e1156 = a482 > 0;
        if(e1156) {
            PAIRS_DEBUG("resizes[0] -> ncells_capacity\n");
            const int a483 = resizes[0];
            const int e1157 = a483 * 2;
            ncells_capacity = e1157;
            pairs->reallocArray(6, &cell_particles, nullptr, (sizeof(int) * (ncells_capacity * cell_capacity)));
            pairs->reallocArray(7, &cell_sizes, nullptr, (sizeof(int) * ncells_capacity));
        }
    }
    pairs::vtk_write_data(pairs, "output/test_cpu_local", 0, nlocal, 0);
    const int e100 = nlocal + nghost;
    pairs::vtk_write_data(pairs, "output/test_cpu_ghost", nlocal, e100, 0);
    for(int i1 = 0; i1 < 101; i1++) {
        if(((i1 % 20) == 0)) {
            nsend_all = 0;
            nghost = 0;
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                determine_exchange_particles0(pairs, nlocal, nghost, send_capacity, &nsend_all, nsend, nrecv, exchg_flag, subdom, pbc, send_map, send_mult, resizes, position);
                const int a487 = resizes[0];
                const bool e1162 = a487 > 0;
                if(e1162) {
                    PAIRS_DEBUG("resizes[0] -> send_capacity\n");
                    const int a488 = resizes[0];
                    const int e1163 = a488 * 2;
                    send_capacity = e1163;
                    pairs->reallocArray(14, &send_buffer, nullptr, (sizeof(double) * (send_capacity * elem_capacity)));
                    pairs->reallocArray(15, &send_map, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(17, &exchg_copy_to, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(18, &send_mult, nullptr, (sizeof(int) * (send_capacity * 3)));
                }
            }
            pairs->communicateSizes(0, nsend, nrecv);
            set_communication_offsets0(pairs, send_offsets, recv_offsets, nsend, nrecv);
            pack_ghost_particles0_0_1_2(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_buffer, send_mult, send_offsets, nsend, mass, position, velocity);
            remove_exchanged_particles_pt1(pairs, nlocal, nsend_all, send_map, exchg_flag, exchg_copy_to);
            remove_exchanged_particles_pt2(pairs, nsend_all, &nlocal, exchg_copy_to, send_map, mass, position, velocity);
            pairs->communicateData(0, 7, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles0_0_1_2(pairs, nlocal, recv_buffer, recv_offsets, nrecv, mass, position, velocity);
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                change_size_after_exchange0(pairs, particle_capacity, &nlocal, nrecv, resizes);
                const int a492 = resizes[0];
                const bool e1175 = a492 > 0;
                if(e1175) {
                    PAIRS_DEBUG("resizes[0] -> particle_capacity\n");
                    const int a493 = resizes[0];
                    const int e1176 = a493 * 2;
                    particle_capacity = e1176;
                    pairs->reallocArray(9, &particle_cell, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocArray(10, &neighborlists, nullptr, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                    pairs->reallocArray(11, &numneighs, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocArray(16, &exchg_flag, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocProperty(0, &mass, nullptr, (0 + particle_capacity));
                    pairs->reallocProperty(1, &position, nullptr, (0 + particle_capacity), 3);
                    pairs->reallocProperty(2, &velocity, nullptr, (0 + particle_capacity), 3);
                    pairs->reallocProperty(3, &force, nullptr, (0 + particle_capacity), 3);
                }
            }
            nsend_all = 0;
            nghost = 0;
            nsend[0] = 0;
            nrecv[0] = 0;
            send_offsets[0] = 0;
            recv_offsets[0] = 0;
            nsend[1] = 0;
            nrecv[1] = 0;
            send_offsets[1] = 0;
            recv_offsets[1] = 0;
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                determine_exchange_particles1(pairs, nlocal, nghost, send_capacity, &nsend_all, nsend, nrecv, exchg_flag, subdom, pbc, send_map, send_mult, resizes, position);
                const int a497 = resizes[0];
                const bool e1183 = a497 > 0;
                if(e1183) {
                    PAIRS_DEBUG("resizes[0] -> send_capacity\n");
                    const int a498 = resizes[0];
                    const int e1184 = a498 * 2;
                    send_capacity = e1184;
                    pairs->reallocArray(14, &send_buffer, nullptr, (sizeof(double) * (send_capacity * elem_capacity)));
                    pairs->reallocArray(15, &send_map, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(17, &exchg_copy_to, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(18, &send_mult, nullptr, (sizeof(int) * (send_capacity * 3)));
                }
            }
            pairs->communicateSizes(1, nsend, nrecv);
            set_communication_offsets1(pairs, nsend, send_offsets, nrecv, recv_offsets);
            pack_ghost_particles1_0_1_2(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_buffer, send_mult, send_offsets, nsend, mass, position, velocity);
            remove_exchanged_particles_pt1(pairs, nlocal, nsend_all, send_map, exchg_flag, exchg_copy_to);
            remove_exchanged_particles_pt2(pairs, nsend_all, &nlocal, exchg_copy_to, send_map, mass, position, velocity);
            pairs->communicateData(1, 7, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles1_0_1_2(pairs, nlocal, recv_buffer, recv_offsets, nrecv, mass, position, velocity);
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                change_size_after_exchange1(pairs, particle_capacity, &nlocal, nrecv, resizes);
                const int a502 = resizes[0];
                const bool e1196 = a502 > 0;
                if(e1196) {
                    PAIRS_DEBUG("resizes[0] -> particle_capacity\n");
                    const int a503 = resizes[0];
                    const int e1197 = a503 * 2;
                    particle_capacity = e1197;
                    pairs->reallocArray(9, &particle_cell, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocArray(10, &neighborlists, nullptr, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                    pairs->reallocArray(11, &numneighs, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocArray(16, &exchg_flag, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocProperty(0, &mass, nullptr, (0 + particle_capacity));
                    pairs->reallocProperty(1, &position, nullptr, (0 + particle_capacity), 3);
                    pairs->reallocProperty(2, &velocity, nullptr, (0 + particle_capacity), 3);
                    pairs->reallocProperty(3, &force, nullptr, (0 + particle_capacity), 3);
                }
            }
            nsend_all = 0;
            nghost = 0;
            nsend[0] = 0;
            nrecv[0] = 0;
            send_offsets[0] = 0;
            recv_offsets[0] = 0;
            nsend[1] = 0;
            nrecv[1] = 0;
            send_offsets[1] = 0;
            recv_offsets[1] = 0;
            nsend[2] = 0;
            nrecv[2] = 0;
            send_offsets[2] = 0;
            recv_offsets[2] = 0;
            nsend[3] = 0;
            nrecv[3] = 0;
            send_offsets[3] = 0;
            recv_offsets[3] = 0;
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                determine_exchange_particles2(pairs, nlocal, nghost, send_capacity, &nsend_all, nsend, nrecv, exchg_flag, subdom, pbc, send_map, send_mult, resizes, position);
                const int a507 = resizes[0];
                const bool e1204 = a507 > 0;
                if(e1204) {
                    PAIRS_DEBUG("resizes[0] -> send_capacity\n");
                    const int a508 = resizes[0];
                    const int e1205 = a508 * 2;
                    send_capacity = e1205;
                    pairs->reallocArray(14, &send_buffer, nullptr, (sizeof(double) * (send_capacity * elem_capacity)));
                    pairs->reallocArray(15, &send_map, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(17, &exchg_copy_to, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(18, &send_mult, nullptr, (sizeof(int) * (send_capacity * 3)));
                }
            }
            pairs->communicateSizes(2, nsend, nrecv);
            set_communication_offsets2(pairs, nsend, send_offsets, nrecv, recv_offsets);
            pack_ghost_particles2_0_1_2(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_buffer, send_mult, send_offsets, nsend, mass, position, velocity);
            remove_exchanged_particles_pt1(pairs, nlocal, nsend_all, send_map, exchg_flag, exchg_copy_to);
            remove_exchanged_particles_pt2(pairs, nsend_all, &nlocal, exchg_copy_to, send_map, mass, position, velocity);
            pairs->communicateData(2, 7, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles2_0_1_2(pairs, nlocal, recv_buffer, recv_offsets, nrecv, mass, position, velocity);
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                change_size_after_exchange2(pairs, particle_capacity, &nlocal, nrecv, resizes);
                const int a512 = resizes[0];
                const bool e1217 = a512 > 0;
                if(e1217) {
                    PAIRS_DEBUG("resizes[0] -> particle_capacity\n");
                    const int a513 = resizes[0];
                    const int e1218 = a513 * 2;
                    particle_capacity = e1218;
                    pairs->reallocArray(9, &particle_cell, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocArray(10, &neighborlists, nullptr, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                    pairs->reallocArray(11, &numneighs, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocArray(16, &exchg_flag, nullptr, (sizeof(int) * particle_capacity));
                    pairs->reallocProperty(0, &mass, nullptr, (0 + particle_capacity));
                    pairs->reallocProperty(1, &position, nullptr, (0 + particle_capacity), 3);
                    pairs->reallocProperty(2, &velocity, nullptr, (0 + particle_capacity), 3);
                    pairs->reallocProperty(3, &force, nullptr, (0 + particle_capacity), 3);
                }
            }
        }
        if(((i1 % 20) == 0)) {
            nsend_all = 0;
            nghost = 0;
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                determine_ghost_particles0(pairs, nlocal, nghost, send_capacity, &nsend_all, nsend, nrecv, subdom, pbc, send_map, send_mult, resizes, position);
                const int a517 = resizes[0];
                const bool e1225 = a517 > 0;
                if(e1225) {
                    PAIRS_DEBUG("resizes[0] -> send_capacity\n");
                    const int a518 = resizes[0];
                    const int e1226 = a518 * 2;
                    send_capacity = e1226;
                    pairs->reallocArray(14, &send_buffer, nullptr, (sizeof(double) * (send_capacity * elem_capacity)));
                    pairs->reallocArray(15, &send_map, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(17, &exchg_copy_to, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(18, &send_mult, nullptr, (sizeof(int) * (send_capacity * 3)));
                }
            }
            pairs->communicateSizes(0, nsend, nrecv);
            set_communication_offsets0(pairs, send_offsets, recv_offsets, nsend, nrecv);
            pack_ghost_particles0_0_1(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_buffer, send_mult, send_offsets, nsend, mass, position);
            pairs->communicateData(0, 4, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles0_0_1(pairs, nlocal, recv_buffer, recv_offsets, nrecv, mass, position);
            const int a24 = nrecv[0];
            const int a25 = nrecv[1];
            const int e66 = a24 + a25;
            const int e67 = nghost + e66;
            nghost = e67;
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                determine_ghost_particles1(pairs, nlocal, nghost, send_capacity, &nsend_all, nsend, nrecv, subdom, pbc, send_map, send_mult, resizes, position);
                const int a522 = resizes[0];
                const bool e1234 = a522 > 0;
                if(e1234) {
                    PAIRS_DEBUG("resizes[0] -> send_capacity\n");
                    const int a523 = resizes[0];
                    const int e1235 = a523 * 2;
                    send_capacity = e1235;
                    pairs->reallocArray(14, &send_buffer, nullptr, (sizeof(double) * (send_capacity * elem_capacity)));
                    pairs->reallocArray(15, &send_map, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(17, &exchg_copy_to, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(18, &send_mult, nullptr, (sizeof(int) * (send_capacity * 3)));
                }
            }
            pairs->communicateSizes(1, nsend, nrecv);
            set_communication_offsets1(pairs, nsend, send_offsets, nrecv, recv_offsets);
            pack_ghost_particles1_0_1(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_buffer, send_mult, send_offsets, nsend, mass, position);
            pairs->communicateData(1, 4, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles1_0_1(pairs, nlocal, recv_buffer, recv_offsets, nrecv, mass, position);
            const int a26 = nrecv[2];
            const int a27 = nrecv[3];
            const int e69 = a26 + a27;
            const int e70 = nghost + e69;
            nghost = e70;
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                determine_ghost_particles2(pairs, nlocal, nghost, send_capacity, &nsend_all, nsend, nrecv, subdom, pbc, send_map, send_mult, resizes, position);
                const int a527 = resizes[0];
                const bool e1243 = a527 > 0;
                if(e1243) {
                    PAIRS_DEBUG("resizes[0] -> send_capacity\n");
                    const int a528 = resizes[0];
                    const int e1244 = a528 * 2;
                    send_capacity = e1244;
                    pairs->reallocArray(14, &send_buffer, nullptr, (sizeof(double) * (send_capacity * elem_capacity)));
                    pairs->reallocArray(15, &send_map, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(17, &exchg_copy_to, nullptr, (sizeof(int) * send_capacity));
                    pairs->reallocArray(18, &send_mult, nullptr, (sizeof(int) * (send_capacity * 3)));
                }
            }
            pairs->communicateSizes(2, nsend, nrecv);
            set_communication_offsets2(pairs, nsend, send_offsets, nrecv, recv_offsets);
            pack_ghost_particles2_0_1(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_buffer, send_mult, send_offsets, nsend, mass, position);
            pairs->communicateData(2, 4, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles2_0_1(pairs, nlocal, recv_buffer, recv_offsets, nrecv, mass, position);
            const int a28 = nrecv[4];
            const int a29 = nrecv[5];
            const int e72 = a28 + a29;
            const int e73 = nghost + e72;
            nghost = e73;
        } else {
            pack_ghost_particles0_1(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_mult, send_buffer, send_offsets, nsend, position);
            pairs->communicateData(0, 3, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles0_1(pairs, nlocal, recv_buffer, recv_offsets, nrecv, position);
            pack_ghost_particles1_1(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_mult, send_buffer, send_offsets, nsend, position);
            pairs->communicateData(1, 3, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles1_1(pairs, nlocal, recv_buffer, recv_offsets, nrecv, position);
            pack_ghost_particles2_1(pairs, grid0_d0_max, grid0_d0_min, grid0_d1_max, grid0_d1_min, grid0_d2_max, grid0_d2_min, send_map, send_mult, send_buffer, send_offsets, nsend, position);
            pairs->communicateData(2, 3, send_buffer, send_offsets, nsend, recv_buffer, recv_offsets, nrecv);
            unpack_ghost_particles2_1(pairs, nlocal, recv_buffer, recv_offsets, nrecv, position);
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                build_cell_lists(pairs, ncells, nlocal, nghost, grid0_d0_min, grid0_d1_min, grid0_d2_min, cell_capacity, cell_sizes, dim_cells, particle_cell, resizes, cell_particles, position);
                const int a532 = resizes[0];
                const bool e1252 = a532 > 0;
                if(e1252) {
                    PAIRS_DEBUG("resizes[0] -> cell_capacity\n");
                    const int a533 = resizes[0];
                    const int e1253 = a533 * 2;
                    cell_capacity = e1253;
                    pairs->reallocArray(6, &cell_particles, nullptr, (sizeof(int) * (ncells_capacity * cell_capacity)));
                }
            }
        }
        if(((i1 % 20) == 0)) {
            resizes[0] = 1;
            while((resizes[0] > 0)) {
                resizes[0] = 0;
                neighbor_lists_build(pairs, nlocal, ncells, cell_capacity, neighborlist_capacity, nstencil, numneighs, particle_cell, stencil, cell_sizes, cell_particles, neighborlists, resizes, position);
                const int a537 = resizes[0];
                const bool e1257 = a537 > 0;
                if(e1257) {
                    PAIRS_DEBUG("resizes[0] -> neighborlist_capacity\n");
                    const int a538 = resizes[0];
                    const int e1258 = a538 * 2;
                    neighborlist_capacity = e1258;
                    pairs->reallocArray(10, &neighborlists, nullptr, (sizeof(int) * (particle_capacity * neighborlist_capacity)));
                }
            }
        }
        reset_volatile_properties(pairs, nlocal, force);
        lj(pairs, neighborlist_capacity, nlocal, numneighs, neighborlists, position, force);
        euler(pairs, nlocal, velocity, force, mass, position);
        const int e82 = i1 + 1;
        pairs::vtk_write_data(pairs, "output/test_cpu_local", 0, nlocal, e82);
        const int e1005 = nlocal + nghost;
        pairs::vtk_write_data(pairs, "output/test_cpu_ghost", nlocal, e1005, e82);
    }
    delete pairs;
    return 0;
}
