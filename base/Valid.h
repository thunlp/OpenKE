#ifndef VALID_H
#define VALID_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

INT lastValidHead = 0;
INT lastValidTail = 0;
	
REAL l_valid_filter_tot = 0;
REAL r_valid_filter_tot = 0;

extern "C"
void validInit() {
    lastValidHead = 0;
    lastValidTail = 0;
    l_valid_filter_tot = 0;
    r_valid_filter_tot = 0;
}

extern "C"
void getValidHeadBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
	ph[i] = i;
	pt[i] = validList[lastValidHead].t;
	pr[i] = validList[lastValidHead].r;
    }
}

extern "C"
void getValidTailBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
	ph[i] = validList[lastValidTail].h;
	pt[i] = i;
	pr[i] = validList[lastValidTail].r;
    }
}

extern "C"
void validHead(REAL *con) {
    INT h = validList[lastValidHead].h;
    INT t = validList[lastValidHead].t;
    INT r = validList[lastValidHead].r;
    REAL minimal = con[h];
    INT l_filter_s = 0;
    for (INT j = 0; j < entityTotal; j++) {
	if (j != h) {
	    REAL value = con[j];
   	    if (value < minimal && ! _find(j, t, r)) {
		l_filter_s += 1;
	    }
	}
    }
    if (l_filter_s < 10) l_valid_filter_tot += 1;
    lastValidHead ++;
  //  printf("head: l_valid_filter_tot = %f | l_filter_hit10 = %f\n", l_valid_filter_tot, l_valid_filter_tot / lastValidHead);
}

extern "C"
void validTail(REAL *con) {
    INT h = validList[lastValidTail].h;
    INT t = validList[lastValidTail].t;
    INT r = validList[lastValidTail].r;
    REAL minimal = con[t];
    INT r_filter_s = 0;
    for (INT j = 0; j < entityTotal; j++) {
	if (j != t) {
	    REAL value = con[j];
	    if (value < minimal && ! _find(h, j, r)) {
	        r_filter_s += 1;
	    }
	}
    }
    if (r_filter_s < 10) r_valid_filter_tot += 1;
    lastValidTail ++;
//    printf("tail: r_valid_filter_tot = %f | r_filter_hit10 = %f\n", r_valid_filter_tot, r_valid_filter_tot / lastValidTail);
}

REAL validHit10 = 0;
extern "C"
REAL  getValidHit10() {
    l_valid_filter_tot /= validTotal;
    r_valid_filter_tot /= validTotal;
    validHit10 = (l_valid_filter_tot + r_valid_filter_tot) / 2;
   // printf("result: %f\n", validHit10);
    return validHit10;
}

#endif
