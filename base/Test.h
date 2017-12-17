#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"

bool _find(INT h, INT t, INT r) {
    INT lef = 0;
    INT rig = tripleTotal - 1;
    INT mid;
    while (lef + 1 < rig) {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}

INT lastHead = 0;
INT lastTail = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0;

extern "C"
void getHeadBatch(INT *ph, INT *pt, INT *pr) {
	for (INT i = 0; i < entityTotal; i++) {
		ph[i] = i;
		pt[i] = testList[lastHead].t;
		pr[i] = testList[lastHead].r;
	}
}

extern "C"
void getTailBatch(INT *ph, INT *pt, INT *pr) {
	for (INT i = 0; i < entityTotal; i++) {
		ph[i] = testList[lastTail].h;
		pt[i] = i;
		pr[i] = testList[lastTail].r;
	}
}

extern "C"
void testHead(REAL *con) {
	INT h = testList[lastHead].h;
	INT t = testList[lastHead].t;
	INT r = testList[lastHead].r;

	REAL minimal = con[h];
	INT l_s = 0;
	INT l_filter_s = 0;
    INT l_s_constrain = 0;

    for (INT j = 0; j <= entityTotal; j++) {
        REAL value = con[j];
        if (j != h && value < minimal) {
            l_s += 1;
            if (not _find(j, t, r))
                l_filter_s += 1;
        }
    }

    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;
	l_filter_rank += (l_filter_s+1);
	l_rank += (1+l_s);
	lastHead++;
	printf("l_filter_s: %ld\n", l_filter_s);
	printf("%f %f %f %f\n", l_tot / lastHead, l_filter_tot / lastHead, l_rank / lastHead, l_filter_rank / lastHead);
}

extern "C"
void testTail(REAL *con) {
	INT h = testList[lastTail].h;
	INT t = testList[lastTail].t;
	INT r = testList[lastTail].r;

	REAL minimal = con[t];
	INT r_s = 0;
	INT r_filter_s = 0;
    INT r_s_constrain = 0;

    for (INT j = 0; j <= entityTotal; j++) {
        REAL value = con[j];
        if (j != t && value < minimal) {
            r_s += 1;
            if (not _find(h, j, r))
                r_filter_s += 1;
        }
    }

	if (r_filter_s < 10) r_filter_tot += 1;
	if (r_s < 10) r_tot += 1;
    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 1) r1_tot += 1;
	r_filter_rank += (1+r_filter_s);
	r_rank += (1+r_s);
	lastTail++;
    printf("r_filter_s: %ld\n", r_filter_s);
	printf("%f %f %f %f\n", r_tot /lastTail, r_filter_tot /lastTail, r_rank /lastTail, r_filter_rank /lastTail);
}

extern "C"
void test() {
	printf("overall results:\n");
	printf("left %f %f %f %f \n", l_rank/ testTotal, l_tot / testTotal, l3_tot / testTotal, l1_tot / testTotal);
	printf("left(filter) %f %f %f %f \n", l_filter_rank/ testTotal, l_filter_tot / testTotal,  l3_filter_tot / testTotal,  l1_filter_tot / testTotal);
	printf("right %f %f %f %f \n", r_rank/ testTotal, r_tot / testTotal,r3_tot / testTotal,r1_tot / testTotal);
	printf("right(filter) %f %f %f %f\n", r_filter_rank/ testTotal, r_filter_tot / testTotal,r3_filter_tot / testTotal,r1_filter_tot / testTotal);
}

#endif