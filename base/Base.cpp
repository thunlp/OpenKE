#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <pthread.h>

extern "C"
void setInPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

struct Parameter {
	INT id;
	INT *batch_h;
	INT *batch_t;
	INT *batch_r;
	REAL *batch_y;
	INT batchSize;
	INT negRate;
	INT negRelRate;
};

void* getBatch(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	INT lef, rig;
	if (batchSize % workThreads == 0) {
		lef = id * (batchSize / workThreads);
		rig = (id + 1) * (batchSize / workThreads);
	} else {
		lef = id * (batchSize / workThreads + 1);
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}
	REAL prob = 500;
	for (INT batch = lef; batch < rig; batch++) {
		INT i = rand_max(id, trainTotal);
		batch_h[batch] = trainList[i].h;
		batch_t[batch] = trainList[i].t;
		batch_r[batch] = trainList[i].r;
		batch_y[batch] = 1;
		INT last = batchSize;
		for (INT times = 0; times < negRate; times ++) {
			if (bernFlag)
				prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
			if (randd(id) % 1000 < prob) {
				batch_h[batch + last] = trainList[i].h;
				batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
				batch_r[batch + last] = trainList[i].r;
			} else {
				batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);;
				batch_t[batch + last] = trainList[i].t;
				batch_r[batch + last] = trainList[i].r;
			}
			batch_y[batch + last] = -1;
			last += batchSize;
		}
		for (INT times = 0; times < negRelRate; times++) {
			batch_h[batch + last] = trainList[i].h;
			batch_t[batch + last] = trainList[i].t;
			batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t);
			batch_y[batch + last] = -1;
			last += batchSize;
		}
	}
	pthread_exit(NULL);
}

extern "C"
void sampling(INT *batch_h, INT *batch_t, INT *batch_r, REAL *batch_y, INT batchSize, INT negRate = 1, INT negRelRate = 0) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_y = batch_y;
		para[threads].batchSize = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);
	free(pt);
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}
