#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"
#include <cstdlib>

unsigned long long *next_random;

extern "C"
void randReset() {
	next_random = (unsigned long long *)calloc(workThreads, sizeof(unsigned long long));
	for (int i = 0; i < workThreads; i++)
		next_random[i] = rand();
}

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

INT rand_max(int id, INT x) {
	INT res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

#endif
