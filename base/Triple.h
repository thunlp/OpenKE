#ifndef TRIPLE_H
#define TRIPLE_H
#include "Setting.h"

struct Triple {

	INT h, r, t;

	static INT minimal(INT a,INT b) {
		if (a > b) return b;
		return a;
	}
	
	static bool cmp_list(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) > minimal(b.h, b.t));
	}

	static bool cmp_head(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}

	static bool cmp_tail(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}

	static bool cmp_rel(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.t < b.t)||(a.h == b.h && a.t == b.t && a.r < b.r);
	}

};

#endif
