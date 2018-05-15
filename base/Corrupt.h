#ifndef CORRUPT_H
#define CORRUPT_H
#include "Random.h"
#include "Triple.h"
#include "Reader.h"

INT corrupt_head(INT id, INT h, INT r) {
	INT lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
    // makes a binary search to find the first triple of the form (h, r, _)
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1; // shifts one bit of the number to the right (the same as dividing by two lol)
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig; // index of the first triple of the form (h, r, _)

	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) { // again, binary search to find the last triple of the form(h, r, _)
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef; // index of the last triple of the form (h, r, _)

	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r) {
	INT lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


INT corrupt_rel(INT id, INT h, INT t) {
	INT lef, rig, mid, ll, rr;
	lef = lefRel[h] - 1;
	rig = rigRel[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t >= t) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefRel[h];
	rig = rigRel[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t <= t) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, relationTotal - (rr - ll + 1));
	if (tmp < trainRel[ll].r) return tmp;
	if (tmp > trainRel[rr].r - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].r - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


bool _find(INT h, INT t, INT r) {
    INT lef = 0;
    INT rig = tripleTotal - 1;
    INT mid;
    // binary search
    while (lef + 1 < rig) {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}

INT corrupt(INT h, INT r){
	INT ll = tail_lef[r]; // gets the first index of the entities ids that are stored in `tail_type` (see Reader.h)
	INT rr = tail_rig[r]; // gets the last  index of the entities ids that are stored in `tail_type` (see Reader.h)
	INT loop = 0;
	INT t;
	while(1) {
		t = tail_type[rand(ll, rr)]; // randomly get one of the observed entities in the tail position (type constrained)
		if (not _find(h, t, r)) { // check if the entity is not observed in the dataset (including train and valid, but not test)
		//	printf("r:%ld\tt:%ld\n", r, t);
			return t;
		} else {
			loop ++;
			if (loop >= 1000){ // if it has tried for more than 1000 times and even then it couldn't find a tail entity that is not observed, then ...
			//	printf("drop\n");
				return corrupt_head(0, h, r);
			}
		}
	}
}
#endif
