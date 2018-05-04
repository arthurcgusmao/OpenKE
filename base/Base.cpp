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
	INT id = para -> id; // id is the processor thread id
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	INT lef, rig;
	INT subBatchSize;
	subBatchSize = batchSize / workThreads;
	if (batchSize % workThreads == 0) {
		lef = id * (subBatchSize);
		rig = (id + 1) * (subBatchSize);
	} else {
		lef = id * (subBatchSize + 1);
		rig = (id + 1) * (subBatchSize + 1);
		if (rig > batchSize) rig = batchSize; // this accounts for the case when there is fewer elements in the last batch portion
	}
	REAL prob = 500;
	INT i;
	if (shuffleFlag) {
		INT start_range = lef + (batchSize * shuffled_trainList_iter);
		INT end_range = rig + (batchSize * shuffled_trainList_iter);
		// printf("iter  %ld,\tthread %ld,\ttrainList range: [%ld to %ld)\n", shuffled_trainList_iter, id, start_range, end_range);
	}
	for (INT batch = lef; batch < rig; batch++) {
		if (shuffleFlag) {
			i = batch + (batchSize * shuffled_trainList_iter);
		} else {
			i = rand_max(id, trainTotal); // randomly gets a number that corresponds to an example from the training set
		}
		batch_h[batch] = trainList[i].h;
		batch_t[batch] = trainList[i].t;
		batch_r[batch] = trainList[i].r;
		batch_y[batch] = 1;
		INT last = batchSize;
		for (INT times = 0; times < negRate; times ++) {
			if (bernFlag) { // bernFlag signals that we should use the Bernoulli distribution for creating negative examples proposed by Wang et al. (2014).
				prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]); // probability of corrupting by replacing the tail
				// with the assumption above:
				//  right_mean: the average number of head entities per tail entity = hpt
				//  left_mean: the average number of tail entities per head entity = tph
			}
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

void verifyReshuffle(INT batchSize) {
	// if we are shuffling the trainList and not sampling random numbers
	if (shuffled_trainList_iter >= trainTotal / batchSize || shuffled_trainList_iter == -1) { // reshuffle trainList when iter >= n_batches
		std::random_shuffle(&trainList[0],&trainList[trainTotal]);
		shuffled_trainList_iter = 0;
	}
}

extern "C"
void sampling(INT *batch_h, INT *batch_t, INT *batch_r, REAL *batch_y, INT batchSize, INT negRate = 1, INT negRelRate = 0) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	if (shuffleFlag) verifyReshuffle(batchSize);
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
	if (shuffleFlag) shuffled_trainList_iter++;
}

int main() {
	importTrainFiles();
	return 0;
}
