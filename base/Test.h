#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"
#include <fstream>

/*=====================================================================================
link prediction
======================================================================================*/
INT lastHead = 0;
INT lastTail = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;

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

    for (INT j = 0; j < entityTotal; j++) {
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
    l_filter_reci_rank += 1.0/(l_filter_s+1);
    l_reci_rank += 1.0/(l_s+1);
    lastHead++;
    printf("l_filter_s: %ld\n", l_filter_s);
    printf("%f %f %f %f \n", l_tot / lastHead, l_filter_tot / lastHead, l_rank / lastHead, l_filter_rank / lastHead);
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

    for (INT j = 0; j < entityTotal; j++) {
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
    r_filter_reci_rank += 1.0/(1+r_filter_s);
    r_reci_rank += 1.0/(1+r_s);
    lastTail++;
    printf("r_filter_s: %ld\n", r_filter_s);
    printf("%f %f %f %f\n", r_tot /lastTail, r_filter_tot /lastTail, r_rank /lastTail, r_filter_rank /lastTail);
}

extern "C"
void test_link_prediction() {
    printf("overall results:\n");
    printf("left %f %f %f %f %f \n", l_rank/ testTotal, l_reci_rank/ testTotal, l_tot / testTotal, l3_tot / testTotal, l1_tot / testTotal);
    printf("left(filter) %f %f %f %f %f \n", l_filter_rank/ testTotal, l_filter_reci_rank/ testTotal, l_filter_tot / testTotal,  l3_filter_tot / testTotal,  l1_filter_tot / testTotal);
    printf("right %f %f %f %f %f \n", r_rank/ testTotal, r_reci_rank/ testTotal, r_tot / testTotal,r3_tot / testTotal,r1_tot / testTotal);
    printf("right(filter) %f %f %f %f %f\n", r_filter_rank/ testTotal, r_filter_reci_rank/ testTotal, r_filter_tot / testTotal,r3_filter_tot / testTotal,r1_filter_tot / testTotal);
}

/*=====================================================================================
triple classification
======================================================================================*/
bool file_exists(const char *filename)
{
  std::ifstream ifile(filename);
  return ifile;
}

Triple *negTestList;
extern "C"
void importNegTestFiles() {
    FILE *fin;
    INT tmp;

    FILE* f_kb4 = fopen((inPath + "test2id_neg.txt").c_str(), "r");
    tmp = fscanf(f_kb4, "%ld", &testTotal);
    negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb4, "%ld", &negTestList[i].h);
        tmp = fscanf(f_kb4, "%ld", &negTestList[i].t);
        tmp = fscanf(f_kb4, "%ld", &negTestList[i].r);
    }
    fclose(f_kb4);

	// the sorting of the negative lists will ensure that triple classification happens correctly
    std::sort(negTestList, negTestList + testTotal, Triple::cmp_rel2);
}
Triple *negValidList;
extern "C"
void importNegValidFiles() {
    FILE *fin;
    INT tmp;

    FILE* f_kb5 = fopen((inPath + "valid2id_neg.txt").c_str(), "r");
    tmp = fscanf(f_kb5, "%ld", &validTotal);
    negValidList = (Triple *)calloc(validTotal, sizeof(Triple));
	for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb5, "%ld", &negValidList[i].h);
        tmp = fscanf(f_kb5, "%ld", &negValidList[i].t);
        tmp = fscanf(f_kb5, "%ld", &negValidList[i].r);
    }
    fclose(f_kb5);

	// the sorting of the negative lists will ensure that triple classification happens correctly
    std::sort(negValidList, negValidList + validTotal, Triple::cmp_rel2);
}

// Triple *negTestList;
extern "C"
void getNegTest() {
    if (file_exists((inPath + "test2id_neg.txt").c_str())) {
        importNegTestFiles(); // function created above
    } else {
        negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
        for (INT i = 0; i < testTotal; i++) {
            negTestList[i] = testList[i];
            negTestList[i].t = corrupt(testList[i].h, testList[i].r);
        }
        FILE* fout = fopen((inPath + "test_neg.txt").c_str(), "w");
        for (INT i = 0; i < testTotal; i++) {
            fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", testList[i].h, testList[i].t, testList[i].r, INT(1));
            fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negTestList[i].h, negTestList[i].t, negTestList[i].r, INT(-1));
        }
        fclose(fout);
    }
    // set value for new variables
}

// Triple *negValidList;
extern "C"
void getNegValid() {
    if (file_exists((inPath + "valid2id_neg.txt").c_str())) {
        importNegValidFiles(); // function created above
    } else {
        negValidList = (Triple *)calloc(validTotal, sizeof(Triple));
        for (INT i = 0; i < validTotal; i++) {
            negValidList[i] = validList[i];
            negValidList[i].t = corrupt(validList[i].h, validList[i].r);
        }
        FILE* fout = fopen((inPath + "valid_neg.txt").c_str(), "w");
        for (INT i = 0; i < validTotal; i++) {
            fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", validList[i].h, validList[i].t, validList[i].r, INT(1));
            fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negValidList[i].h, negValidList[i].t, negValidList[i].r, INT(-1));
        }
        fclose(fout);
    }
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}

extern "C"
void getValidBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegValid();
    for (INT i = 0; i < validTotal; i++) {
        ph[i] = validList[i].h;
        pt[i] = validList[i].t;
        pr[i] = validList[i].r;
        nh[i] = negValidList[i].h;
        nt[i] = negValidList[i].t;
        nr[i] = negValidList[i].r;
    }
}

REAL *relThresh;
REAL threshEntire;
REAL validAcc;
extern "C"
void getBestThreshold(REAL *score_pos, REAL *score_neg) {
    relThresh = (REAL *)calloc(relationTotal, sizeof(REAL));
    REAL interval, min_score, max_score, bestThresh, tmpThresh, bestAcc, tmpAcc;
    INT n_interval, correct, total;
    INT total_all_relations, correct_all_relations, bestCorrect;
    total_all_relations = 0;
    correct_all_relations = 0;
    n_interval = 1000;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1) continue;
        total = (validRig[r] - validLef[r] + 1) * 2;
        total_all_relations += total;
        min_score = score_pos[validLef[r]];
        max_score = score_pos[validLef[r]];
        for (INT i = validLef[r]; i <= validRig[r]; i++) {
            if(score_pos[i] < min_score) min_score = score_pos[i];
            if(score_pos[i] > max_score) max_score = score_pos[i];
            if(score_neg[i] < min_score) min_score = score_neg[i];
            if(score_neg[i] > max_score) max_score = score_neg[i];
        }
        interval = (max_score - min_score)/n_interval; // defining the interval this way is more portable accross different kinds of models
        for (INT i = 0; i <= n_interval+1; i++) { // we should start the search BEFORE the min score and end it AFTER the max score, in case validation triples are all positive or negative.
            tmpThresh = min_score + i * interval;
            correct = 0;
            for (INT j = validLef[r]; j <= validRig[r]; j++) {
                if (score_pos[j] < tmpThresh) correct ++;
                if (score_neg[j] >= tmpThresh) correct ++;
            }
            tmpAcc = 1.0 * correct / total;
            if (i == 0) {
                bestThresh = tmpThresh;
                bestAcc = tmpAcc;
                bestCorrect = correct;
            } else if (tmpAcc > bestAcc) {
                bestAcc = tmpAcc;
                bestThresh = tmpThresh;
                bestCorrect = correct;
            }
        }
        relThresh[r] = bestThresh;
        // printf("relation %ld: bestThresh is %lf, bestAcc is %lf, min/max scores are [%lf, %lf]\n", r, bestThresh, bestAcc, min_score, max_score);
        printf("relation %li,\tbestCorrect:   %li,\tvalidLef[r]: %li,  \tvalidRig[r]: %li,\tinterval: %lf,\tn_interval: %li\tmin_score: %lf,\tmax_score: %lf\n", r, bestCorrect, validLef[r], validRig[r], interval, n_interval, min_score, max_score);
        correct_all_relations += bestCorrect;
    }
    validAcc = 1.0 * correct_all_relations / total_all_relations;
    printf("correct_all_relations: %li\ntotal_all_relations: %li\n\n", correct_all_relations, total_all_relations);
    // printf("validation best accuracy is %lf\n", validAcc);
}

REAL threshold_for_relation;
extern "C"
void update_threshold_for_relation(INT r) {
    threshold_for_relation = relThresh[r];
}

REAL *testAcc;
REAL aveAcc;
extern "C"
void test_triple_classification(REAL *score_pos, REAL *score_neg) {
    testAcc = (REAL *)calloc(relationTotal, sizeof(REAL));
    INT aveCorrect = 0, aveTotal = 0;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1 || testLef[r] ==-1) continue;
        INT correct = 0, total = 0;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos[i] <= relThresh[r]) correct++;
            if (score_neg[i] > relThresh[r]) correct++;
            total += 2;
        }
        testAcc[r] = 1.0 * correct / total;
        aveCorrect += correct;
        aveTotal += total;
        // printf("relation %ld: triple classification accuracy is %lf\n", r, testAcc[r]);
    }
    aveAcc = 1.0 * aveCorrect / aveTotal;
    // printf("average accuracy is %lf\n", aveAcc);
}

#endif
