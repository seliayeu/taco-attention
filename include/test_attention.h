#ifndef TEST_SPMM_H
#define TEST_SPMM_H

#include <string>

int parseArgumentsAndRun(int argc, char *argv[]);

void run(int n_q, int n_k, int d_k, int d_v);

void runTests(int argc, char *argv[]);

#endif
