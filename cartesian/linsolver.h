// Linear algebra solver (HYPRE) wrapper

#ifndef LINSOLVER_H
#define LINSOLVER_H

#include <vector>

typedef void (*GetPortraitFunc)(int ic, int& PortraitSize, int* Portrait, const double* datain, double* dataout);
void LinearSolverInit();
void LinearSolverAlloc(int WhatSystem, const int* N, int ROW_SIZE_MAX = 7, GetPortraitFunc MyGetPortrait = NULL);
void LinearSolverDealloc();
void LinearSolverFinalize();
void LinearSystemInit(int WhatSystem, const vector<double>& M); // WhatSystem=0 for pressure, =1 for velocities
int LinearSystemSolve(int WhatSystem, vector<double>& X, const vector<double>& RHS, double AMG_Tolerance, int AMG_MaxIters);

#endif
