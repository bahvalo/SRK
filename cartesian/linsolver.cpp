#include <_hypre_IJ_mv.h>
#include <_hypre_parcsr_ls.h>
#include <vector>
using namespace std;
#include "asrk.h"
#include "base.h" // for default GetPortrait

#include "linsolver.h"

#define MAX_NUM_SYSTEMS 2

// Debug flags
int PRINT_MATRICES = 0; // print matrices and vectors to file
int PRINT_SOLVER_LOG = 0; // print all solver log

// Structure for the data related to the linear algebra solver
struct tHYPRE_Data{
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;

    HYPRE_Solver solver, precond;

    vector<int> rows;
    int ROW_SIZE_MAX = 7;

    int NN = 0; // number of unknowns
    int Allocated = 0;
    int SolverAllocated = 0;
    int num_iterations = 0; // statistics from the last Solve call
    double final_res_norm = 0.; // statistics from the last Solve call

    void AllocMatrix(const int* N);
    void Init(const vector<double>& Mat); // of size NN*ROW_SIZE_MAX
    void Solve(vector<double>& X, const vector<double>& RHS, double tol, int maxit);
    void ReleaseMem();
    GetPortraitFunc MyGetPortrait = NULL;

    tHYPRE_Data(){}
    ~tHYPRE_Data(){ ReleaseMem(); }
};
static tHYPRE_Data HYPRE_Data[MAX_NUM_SYSTEMS]; // [0] for pressure, [1] for velocities


// Get a row of the matrix portrait
// Optionally, compress the data (in the input, there are spaces related to the non-existing nodes)
// Default subroutine -- for structured data based on tIndex and 5- (7-) point stencil
void GetPortrait(int ic, int& PortraitSize, int* Portrait, const double* datain, double* dataout){
    tIndex in(ic);

    PortraitSize = 1;
    Portrait[0]=ic;
    if(datain) dataout[0]=datain[0];
    for(int idir=0; idir<tIndex::Dim; idir++){
        if(in.i[idir]>0 || tIndex::IsPer[idir]){
            if(datain) dataout[PortraitSize]=datain[idir*2+1];
            Portrait[PortraitSize++] = in.Neighb(idir,0);
        }
        if(in.i[idir]<tIndex::N[idir]-1 || tIndex::IsPer[idir]){
            if(datain) dataout[PortraitSize]=datain[idir*2+2];
            Portrait[PortraitSize++] = in.Neighb(idir,1);
        }
    }
}


// Preprocess knowing the matrix portrait only
void tHYPRE_Data::AllocMatrix(const int* N){
    for(int idir=0; idir<3; idir++) if(tIndex::IsPer[idir] && N[idir]<3) { printf("At least three nodes for each periodic direction!\n"); exit(0); }
    NN = N[0]*N[1]*N[2];

    /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, NN-1, 0, NN-1, &A);

    /* Choose a parallel csr format storage (see the User's Manual) */
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

    /* Initialize before setting coefficients */
    HYPRE_IJMatrixInitialize(A);

    vector<double> null_vector(NN, 0.);

    int PortraitSize;
    vector<int> Portrait(ROW_SIZE_MAX);
    for(int ic=0; ic<NN; ic++){
        if(!MyGetPortrait) GetPortrait(ic, PortraitSize, Portrait.data(), NULL, NULL);
        else MyGetPortrait(ic, PortraitSize, Portrait.data(), NULL, NULL);
        HYPRE_IJMatrixSetValues(A, 1, &PortraitSize, &ic, Portrait.data(), null_vector.data());
    }

    /* Assemble after setting the coefficients */
    HYPRE_IJMatrixAssemble(A);
    /* Get the parcsr matrix object to use */
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);

    /* Create the rhs and solution */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, NN-1, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, NN-1, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    rows.resize(NN);
    for(int i=0; i<NN; i++) rows[i]=i;

    HYPRE_IJVectorSetValues(b, NN, rows.data(), null_vector.data());
    HYPRE_IJVectorSetValues(x, NN, rows.data(), null_vector.data());

    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void **) &par_x);

    Allocated = 1;
}

void tHYPRE_Data::ReleaseMem(){
    if(Allocated){
        rows = vector<int>();
        HYPRE_IJMatrixDestroy(A);
        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
        Allocated = 0;
    }
    if(SolverAllocated){
        HYPRE_BoomerAMGDestroy(solver); // Destroy solver
        SolverAllocated = 0;
    }
}

// Preprocess knowing the matrix but not the right-hand side
void tHYPRE_Data::Init(const vector<double>& Mat){
    HYPRE_IJMatrixInitialize(A);

    int PortraitSize;
    vector<int> Portrait(ROW_SIZE_MAX);
    vector<double> MatRow(ROW_SIZE_MAX);
    for(int ic=0; ic<NN; ic++){
        if(!MyGetPortrait) GetPortrait(ic, PortraitSize, Portrait.data(), Mat.data()+ic*ROW_SIZE_MAX, MatRow.data());
        else MyGetPortrait(ic, PortraitSize, Portrait.data(), Mat.data()+ic*ROW_SIZE_MAX, MatRow.data());
        HYPRE_IJMatrixSetValues(A, 1, &PortraitSize, &ic, Portrait.data(), MatRow.data());
    }

    HYPRE_IJMatrixAssemble(A);

    if(1){
        /* Create solver */
        if(SolverAllocated) HYPRE_BoomerAMGDestroy(solver); // Destroy the previous one
        HYPRE_BoomerAMGCreate(&solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_BoomerAMGSetPrintLevel(solver, PRINT_SOLVER_LOG?3:0);  /* print solve info + parameters */
        HYPRE_BoomerAMGSetOldDefault(solver); /* Falgout coarsening with modified classical interpolaiton */
        HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
        HYPRE_BoomerAMGSetRelaxOrder(solver, 1);   /* uses C/F relaxation */
        HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
        HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
        if(PRINT_SOLVER_LOG) printf("Set params done\n");

        /* Now setup */
        HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
    }
    if(PRINT_SOLVER_LOG) printf("Setup done\n");

    SolverAllocated = 1;
}

void tHYPRE_Data::Solve(vector<double>& X, const vector<double>& RHS, double tol, int maxit){
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorSetValues(b, NN, rows.data(), RHS.data());
    HYPRE_IJVectorAssemble(b);

    HYPRE_IJVectorInitialize(x);
    HYPRE_IJVectorSetValues(x, NN, rows.data(), X.data());
    HYPRE_IJVectorAssemble(x);

    /*  Print out the system */
    if(PRINT_MATRICES){
        char fnameA[128], fnameB[128];
        static int counter = 0;
        counter++;
        sprintf(fnameA, "IJ.out.A.%08i", counter);
        sprintf(fnameB, "IJ.out.b.%08i", counter);
        HYPRE_IJMatrixPrint(A, fnameA);
        HYPRE_IJVectorPrint(b, fnameB);
    }

    HYPRE_BoomerAMGSetTol(solver, tol);  // conv. tolerance
    HYPRE_BoomerAMGSetMaxIter(solver, maxit); // max number of iters
    HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
    if(PRINT_SOLVER_LOG) printf("Solver done\n");

    /* Run info - needed logging turned on */
    HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

    if(PRINT_SOLVER_LOG){
        printf("\n");
        printf("Iterations = %d\n", num_iterations);
        printf("Final Relative Residual Norm = %e\n", final_res_norm);
        printf("\n");
    }

    /* get the local solution */
    HYPRE_IJVectorGetValues(x, NN, rows.data(), X.data());
}



void LinearSystemInit(int WhatSystem, const vector<double>& M){
    HYPRE_Data[WhatSystem].Init(M);
}

int LinearSystemSolve(int WhatSystem, vector<double>& X, const vector<double>& RHS, double AMG_Tolerance, int AMG_MaxIters){
    HYPRE_Data[WhatSystem].Solve(X, RHS, AMG_Tolerance, AMG_MaxIters);
    return HYPRE_Data[WhatSystem].num_iterations;
}

// General init and uninit of the linear solver
void LinearSolverInit(){
    HYPRE_Initialize();
}
void LinearSolverAlloc(int WhatSystem, const int* N, int ROW_SIZE_MAX, GetPortraitFunc MyGetPortrait){
    HYPRE_Data[WhatSystem].ROW_SIZE_MAX = ROW_SIZE_MAX;
    HYPRE_Data[WhatSystem].MyGetPortrait = MyGetPortrait;
    HYPRE_Data[WhatSystem].AllocMatrix(N); // alloc memory for pressure system
}
void LinearSolverDealloc(){
    for(int i=0; i<MAX_NUM_SYSTEMS; i++) HYPRE_Data[i].ReleaseMem();
}
void LinearSolverFinalize(){
    LinearSolverDealloc();
    HYPRE_Finalize();
}
