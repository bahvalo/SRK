// Test case from Colomes&Badia
// Domain is [0,1]x[0,1]
// Exact solution is u=(x,-y)*g(t), p=x+y
// Source terms and boundary conditions are according to the exact solution

#include "asrk.h"
#include "fd2.h"
#include "linsolver.h"
#include <omp.h>

// Do not specify values here!
static vector<double> X[3]; // nodal coordinates
static double3 GetCoor(const tIndex& ind) { return double3(X[0][ind.i[0]], X[1][ind.i[1]], X[2][ind.i[2]]); }
static int NN = 0; // number of mesh nodes
static int EnableConvection = 1; // convection (enabled or not)

#if 1
// Solution dependency on time: functions from the original test
static const double Pi = 3.14159265358979323846;
static double g(double t){ return sin(0.1*Pi*t) * exp(0.04*t); }
static double dgdt(double t) { return 0.1*Pi*cos(0.1*Pi*t)*exp(0.04*t) + 0.04*g(t); }
#else
// Solution dependency on time: linear (for debugging)
static double g(double t){ return t; }
static double dgdt(double t) { return 1.; }
#endif


static void SourceConv(double t, vector<double3>& f){
    double _g = g(t);
    for(tIndex in=0; in<NN; ++in){
        double3 coor = GetCoor(in);
        if(in.IsWall()) continue;
        // Pressure gradient compensation
        f[in][0] += 1.;
        f[in][1] += 1.;
        // Convective term compensation
        if(EnableConvection){
            f[in][0] += coor[0]*_g*_g;
            f[in][1] += coor[1]*_g*_g;
        }
    }
}

static void SourceUnst(double t, vector<double3>& f){
    double gt = dgdt(t);
    for(tIndex in=0; in<NN; ++in){
        double3 coor = GetCoor(in);
        // Unsteady term compensation
        f[in][0] += coor[0]*gt;
        f[in][1] += -coor[1]*gt;
    }
}

static void SourceBoth(double t, vector<double3>& f){
    SourceConv(t,f);
    SourceUnst(t,f);
}


static double3 MyBoundaryValue(double t, const double3& coor){
    double _g = g(t);
    return double3(coor[0]*_g, -coor[1]*_g, 0.);
}
static double3 dMyBoundaryValue(double t, const double3& coor){
    double gt = dgdt(t);
    return double3(coor[0]*gt, -coor[1]*gt, 0.);
}


static void ctest(const tIMEXMethod& IMEX_Method, tTimeIntMethod TimeIntMethod, int StabType, double alpha_fraction, double nu,
                  double tau, double& max_err_u, double& max_err_p, bool PlotVtk=0){
    // Computational mesh
    int IsPer[3] = {0,0,0}; // 1 if there are periodic conditions for the corresponding direction
    int N[3] = {11,11,1}; // number of nodes for each direction (for non-periodic BCs, includes boundary nodes; for periodic BCs, no images)
    tIndex::Init(N, IsPer); // init of structure used for loops over nodes, to get neighboring nodes, etc.

    NN = N[0]*N[1];
    for(int idir=0; idir<2; idir++) for(int i=0; i<N[idir]; i++) X[idir].push_back(double(i)/(N[idir]-1));
    X[2].push_back(0.);

    tSRKTimeIntegrator<S_FD2> FlowSolver(IMEX_Method, StabType);
    FlowSolver.TimeIntMethod = TimeIntMethod; // time integration method
    FlowSolver.EnableConvection = EnableConvection = 1; // include the convective term or not
    FlowSolver.visc = nu; // viscosity coefficient
    FlowSolver.ViscScheme = tViscScheme::CROSS; // discretization of viscous terms
    FlowSolver.ConvMethod = tConvMethod::SKEWSYMMETRIC;
    FlowSolver.X[0] = X[0]; // pass the nodal coordinates to the solver
    FlowSolver.X[1] = X[1];
    FlowSolver.X[2] = X[2];
    FlowSolver.NumIters_IMEX = 2; // unnecessary; 1 is enough
    FlowSolver.NumIters_Impl = 10; // some large number for nonlinear iterations
    if(TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX){
        FlowSolver.SourceE = SourceBoth;
        FlowSolver.SourceI = NULL;

        if(0){ // We can split the source term between the explicit and implicit parts
            FlowSolver.SourceE = SourceConv;
            FlowSolver.SourceI = SourceUnst;
        }
    }
    else{
        FlowSolver.SourceE = NULL;
        FlowSolver.SourceI = SourceBoth;
    }
    FlowSolver.DoNotUseDBoundaryValue = 0;
    FlowSolver.BoundaryValue = MyBoundaryValue;
    FlowSolver.dBoundaryValue = dMyBoundaryValue;
    FlowSolver.AMG_Tolerance_P = FlowSolver.AMG_Tolerance_U = 1e-12; // tolerance for HYPRE AMG solver
    FlowSolver.alpha_tau = alpha_fraction*IMEX_Method.alpha_tau_max[StabType];
    FlowSolver.Init();

    FlowSolver.AllocDataArrays();

    // Initial data
    vector<double3> velocity(NN);
    vector<double> pressure(NN), qressure(NN);
    for(tIndex in=0; in<NN; ++in){
        double3 coor = GetCoor(in);
        velocity[in] = double3();
        pressure[in] = coor[0]+coor[1];
        qressure[in] = 0.; // dp/dt -- will be used for StabType==1 only
    }

    // Get the number of timesteps and adjust the timestep size
    double TimeMax = 0.1;
    int NumTimeSteps = TimeMax / tau;
    if(tau*NumTimeSteps < TimeMax) NumTimeSteps++;
    tau = TimeMax / NumTimeSteps;

    // Time integration
    //printf("Start of time integration (%i steps)\n", NumTimeSteps);
    for(int iTimeStep=0; iTimeStep < NumTimeSteps; iTimeStep++){
        FlowSolver.Step(iTimeStep*tau, tau, velocity, pressure, qressure);

        // Reapply boundary conditions here -- if they were applied in the form of the time derivative
        int DoReapplyBCs = (FlowSolver.DoNotUseDBoundaryValue==0);
        if(DoReapplyBCs){
            double t = (iTimeStep+1)*tau;
            double _g = g(t);
            for(tIndex in=0; in<NN; ++in){
                if(!in.IsWall()) continue;
                double3 coor = GetCoor(in);
                velocity[in] = double3(coor[0]*_g, -coor[1]*_g, 0.);
            }
        }
    }

    // Substract the exact solution
    for(tIndex in=0; in<NN; ++in){
        double _g = g(TimeMax);
        double3 coor = GetCoor(in);
        double3 uex(coor[0]*_g, -coor[1]*_g, 0.);
        double pex = coor[0] + coor[1];
        velocity[in] = velocity[in] - uex;
        pressure[in] -= pex;
    }
    FlowSolver.NormalizePressure(pressure);

    // Calculating a norm of the solution error
    max_err_u = 0., max_err_p = 0.;
    for(int in=0; in<NN; ++in){
        double uerr = abs(velocity[in]);
        double perr = fabs(pressure[in]);
        if(uerr>max_err_u) max_err_u = uerr;
        if(perr>max_err_p) max_err_p = perr;
    }

    LinearSolverDealloc();

    if(PlotVtk){
        FILE* f = fopen("output.vtk", "wt");
        fprintf(f, "# vtk DataFile Version 2.0\n");
        fprintf(f, "Volume example\n");
        fprintf(f, "ASCII\n");
        fprintf(f, "DATASET STRUCTURED_POINTS\n");
        fprintf(f, "DIMENSIONS %i %i 1\n", N[0], N[1]);
        fprintf(f, "ASPECT_RATIO 1 1 1\n");
        fprintf(f, "ORIGIN 0 0 0\n");
        fprintf(f, "POINT_DATA %i\n", NN);
        fprintf(f, "SCALARS volume_scalars float 6\n");
        fprintf(f, "LOOKUP_TABLE default\n");
        for(tIndex in=0; in<NN; ++in){
            double _g = g(TimeMax);
            double3 coor = GetCoor(in);
            double3 uex(coor[0]*_g, -coor[1]*_g, 0.);
            double pex = coor[0] + coor[1];
            fprintf(f, "%e %e %e %e %e %e\n", velocity[in][0]+uex[0], velocity[in][1]+uex[1], pressure[in]+pex, velocity[in][0], velocity[in][1], pressure[in]);
        }
        fclose(f);
    }
}


void ctest_series(const tIMEXMethod& IMEX_Method, tTimeIntMethod TimeIntMethod, double alpha_fraction, double nu){
    const int NNN = 4;
    double max_err_u[2][NNN], max_err_p[2][NNN];
    for(int StabType=0; StabType<=1; StabType++){
        for(int i=0; i<NNN; i++){
            double tau = 0.1 * pow(0.5, i);
            ctest(IMEX_Method, TimeIntMethod, StabType, alpha_fraction, nu, tau, max_err_u[StabType][i], max_err_p[StabType][i]);
        }
    }

    for(int i=0; i<NNN; i++){
        double tau = 0.1 * pow(0.5, i);
        printf("%.04f %e %e %e %e\n", tau, max_err_u[0][i], max_err_p[0][i], max_err_u[1][i], max_err_p[1][i]);
    }
}

#if 0
void main_ctest(){
    omp_set_num_threads(1);

    // HYPRE initialization
    LinearSolverInit();

    double alpha_fraction = 0.5;
    double nu = 0.01;
    tIMEXMethod IMEX_Method = SSP2_332();

    printf("EXPLICIT\n");
    //ctest_series(IMEX_Method, tTimeIntMethod::EXPLICIT, alpha_fraction, nu);
    printf("IMEX\n");
    //ctest_series(IMEX_Method, tTimeIntMethod::IMEX, alpha_fraction, nu);
    printf("IMPLICIT\n");
    ctest_series(IMEX_Method, tTimeIntMethod::IMPLICIT, alpha_fraction, nu);

    LinearSolverFinalize();
}
#endif

#if 0
void main_ctest(){
    omp_set_num_threads(1);

    // HYPRE initialization
    LinearSolverInit();

    double alpha_fraction = 0.5;
    double nu = 0.0;
    tIMEXMethod IMEX_Method = SI_IMEX_332();

    //printf("EXPLICIT\n");
    //ctest_series(IMEX_Method, tTimeIntMethod::EXPLICIT, alpha_fraction, nu);
    //printf("IMEX\n");
    //ctest_series(IMEX_Method, tTimeIntMethod::IMEX, alpha_fraction, nu);
    //printf("IMPLICIT\n");
    //ctest_series(IMEX_Method, tTimeIntMethod::IMPLICIT, alpha_fraction, nu);

    double erru, errp;
    ctest(IMEX_Method, tTimeIntMethod::IMPLICIT, 0, alpha_fraction, nu, 0.1, erru, errp, true);
    printf("erru = %e, errp = %e\n", erru, errp);

    LinearSolverFinalize();
}
#endif

#if 1
void main_ctest(){
    omp_set_num_threads(1);

    // HYPRE initialization
    LinearSolverInit();

    double alpha_fraction = 0.5;
    double nu = 0.01;
    const int NNN = 5; // number of different values of tau

    tIMEXMethod T[20] = {ARS_121(), ARS_232(), ARS_343(), ARS_111(), ARS_222(), ARS_443(), MARS_343(),
                         ARK3(), ARK4(), ARK5(), MARK3(), BHR_553(),
                         SSP2_322(), SI_IMEX_332(), SI_IMEX_443(), SI_IMEX_433(),
                         SSP2_322_A1(), SI_IMEX_332_A1(), SI_IMEX_443_A1(), SI_IMEX_433_A1()};
    const char fnames[2][3][16] = {{"st0_expl.dat", "st0_imex.dat", "st0_impl.dat"}, {"st1_expl.dat", "st1_imex.dat", "st1_impl.dat"}};

    for(int StabType=0; StabType<=1; StabType++)
    for(int TM = 0; TM<3; TM++){
        FILE* F = fopen(fnames[StabType][TM], "wt");
        for(int i=0; i<NNN; i++){
            double tau = 0.1 * pow(0.5, i);
            fprintf(F, "%.04f", tau);
            for(int imethod=0; imethod<20; imethod++){
                double err_u, err_p;
                ctest(T[imethod], tTimeIntMethod(TM), StabType, alpha_fraction, nu, tau, err_u, err_p);
                fprintf(F, " %e %e", err_u, err_p);
            }
            fprintf(F, "\n");
        }
        fclose(F);
        printf("TM %i done\n", TM);
    }

    LinearSolverFinalize();
}
#endif
