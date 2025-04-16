// 2D Taylor-Green vortex
#include "asrk.h"
#include "fd.h"
#include "gal1.h"
#include "fd2.h"
#include "linsolver.h"
#include <omp.h>

static const double Pi = 3.14159265358979323846;

static void GetExactSol(vector<double3>& u, vector<double>& p, vector<double>& q, const S_Base& FS, double3 h, double t, double3 Uinf){
    u.resize(FS.NN);
    p.resize(FS.NN);
    q.resize(FS.NN);

    double ee = exp(-2.*FS.visc*t); // decay due to viscosity
    double vx = Uinf[0], vy = Uinf[1];
    for(tIndex in=0; in<FS.NN; ++in){
        double x = in.i[0]*h[0]-vx*t, y = in.i[1]*h[1]-vy*t;
        u[in] = double3(sin(x)*cos(y)*ee, -cos(x)*sin(y)*ee, 0.) + Uinf; // u exact
        p[in] = (cos(2*y)+cos(2*x))/4. * ee*ee; // p exact
        q[in] = (vx*sin(2*x)+vy*sin(2*y))/2. * ee*ee; // dpdt exact
    }
    FS.NormalizePressure(p);
}


void main_tgv2d_FD(int N1D, double tau_requested, tTimeIntMethod TimeIntMethod, const tIMEXMethod& IMEX, int StabType, double visc, double TimeMax,
                   double& eu, double& ep, double& ek){
    // Computational mesh
    const int USE_3D_MESH = 0; // switch to 1 for the use of 3D mesh (for test purpose)
    int N[3] = {N1D,N1D,1}; // number of nodes for each direction
    if(!USE_3D_MESH) N[2]=1;
    int NN = N[0]*N[1]*N[2];
    const double3 h(2.*Pi/N[0], 2.*Pi/N[1], 1./N[2]);
    int IsPer[3] = {1,1,USE_3D_MESH};
    tIndex::Init(N, IsPer); // init of structure used for loops over nodes, to get neighboring nodes, etc.

    // Model parameters
    double3 BackgroundFlow(1.,0.,0.);

    // Get the number of timesteps and adjust the timestep size
    int NumTimeSteps = TimeMax / tau_requested;
    if(tau_requested*NumTimeSteps < TimeMax) NumTimeSteps++;
    double tau = TimeMax / NumTimeSteps;

    // Time integration
    tSRKTimeIntegrator<S_FD> FlowSolver(IMEX, StabType);
    for(int i=0; i<=N[0]; i++) FlowSolver.X[0].push_back(i*h[0]);
    for(int i=0; i<=N[1]; i++) FlowSolver.X[1].push_back(i*h[1]);
    for(int i=0; i<=N[2]; i++) FlowSolver.X[2].push_back(i*h[2]);
    FlowSolver.TimeIntMethod = TimeIntMethod;
    FlowSolver.GradDivOrder = FlowSolver.ConvOrder = FlowSolver.ViscOrder = 6;
    FlowSolver.MeshStep = h;
    FlowSolver.visc = visc;
    FlowSolver.ConvMethod = tConvMethod::SKEWSYMMETRIC;
    FlowSolver.ViscScheme = tViscScheme::CROSS;
    FlowSolver.AMG_Tolerance_U = 1e-10;
    FlowSolver.Init(); // viscosity coefficient

    // Numerical methods initialization
    FlowSolver.AllocDataArrays();

    // Initial data
    vector<double3> velocity(NN);
    vector<double> pressure(NN), qressure(NN);
    GetExactSol(velocity, pressure, qressure, FlowSolver, h, 0., BackgroundFlow);
    if(FlowSolver.ConvMethod==tConvMethod::EMAC) for(tIndex in=0; in<NN; ++in) pressure[in] -= 0.5*DotProd(velocity[in], velocity[in]);

    // Time integration
    //printf("Start of time integration (%i steps)\n", NumTimeSteps);
    double Kini = FlowSolver.CalcKineticEnergy(velocity);
    for(int iTimeStep=0; iTimeStep < NumTimeSteps; iTimeStep++){
        FlowSolver.Step(iTimeStep*tau, tau, velocity, pressure, qressure);
    }
    double Kfin = FlowSolver.CalcKineticEnergy(velocity);
    ek = Kfin-Kini;

    // Back to static pressure
    if(FlowSolver.ConvMethod==tConvMethod::EMAC) for(tIndex in=0; in<NN; ++in) pressure[in] += 0.5*DotProd(velocity[in], velocity[in]);

    vector<double3> uex;
    vector<double> pex, qex;
    GetExactSol(uex, pex, qex, FlowSolver, h, TimeMax, BackgroundFlow);

    FlowSolver.NormalizePressure(pressure);
    eu = ep = 0.;
    for(tIndex in=0; in<NN; ++in){
        eu = std::max(eu, abs(velocity[in] - uex[in]));
        ep = std::max(ep, fabs(pressure[in] - pex[in]));
    }

    if(0 && !USE_3D_MESH){
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
            fprintf(f, "%e %e %e %e %e %e\n", velocity[in][0], velocity[in][1], pressure[in], uex[in][0], uex[in][1], pex[in]);
        }
        fclose(f);
    }
}


void main_tgv2d_Galerkin(int N1D, double tau_requested, tTimeIntMethod TimeIntMethod, const tIMEXMethod& IMEX, int StabType, double visc, double TimeMax,
                   double& eu, double& ep, double& ek){
    // Computational mesh
    const int USE_3D_MESH = 0; // switch to 1 for the use of 3D mesh (for test purpose)
    int N[3] = {N1D,N1D,1}; // number of nodes for each direction
    if(!USE_3D_MESH) N[2]=1;
    int NN = N[0]*N[1]*N[2];
    const double3 h(2.*Pi/N[0], 2.*Pi/N[1], 1./N[2]);
    int IsPer[3] = {1,1,USE_3D_MESH};
    tIndex::Init(N, IsPer); // init of structure used for loops over nodes, to get neighboring nodes, etc.

    // Model parameters
    double3 BackgroundFlow(1.,0.,0.);

    // Get the number of timesteps and adjust the timestep size
    int NumTimeSteps = TimeMax / tau_requested;
    if(tau_requested*NumTimeSteps < TimeMax) NumTimeSteps++;
    double tau = TimeMax / NumTimeSteps;

    // Time integration
    tSRKTimeIntegrator<S_Gal1B> FlowSolver(IMEX, StabType);
    for(int i=0; i<=N[0]; i++) FlowSolver.X[0].push_back(i*h[0]);
    for(int i=0; i<=N[1]; i++) FlowSolver.X[1].push_back(i*h[1]);
    for(int i=0; i<=N[2]; i++) FlowSolver.X[2].push_back(i*h[2]);
    FlowSolver.TimeIntMethod = TimeIntMethod;
    FlowSolver.visc = visc;
    FlowSolver.EnableConvection=1;
    FlowSolver.AMG_Tolerance_P = FlowSolver.AMG_Tolerance_U = 1e-10;
    FlowSolver.IsFourier[0]=FlowSolver.IsFourier[1]=1;
    FlowSolver.IsFourier[2]=USE_3D_MESH;
    FlowSolver.ConvMethod = tConvMethod::CONSERVATIVE; // or tConvMethod::EMAC
    FlowSolver.Init(); // viscosity coefficient

    // Check that the convective term is energy-preserving
    #if 0
    if(1){
        vector<double3> u(NN), f(NN);
        for(tIndex in=0; in<NN; ++in) u[in]=double3(rand(), rand(), 0.);
        FlowSolver.CalcConvTerm(u, f, true);
        double sum = 0.;
        for(tIndex in=0; in<NN; ++in) sum += DotProd(u[in], f[in]);
        printf("sum = %e\n", sum);
    }
    #endif

    // Numerical methods initialization
    FlowSolver.AllocDataArrays();

    vector<double3> velocity(NN);
    vector<double> pressure(NN), qressure(NN);

    // Initial data
    GetExactSol(velocity, pressure, qressure, FlowSolver, h, 0., BackgroundFlow);
    if(FlowSolver.ConvMethod==tConvMethod::EMAC) FlowSolver.PressureToEffectivePressure(velocity, pressure, -0.5);

    // Time integration
    //printf("Start of time integration (%i steps)\n", NumTimeSteps);
    double Kini = FlowSolver.CalcKineticEnergy(velocity);
    for(int iTimeStep=0; iTimeStep < NumTimeSteps; iTimeStep++){
        FlowSolver.Step(iTimeStep*tau, tau, velocity, pressure, qressure);
    }
    double Kfin = FlowSolver.CalcKineticEnergy(velocity);
    ek = Kfin-Kini;
    //printf("Kinetic energy loss = %e\n", Kini-Kfin);

    // Back to static pressure
    if(FlowSolver.ConvMethod==tConvMethod::EMAC) FlowSolver.PressureToEffectivePressure(velocity, pressure, 0.5);

    vector<double3> uex;
    vector<double> pex, qex;
    GetExactSol(uex, pex, qex, FlowSolver, h, TimeMax, BackgroundFlow);

    FlowSolver.NormalizePressure(pressure);
    eu = ep = 0.;
    for(tIndex in=0; in<NN; ++in){
        eu = std::max(eu, abs(velocity[in] - uex[in]));
        ep = std::max(ep, fabs(pressure[in] - pex[in]));
    }
}


void main_tgv2d_visc_dominant(){
    //omp_set_num_threads(1);
    LinearSolverInit(); // Initialization of HYPRE

    int StabType = 0;
    double visc = 0.5;
    double TimeMax = 2.;

    tIMEXMethod T[12] = {ARS_232(), ARS_343(), ARS_443(), MARS_343(),
                         ARK3(), ARK4(), ARK5(), MARK3(), BHR_553(),
                         SSP2_322(), SI_IMEX_443(), SI_IMEX_433()};

    // EXPLICIT
    if(1){
        FILE* F = fopen("expl.dat", "wt");
        for(int N1D = 32; N1D<=128; N1D*=2){
            const double h = 2.*Pi/N1D;
            const double CFL = 1.;
            double tau = CFL / (2./h + 6.*visc/h/h);

            fprintf(F, "%e", h);
            for(int imethod=0; imethod<12; imethod++){
                double eu, ep, ek;
                main_tgv2d_FD(N1D, tau, tTimeIntMethod::EXPLICIT, T[imethod], StabType, visc, TimeMax, eu, ep, ek);
                fprintf(F, " %.3e %.3e", eu, ep);
            }
            fprintf(F, "\n");
            printf("Done: explicit, N1D=%i\n", N1D);
        }
        fclose(F);
    }

    // IMEX
    if(1){
        FILE* F = fopen("imex.dat", "wt");
        for(int N1D = 32; N1D<=128; N1D*=2){
            const double h = 2.*Pi/N1D;
            const double CFL = 0.5;
            double tau = CFL / (2./h); // now viscosity does not contibute to the timestep restriction

            fprintf(F, "%e", h);
            for(int imethod=0; imethod<12; imethod++){
                double eu, ep, ek;
                main_tgv2d_FD(N1D, tau, tTimeIntMethod::IMEX, T[imethod], StabType, visc, TimeMax, eu, ep, ek);
                fprintf(F, " %.3e %.3e", eu, ep);
            }
            fprintf(F, "\n");
            printf("Done: IMEX, N1D=%i\n", N1D);
        }
        fclose(F);
    }

    // IMPLICIT
    if(1){
        FILE* F = fopen("impl.dat", "wt");
        for(int N1D = 32; N1D<=128; N1D*=2){
            const double h = 2.*Pi/N1D;
            const double CFL = 1.;
            double tau = CFL / (2./h); // now viscosity does not contibute to the timestep restriction

            fprintf(F, "%e", h);
            for(int imethod=0; imethod<12; imethod++){
                double eu, ep, ek;
                main_tgv2d_FD(N1D, tau, tTimeIntMethod::IMPLICIT, T[imethod], StabType, visc, TimeMax, eu, ep, ek);
                fprintf(F, " %.3e %.3e", eu, ep);
            }
            fprintf(F, "\n");
            printf("Done: implicit, N1D=%i\n", N1D);
        }
        fclose(F);
    }

    LinearSolverFinalize();
}


double main_tgv2d_stab_check(int N1D, double tau, const tIMEXMethod& IMEX, int StabType, double TimeMax){
    // Computational mesh
    const int USE_3D_MESH = 0; // switch to 1 for the use of 3D mesh (for test purpose)
    int N[3] = {N1D,N1D,1}; // number of nodes for each direction
    if(!USE_3D_MESH) N[2]=1;
    int NN = N[0]*N[1]*N[2];
    const double3 h(2.*Pi/N[0], 2.*Pi/N[1], 1./N[2]);
    int IsPer[3] = {1,1,USE_3D_MESH};
    tIndex::Init(N, IsPer); // init of structure used for loops over nodes, to get neighboring nodes, etc.

    // Time integration
    tSRKTimeIntegrator<S_FD> FlowSolver(IMEX, StabType);
    for(int i=0; i<=N[0]; i++) FlowSolver.X[0].push_back(i*h[0]);
    for(int i=0; i<=N[1]; i++) FlowSolver.X[1].push_back(i*h[1]);
    for(int i=0; i<=N[2]; i++) FlowSolver.X[2].push_back(i*h[2]);
    FlowSolver.TimeIntMethod = tTimeIntMethod::EXPLICIT;
    FlowSolver.GradDivOrder = FlowSolver.ConvOrder = FlowSolver.ViscOrder = 6;
    FlowSolver.MeshStep = h;
    FlowSolver.visc = 0.;
    FlowSolver.ConvMethod = tConvMethod::SKEWSYMMETRIC;
    FlowSolver.ViscScheme = tViscScheme::CROSS;
    FlowSolver.Init(); // viscosity coefficient

    // Numerical methods initialization
    FlowSolver.AllocDataArrays();

    // Initial data
    vector<double3> velocity(NN);
    vector<double> pressure(NN), qressure(NN);
    GetExactSol(velocity, pressure, qressure, FlowSolver, h, 0., double3(1.,0.,0.));
    if(FlowSolver.ConvMethod == tConvMethod::EMAC) for(tIndex in=0; in<NN; ++in) pressure[in] -= 0.5*DotProd(velocity[in], velocity[in]);

    // Time integration
    //printf("Start of time integration (%i steps)\n", NumTimeSteps);
    int NumTimeSteps = TimeMax / tau + 1;
    double Kini = FlowSolver.CalcKineticEnergy(velocity);
    for(int iTimeStep=0; iTimeStep < NumTimeSteps; iTimeStep++){
        FlowSolver.Step(iTimeStep*tau, tau, velocity, pressure, qressure);
        double K = FlowSolver.CalcKineticEnergy(velocity);
        if(K > Kini*2. || !(K>0.)) return 1e10;
    }
    return FlowSolver.CalcKineticEnergy(velocity) - Kini;
}



void main_tgv2d_no_visc(){
    LinearSolverInit(); // Initialization of HYPRE

    const int N1D = 64;
    const double h = 2.*Pi/N1D;
    int StabType = 0;
    double TimeMax = 1000.;

    tIMEXMethod T[12] = {ARS_232(), ARS_343(), ARS_443(), MARS_343(),
                         ARK3(), ARK4(), ARK5(), MARK3(), BHR_553(),
                         SSP2_322(), SI_IMEX_443(), SI_IMEX_433()};

    // EXPLICIT
    FILE* F = fopen("stab_check.dat", "wt");
    for(int imethod=0; imethod<12; imethod++){
        double tau_min = 0., tau_max = 1.5*h;
        for(int iter=0; iter<10; iter++){ // 3 decimal signs
            double tau = 0.5*(tau_min + tau_max);
            double ek = main_tgv2d_stab_check(N1D, tau, T[imethod], StabType, TimeMax);
            if(ek > 1e9) tau_max = tau;
            else tau_min = tau;
        }
        double tau = 0.5*(tau_min + tau_max);
        fprintf(F, "Method %i, CFLmax=%f\n", imethod, tau/h * 2.*1.586);
    }
    fclose(F);

    LinearSolverFinalize();
}

void main_tgv2d() {
    return main_tgv2d_visc_dominant();
}
