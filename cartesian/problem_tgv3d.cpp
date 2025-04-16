// 3D Taylor-Green vortex
#include "asrk.h"
#include "fd.h"
#include "gal1.h"
#include "linsolver.h"
#include <omp.h>

static const double Pi = 3.14159265358979323846;

void main_tgv3d(){
    // Computational mesh
    const int N = 128; // number of nodes in one direction. Should be a power of 2 to use FFT
    const int NN = N*N*N;
    const double h = 2.*Pi/N;
    const int IsPer[3] = {1,1,1};
    const int _N[3] = {N, N, N};
    tIndex::Init(_N, IsPer); // init of structure used for loops over nodes, to get neighboring nodes, etc.

    //omp_set_num_threads(8);

    // Model parameters
    double3 BackgroundFlow(0.,0.,0.);
    double TimeMax = 10.; // maximal integration time
    double CFL_requested = 0.4;
    double visc = 1./800.; // viscosity coefficient

    // Get the number of timesteps and adjust the timestep size
    double tau = CFL_requested*h; // will be adjusted to get integer number of timesteps
    int NumTimeSteps = TimeMax / tau;
    if(tau*NumTimeSteps < TimeMax) NumTimeSteps++;
    tau = TimeMax / NumTimeSteps;

    LinearSolverInit(); // HYPRE init. Used only for implicit or IMEX schemes

    vector<double3> velocity(NN);
    vector<double> pressure(NN), qressure(NN);

    // Initial data
    for(tIndex in=0; in<NN; ++in){
        double x = in.i[0]*h, y = in.i[1]*h, z = in.i[2]*h;
        velocity[in][0] = cos(x)*sin(y)*sin(z);
        velocity[in][1] = -sin(x)*cos(y)*sin(z);
        velocity[in][2] = 0.;
        velocity[in] += BackgroundFlow;
        pressure[in] = (cos(2*y)+cos(2*x))*(2.+cos(2*z))/16.;
        qressure[in] = 0.; // (vx*sin(2*x)+vy*sin(2*x))/4.;
    }


    if(1){ // Finite-difference method
        // Time integration
        tSRKTimeIntegrator<S_FD> FlowSolver(ARS_121(), 0 /*StabType*/);
        FlowSolver.TimeIntMethod = tTimeIntMethod::EXPLICIT;
        //FlowSolver.ViscScheme = tViscScheme::CROSS; FlowSolver.ViscOrder = 6;
        FlowSolver.ViscScheme = tViscScheme::CROSS;
        FlowSolver.GradDivOrder = FlowSolver.ConvOrder = 6;
        FlowSolver.MeshStep = double3(h,h,h);
        FlowSolver.visc = visc;
        FlowSolver.ConvMethod = tConvMethod::SKEWSYMMETRIC;
        FlowSolver.Init();
        FlowSolver.AllocDataArrays();

        FILE* F = fopen("k.dat", "wt");
        printf("Start of time integration (%i steps)\n", NumTimeSteps);
        const double C = 1./NN;
        double Kprev = FlowSolver.CalcKineticEnergy(velocity) / (8.*Pi*Pi*Pi);
        for(int iTimeStep=0; iTimeStep < NumTimeSteps; iTimeStep++){
            FlowSolver.Step(iTimeStep*tau, tau, velocity, pressure, qressure);
            double K = FlowSolver.CalcKineticEnergy(velocity) / (8.*Pi*Pi*Pi);

            double dKdt_output = (K-Kprev)/tau; // actual dK/dt (with O(tau) approximation)

            // Contribution of separate terms to dKdt
            double t = (iTimeStep+1)*tau;
            vector<double3> f(NN);
            FlowSolver.CalcFluxTerm_FD(t, velocity, pressure, true, false, f);
            double dKdt_conv = 0.0; // contribution of the convective term to dK/dt
            for(tIndex in=0; in<NN; ++in) dKdt_conv += C*DotProd(velocity[in],f[in]);
            FlowSolver.CalcFluxTerm_FD(t, velocity, pressure, false, true, f);
            double dKdt_visc = 0.0; // contribution of the viscous term to dK/dt
            for(tIndex in=0; in<NN; ++in) dKdt_visc += C*DotProd(velocity[in],f[in]);
            FlowSolver.ApplyGradient(pressure, f);
            double dKdt_pres = 0.0; // contribution of the pressure gradient term to dK/dt
            for(tIndex in=0; in<NN; ++in) dKdt_pres += C*DotProd(velocity[in],f[in]);
            printf("K = %.10f, dKdt = %.10f %.10f %.10f\n", K, dKdt_output, dKdt_visc, dKdt_conv+dKdt_pres);
            fprintf(F, "%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", (iTimeStep+1)*tau, K, dKdt_output, dKdt_conv, dKdt_visc, dKdt_pres, dKdt_output-(dKdt_conv+dKdt_visc+dKdt_pres));

            Kprev = K;
        }
        fclose(F);
    }
    else{ // P1-Galerkin method
        tSRKTimeIntegrator<S_Gal1B> FlowSolverGalerkin(ARS_343(), 1 /*StabType*/);
        FlowSolverGalerkin.IsFourier[0]=FlowSolverGalerkin.IsFourier[1]=FlowSolverGalerkin.IsFourier[2]=1;
        for(int i=0; i<=N; i++) FlowSolverGalerkin.X[0].push_back(double(i)*h);
        FlowSolverGalerkin.X[1]=FlowSolverGalerkin.X[0];
        FlowSolverGalerkin.X[2]=FlowSolverGalerkin.X[0];
        FlowSolverGalerkin.TimeIntMethod = tTimeIntMethod::EXPLICIT;
        FlowSolverGalerkin.visc = visc;
        FlowSolverGalerkin.ViscScheme = tViscScheme::GALERKIN;
        FlowSolverGalerkin.Init();
        FlowSolverGalerkin.AllocDataArrays();

        // Time integration
        FILE* F = fopen("k.dat", "wt");
        printf("Start of time integration (%i steps)\n", NumTimeSteps);
        vector<double3> f(NN), Mf(NN);
        const double C = 1./(8.*Pi*Pi*Pi);
        double Kprev = C*FlowSolverGalerkin.CalcKineticEnergy(velocity);
        for(int iTimeStep=0; iTimeStep < NumTimeSteps; iTimeStep++){
            FlowSolverGalerkin.Step(iTimeStep*tau, tau, velocity, pressure, qressure);
            double K = C*FlowSolverGalerkin.CalcKineticEnergy(velocity);

            double dKdt_output = (K-Kprev)/tau; // actual dK/dt (with O(tau) approximation)

            // Contribution of separate terms to dKdt
            double t = (iTimeStep+1)*tau;
            FlowSolverGalerkin.CalcConvTerm(velocity, f, true);
            double dKdt_conv = 0.0;
            for(tIndex in=0; in<NN; ++in) dKdt_conv += C*DotProd(velocity[in],f[in]);
            FlowSolverGalerkin.CalcFluxTerm(t, velocity, pressure, false, true, false, false, f); // viscous term - according to ViscScheme
            if(FlowSolverGalerkin.DoNotApplyMInvToVisc){ // nvm
                for(tIndex in=0; in<NN; ++in) f[in] /= (FlowSolverGalerkin.GetCellVolume(in));
                FlowSolverGalerkin.ApplyMassMatrix(f, Mf);
                f = Mf;
            }
            double dKdt_visc = 0.0;
            for(tIndex in=0; in<NN; ++in) dKdt_visc += C*DotProd(velocity[in],f[in]);
            FlowSolverGalerkin.ApplyGradient(pressure, f); // M^{-1} G p
            FlowSolverGalerkin.ApplyMassMatrix(f, Mf); // G p
            double dKdt_pres = 0.0;
            for(tIndex in=0; in<NN; ++in) dKdt_pres += C*DotProd(velocity[in],Mf[in]);
            printf("K = %.10f, dKdt = %.10f %.10f %.10f\n", K, dKdt_output, dKdt_visc, dKdt_conv+dKdt_pres);
            fprintf(F, "%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", (iTimeStep+1)*tau, K, dKdt_output, dKdt_conv, dKdt_visc, dKdt_pres, dKdt_output-(dKdt_conv+dKdt_visc+dKdt_pres));

            Kprev = K;
        }
        fclose(F);
    }

    FILE* f = fopen("output.vtk", "wt");
    fprintf(f, "# vtk DataFile Version 2.0\n");
    fprintf(f, "Volume example\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");
    fprintf(f, "DIMENSIONS %i %i %i\n", N, N, N);
    fprintf(f, "ASPECT_RATIO 1 1 1\n");
    fprintf(f, "ORIGIN 0 0 0\n");
    fprintf(f, "POINT_DATA %i\n", NN);
    fprintf(f, "SCALARS volume_scalars float 4\n");
    fprintf(f, "LOOKUP_TABLE default\n");
    for(tIndex in=0; in<NN; ++in){
        fprintf(f, "%e %e %e %e\n", velocity[in][0], velocity[in][1], velocity[in][2], pressure[in]);
    }
    fclose(f);

    LinearSolverFinalize();
}

// Here we check
//      conservation of the kinetic energy,
//      stability in the zero-viscosity limit,
//      zero-divergence condition
// TGV is just the initial data
void main_tgv3d_novisc(int N1D, const tIMEXMethod& IMEX, int StabType, const char* fname){
    // Computational mesh
    const int N = N1D; // number of nodes in one direction. Should be a power of 2 to use FFT
    const int NN = N*N*N;
    const double h = 2.*Pi/N;
    const int IsPer[3] = {1,1,1};
    const int _N[3] = {N, N, N};
    tIndex::Init(_N, IsPer); // init of structure used for loops over nodes, to get neighboring nodes, etc.

    //omp_set_num_threads(8);

    // Model parameters
    double3 BackgroundFlow(0.,0.,0.);
    double TimeMax = 6.; // maximal integration time (may be exceeded by ont timestep)
    double tau = StabType==0 ? h : 0.7*h;
    int NumTimeSteps = TimeMax / tau + 1;

    //LinearSolverInit(); // HYPRE init. Used only for implicit or IMEX schemes

    vector<double3> velocity(NN);
    vector<double> pressure(NN), qressure(NN);

    // Initial data
    for(tIndex in=0; in<NN; ++in){
        double x = in.i[0]*h, y = in.i[1]*h, z = in.i[2]*h;
        velocity[in][0] = cos(x)*sin(y)*sin(z);
        velocity[in][1] = -sin(x)*cos(y)*sin(z);
        velocity[in][2] = 0.;
        velocity[in] += BackgroundFlow;
        pressure[in] = (cos(2*y)+cos(2*x))*(2.+cos(2*z))/16.;
        qressure[in] = 0.; // (vx*sin(2*x)+vy*sin(2*x))/4.;
    }

    // Time integration
    tSRKTimeIntegrator<S_FD> FlowSolver(IMEX, StabType);
    FlowSolver.TimeIntMethod = tTimeIntMethod::EXPLICIT;
    FlowSolver.ViscScheme = tViscScheme::CROSS; FlowSolver.ViscOrder = 6;
    //FlowSolver.ViscScheme = tViscScheme::AES;
    FlowSolver.GradDivOrder = FlowSolver.ConvOrder = 6;
    FlowSolver.MeshStep = double3(h,h,h);
    FlowSolver.visc = 0.;
    FlowSolver.ConvMethod = tConvMethod::SKEWSYMMETRIC;
    FlowSolver.Init();
    FlowSolver.AllocDataArrays();

    vector<double> MassResid(NN);

    printf("Start of time integration (%i steps)\n", NumTimeSteps);
    double K0 = FlowSolver.CalcKineticEnergy(velocity) / (8.*Pi*Pi*Pi);
    FILE* f = fopen(fname, "wt");
    for(int iTimeStep=0; iTimeStep < NumTimeSteps; iTimeStep++){
        FlowSolver.Step(iTimeStep*tau, tau, velocity, pressure, qressure);
        double K = FlowSolver.CalcKineticEnergy(velocity) / (8.*Pi*Pi*Pi);
        if(iTimeStep<=1) K0 = K;
        double R0 = 0., R1 = 0.;
        FlowSolver.CalcContinuityResidual(velocity, vector<double>(), MassResid);
        for(int in=0; in<NN; in++) R0 += MassResid[in]*MassResid[in];
        FlowSolver.CalcContinuityResidual(velocity, StabType ? qressure : pressure, MassResid);
        for(int in=0; in<NN; in++) R1 += MassResid[in]*MassResid[in];
        R0 = sqrt(R0/NN);
        R1 = sqrt(R0/NN);
        printf("%.4f %f %f %f\n", (iTimeStep+1)*tau, K/K0, R0, R1);
        fprintf(f, "%.4f %f %e %e\n", (iTimeStep+1)*tau, K/K0, R0, R1);
    }
    fclose(f);
    //LinearSolverFinalize();
}

void main_tgv3d_novisc(const tIMEXMethod& IMEX, const char* fname){
    char _fname[255];
    for(int StabType=1; StabType<=1; StabType++){
        sprintf(_fname, "32_ST%c_%s", '0'+StabType, fname);
        main_tgv3d_novisc(32, IMEX, StabType, _fname);
        sprintf(_fname, "64_ST%c_%s", '0'+StabType, fname);
        main_tgv3d_novisc(64, IMEX, StabType, _fname);
        sprintf(_fname, "128_ST%c_%s", '0'+StabType, fname);
        main_tgv3d_novisc(128, IMEX, StabType, _fname);
    }
}

void main_tgv3d_novisc(){
    LinearSolverInit();
    main_tgv3d_novisc(ARS_343(), "ARS343.dat");
    main_tgv3d_novisc(ARK4(), "ARK4.dat");
    main_tgv3d_novisc(SI_IMEX_433(), "SI_IMEX_433.dat");
    LinearSolverFinalize();
}
