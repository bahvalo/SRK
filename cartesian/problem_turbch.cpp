#include "asrk.h"
#include "fd2.h"
#include "gal1.h"
#include <omp.h>
#include <fstream>


static const int Coor_NonPeriodic = 2;
static const int Coor_PressureGrad = 0;
static double sour_dpdx = 0.;

void SourPressureGradient(double t, vector<double3>& f){
    const int NN = tIndex::N[0]*tIndex::N[1]*tIndex::N[2];
    #pragma omp parallel for
    for(int in=0; in<NN; ++in) f[in][Coor_PressureGrad] += sour_dpdx;
}


void DumpData(const char* fname, const S_Base& FlowSolver, const vector<double3>& velocity, const vector<double>& pressure, const vector<double>& qressure){
    const int* N = tIndex::N;
    const int NN = N[0]*N[1]*N[2];
    FILE* f = fopen(fname, "wt");
    fprintf(f, "# vtk DataFile Version 2.0\n");
    fprintf(f, "Volume example\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET RECTILINEAR_GRID\n");
    fprintf(f, "DIMENSIONS %i %i %i\n", N[0], N[1], N[2]);
    for(int idir=0; idir<3; idir++){
        fprintf(f, "%c_COORDINATES %i float\n", 'X'+idir, N[idir]);
        for(int i=0; i<N[idir]; i++) fprintf(f, "%e ", FlowSolver.X[idir][i]);
        fprintf(f, "\n");
    }
    fprintf(f, "POINT_DATA %i\n", NN);
    fprintf(f, "SCALARS volume_scalars float %i\n", FlowSolver.visc_array.size() ? 6 : 5);
    fprintf(f, "LOOKUP_TABLE default\n");
    for(tIndex in=0; in<NN; ++in){
        fprintf(f, "%e %e %e %e %e", velocity[in][0], velocity[in][1], velocity[in][2], pressure[in], qressure[in]);
        if(FlowSolver.visc_array.size()) fprintf(f, " %e", FlowSolver.AverageE2N(FlowSolver.visc_array, in));
        fprintf(f, "\n");
    }
    fclose(f);
}

// Reads input data from a file
// If the data is in the file corresponds to a coarser mesh, then repeats each value (assuming that the domain size is unchanged)
static void ReadData(const char* fname, const S_Base& FlowSolver, vector<double3>& velocity, vector<double>& pressure, vector<double>& qressure){
    std::ifstream file(fname);
    if(!file.is_open()) { printf("Error opening input file\n"); exit(0); }
    std::string line;
    int iz=-1, n[3]={-1,-1,-1};
    while(std::getline(file, line)) {
        if(line.substr(0, 10)==string("DIMENSIONS")){
            line = line.substr(11);
            sscanf(line.c_str(), "%i%i%i", n, n+1, n+2);
            if(n[2]!=FlowSolver.N[2]){ printf("NZ mismatch\n"); exit(0); }
            if(FlowSolver.N[0]%n[0] || FlowSolver.N[1]%n[1]) { printf("NX or NY is wrong\n"); exit(0); }
        }
        if(line.substr(0, 10)==string("LOOKUP_TAB")) {iz=0; break;}
    }
    if(iz==-1 || n[0]<0 || n[1]<0 || n[2]<0) { printf("Unexpected EOF (no data read)\n"); exit(0); }

    int MX = FlowSolver.N[0]/n[0], MY = FlowSolver.N[1]/n[1];
    for(iz=0; iz<n[2]; iz++){
        for(int iy=0; iy<n[1]; iy++){
            for(int ix=0; ix<n[0]; ix++){
                if(!std::getline(file, line)) { printf("Unexpected EOF\n"); exit(0); }
                int in=iz*FlowSolver.N[1]*FlowSolver.N[0] + iy*FlowSolver.N[0] + ix*MX;
                sscanf(line.c_str(), "%lf%lf%lf%lf%lf", &(velocity[in][0]), &(velocity[in][1]), &(velocity[in][2]), &(pressure[in]), &(qressure[in]));
                for(int iiy=0; iiy<MY; iiy++) for(int iix=0; iix<MX; iix++){
                    if(iiy==0 && iix==0) continue;
                    int jn = in + iiy*FlowSolver.N[0] + iix;
                    velocity[jn] = velocity[in];
                    pressure[jn] = pressure[in];
                    qressure[jn] = qressure[in];
                }
            }
        }
    }
    file.close();
    printf("Initial data read\n");
}


struct double9{
    double3 u;
    double uu[6]={0.,0.,0.,0.,0.,0.}; // XX, YY, ZZ, XY, XZ, YZ
    double nu_t = 0.;
};

static void IncrementAv(const vector<double3>& velocity, const vector<double>& nu_t, vector<double9>& av, double dt){
    const int NXY = tIndex::N[0]*tIndex::N[1];
    const double m = dt/NXY;
    #pragma omp parallel for
    for(int iz=0; iz<tIndex::N[2]; iz++){
        for(int in=iz*NXY; in<(iz+1)*NXY; in++){
            av[iz].u += velocity[in]*m;
            av[iz].uu[0] += velocity[in][0]*velocity[in][0]*m;
            av[iz].uu[1] += velocity[in][1]*velocity[in][1]*m;
            av[iz].uu[2] += velocity[in][2]*velocity[in][2]*m;
            av[iz].uu[3] += velocity[in][0]*velocity[in][1]*m;
            av[iz].uu[4] += velocity[in][0]*velocity[in][2]*m;
            av[iz].uu[5] += velocity[in][1]*velocity[in][2]*m;
            if(nu_t.size()) av[iz].nu_t += nu_t[in]*m;
        }
    }
}

static double CalcUfric(const vector<double9>& av, const vector<double>& Z, double t, double visc){
    double dudy = (av[1].u[0]-0.)/(Z[1]-Z[0]);
    if(dudy<0.) return 0.;
    if(t<1e-50) return 0.;
    dudy /= t;
    return sqrt(dudy*visc);
}

static double CalcUfric(const vector<double3>& velocity, const vector<double>& Z, double visc){
    const int NXY = tIndex::N[0]*tIndex::N[1];
    double u[3]={0.,0.,0.};
    for(int iz=1; iz<=2; iz++){
        for(int in=iz*NXY; in<(iz+1)*NXY; in++) u[iz] += velocity[in][0];
    }
    u[1] /= NXY;
    u[2] /= NXY;
    double dudy = (u[1]-0.)/(Z[1]-Z[0]);
    if(dudy<0.) return 0.;
    return sqrt(dudy*visc);
}

static void DumpProfiles(const vector<double>& Z, const vector<double9>& av, double visc, double u_fric, const char* fname, int IsZplus=0){
    FILE* F = fopen(fname, "wt");

    for(int iz=0; iz<int(Z.size()); iz++){
        double H = 0.5*(Z[Z.size()-1] - Z[0]);
        double dist_to_wall = (Z[iz]<H) ? Z[iz] : 2.*H-Z[iz];
        double y_plus = dist_to_wall*u_fric;
        if(!IsZplus) y_plus /= visc;

        double3 u = av[iz].u / u_fric;
        double uu[6];
        for(int i=0; i<6; i++) uu[i] = av[iz].uu[i] / (u_fric*u_fric);
        uu[0] = std::max(0., uu[0]-u[0]*u[0]);
        uu[1] = std::max(0., uu[1]-u[1]*u[1]);
        uu[2] = std::max(0., uu[2]-u[2]*u[2]);
        uu[3] -= u[0]*u[1];
        uu[4] -= u[0]*u[2];
        uu[5] -= u[1]*u[2];

        double nu_t; // turbunent viscosity is defined at half-integer points, need to average
        if(iz==0) nu_t = av[0].nu_t;
        else if(iz==int(Z.size())-1) nu_t = av[iz-1].nu_t;
        else nu_t = 0.5*(av[iz].nu_t + av[iz-1].nu_t);

        fprintf(F, "%e %e %e %e %e %e %e\n", y_plus, u[0], sqrt(uu[0]), sqrt(uu[1]), sqrt(uu[2]), uu[4], nu_t/visc);
    }
    fclose(F);
}


// DNS of the turbulent channel flow
// Here X is the streamwise direction, Y is the spanwise one, Z is the normal one
void main_turbch_180(){
    const double Re_tau = 180.; // u_tau*H/visc
    const double H = 1.; // channel half-height
    const double visc = 1./180.; // kinematic viscosity (affects the velocity scale only)
    const double u_tau = Re_tau*visc; // expected friction velocity
    sour_dpdx = u_tau*u_tau / H; // source in the momentum equation per unit volume

    // Creating mesh
    const int IsPer[3] = {1, 1, 0}; // periodic boundary conditions in X and Y, Dirichlet in Z
    const double Pi = 3.14159265358979323846;
    const double Hz = 2.*H, zmin = 0.;

    vector<double> Z; // mesh coordinates in normal direction ("y+")
    {
        const double y1 = 0.49; // first step
        const double ybulk = 2.2; // max step
        const double coeff = 1.05; // geometric progression coefficient

        Z.push_back(0.);
        Z.push_back(y1);
        while(1){
            double hz = (Z[Z.size()-1]-Z[Z.size()-2]) * coeff;
            if(hz>=ybulk) { printf("Mesh generation: max step reached at y+ = %.2f\n", Z[Z.size()-1]); break; }
            Z.push_back(Z[Z.size()-1] + hz);
        };
        int n = (Re_tau - Z[Z.size()-1]) / ybulk + 1;
        double hz_bulk = (Re_tau - Z[Z.size()-1])/n;
        for(int i=0; i<n-1; i++) Z.push_back(Z[Z.size()-1] + hz_bulk);
        Z.push_back(Re_tau);
        for(int i=Z.size()-2; i>=0; i--) Z.push_back(2.*Re_tau-Z[i]);
    }

    int N[3] = {256, 128, int(Z.size())}; // number of mesh nodes per direction. Numbers in X and Y should be powers of 2
    const double Hx = 4.*Pi*H, Hy = 4.*Pi*H/3; // domain size
    const double xmin = 0., ymin = -0.5*Hy; // offset, does not matter
    const int NN=N[0]*N[1]*N[2];
    tIndex::Init(N, IsPer); // init of structure used for loops over nodes, to get neighboring nodes, etc.

    omp_set_num_threads(96);

    // Choose a solver to use
    // General schemes
        //tSRKTimeIntegrator<S_FD2> FlowSolver(ARS_343(), 1 /*StabType*/);
        //tSRKTimeIntegrator<S_Gal1B> FlowSolver(ARS_343(), 1 /*StabType*/);
    // Special semi-implicit schemes (stresses across XY plane are taken implicitly, everything else is taken explicitly)
        tSRKTimeIntegrator<S_Gal1Bexim> FlowSolver(ARS_343(), 1 /*StabType*/);
        //tSRKTimeIntegrator<S_FD2exim> FlowSolver(ARS_232(), 1 /*StabType*/);

    // Specification of the method parameters
    FlowSolver.IsFourier[0] = FlowSolver.IsFourier[1] = 1; // Use FFT in X and Y to solve the pressure equation
    FlowSolver.ConvMethod = tConvMethod::SKEWSYMMETRIC;   // Convective fluxes approximation (may be changed)
    FlowSolver.TimeIntMethod = tTimeIntMethod::EXPLICIT; // The only choice for FD2 and Gal1B; no effect for *exim solvers
    FlowSolver.visc = visc;                              // Kinematic viscosity coefficient
    FlowSolver.SourceE = SourPressureGradient;
    FlowSolver.ViscScheme = tViscScheme::AES;

    // Pass the mesh to the solver
    for(int i=0; i<=N[0]; i++) FlowSolver.X[0].push_back(xmin+i*Hx/N[0]);
    for(int i=0; i<=N[1]; i++) FlowSolver.X[1].push_back(ymin+i*Hy/N[1]);
    for(int i=0; i<N[2]; i++) FlowSolver.X[2].push_back(zmin+Z[i]*visc); // i<N[2] is not a mistake (because there are no periodic conditions in Z)

    FlowSolver.Init();
    FlowSolver.AllocDataArrays();

    // Get the number of timesteps and adjust the timestep size
    double TimeMax = 10; // maximal integration time
    double tau = 0.8e-3;
    int MaxTimeSteps = 10000000;

    // Initial data
    vector<double3> velocity(NN);
    vector<double> pressure(NN), qressure(NN);
    if(1){ // Prescribe the initial data manually
        for(tIndex in=0; in<NN; ++in){
            velocity[in] = double3();
            double yplus = std::min(Z[in.i[2]], Re_tau*2-Z[in.i[2]]);
            double z = Z[in.i[2]]*visc;
            velocity[in][0] = 0.5*(z-zmin)*(zmin+Hz-z)*sour_dpdx/visc;
            velocity[in][1] = 1e-1*sin(in.i[0]*2.*Pi/N[0])*yplus*exp(-0.01*yplus*yplus);
            for(int idir=0; idir<3; idir++) if(!in.IsWall()) velocity[in][idir] += 1e-1*double(rand())/RAND_MAX;
            pressure[in] = 0.;
            qressure[in] = 0.; // dp/dt (used for StabType==1 only)
        }
    }
    if(1){ // Read from file
        ReadData("input.vtk", FlowSolver, velocity, pressure, qressure);
    }

    // Time averaged data
    vector<double9> av(N[2]);

    // Time integration
    printf("Start of time integration (N=%i,%i,%i)\n", N[0], N[1], N[2]);
    FILE* FILE_LOG = fopen("log.dat", "wt");
    const double TotalVolume = Hx * Hy * Hz;
    double t = 0.;
    int iTimeStep=0, isFinalStep=0;
    double3 FlowRate_prev = FlowSolver.CalcIntegral(velocity)/TotalVolume;
    for(iTimeStep=0; iTimeStep < MaxTimeSteps && !isFinalStep; iTimeStep++){
        if(t+tau>=TimeMax) { isFinalStep=1; }

        IncrementAv(velocity, FlowSolver.visc_array, av, 0.5*tau); // time averaging
        FlowSolver.Step(t, tau, velocity, pressure, qressure);
        IncrementAv(velocity, FlowSolver.visc_array, av, 0.5*tau); // time averaging

        t+=tau;

        double3 FlowRate = FlowSolver.CalcIntegral(velocity)/TotalVolume;

        int DoPrint = (iTimeStep%10==0) || isFinalStep;
        int DoPrintFile = (iTimeStep%10==0) || isFinalStep;
        if(DoPrint || DoPrintFile){
            double Ubulk = 0.;
            for(int in=0; in<NN; in++) Ubulk += velocity[in][0]*FlowSolver.GetCellVolume(in);
            Ubulk /= (Hx*Hy*Hz);
            double uf1 = CalcUfric(velocity, Z, 1.); // since Z is already normalized to "plus" values, pass 1. instead of viscosity coeff
            double uf2 = CalcUfric(av, Z, t, 1.); // same
            if(DoPrint) printf("T=%f Ubulk=%.04f Ufric=%.04f UfricAv=%.04f\n", t, Ubulk, uf1, uf2);
            if(DoPrintFile) fprintf(FILE_LOG, "%f %f %f %f\n", t, Ubulk, uf1, uf2);
        }

        int DoWriteOutput1 = (iTimeStep!=0 && iTimeStep%100==0) || isFinalStep;
        int DoWriteOutput2 = DoWriteOutput1;//(iTimeStep!=0 && iTimeStep%200==0) || isFinalStep;
        if(DoWriteOutput1){
            char fname[256];
            sprintf(fname, "q%05i.vtk", iTimeStep);
            DumpData(fname, FlowSolver, velocity, pressure, qressure);
        }
        if(DoWriteOutput2){
            char fname[256];
            vector<double9> av_cur(N[2]);
            IncrementAv(velocity, FlowSolver.visc_array, av_cur, 1.);
            sprintf(fname, "r%05i.dat", iTimeStep);
            DumpProfiles(Z, av_cur, visc, u_tau, fname, true /*Z are non-dimentional*/);
        }
        FlowRate_prev = FlowRate;
    }

    // For the EMAC method, convert pressure to static
    FlowSolver.NormalizePressure(pressure);
    DumpData("output.vtk", FlowSolver, velocity, pressure, qressure);
    fclose(FILE_LOG);

    if(t>0.){
        for(int iz=0; iz<N[2]; iz++){
            av[iz].u /= t;
            for(int i=0; i<6; i++) av[iz].uu[i] /= t;
            av[iz].nu_t /= t;
        }
        double dudy = 1.5*(av[1].u[0]-0.)/(Z[1]-Z[0])-0.5*(av[2].u[0]-av[1].u[0])/(Z[2]-Z[1]);
        if(dudy<0.) { printf("Error: negative friction\n"); exit(0); }
        double u_fric = sqrt(dudy);
        printf("u_fric: obtained=%.6f, expected=%.6f\n", u_fric, u_tau);
        DumpProfiles(Z, av, visc, u_tau, "res.dat", true);
    }
}
