// High-order finite difference discretization on uniform meshes (2D and 3D) with periodic boundary conditions
#include "asrk.h"
#include "fd.h"
#include "linsolver.h"
#include <omp.h>

const int m_max = 4; // max stencil size = m_max*2+1

static const double Pi = 3.14159265358979323846;
static const double C2_3 = 2./3.;
static const double C4_3 = 4./3.;
static const double C1_12 = 1./12.;
static const double C1_90 = 1./90.;
static const double C49_18 = 49./18.;


// Calculate terms in the momentum equation (divided by cell volume)
void S_FD::CalcFluxTerm_FD(double t, const vector<double3>& u, const vector<double>&, bool DoConv, bool DoVisc, vector<double3>& f){
    if(!(visc>0. || visc_array.size())) DoVisc = false;

    const int m = ConvOrder / 2;
    //for(tIndex in=0; in<NN; ++in){
    #pragma omp parallel for
    for(int _in=0; _in<NN; ++_in){
        tIndex in(_in);

        double3 sumflux;
        for(int idir=0; idir<Dim; idir++){
            const double invh = inv_MeshStep[idir];
            // Stencil
            tIndex inL[m_max], inR[m_max];
            inL[0] = in.Neighb(idir,0);
            inR[0] = in.Neighb(idir,1);
            for(int i=1; i<m; i++) { inL[i] = inL[i-1].Neighb(idir,0); inR[i] = inR[i-1].Neighb(idir,1); }

            const tIndex &jn=inL[0], &kn=inR[0], &jjn=inL[1], &kkn=inR[1], &jjjn=inL[2], &kkkn=inR[2];

            if(DoConv && EnableConvection){
                if(ConvMethod==tConvMethod::EMAC){
                    double3 conv_div;
                    double conv_add;
                    if(ConvOrder==6){
                        conv_div = 3./4.*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]) - 3./20.*(u[kkn]*u[kkn][idir] - u[jjn]*u[jjn][idir]) + 1./60.*(u[kkkn]*u[kkkn][idir] - u[jjjn]*u[jjjn][idir]);
                        conv_add = DotProd(3./4.*(u[kn] - u[jn]) - 3./20.*(u[kkn]-u[jjn]) + 1./60.*(u[kkkn]-u[jjjn]), u[in]);
                    }
                    else if(ConvOrder==4){
                        conv_div = 2./3.*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]) - 1./12.*(u[kkn]*u[kkn][idir] - u[jjn]*u[jjn][idir]);
                        conv_add = DotProd(2./3.*(u[kn] - u[jn]) - 1./12.*(u[kkn]-u[jjn]), u[in]);
                    }
                    else{
                        conv_div = 0.5*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]);
                        conv_add = 0.5*DotProd(u[kn] - u[jn], u[in]);
                    }
                    sumflux += conv_div * invh;
                    sumflux[idir] += conv_add * invh;
                }
                else{
                    double3 conv_div, conv_adv;
                    if(ConvOrder==6){
                        conv_div = 3./4.*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]) - 3./20.*(u[kkn]*u[kkn][idir] - u[jjn]*u[jjn][idir]) + 1./60.*(u[kkkn]*u[kkkn][idir] - u[jjjn]*u[jjjn][idir]);
                        conv_adv = u[in][idir] * (3./4.*(u[kn]-u[jn]) - 3./20.*(u[kkn]-u[jjn]) + 1./60.*(u[kkkn]-u[jjjn]));
                    }
                    else if(ConvOrder==4){
                        conv_div = 2./3.*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]) - 1./12.*(u[kkn]*u[kkn][idir] - u[jjn]*u[jjn][idir]);
                        conv_adv = u[in][idir] * (2./3.*(u[kn]-u[jn]) - 1./12.*(u[kkn]-u[jjn]));
                    }
                    else{
                        conv_div = 0.5*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]);
                        conv_adv = u[in][idir] * 0.5*(u[kn]-u[jn]);
                    }
                    if(ConvMethod==tConvMethod::SKEWSYMMETRIC){
                        // Skew-symmetric (energy-conservative) convection discretization
                        // If the divergence approximation is consistent (i. e. based on the same derivative approximation), it is also momentum-conservative
                        sumflux += 0.5*(conv_div + conv_adv) * invh;
                    }
                    else if(ConvMethod==tConvMethod::CONVECTIVE){
                        // This formulation is not energy-preserving even if the divergence approximation is consistent with the fluxes approximation
                        sumflux += conv_adv * invh;
                    }
                    else if(ConvMethod==tConvMethod::CONSERVATIVE){
                        // This formulation is not energy-preserving even if the divergence approximation is consistent with the fluxes approximation
                        sumflux += conv_div * invh;
                    }
                    else{ printf("Unknown configuration\n"); exit(0); }
                }
            }

            // Viscosity
            if(DoVisc && ViscScheme==tViscScheme::CROSS){
                if(visc_array.size()) { printf("tViscScheme::CROSS does not work for variable viscosity\n"); exit(0); }
                double c = visc*invh*invh;
                if(ViscOrder==6){
                    sumflux -= c*(C1_90*(u[kkkn]+u[jjjn])-0.15*(u[kkn]+u[jjn])+1.5*(u[kn]+u[jn])-C49_18*u[in]);
                }
                else if(ViscOrder==4){
                    sumflux -= c*(C4_3*(u[kn]+u[jn]) - C1_12*(u[kkn]+u[jjn]) - 2.5*u[in]);
                }
                else{
                    sumflux -= c*(u[kn]+u[jn]-2.*u[in]);
                }
            }
        }
        f[in] = sumflux*(-1.);
    }

    // Non-finite-difference approximation of viscous terms (necessary for variable viscosity)
    if(DoVisc && ViscScheme!=tViscScheme::CROSS){
        vector<double3> fv(NN);
        if(ViscScheme==tViscScheme::GALERKIN) CalcViscTermGalerkin(u, visc, visc_array, fv, true);
        if(ViscScheme==tViscScheme::AES) CalcViscTermAES(u, visc, visc_array, fv, true);
        for(tIndex in=0; in<NN; ++in) f[in] += fv[in] / GetCellVolume(in);
    }
}


void S_FD::ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat){
    if(TimeIntMethod==tTimeIntMethod::IMPLICIT) {
        for(int in=0; in<NN; in++) kuhat[in]=double3();
    }
    else{
        CalcFluxTerm_FD(t, u, p, true, TimeIntMethod==tTimeIntMethod::EXPLICIT, kuhat);
    }
}

void S_FD::ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku){
    if(TimeIntMethod==tTimeIntMethod::EXPLICIT) {
        for(int in=0; in<NN; in++) ku[in]=double3();
    }
    else{
        CalcFluxTerm_FD(t, u, p, TimeIntMethod==tTimeIntMethod::IMPLICIT, true, ku);
    }
}

// Implicit velocity stage
void S_FD::ImplicitStage(double time_stage, double tau_stage, const vector<double3>& ustar, const vector<double>&, vector<double3>& u){
    if(TimeIntMethod==tTimeIntMethod::EXPLICIT) {
        u = ustar;
        return;
    }

    vector<double> L(NN*7); // matrix of the velocity equation
    vector<double> f[3]; // right-hand side
    for(int idir=0; idir<Dim; idir++) f[idir].resize(NN);
    vector<double> sol(NN);

    u = ustar;

    const int NumIters_Impl = 20;
    int NumIters_loc = (TimeIntMethod==tTimeIntMethod::IMPLICIT) ? NumIters_Impl : (ViscOrder/2);
    for(int iter=0; iter<NumIters_loc; iter++){
        vector<double3> R;
        R.resize(NN);
        // Right-hand side
        bool DoConv = TimeIntMethod==tTimeIntMethod::IMPLICIT;
        CalcFluxTerm_FD(time_stage, u, vector<double>(), DoConv, true, R);
        double RHS_norm = 0.; // just a norm, not the L2 norm
        for(tIndex in=0; in<NN; ++in){
            for(int jdir=0; jdir<Dim; jdir++){
                f[jdir][in] = -R[in][jdir]*tau_stage + (u[in][jdir]-ustar[in][jdir]);
            }
            RHS_norm += f[0][in]*f[0][in] + f[1][in]*f[1][in];
        }
        //printf("iter=%i, residual norm=%e\n", iter, sqrt(RHS_norm));

        // Matrix
        for(tIndex in=0; in<NN; ++in){
            L[in*7] = 1.;
            L[in*7+1]=L[in*7+2]=L[in*7+3]=L[in*7+4]=L[in*7+5]=L[in*7+6]=0.;
            if(in.IsWall()) continue;

            const double3& hbar = MeshStep;
            for(int idir=0; idir<Dim; idir++){
                // Neighboring nodes -- well defined because wall nodes are skipped
                tIndex jn = in.Neighb(idir,0);
                tIndex kn = in.Neighb(idir,1);

                // Viscosity
                const vector<double>& visc_coeff = visc_array;
                if(visc>0. || visc_coeff.size()){
                    double viscL = visc, viscR = visc;
                    if(ViscScheme!=tViscScheme::CROSS) { printf("S_FD: implicit viscosity is implemented for ViscScheme::CROSS only\n"); exit(0);}
                    double mmL = viscL*tau_stage*inv_MeshStep[idir]*inv_MeshStep[idir];
                    double mmR = viscR*tau_stage*inv_MeshStep[idir]*inv_MeshStep[idir];
                    L[in*7+idir*2+1] -= mmL;
                    L[in*7+idir*2+2] -= mmR;
                    L[in*7]          += mmL+mmR;
                }

                // Convection
                if(EnableConvection && DoConv){
                    double un = 0.5*(u[kn][idir]+u[in][idir])*tau_stage/hbar[idir];
                    if(un>=0) L[in*7] += un;
                    else L[in*7+idir*2+2] += un;
                    un = 0.5*(u[jn][idir]+u[in][idir])*tau_stage/hbar[idir];
                    if(un>=0) L[in*7] += un;
                    else L[in*7+idir*2+1] += un;
                }
            }
        }

        LinearSystemInit(1, L); // pass the matrix to the external linear algebra solver

        for(int jdir=0; jdir<Dim; jdir++){
            for(int in=0; in<NN; in++) sol[in]=0.;
            LinearSystemSolve(1 /*velocity system*/, sol, f[jdir], AMG_Tolerance_U, AMG_MaxIters_U);
            for(int in=0; in<NN; in++) u[in][jdir]-=sol[in];
        }
    }
}


void S_FD::ApplyGradient(const vector<double>& a, vector<double3>& R) {
    const int m = GradDivOrder / 2;

    #pragma omp parallel for
    for(int _in=0; _in<NN; ++_in){
        tIndex in(_in);
//    for(tIndex in=0; in<NN; ++in){
        for(int idir=0; idir<Dim; idir++){
            tIndex inL[m_max], inR[m_max];
            inL[0] = in.Neighb(idir,0);
            inR[0] = in.Neighb(idir,1);
            for(int i=1; i<m; i++) { inL[i] = inL[i-1].Neighb(idir,0); inR[i] = inR[i-1].Neighb(idir,1); }
            const tIndex &jn=inL[0], &kn=inR[0], &jjn=inL[1], &kkn=inR[1], &jjjn=inL[2], &kkkn=inR[2];

            if(GradDivOrder==6){
                R[in][idir] = 0.75*(a[kn]-a[jn]) - 0.15*(a[kkn] - a[jjn]) + (1./60.)*(a[kkkn]-a[jjjn]);
            }
            else if(GradDivOrder==4){
                R[in][idir] = C2_3*(a[kn] - a[jn]) - C1_12*(a[kkn] - a[jjn]);
            }
            else{
                R[in][idir] = 0.5*(a[kn] - a[jn]);
            }
            R[in][idir] *= inv_MeshStep[idir];
        }
    }
}

void S_FD::UnapplyGradient(const vector<double3>& v, vector<double>& Lm1Dv){
    const int m = GradDivOrder / 2;
    vector<double> Dv(NN);

    #pragma omp parallel for
    for(int _in=0; _in<NN; ++_in){
        tIndex in(_in);
//    for(tIndex in=0; in<NN; ++in){
        Dv[in] = 0.;
        for(int idir=0; idir<Dim; idir++){
            tIndex inL[m_max], inR[m_max];
            inL[0] = in.Neighb(idir,0);
            inR[0] = in.Neighb(idir,1);
            for(int i=1; i<m; i++) { inL[i] = inL[i-1].Neighb(idir,0); inR[i] = inR[i-1].Neighb(idir,1); }
            const tIndex &jn=inL[0], &kn=inR[0], &jjn=inL[1], &kkn=inR[1], &jjjn=inL[2], &kkkn=inR[2];

            double dDv;
            if(GradDivOrder==6){
                dDv = 0.75*(v[kn][idir]-v[jn][idir]) - 0.15*(v[kkn][idir] - v[jjn][idir]) + (1./60.)*(v[kkkn][idir]-v[jjjn][idir]);
            }
            else if(GradDivOrder==4){
                dDv = C2_3*(v[kn][idir] - v[jn][idir]) - C1_12*(v[kkn][idir] - v[jjn][idir]);
            }
            else{
                dDv = 0.5*(v[kn][idir] - v[jn][idir]);
            }
            Dv[in] += dDv * inv_MeshStep[idir];
        }
    }

    LinearSystemSolveFFT(Lm1Dv, Dv);
}




void S_FD::LinearSystemSolveFFT_Init(){
    const int* N = tIndex::N;
    Vphys = (fftw_complex*) fftw_malloc(NN * sizeof(fftw_complex));
    Vspec = (fftw_complex*) fftw_malloc(NN * sizeof(fftw_complex));
    if(tIndex::Dim==3){
        p1 = fftw_plan_dft_3d(N[2], N[1], N[0], Vphys, Vspec, FFTW_FORWARD, FFTW_MEASURE);
        p2 = fftw_plan_dft_3d(N[2], N[1], N[0], Vspec, Vphys, FFTW_BACKWARD, FFTW_MEASURE);
    }
    else{
        p1 = fftw_plan_dft_2d(N[1], N[0], Vphys, Vspec, FFTW_FORWARD, FFTW_MEASURE);
        p2 = fftw_plan_dft_2d(N[1], N[0], Vspec, Vphys, FFTW_BACKWARD, FFTW_MEASURE);
    }
    FFTW_initialized = 1;
}

void S_FD::LinearSystemSolveFFT(vector<double>& X, const vector<double>& f){
    for(int in=0; in<NN; in++) { Vphys[in][0]=f[in]; Vphys[in][1]=0.; }
    fftw_execute(p1);

    vector<double> cosphi[3];
    for(int idir=0; idir<3; idir++){
        cosphi[idir].resize(tIndex::N[idir]*4); // it has the period N[idir]
        double phi_over_i = 2.*Pi/tIndex::N[idir];
        for(int i=0; i<tIndex::N[idir]*4; i++) cosphi[idir][i] = cos(phi_over_i*i);
    }

    Vspec[0][0] = Vspec[0][1] = 0.; // zeroth mode, which corresponds to the average pressure
    const double mx=inv_MeshStep[0]*inv_MeshStep[0];
    const double my=inv_MeshStep[1]*inv_MeshStep[1];
    const double mz=Dim==2 ? 1. : inv_MeshStep[2]*inv_MeshStep[2];
    const int* N = tIndex::N;
    for(int kx=0; kx<N[0]; kx++) for(int ky=0; ky<N[1]; ky++) for(int kz=0; kz<N[2]; kz++){
        if(kx==0 && ky==0 && kz==0) continue;
        double m = 0.;
        if(GradDivOrder==6){
            m += 1.5*mx*(cosphi[0][kx]-1.) - 0.15*mx*(cosphi[0][kx*2]-1.) + C1_90*mx*(cosphi[0][kx*3]-1.);
            m += 1.5*my*(cosphi[1][ky]-1.) - 0.15*my*(cosphi[0][ky*2]-1.) + C1_90*my*(cosphi[0][ky*3]-1.);
            m += 1.5*mz*(cosphi[2][kz]-1.) - 0.15*mz*(cosphi[0][kz*2]-1.) + C1_90*mz*(cosphi[0][kz*3]-1.);
        }
        else if(GradDivOrder==4){
            m += C4_3*mx*(cosphi[0][kx]-1.) - C1_12*mx*(cosphi[0][kx*2]-1.);
            m += C4_3*my*(cosphi[1][ky]-1.) - C1_12*my*(cosphi[0][ky*2]-1.);
            m += C4_3*mz*(cosphi[2][kz]-1.) - C1_12*mz*(cosphi[0][kz*2]-1.);
        }
        else{
            m += mx*(cosphi[0][kx]-1.);
            m += my*(cosphi[1][ky]-1.);
            m += mz*(cosphi[2][kz]-1.);
        }
        double inv_m = 1./(2.*m);
        Vspec[kz*N[1]*N[0]+ky*N[0]+kx][0] *= inv_m;
        Vspec[kz*N[1]*N[0]+ky*N[0]+kx][1] *= inv_m;
    }

    fftw_execute(p2);

    double inv_NN = 1./NN;
    for(int in=0; in<NN; in++) X[in]=Vphys[in][0]*inv_NN;
}


void S_FD::Init(){
    InitBase();
    for(int idir=0; idir<Dim; idir++){
        if(!IsPer[idir]){ printf("Only for periodic conditions\n"); exit(0); }
        inv_MeshStep[idir] = 1./MeshStep[idir];
        // Fill coordinates if they are not filled yet (they are used, for instance, for FE viscosity approximations)
        if(!X[idir].size()) for(int i=0; i<=N[idir]; i++) X[idir].push_back(i*MeshStep[idir]);
    }

    LinearSystemSolveFFT_Init();
    LinearSolverAlloc(1, tIndex::N); // HYPRE data for the velocity system
}

// Shift pressure so that is has zero average
void S_FD::NormalizePressure(vector<double>& p) const{
    double psum = 0.;
    for(int in=0; in<NN; ++in) psum += p[in];
    double p_shift = psum / NN;
    for(int in=0; in<NN; ++in) p[in] -= p_shift;
}

double S_FD::CalcKineticEnergy(const vector<double3>& u) const{
    double V = MeshStep[0]*MeshStep[1]*MeshStep[2];
    double sum=0.;
    for(int in=0; in<NN; ++in) sum+=DotProd(u[in],u[in]);
    return 0.5*sum*V;
}
double3 S_FD::CalcIntegral(const vector<double3>& u) const{
    double V = MeshStep[0]*MeshStep[1]*MeshStep[2];
    double3 sum;
    for(int in=0; in<NN; ++in) sum+=u[in];
    return sum*V;
}

