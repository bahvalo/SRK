// P1-Galerkin method (not lumped)
// Dirichlet (u=0) or periodic boundary conditions
#include "asrk.h"
#include "gal1.h"

#include "linsolver.h"

static const double C1_9 = 1./9.;
static const double C1_18 = 1./18.;
static const double C1_36 = 1./36.;

static const double C1_27 = 1./27.;
static const double C1_54 = 1./54.;
static const double C1_108 = 1./108.;
static const double C1_216 = 1./216.;

static const double C1_6 = 1./6.;
static const double C1_12 = 1./12.;
static const double C1_72 = 1./72.;

static const double C1_3 = 1./3.;
static const double C2_3 = 1./3.;
static const double C1_864 = 1./864.;




// Implicit velocity stage
void S_Gal1A::ImplicitStage(double time_stage, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u){
    return ImplicitStage_MainGalerkin(time_stage, tau_stage, ustar, p, u);
}


// Explicit velocity stage
void S_Gal1A::ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat){
    bool DoConv = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoVisc = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    bool DoSourceE = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoSourceI = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    CalcFluxTerm(t, u, p, DoConv, DoVisc, DoSourceE, DoSourceI, kuhat);
    ApplyMassMatrixInv(kuhat, tZeroAtBoundary::NOTHING);
}

// Implicit velocity term - for methods of type CK (excluding ARS). Pressure should not be used
void S_Gal1A::ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku){
    vector<double3> buf(NN);
    bool DoConv = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoVisc = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    bool DoSourceE = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoSourceI = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    CalcFluxTerm(t, u, p, !DoConv, !DoVisc, !DoSourceE, !DoSourceI, ku);
    ApplyMassMatrixInv(ku, tZeroAtBoundary::NOTHING);
}

// Apply the gradient operator. No zero on walls!
void S_Gal1A::ApplyGradientMatrix(const vector<double>& f, vector<double3>& Gf) const{
    #pragma omp parallel for
    for(int ic=0; ic<NN; ic++){ tIndex in(ic);
        Gf[in] = double3();
        double h[3][2];
        for(int idir=0; idir<tIndex::Dim; idir++){
            h[idir][0] = HLeft(in, idir);
            h[idir][1] = HRight(in, idir);
        }

        if(tIndex::Dim==2){
            for(int oy=0; oy<=1; oy++){
                tIndex in_y = in.Neighb(1, oy);
                if(in_y==-1) continue;
                for(int ox=0; ox<=1; ox++){
                    tIndex in_x = in.Neighb(0, ox);
                    tIndex in_xy = in_y.Neighb(0, ox);
                    if(in_x==-1) continue;
                    double Fx = C1_6*(f[in_x]-f[in]) + C1_12*(f[in_xy]-f[in_y]);
                    double Fy = C1_6*(f[in_y]-f[in]) + C1_12*(f[in_xy]-f[in_x]);
                    if(!ox) Fx*=-1.;
                    if(!oy) Fy*=-1.;
                    Gf[in][0] += h[1][oy]*Fx;
                    Gf[in][1] += h[0][ox]*Fy;
                }
            }
        }
        else{
            for(int oz=0; oz<=1; oz++){
                if(h[2][oz]==0.) continue;
                tIndex in_z = in.Neighb(2, oz);
                for(int oy=0; oy<=1; oy++){
                    if(h[1][oy]==0.) continue;
                    tIndex in_y = in.Neighb(1, oy);
                    tIndex in_yz = in_z.Neighb(1, oy);
                    for(int ox=0; ox<=1; ox++){
                        if(h[0][ox]==0.) continue;
                        tIndex in_x = in.Neighb(0, ox);
                        tIndex in_xy = in_y.Neighb(0, ox);
                        tIndex in_xz = in_z.Neighb(0, ox);
                        tIndex in_xyz = in_yz.Neighb(0, ox);
                        double Fx = C1_18*(f[in_x]-f[in]) + C1_36*(f[in_xy]-f[in_y]) + C1_36*(f[in_xz]-f[in_z]) + C1_72*(f[in_xyz]-f[in_yz]);
                        double Fy = C1_18*(f[in_y]-f[in]) + C1_36*(f[in_xy]-f[in_x]) + C1_36*(f[in_yz]-f[in_z]) + C1_72*(f[in_xyz]-f[in_xz]);
                        double Fz = C1_18*(f[in_z]-f[in]) + C1_36*(f[in_xz]-f[in_x]) + C1_36*(f[in_yz]-f[in_y]) + C1_72*(f[in_xyz]-f[in_xy]);
                        if(!ox) Fx*=-1.;
                        if(!oy) Fy*=-1.;
                        if(!oz) Fz*=-1.;
                        Gf[in][0] += h[1][oy]*h[2][oz]*Fx;
                        Gf[in][1] += h[2][oz]*h[0][ox]*Fy;
                        Gf[in][2] += h[0][ox]*h[1][oy]*Fz;
                    }
                }
            }
        }
    }
}

void S_Gal1A::ApplyGradient(const vector<double>& f, vector<double3>& R) {
    ApplyGradientMatrix(f, R);
    ApplyMassMatrixInv(R, tZeroAtBoundary::NOTHING);
}

// Apply the divergence operator. D = -G^*, and the velocity space admits nonzero values on boundary
void S_Gal1A::UnapplyGradient(const vector<double3>& f, vector<double>& InvDf){
    vector<double> Df(NN);
    for(tIndex in=0; in<NN; ++in){
        double h[3][2];
        for(int idir=0; idir<tIndex::Dim; idir++){
            h[idir][0] = HLeft(in, idir);
            h[idir][1] = HRight(in, idir);
        }

        if(tIndex::Dim==2){
            for(int oy=0; oy<=1; oy++){
                if(h[1][oy]==0.) continue;
                tIndex in_y = in.Neighb(1, oy);
                for(int ox=0; ox<=1; ox++){
                    if(h[0][ox]==0.) continue;
                    tIndex in_x = in.Neighb(0, ox);
                    tIndex in_xy = in_y.Neighb(0, ox);
                    double Fx = C1_6*(f[in_x][0]+f[in][0]) + C1_12*(f[in_xy][0]+f[in_y][0]);
                    double Fy = C1_6*(f[in_y][1]+f[in][1]) + C1_12*(f[in_xy][1]+f[in_x][1]);
                    if(!ox) Fx*=-1.;
                    if(!oy) Fy*=-1.;
                    Df[in] += h[1][oy]*Fx;
                    Df[in] += h[0][ox]*Fy;
                }
            }
        }
        else{
            for(int oz=0; oz<=1; oz++){
                if(h[2][oz]==0.) continue;
                tIndex in_z = in.Neighb(2, oz);
                for(int oy=0; oy<=1; oy++){
                    if(h[1][oy]==0.) continue;
                    tIndex in_y = in.Neighb(1, oy);
                    tIndex in_yz = in_z.Neighb(1, oy);
                    for(int ox=0; ox<=1; ox++){
                        if(h[0][ox]==0.) continue;
                        tIndex in_x = in.Neighb(0, ox);
                        tIndex in_xy = in_y.Neighb(0, ox);
                        tIndex in_xz = in_z.Neighb(0, ox);
                        tIndex in_xyz = in_yz.Neighb(0, ox);
                        double Fx = C1_18*(f[in_x][0]+f[in][0]) + C1_36*(f[in_xy][0]+f[in_y][0]) + C1_36*(f[in_xz][0]+f[in_z][0]) + C1_72*(f[in_xyz][0]+f[in_yz][0]);
                        double Fy = C1_18*(f[in_y][1]+f[in][1]) + C1_36*(f[in_xy][1]+f[in_x][1]) + C1_36*(f[in_yz][1]+f[in_z][1]) + C1_72*(f[in_xyz][1]+f[in_xz][1]);
                        double Fz = C1_18*(f[in_z][2]+f[in][2]) + C1_36*(f[in_xz][2]+f[in_x][2]) + C1_36*(f[in_yz][2]+f[in_y][2]) + C1_72*(f[in_xyz][2]+f[in_xy][2]);
                        if(!ox) Fx*=-1.;
                        if(!oy) Fy*=-1.;
                        if(!oz) Fz*=-1.;
                        Df[in] += h[1][oy]*h[2][oz]*Fx;
                        Df[in] += h[2][oz]*h[0][ox]*Fy;
                        Df[in] += h[0][ox]*h[1][oy]*Fz;
                    }
                }
            }
        }
    }

    // Solve the system with the scaled Laplace operator (in FEM, with the stiffness matrix)
    LinearSystemSolve(0 /*pressure system*/, InvDf, Df, AMG_Tolerance_P, AMG_MaxIters_P);
}


void S_Gal1A::Init(){
    InitBase();

    // Init of pressure system
    PassDefaultPressureSystemToHYPRE(false);

    // Init of velocity system
    InitVelocitySystem();
}





void CheckThomas(){
    {
        const int n = 10;
        vector<double> c(n);
        c[0] = 0.;
        for(int i=1; i<n; i++) c[i]=c[i-1]+2.+sin(double(i*i));
        vector<double> xexact(n), x(n), y(n);
        xexact[0] = xexact[n-1] = 0.;
        for(int i=1; i<n-1; i++) xexact[i] = cos(double(i*i+10*i+4));
        for(int i=1; i<n-1; i++) x[i] = C1_6*(c[i]-c[i-1])*xexact[i-1] + C1_3*(c[i+1]-c[i-1])*xexact[i] + C1_6*(c[i+1]-c[i])*xexact[i+1];
        ThomasForM_Dirichlet(c, x);
        double sumerr = 0.;
        for(int i=0; i<n; i++) sumerr += fabs(x[i]-xexact[i]);
        printf("sumerr = %e\n", sumerr);
    }
    {
        const int n = 10;
        vector<double> c(n+1);
        c[0] = 0.;
        for(int i=1; i<=n; i++) c[i]=c[i-1]+2.+sin(double(i*i));
        vector<double> xexact(n), x(n), y(n);
        for(int i=0; i<n; i++) xexact[i] = cos(double(i*i+10*i+4));
        //for(int i=0; i<n; i++) xexact[i] = 1.;
        for(int i=0; i<n; i++){
            y[i]=0.;
            if(i==0) y[i] += C1_6*(c[n]-c[n-1])*(2*xexact[0]+xexact[n-1]);
            else y[i] += C1_6*(c[i]-c[i-1])*(2*xexact[i]+xexact[i-1]);
            if(i==n-1) y[i] += C1_6*(c[i+1]-c[i])*(2*xexact[i]+xexact[0]);
            else y[i] += C1_6*(c[i+1]-c[i])*(2*xexact[i]+xexact[i+1]);
        }
        ThomasForM_Periodic(c, y, x);
        double sumerr = 0.;
        for(int i=0; i<n; i++) sumerr += fabs(x[i]-xexact[i]);
        printf("sumerr = %e\n", sumerr);
    }
}


