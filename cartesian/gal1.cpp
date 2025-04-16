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

static double MAX(double x, double y) { return (x>y) ? x:y; }


// Set zero normal component on boundaries (for a residual vector)
void S_Gal1::SetZeroNormalComponent(vector<double3>& R) const{
    if(!IsPer[0]){
        for(int iyz=0; iyz<N[1]*N[2]; iyz++) R[iyz*N[0]][0] = R[iyz*N[0]+N[0]-1][0] = 0.;
    }
    if(!IsPer[1]){
        for(int iz=0; iz<N[2]; iz++) for(int ix=0; ix<N[0]; ix++) R[iz*N[0]*N[1]+ix][1] = R[iz*N[0]*N[1]+N[0]*(N[1]-1)+ix][1] = 0.;
    }
    if(!IsPer[2]){
        for(int ixy=0; ixy<N[0]*N[1]; ixy++) R[ixy][2] = 0.;
        const int oxy = N[0]*N[1]*(N[2]-1);
        for(int ixy=0; ixy<N[0]*N[1]; ixy++) R[oxy+ixy][2] = 0.;
    }

    // Same buf in one loop
    #if 0
    for(tIndex in=0; in<NN; ++in){
        for(int idir=0; idir<Dim; idir++){
            if(IsPer[idir]) continue;
            if(in.i[idir]==0 || in.i[idir]==N[idir]-1) R[in][idir]=0.;
        }
    }
    #endif
}


void S_Gal1::FillMassMatrix(vector<double>& _L) const{
    int MAX_ROW_SIZE = (Dim==2) ? 9 : 27;
    for(tIndex in=0; in<NN; ++in){
        double* L = _L.data()+in*MAX_ROW_SIZE;
        for(int i=0; i<MAX_ROW_SIZE; i++) L[i]=0.;

        double h[3][2];
        for(int idir=0; idir<tIndex::Dim; idir++){
            h[idir][0] = HLeft(in, idir);
            h[idir][1] = HRight(in, idir);
        }

        if(tIndex::Dim==2){
            for(int oy=0; oy<=1; oy++){
                if(h[1][oy]==0.) continue;
                for(int ox=0; ox<=1; ox++){
                    if(h[0][ox]==0.) continue;
                    double V = h[0][ox]*h[1][oy]; // rectangle area
                    L[4] += C1_9*V;
                    L[4+(ox?1:-1)] += C1_18*V;
                    L[4+3*(oy?1:-1)] += C1_18*V;
                    L[4+3*(oy?1:-1)+(ox?1:-1)] += C1_36*V;
                }
            }
        }
        else{
            for(int oz=0; oz<=1; oz++){
                if(h[2][oz]==0.) continue;
                for(int oy=0; oy<=1; oy++){
                    if(h[1][oy]==0.) continue;
                    for(int ox=0; ox<=1; ox++){
                        if(h[0][ox]==0.) continue;
                        double V = h[0][ox]*h[1][oy]*h[2][oz]; // parallelepiped volume
                        L[13] += C1_27*V;
                        L[13+(ox?1:-1)] += C1_54*V;
                        L[13+3*(oy?1:-1)] += C1_54*V;
                        L[13+9*(oz?1:-1)] += C1_54*V;
                        L[13+3*(oy?1:-1)+(ox?1:-1)] += C1_108*V;
                        L[13+9*(oz?1:-1)+(ox?1:-1)] += C1_108*V;
                        L[13+9*(oz?1:-1)+3*(oy?1:-1)] += C1_108*V;
                        L[13+9*(oz?1:-1)+3*(oy?1:-1)+(ox?1:-1)] += C1_216*V;
                    }
                }
            }
        }
    }
}


void S_Gal1::ApplyMassMatrix(const vector<double3>& f, vector<double3>& Mf) const{
    #pragma omp parallel for
    for(int ic=0; ic<NN; ++ic){
        tIndex in(ic);
        Mf[in] = double3();
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
                    double V = h[0][ox]*h[1][oy]; // rectangle area
                    double3 F = C1_9*f[in] + C1_18*(f[in_x]+f[in_y]) + C1_36*f[in_xy];
                    Mf[in] += V*F;
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
                        double V = h[0][ox]*h[1][oy]*h[2][oz]; // parallelepiped volume
                        double3 F = C1_27*f[in] + C1_54*(f[in_x]+f[in_y]+f[in_z]) + C1_108*(f[in_xy]+f[in_xz]+f[in_yz]) + C1_216*f[in_xyz];
                        Mf[in] += V*F;
                    }
                }
            }
        }
    }
}


// Thomas algorithm for the 1D (in direction 'idir') mass matrix
// Matrix coefficients are h_{j-1/2}/6 (h_{j-1/2}+h_{j+1/2})*1/3 h_{j+1/2}/6
// Version for the Dirichlet boundary conditions (boundary values of x are zero)
void ThomasForM_Dirichlet(const vector<double>& c, vector<double>& x){
    const int n = c.size();
    vector<double> p(n);
    x[0] = 0.;
    p[0] = 0.;
    for(int i=1; i<n-1; i++){
        double delta = C1_3*(c[i+1]-c[i-1]) - C1_6*(c[i]-c[i-1])*p[i-1];
        delta = 1./delta;
        p[i] = C1_6*(c[i+1]-c[i])*delta;
        x[i] = (x[i]-C1_6*(c[i]-c[i-1])*x[i-1])*delta;
    }
    x[n-1] = 0.;
    for(int i=n-2; i>=0; i--){
        x[i] -= x[i+1]*p[i];
    }
}

void ThomasForM_Periodic(const vector<double>& c, const vector<double>& y, vector<double>& x){
    const int n = c.size()-1;
    if(n==1) { x[0]=y[0]/(c[1]-c[0]); return; }
    vector<double> p(n), xx(n);
    x[0] = 0.;
    xx[0] = 1.;
    p[0] = 0.;
    for(int i=1; i<n; i++){
        double delta = C1_3*(c[i+1]-c[i-1]) - C1_6*(c[i]-c[i-1])*p[i-1];
        delta = 1./delta;
        p[i] = C1_6*(c[i+1]-c[i])*delta;
        double ci = C1_6*(c[i]-c[i-1]);
        x[i] = (y[i]-ci*x[i-1])*delta;
        xx[i] = (y[i]-ci*xx[i-1])*delta;
    }
    xx[n-1] -= p[n-1];
    for(int i=n-2; i>=0; i--){
        x[i] -= x[i+1]*p[i];
        xx[i] -= xx[i+1]*p[i];
    }
    double a0 = C1_6*(c[n]-c[n-1]);
    double c0 = C1_6*(c[1]-c[0]);
    double b0 = 2.*(a0+c0);
    double alpha = -(a0*x[n-1]+b0*x[0]+c0*x[1] - y[0]) / (a0*(xx[n-1]-x[n-1])+b0*(xx[0]-x[0])+c0*(xx[1]-x[1]));
    for(int i=0; i<n; i++) x[i]=(1.-alpha)*x[i]+alpha*xx[i];
}


// Apply the inversed mass matrix (the result overrides the input)
// Mass matrix is the product Mx*My*Mz, so we apply the inversed 1D mass matrices one by one
void S_Gal1::ApplyMassMatrixInv(vector<double3>& f, tZeroAtBoundary Flag) const{
    if(Dim==2 && N[2]!=1) { printf("Dim==2 && N[2]!=1\n"); exit(0); }
    const int MaxN = MAX(MAX(N[0], N[1]), N[2]);

    #pragma omp parallel
    {
        vector<double> x(MaxN), y(MaxN);

        for(int idir=0; idir<Dim; idir++){
            const int jdir = (idir+1)%Dim;
            const int kdir = Dim==3 ? (idir+2)%3 : 2;
            #pragma omp for
            for(int jslice=0; jslice<N[jdir]; jslice++){
                for(int kslice=0; kslice<N[kdir]; kslice++){ // for 2D, this is a dummy loop
                    const int in0 = jslice*tIndex::shift[jdir] + kslice*tIndex::shift[kdir];
                    const int s = tIndex::shift[idir];

                    // Loop over velocity components
                    for(int icomponent=0; icomponent<Dim; icomponent++){
                        if(IsPer[idir]){
                            for(int i=0; i<N[idir]; i++) y[i] = f[in0+i*s][icomponent];
                            ThomasForM_Periodic(X[idir], y, x);
                        }
                        else{
                            for(int i=0; i<N[idir]; i++) x[i] = f[in0+i*s][icomponent];
                            int ZeroFlag = (Flag==tZeroAtBoundary::VECTOR) || (Flag==tZeroAtBoundary::NORMAL && icomponent==idir);
                            if(ZeroFlag) ThomasForM_Dirichlet(X[idir], x);
                            else ThomasForM(X[idir], x);
                        }
                        for(int i=0; i<N[idir]; i++) f[in0+i*s][icomponent] = x[i];
                    }
                }
            }
            #pragma omp barrier
        }
    }
}


// Apply the inversed mass matrix (the result overrides the input)
// Mass matrix is the product Mx*My*Mz, so we apply the inversed 1D mass matrices one by one
// Same as above, but for a scalar field (used for pressure conversion only)
void S_Gal1::ApplyMassMatrixInv(vector<double>& f) const{
    if(Dim==2 && N[2]!=1) { printf("Dim==2 && N[2]!=1\n"); exit(0); }
    const int MaxN = MAX(MAX(N[0], N[1]), N[2]);

    #pragma omp parallel
    {
        vector<double> x(MaxN), y(MaxN);

        for(int idir=0; idir<Dim; idir++){
            const int jdir = (idir+1)%Dim;
            const int kdir = Dim==3 ? (idir+2)%3 : 2;
            #pragma omp for
            for(int jslice=0; jslice<N[jdir]; jslice++){
                for(int kslice=0; kslice<N[kdir]; kslice++){ // for 2D, this is a dummy loop
                    const int in0 = jslice*tIndex::shift[jdir] + kslice*tIndex::shift[kdir];
                    const int s = tIndex::shift[idir];

                    if(IsPer[idir]){
                        for(int i=0; i<N[idir]; i++) y[i] = f[in0+i*s];
                        ThomasForM_Periodic(X[idir], y, x);
                    }
                    else{
                        for(int i=0; i<N[idir]; i++) x[i] = f[in0+i*s];
                        ThomasForM(X[idir], x);
                    }
                    for(int i=0; i<N[idir]; i++) f[in0+i*s] = x[i];
                }
            }
            #pragma omp barrier
        }
    }
}


double S_Gal1::CalcKineticEnergy(const vector<double3>& u) const{
    vector<double3> Mu(NN);
    ApplyMassMatrix(u, Mu);
    double sum=0.;
    for(int in=0; in<NN; in++) sum+=DotProd(u[in],Mu[in]);
    return 0.5*sum;
}

double3 S_Gal1::CalcIntegral(const vector<double3>& u) const{
    double3 sum;
    for(tIndex in=0; in<NN; ++in) sum+=u[in]*GetCellVolume(in);
    return sum;
}

// Normalize pressure
void S_Gal1::NormalizePressure(vector<double>& p) const{
    double psum = 0., ssum=0.;
    for(tIndex in=0; in<NN; ++in){
        double3 hbar = GetHbar(in);
        double V = hbar[0]*hbar[1]*hbar[2];
        psum += p[in]*V;
        ssum += V;
    }
    double p_shift = psum / ssum;
    for(tIndex in=0; in<NN; ++in){
        p[in] -= p_shift;
    }
}


// Kinematic pressure to effective pressure conversion (p += m*u^2)
// 3D not verfied yet
void S_Gal1::PressureToEffectivePressure(const vector<double3>& u, vector<double>& p, double m) const{
    vector<double> dp(NN);

    #pragma omp parallel for
    for(int ic=0; ic<NN; ic++){
        tIndex in(ic);
        double sum = 0.;

        double h[3][2];
        for(int idir=0; idir<tIndex::Dim; idir++){
            h[idir][0] = HLeft(in, idir);
            h[idir][1] = HRight(in, idir);
        }

        if(Dim==2){
            for(int oy=0; oy<=1; oy++){
                tIndex in_y = in.Neighb(1, oy);
                if(in_y==-1) continue;
                for(int ox=0; ox<=1; ox++){
                    tIndex in_x = in.Neighb(0, ox);
                    tIndex in_xy = in_y.Neighb(0, ox);
                    if(in_x==-1) continue;

                    double b = 9.*DotProd(u[in],u[in]) + 3.*DotProd(u[in_x],u[in_x]) + 3.*DotProd(u[in_y],u[in_y]) + DotProd(u[in_xy],u[in_xy]) +
                               6.*DotProd(u[in],u[in_x]) + 6.*DotProd(u[in],u[in_y]) + 2.*DotProd(u[in],u[in_xy]) +
                               2.*DotProd(u[in_x],u[in_y]) + 2.*DotProd(u[in_x],u[in_xy]) + 2.*DotProd(u[in_y],u[in_xy]);
                    sum += b * h[0][ox]*h[1][oy] / 144.;
                }
            }
        }
        else {
            tIndex i[8];
            i[0] = in;
            for(int oz=0; oz<=1; oz++){
                i[4] = in.Neighb(2, oz);
                if(i[4]==-1) continue;
                for(int oy=0; oy<=1; oy++){
                    i[2] = in.Neighb(1, oy);
                    if(i[2]==-1) continue;
                    i[6] = i[4].Neighb(1, oy);
                    for(int ox=0; ox<=1; ox++){
                        i[1] = in.Neighb(0, ox);
                        if(i[1]==-1) continue;
                        i[3] = i[2].Neighb(0, ox);
                        i[5] = i[4].Neighb(0, ox);
                        i[7] = i[6].Neighb(0, ox);

                        double b = 0.;
                        for(int ii=0; ii<8; ii++) for(int jj=0; jj<8; jj++){
                            double c = 1.;
                            if(!(ii&1) && !(jj&1)) c*=3.;
                            if(!(ii&2) && !(jj&2)) c*=3.;
                            if(!(ii&4) && !(jj&4)) c*=3.;
                            b += c*DotProd(u[i[ii]], u[i[jj]]);
                        }
                        sum += b * h[0][ox]*h[1][oy]*h[2][oz] / (12.*12.*12.);
                    }
                }
            }
        }
        dp[ic] = sum;
    }

    ApplyMassMatrixInv(dp);
    for(int in=0; in<NN; in++) p[in] += m*dp[in];
}



void S_Gal1::CalcConvTerm(const vector<double3>& u, vector<double3>& ConvTerm, int Nullify) const{
    double mult_divu = 0.;
    if(ConvMethod==tConvMethod::CONSERVATIVE || ConvMethod==tConvMethod::EMAC) mult_divu = 1.;
    if(ConvMethod==tConvMethod::SKEWSYMMETRIC) mult_divu = 0.5;

    #pragma omp parallel for
    for(int ic=0; ic<NN; ic++){
        tIndex in(ic);
        double3 sumflux;

        double h[3][2];
        for(int idir=0; idir<tIndex::Dim; idir++){
            h[idir][0] = HLeft(in, idir);
            h[idir][1] = HRight(in, idir);
        }

        if(Dim==2){
            for(int oy=0; oy<=1; oy++){
                tIndex in_y = in.Neighb(1, oy);
                if(in_y==-1) continue;
                for(int ox=0; ox<=1; ox++){
                    tIndex in_x = in.Neighb(0, ox);
                    tIndex in_xy = in_y.Neighb(0, ox);
                    if(in_x==-1) continue;
                    // Convective part
                    double3 gx = C1_72*(6.*u[in][0]+3.*u[in_x][0]+2.*u[in_y][0]+u[in_xy][0])*(u[in_x]-u[in]) +
                                 C1_72*(2.*u[in][0]+   u[in_x][0]+2.*u[in_y][0]+u[in_xy][0])*(u[in_xy]-u[in_y]);
                    double3 gy = C1_72*(6.*u[in][1]+3.*u[in_y][1]+2.*u[in_x][1]+u[in_xy][1])*(u[in_y]-u[in]) +
                                 C1_72*(2.*u[in][1]+   u[in_y][1]+2.*u[in_x][1]+u[in_xy][1])*(u[in_xy]-u[in_x]);
                    // Divu-part
                    double3 ggx = C1_72*(6.*u[in]+3.*u[in_x]+2.*u[in_y]+u[in_xy])*(u[in_x][0]-u[in][0]) +
                                  C1_72*(2.*u[in]+   u[in_x]+2.*u[in_y]+u[in_xy])*(u[in_xy][0]-u[in_y][0]);
                    double3 ggy = C1_72*(6.*u[in]+3.*u[in_y]+2.*u[in_x]+u[in_xy])*(u[in_y][1]-u[in][1]) +
                                  C1_72*(2.*u[in]+   u[in_y]+2.*u[in_x]+u[in_xy])*(u[in_xy][1]-u[in_x][1]);

                    gx += mult_divu*ggx;
                    gy += mult_divu*ggy;

                    if(!ox) { gx*=-1.; }
                    if(!oy) { gy*=-1.; }

                    gx *= h[1][oy];
                    gy *= h[0][ox];

                    sumflux += gx;
                    sumflux += gy;

                    if(ConvMethod==tConvMethod::EMAC){
                        double gggx = C1_72*DotProd(6.*u[in]+3.*u[in_x]+2.*u[in_y]+u[in_xy], u[in_x]-u[in]) +
                                      C1_72*DotProd(2.*u[in]+   u[in_x]+2.*u[in_y]+u[in_xy], u[in_xy]-u[in_y]);
                        double gggy = C1_72*DotProd(6.*u[in]+3.*u[in_y]+2.*u[in_x]+u[in_xy], u[in_y]-u[in]) +
                                      C1_72*DotProd(2.*u[in]+   u[in_y]+2.*u[in_x]+u[in_xy], u[in_xy]-u[in_x]);
                        if(!ox) gggx*=-1.;
                        if(!oy) gggy*=-1.;
                        gggx *= h[1][oy];
                        gggy *= h[0][ox];
                        sumflux += double3(gggx, gggy, 0.);
                    }
                }
            }
        }
        else {
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

                        #if 1
                        // Convective part
                        double3 gx = C1_864*(18.*u[in][0]+9.*u[in_x][0]+6.*u[in_y][0]+6.*u[in_z][0]+3.*u[in_xy][0]+3.*u[in_xz][0]+2.*u[in_yz][0]+u[in_xyz][0])*(u[in_x]-u[in]) +
                                     C1_864*( 6.*u[in][0]+3.*u[in_x][0]+6.*u[in_y][0]+2.*u[in_z][0]+3.*u[in_xy][0]+   u[in_xz][0]+2.*u[in_yz][0]+u[in_xyz][0])*(u[in_xy]-u[in_y]) +
                                     C1_864*( 6.*u[in][0]+3.*u[in_x][0]+2.*u[in_y][0]+6.*u[in_z][0]+   u[in_xy][0]+3.*u[in_xz][0]+2.*u[in_yz][0]+u[in_xyz][0])*(u[in_xz]-u[in_z]) +
                                     C1_864*( 2.*u[in][0]+   u[in_x][0]+2.*u[in_y][0]+2.*u[in_z][0]+   u[in_xy][0]+   u[in_xz][0]+2.*u[in_yz][0]+u[in_xyz][0])*(u[in_xyz]-u[in_yz]);
                        double3 gy = C1_864*(18.*u[in][1]+9.*u[in_y][1]+6.*u[in_x][1]+6.*u[in_z][1]+3.*u[in_xy][1]+3.*u[in_yz][1]+2.*u[in_xz][1]+u[in_xyz][1])*(u[in_y]-u[in]) +
                                     C1_864*( 6.*u[in][1]+3.*u[in_y][1]+6.*u[in_x][1]+2.*u[in_z][1]+3.*u[in_xy][1]+   u[in_yz][1]+2.*u[in_xz][1]+u[in_xyz][1])*(u[in_xy]-u[in_x]) +
                                     C1_864*( 6.*u[in][1]+3.*u[in_y][1]+2.*u[in_x][1]+6.*u[in_z][1]+   u[in_xy][1]+3.*u[in_yz][1]+2.*u[in_xz][1]+u[in_xyz][1])*(u[in_yz]-u[in_z]) +
                                     C1_864*( 2.*u[in][1]+   u[in_y][1]+2.*u[in_x][1]+2.*u[in_z][1]+   u[in_xy][1]+   u[in_yz][1]+2.*u[in_xz][1]+u[in_xyz][1])*(u[in_xyz]-u[in_xz]);
                        double3 gz = C1_864*(18.*u[in][2]+9.*u[in_z][2]+6.*u[in_y][2]+6.*u[in_x][2]+3.*u[in_yz][2]+3.*u[in_xz][2]+2.*u[in_xy][2]+u[in_xyz][2])*(u[in_z]-u[in]) +
                                     C1_864*( 6.*u[in][2]+3.*u[in_z][2]+6.*u[in_y][2]+2.*u[in_x][2]+3.*u[in_yz][2]+   u[in_xz][2]+2.*u[in_xy][2]+u[in_xyz][2])*(u[in_yz]-u[in_y]) +
                                     C1_864*( 6.*u[in][2]+3.*u[in_z][2]+2.*u[in_y][2]+6.*u[in_x][2]+   u[in_yz][2]+3.*u[in_xz][2]+2.*u[in_xy][2]+u[in_xyz][2])*(u[in_xz]-u[in_x]) +
                                     C1_864*( 2.*u[in][2]+   u[in_z][2]+2.*u[in_y][2]+2.*u[in_x][2]+   u[in_yz][2]+   u[in_xz][2]+2.*u[in_xy][2]+u[in_xyz][2])*(u[in_xyz]-u[in_xy]);
                        // Divu-part
                        double3 ggx = C1_864*(18.*u[in]+9.*u[in_x]+6.*u[in_y]+6.*u[in_z]+3.*u[in_xy]+3.*u[in_xz]+2.*u[in_yz]+u[in_xyz])*(u[in_x][0]-u[in][0]) +
                                      C1_864*( 6.*u[in]+3.*u[in_x]+6.*u[in_y]+2.*u[in_z]+3.*u[in_xy]+   u[in_xz]+2.*u[in_yz]+u[in_xyz])*(u[in_xy][0]-u[in_y][0]) +
                                      C1_864*( 6.*u[in]+3.*u[in_x]+2.*u[in_y]+6.*u[in_z]+   u[in_xy]+3.*u[in_xz]+2.*u[in_yz]+u[in_xyz])*(u[in_xz][0]-u[in_z][0]) +
                                      C1_864*( 2.*u[in]+   u[in_x]+2.*u[in_y]+2.*u[in_z]+   u[in_xy]+   u[in_xz]+2.*u[in_yz]+u[in_xyz])*(u[in_xyz][0]-u[in_yz][0]);
                        double3 ggy = C1_864*(18.*u[in]+9.*u[in_y]+6.*u[in_x]+6.*u[in_z]+3.*u[in_xy]+3.*u[in_yz]+2.*u[in_xz]+u[in_xyz])*(u[in_y][1]-u[in][1]) +
                                      C1_864*( 6.*u[in]+3.*u[in_y]+6.*u[in_x]+2.*u[in_z]+3.*u[in_xy]+   u[in_yz]+2.*u[in_xz]+u[in_xyz])*(u[in_xy][1]-u[in_x][1]) +
                                      C1_864*( 6.*u[in]+3.*u[in_y]+2.*u[in_x]+6.*u[in_z]+   u[in_xy]+3.*u[in_yz]+2.*u[in_xz]+u[in_xyz])*(u[in_yz][1]-u[in_z][1]) +
                                      C1_864*( 2.*u[in]+   u[in_y]+2.*u[in_x]+2.*u[in_z]+   u[in_xy]+   u[in_yz]+2.*u[in_xz]+u[in_xyz])*(u[in_xyz][1]-u[in_xz][1]);
                        double3 ggz = C1_864*(18.*u[in]+9.*u[in_z]+6.*u[in_y]+6.*u[in_x]+3.*u[in_yz]+3.*u[in_xz]+2.*u[in_xy]+u[in_xyz])*(u[in_z][2]-u[in][2]) +
                                      C1_864*( 6.*u[in]+3.*u[in_z]+6.*u[in_y]+2.*u[in_x]+3.*u[in_yz]+   u[in_xz]+2.*u[in_xy]+u[in_xyz])*(u[in_yz][2]-u[in_y][2]) +
                                      C1_864*( 6.*u[in]+3.*u[in_z]+2.*u[in_y]+6.*u[in_x]+   u[in_yz]+3.*u[in_xz]+2.*u[in_xy]+u[in_xyz])*(u[in_xz][2]-u[in_x][2]) +
                                      C1_864*( 2.*u[in]+   u[in_z]+2.*u[in_y]+2.*u[in_x]+   u[in_yz]+   u[in_xz]+2.*u[in_xy]+u[in_xyz])*(u[in_xyz][2]-u[in_xy][2]);

                        gx += mult_divu*ggx;
                        gy += mult_divu*ggy;
                        gz += mult_divu*ggz;

                        if(!ox) { gx*=-1.; }
                        if(!oy) { gy*=-1.; }
                        if(!oz) { gz*=-1.; }

                        gx *= h[1][oy]*h[2][oz];
                        gy *= h[0][ox]*h[2][oz];
                        gz *= h[0][ox]*h[1][oy];

                        sumflux += gx;
                        sumflux += gy;
                        sumflux += gz;

                        if(ConvMethod==tConvMethod::EMAC){
                            double gggx = C1_864*DotProd(18.*u[in]+9.*u[in_x]+6.*u[in_y]+6.*u[in_z]+3.*u[in_xy]+3.*u[in_xz]+2.*u[in_yz]+u[in_xyz], u[in_x]-u[in]) +
                                          C1_864*DotProd( 6.*u[in]+3.*u[in_x]+6.*u[in_y]+2.*u[in_z]+3.*u[in_xy]+   u[in_xz]+2.*u[in_yz]+u[in_xyz], u[in_xy]-u[in_y]) +
                                          C1_864*DotProd( 6.*u[in]+3.*u[in_x]+2.*u[in_y]+6.*u[in_z]+   u[in_xy]+3.*u[in_xz]+2.*u[in_yz]+u[in_xyz], u[in_xz]-u[in_z]) +
                                          C1_864*DotProd( 2.*u[in]+   u[in_x]+2.*u[in_y]+2.*u[in_z]+   u[in_xy]+   u[in_xz]+2.*u[in_yz]+u[in_xyz], u[in_xyz]-u[in_yz]);
                            double gggy = C1_864*DotProd(18.*u[in]+9.*u[in_y]+6.*u[in_x]+6.*u[in_z]+3.*u[in_xy]+3.*u[in_yz]+2.*u[in_xz]+u[in_xyz], u[in_y]-u[in]) +
                                          C1_864*DotProd( 6.*u[in]+3.*u[in_y]+6.*u[in_x]+2.*u[in_z]+3.*u[in_xy]+   u[in_yz]+2.*u[in_xz]+u[in_xyz], u[in_xy]-u[in_x]) +
                                          C1_864*DotProd( 6.*u[in]+3.*u[in_y]+2.*u[in_x]+6.*u[in_z]+   u[in_xy]+3.*u[in_yz]+2.*u[in_xz]+u[in_xyz], u[in_yz]-u[in_z]) +
                                          C1_864*DotProd( 2.*u[in]+   u[in_y]+2.*u[in_x]+2.*u[in_z]+   u[in_xy]+   u[in_yz]+2.*u[in_xz]+u[in_xyz], u[in_xyz]-u[in_xz]);
                            double gggz = C1_864*DotProd(18.*u[in]+9.*u[in_z]+6.*u[in_y]+6.*u[in_x]+3.*u[in_yz]+3.*u[in_xz]+2.*u[in_xy]+u[in_xyz], u[in_z]-u[in]) +
                                          C1_864*DotProd( 6.*u[in]+3.*u[in_z]+6.*u[in_y]+2.*u[in_x]+3.*u[in_yz]+   u[in_xz]+2.*u[in_xy]+u[in_xyz], u[in_yz]-u[in_y]) +
                                          C1_864*DotProd( 6.*u[in]+3.*u[in_z]+2.*u[in_y]+6.*u[in_x]+   u[in_yz]+3.*u[in_xz]+2.*u[in_xy]+u[in_xyz], u[in_xz]-u[in_x]) +
                                          C1_864*DotProd( 2.*u[in]+   u[in_z]+2.*u[in_y]+2.*u[in_x]+   u[in_yz]+   u[in_xz]+2.*u[in_xy]+u[in_xyz], u[in_xyz]-u[in_xy]);
                            if(!ox) gggx*=-1.;
                            if(!oy) gggy*=-1.;
                            if(!oz) gggz*=-1.;
                            gggx *= h[1][oy]*h[2][oz];
                            gggy *= h[0][ox]*h[2][oz];
                            gggz *= h[0][ox]*h[1][oy];
                            sumflux += double3(gggx, gggy, gggz);
                        }

                        #else // identical version with less FLOP but working slower for some reason
                        const double q = EnableEMAC ? 1. : 0.5;
                        double3 g, aux;

                        aux = C1_864*(18.*u[in]+9.*u[in_x]+6.*u[in_y]+6.*u[in_z]+3.*u[in_xy]+3.*u[in_xz]+2.*u[in_yz]+u[in_xyz]);
                        g  = aux[0]*(u[in_x]-u[in]) + q*aux*(u[in_x][0]-u[in][0]);
                        if(EnableEMAC) g[0] += DotProd(aux, u[in_x]-u[in]);
                        aux = C1_864*( 6.*u[in]+3.*u[in_x]+6.*u[in_y]+2.*u[in_z]+3.*u[in_xy]+   u[in_xz]+2.*u[in_yz]+u[in_xyz]);
                        g += aux[0]*(u[in_xy]-u[in_y]) + q*aux*(u[in_xy][0]-u[in_y][0]);
                        if(EnableEMAC) g[0] += DotProd(aux, u[in_xy]-u[in_y]);
                        aux = C1_864*( 6.*u[in]+3.*u[in_x]+2.*u[in_y]+6.*u[in_z]+   u[in_xy]+3.*u[in_xz]+2.*u[in_yz]+u[in_xyz]);
                        g += aux[0]*(u[in_xz]-u[in_z]) + q*aux*(u[in_xz][0]-u[in_z][0]);
                        if(EnableEMAC) g[0] += DotProd(aux, u[in_xz]-u[in_z]);
                        aux = C1_864*( 2.*u[in]+   u[in_x]+2.*u[in_y]+2.*u[in_z]+   u[in_xy]+   u[in_xz]+2.*u[in_yz]+u[in_xyz]);
                        g += aux[0]*(u[in_xyz]-u[in_yz]) + q*aux*(u[in_xyz][0]-u[in_yz][0]);
                        if(EnableEMAC) g[0] += DotProd(aux, u[in_xyz]-u[in_yz]);
                        if(!ox) g*=-1.;
                        g *= h[1][oy]*h[2][oz];
                        sumflux += g;

                        aux = C1_864*(18.*u[in]+9.*u[in_y]+6.*u[in_x]+6.*u[in_z]+3.*u[in_xy]+3.*u[in_yz]+2.*u[in_xz]+u[in_xyz]);
                        g  = aux[1]*(u[in_y]-u[in]) + q*aux*(u[in_y][1]-u[in][1]);
                        if(EnableEMAC) g[1] += DotProd(aux, u[in_y]-u[in]);
                        aux = C1_864*( 6.*u[in]+3.*u[in_y]+6.*u[in_x]+2.*u[in_z]+3.*u[in_xy]+   u[in_yz]+2.*u[in_xz]+u[in_xyz]);
                        g += aux[1]*(u[in_xy]-u[in_x]) + q*aux*(u[in_xy][1]-u[in_x][1]);
                        if(EnableEMAC) g[1] += DotProd(aux, u[in_xy]-u[in_x]);
                        aux = C1_864*( 6.*u[in]+3.*u[in_y]+2.*u[in_x]+6.*u[in_z]+   u[in_xy]+3.*u[in_yz]+2.*u[in_xz]+u[in_xyz]);
                        g += aux[1]*(u[in_yz]-u[in_z]) + q*aux*(u[in_yz][1]-u[in_z][1]);
                        if(EnableEMAC) g[1] += DotProd(aux, u[in_yz]-u[in_z]);
                        aux = C1_864*( 2.*u[in]+   u[in_y]+2.*u[in_x]+2.*u[in_z]+   u[in_xy]+   u[in_yz]+2.*u[in_xz]+u[in_xyz]);
                        g += aux[1]*(u[in_xyz]-u[in_xz]) + q*aux*(u[in_xyz][1]-u[in_xz][1]);
                        if(EnableEMAC) g[1] += DotProd(aux, u[in_xyz]-u[in_xz]);
                        if(!oy) g*=-1.;
                        g *= h[0][ox]*h[2][oz];
                        sumflux += g;

                        aux = C1_864*(18.*u[in]+9.*u[in_z]+6.*u[in_y]+6.*u[in_x]+3.*u[in_yz]+3.*u[in_xz]+2.*u[in_xy]+u[in_xyz]);
                        g  = aux[2]*(u[in_z]-u[in]) + q*aux*(u[in_z][2]-u[in][2]);
                        if(EnableEMAC) g[2] += DotProd(aux, u[in_z]-u[in]);
                        aux = C1_864*( 6.*u[in]+3.*u[in_z]+6.*u[in_y]+2.*u[in_x]+3.*u[in_yz]+   u[in_xz]+2.*u[in_xy]+u[in_xyz]);
                        g += aux[2]*(u[in_yz]-u[in_y]) + q*aux*(u[in_yz][2]-u[in_y][2]);
                        if(EnableEMAC) g[2] += DotProd(aux, u[in_yz]-u[in_y]);
                        aux = C1_864*( 6.*u[in]+3.*u[in_z]+2.*u[in_y]+6.*u[in_x]+   u[in_yz]+3.*u[in_xz]+2.*u[in_xy]+u[in_xyz]);
                        g += aux[2]*(u[in_xz]-u[in_x]) + q*aux*(u[in_xz][2]-u[in_x][2]);
                        if(EnableEMAC) g[2] += DotProd(aux, u[in_xz]-u[in_x]);
                        aux = C1_864*( 2.*u[in]+   u[in_z]+2.*u[in_y]+2.*u[in_x]+   u[in_yz]+   u[in_xz]+2.*u[in_xy]+u[in_xyz]);
                        g += aux[2]*(u[in_xyz]-u[in_xy]) + q*aux*(u[in_xyz][2]-u[in_xy][2]);
                        if(EnableEMAC) g[2] += DotProd(aux, u[in_xyz]-u[in_xy]);
                        if(!oz) g*=-1.;
                        g *= h[0][ox]*h[1][oy];
                        sumflux += g;
                        #endif
                    }
                }
            }
        }
        if(Nullify) ConvTerm[in] = sumflux * (-1.);
        else ConvTerm[in] -= sumflux;
    }
}


// Gradient - with zero normal component on boundary
void S_Gal1::ApplyGradient_MainGalerkin(const vector<double>& f, vector<double3>& R){
    const int MaxN = MAX(MAX(N[0], N[1]), N[2]);
    #pragma omp parallel
    {
        vector<double> x(MaxN), y(MaxN);

        for(int idir=0; idir<Dim; idir++){
            int jdir = (idir+1)%Dim;
            int kdir = Dim==3 ? (idir+2)%3 : 2;
            #pragma omp for
            for(int jslice=0; jslice<N[jdir]; jslice++){
                for(int kslice=0; kslice<N[kdir]; kslice++){ // for 2D, this is a dummy loop
                    int in0 = jslice*tIndex::shift[jdir] + kslice*tIndex::shift[kdir];
                    int s = tIndex::shift[idir];

                    if(IsPer[idir]){
                        y[0] = 0.5*(f[in0+1*s] - f[in0+(N[idir]-1)*s]);
                        for(int i=1; i<N[idir]-1; i++) y[i]=0.5*(f[in0+(i+1)*s] - f[in0+(i-1)*s]);
                        y[N[idir]-1] = 0.5*(f[in0] - f[in0+(N[idir]-2)*s]);
                        ThomasForM_Periodic(X[idir], y, x);
                    }
                    else{
                        x[0] = 0.;
                        for(int i=1; i<N[idir]-1; i++) x[i]=0.5*(f[in0+(i+1)*s] - f[in0+(i-1)*s]);
                        x[N[idir]-1] = 0.;
                        ThomasForM_Dirichlet(X[idir], x);
                    }
                    for(int i=0; i<N[idir]; i++) R[in0+i*s][idir]=x[i];
                }
            }
            #pragma omp barrier
        }
    }
}

// Divergence
void S_Gal1::ApplyDivergence_MainGalerkin(const vector<double3>& f, vector<double>& Df){
    #pragma omp parallel for
    for(int ic=0; ic<NN; ic++){
        tIndex in(ic);
        Df[in] = 0.;

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
                    if(in_x==-1) continue;
                    tIndex in_xy = in_y.Neighb(0, ox);
                    double Fx = C1_6*(f[in_x][0]-f[in][0]) + C1_12*(f[in_xy][0]-f[in_y][0]);
                    double Fy = C1_6*(f[in_y][1]-f[in][1]) + C1_12*(f[in_xy][1]-f[in_x][1]);
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
                        double Fx = C1_18*(f[in_x][0]-f[in][0]) + C1_36*(f[in_xy][0]-f[in_y][0]) + C1_36*(f[in_xz][0]-f[in_z][0]) + C1_72*(f[in_xyz][0]-f[in_yz][0]);
                        double Fy = C1_18*(f[in_y][1]-f[in][1]) + C1_36*(f[in_xy][1]-f[in_x][1]) + C1_36*(f[in_yz][1]-f[in_z][1]) + C1_72*(f[in_xyz][1]-f[in_xz][1]);
                        double Fz = C1_18*(f[in_z][2]-f[in][2]) + C1_36*(f[in_xz][2]-f[in_x][2]) + C1_36*(f[in_yz][2]-f[in_y][2]) + C1_72*(f[in_xyz][2]-f[in_xy][2]);
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
}



// Get a row of the matrix portrait -- for the full-stencil (9- in 2D, 27- in 3D) discretization
// Optionally, compress the data (in the input, there are spaces related to the non-existing nodes)
static void MyGetPortrait2D(int ic, int& PortraitSize, int* Portrait, const double* datain, double* dataout){
    tIndex in(ic);

    PortraitSize = 0;
    for(int oy=-1; oy<=1; oy++){
        tIndex in_y = (oy==-1) ? in.Neighb(1, 0) : ((oy==0) ? in : in.Neighb(1, 1));
        if(in_y==-1) continue;
        for(int ox=-1; ox<=1; ox++){
            tIndex in_xy = (ox==-1) ? in_y.Neighb(0, 0) : ((ox==0) ? in_y : in_y.Neighb(0, 1));
            if(in_xy==-1) continue;

            if(datain) dataout[PortraitSize]=datain[(oy+1)*3+ox+1];
            Portrait[PortraitSize++] = in_xy;
        }
    }
}
static void MyGetPortrait3D(int ic, int& PortraitSize, int* Portrait, const double* datain, double* dataout){
    tIndex in(ic);

    PortraitSize = 0;
    for(int oz=-1; oz<=1; oz++){
        tIndex in_z = (oz==-1) ? in.Neighb(2, 0) : ((oz==0) ? in : in.Neighb(2, 1));
        if(in_z==-1) continue;
        for(int oy=-1; oy<=1; oy++){
            tIndex in_y = (oy==-1) ? in_z.Neighb(1, 0) : ((oy==0) ? in_z : in_z.Neighb(1, 1));
            if(in_y==-1) continue;
            for(int ox=-1; ox<=1; ox++){
                tIndex in_xy = (ox==-1) ? in_y.Neighb(0, 0) : ((ox==0) ? in_y : in_y.Neighb(0, 1));
                if(in_xy==-1) continue;

                if(datain) dataout[PortraitSize]=datain[(oz+1)*9+(oy+1)*3+ox+1];
                Portrait[PortraitSize++] = in_xy;
            }
        }
    }
}

void S_Gal1::InitVelocitySystem(){
    // HYPRE data for the velocity system
    if(SimplifiedImplSystem){ // stencil = cross
        LinearSolverAlloc(1, N);
    }
    else{
        if(Dim==2) LinearSolverAlloc(1, N, 9, MyGetPortrait2D);
        else LinearSolverAlloc(1, N, 27, MyGetPortrait3D);
    }
}



// Calculate terms in the momentum equation (divided by cell volume)
void S_Gal1::CalcFluxTerm(double t, const vector<double3>& u, const vector<double>& p, bool DoConv, bool DoVisc, bool DoSourceE, bool DoSourceI, vector<double3>& f){
    for(int in=0; in<NN; in++) f[in]=double3();

    // Convection
    if(EnableConvection && DoConv){
        CalcConvTerm(u, f, false);
    }

    // Viscosity
    if(DoVisc && (visc>0. || visc_array.size())){
        if(ViscScheme==tViscScheme::CROSS) CalcViscTermCross(u, visc, f, true, false);
        if(ViscScheme==tViscScheme::AES) CalcViscTermAES(u, visc, visc_array, f, false);
        if(ViscScheme==tViscScheme::GALERKIN) CalcViscTermGalerkin(u, visc, visc_array, f, false);
    }

    int HasSource = (DoSourceE && SourceE!=NULL) || (DoSourceI && SourceI!=NULL);
    if(HasSource){
        vector<double3> s(NN);
        if(DoSourceE && SourceE!=NULL) SourceE(t, s);
        if(DoSourceI && SourceI!=NULL) SourceI(t, s);
        for(tIndex in=0; in<NN; ++in) f[in]+=s[in]*GetCellVolume(in);
    }
}


void S_Gal1::ImplicitStage_MainGalerkin(double time_stage, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u){
    if(TimeIntMethod==tTimeIntMethod::EXPLICIT) {
        u = ustar;
        // ??? Set boundary values to zero
        return;
    }


    int ROW_SIZE = SimplifiedImplSystem ? 7 : ((Dim==2) ? 9 : 27);
    vector<double> L(NN*ROW_SIZE); // matrix of the velocity equation
    vector<double> f[3]; // right-hand side
    for(int idir=0; idir<Dim; idir++) f[idir].resize(NN);
    vector<double> sol(NN);
    vector<double3> R(NN);
    vector<double3> MDU(NN);

    u = ustar;

    // u = ustar + tau_stage*M^{-1}*(L*u + s)
    // or, equivalently,
    // (M-tau_stage*L) (u - ustar) = tau_stage*(L*ustar + s)
    // Since our matrix inverse is only approximate, we write
    // u^{(n+1)} = u^{(n)} - [[(M-tau_stage*L)^{-1}]] R
    // where [[...]] is an approximating operator and
    // R = (M-tau_stage*L) (u - ustar) - tau_stage*(L*ustar + s) = M(u-ustar) - tau_stage*(L*u+s)

    int NumIters_loc = (TimeIntMethod==tTimeIntMethod::IMPLICIT) ? NumIters_Impl : NumIters_IMEX;
    for(int iter=0; iter<NumIters_loc; iter++){
        // Correction from the previous iterations
        for(int in=0; in<NN; ++in) R[in] = u[in]-ustar[in];
        ApplyMassMatrix(R, MDU);

        // Right-hand side
        bool DoConv = TimeIntMethod==tTimeIntMethod::IMPLICIT;
        CalcFluxTerm(time_stage, u, vector<double>(), DoConv, true, DoConv, true, R);

        double RHS_norm = 0.; // just a norm, not the L2 norm -- for debugging
        for(tIndex in=0; in<NN; ++in){
            if(in.IsWall()){ // On the Dirichlet boundary, we set DeltaU = u_bnd[in]-ustar[in]
                double3 ubnd = BoundaryValue ? BoundaryValue(time_stage, GetCoor(in)) : double3();
                for(int jdir=0; jdir<Dim; jdir++){
                    f[jdir][in] = -(ubnd[jdir]-u[in][jdir])*GetCellVolume(in);
                }
            }
            else{
                for(int jdir=0; jdir<Dim; jdir++){
                    f[jdir][in] = -R[in][jdir]*tau_stage + MDU[in][jdir];
                }
            }
            RHS_norm += f[0][in]*f[0][in] + f[1][in]*f[1][in];
            if(Dim==3) RHS_norm += f[2][in]*f[2][in];
        }
        //printf("iter=%i, residual norm=%e\n", iter, sqrt(RHS_norm));

        // Matrix
        if(SimplifiedImplSystem){
            for(tIndex in=0; in<NN; ++in){
                for(int i=0; i<ROW_SIZE; i++) L[in*ROW_SIZE+i]=0.;
                L[in*ROW_SIZE] = GetCellVolume(in);
            }
        }
        else{
            if(1){
                FillMassMatrix(L);
            }
            else{ // Test - use lumping even if we allocated the memory for a non-lumped matrix
                for(tIndex in=0; in<NN; ++in){
                    for(int i=0; i<ROW_SIZE; i++) L[in*ROW_SIZE+i]=0.;
                    L[in*ROW_SIZE + ROW_SIZE/2] = GetCellVolume(in);
                }
            }
        }

        int O[3]={1,3,9};
        int o[3]={-1,-3,-9};
        if(SimplifiedImplSystem){
            o[0]=1; o[1]=3; o[2]=5;
            O[0]=2; O[1]=4; O[2]=6;
        }

        if(ViscScheme==tViscScheme::CROSS){
            #pragma omp parallel for
            for(int ic=0; ic<NN; ic++){
                tIndex in(ic);
                double* pdiag = L.data() + in*ROW_SIZE + (SimplifiedImplSystem ? 0 : ROW_SIZE/2);
                if(in.IsWall()){
                    for(int i=0; i<ROW_SIZE; i++) L[in*ROW_SIZE+i]=0.;
                    *pdiag = GetCellVolume(in);
                    continue;
                }

                double3 Sbar = GetSbar(in);
                for(int idir=0; idir<Dim; idir++){
                    // Neighboring nodes -- well defined because wall nodes are skipped
                    tIndex jn = in.Neighb(idir,0);
                    tIndex kn = in.Neighb(idir,1);

                    double hL = HLeft(in, idir), hR = HRight(in, idir);
                    //double hbar = 0.5*(hL+hR);

                    // Viscosity
                    if(visc>0. || visc_array.size()){
                        double viscL = visc, viscR = visc;
                        if(visc_array.size()) { // this can't be for tViscScheme:CROSS, so nvm
                            viscL += AverageE2Edge(visc_array, jn, idir);
                            viscR += AverageE2Edge(visc_array, in, idir);
                        }
                        double mmL = viscL*tau_stage / hL * Sbar[idir];
                        double mmR = viscR*tau_stage / hR * Sbar[idir];
                        pdiag[o[idir]] -= mmL;
                        pdiag[O[idir]]  -= mmR;
                        pdiag[0]        += mmL+mmR;
                    }

                    // Convection
                    if(EnableConvection && DoConv){
                        double un = 0.5*(u[kn][idir]+u[in][idir])*tau_stage * Sbar[idir];
                        if(un>=0) pdiag[0] += un;
                        else pdiag[O[idir]] += un;
                        un = -0.5*(u[jn][idir]+u[in][idir])*tau_stage * Sbar[idir];
                        if(un>=0) pdiag[0] += un;
                        else pdiag[o[idir]] += un;
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
        else{
            for(int jdir=0; jdir<Dim; jdir++){
                #pragma omp parallel for
                for(int ic=0; ic<NN; ic++){
                    tIndex in(ic);
                    double* pdiag = L.data() + in*ROW_SIZE + (SimplifiedImplSystem ? 0 : ROW_SIZE/2);
                    if(in.IsWall()){
                        for(int i=0; i<ROW_SIZE; i++) L[in*ROW_SIZE+i]=0.;
                        *pdiag = GetCellVolume(in);
                        continue;
                    }

                    double3 Sbar = GetSbar(in);
                    for(int idir=0; idir<Dim; idir++){
                        // Neighboring nodes -- well defined because wall nodes are skipped
                        tIndex jn = in.Neighb(idir,0);
                        tIndex kn = in.Neighb(idir,1);

                        double hL = HLeft(in, idir), hR = HRight(in, idir);
                        //double hbar = 0.5*(hL+hR);

                        // Viscosity
                        if(visc>0. || visc_array.size()){
                            double viscL = visc, viscR = visc;
                            if(visc_array.size()) {
                                viscL += AverageE2Edge(visc_array, jn, idir);
                                viscR += AverageE2Edge(visc_array, in, idir);
                            }
                            if(idir==jdir) { viscL*=2.; viscR*=2.; }
                            double mmL = viscL*tau_stage / hL * Sbar[idir];
                            double mmR = viscR*tau_stage / hR * Sbar[idir];
                            pdiag[o[idir]] -= mmL;
                            pdiag[O[idir]] -= mmR;
                            pdiag[0]       += mmL+mmR;
                        }

                        // Convection
                        if(EnableConvection && DoConv){
                            double un = 0.5*(u[kn][idir]+u[in][idir])*tau_stage * Sbar[idir];
                            if(un>=0) pdiag[0] += un;
                            else pdiag[O[idir]] += un;
                            un = -0.5*(u[jn][idir]+u[in][idir])*tau_stage * Sbar[idir];
                            if(un>=0) pdiag[0] += un;
                            else pdiag[o[idir]] += un;
                        }
                    }
                }

                LinearSystemInit(1, L); // pass the matrix to the external linear algebra solver

                for(int in=0; in<NN; in++) sol[in]=0.;
                LinearSystemSolve(1 /*velocity system*/, sol, f[jdir], AMG_Tolerance_U, AMG_MaxIters_U);
                for(int in=0; in<NN; in++) u[in][jdir]-=sol[in];
            }
        }
    }
}

