// Abstract base class for FD2 and Gal1 with common methods
#include "base.h"

#include "linsolver.h"

int tIndex::Dim, tIndex::N[3], tIndex::IsPer[3], tIndex::shift[3], tIndex::Shift[3]; // static members of tIndex

static const double Pi = 3.14159265358979323846;
static const double C1_3 = 1./3.;
static const double C1_6 = 1./6.;
static const double C1_9 = 1./9.;
static const double C1_18 = 1./18.;
static const double C1_36 = 1./36.;

// Returns the index of an element near node `ind`. ox,oy,oz=0,1 are directional markers
int GetElementIndex(const tIndex& ind, int ox, int oy, int oz){
    tIndex in = ind;
    if(!ox) in = in.Neighb(0, 0);
    if(!oy) in = in.Neighb(1, 0);
    if(tIndex::Dim==3 && !oz) in = in.Neighb(2, 0);
    return in.in;
}

double S_Base::AverageE2N(const vector<double>& A, const tIndex& in) const{
    double h[3][2];
    for(int idir=0; idir<Dim; idir++){
        h[idir][0] = HLeft(in, idir);
        h[idir][1] = HRight(in, idir);
    }

    double sum = 0.;
    for(int oz=0; oz<=1; oz++){
        if(oz==0 && Dim==2) continue;
        for(int oy=0; oy<=1; oy++)for(int ox=0; ox<=1; ox++){
            tIndex jn = in;
            if(!ox){ jn = jn.Neighb(0, 0); if(jn==-1) continue; }
            if(!oy){ jn = jn.Neighb(1, 0); if(jn==-1) continue; }
            if(!oz){ jn = jn.Neighb(2, 0); if(jn==-1) continue; }
            sum += A[jn]*h[0][ox]*h[1][oy]*h[2][oz];
        }
    }
    double V = (h[0][0]+h[0][1])*(h[1][0]+h[1][1])*(h[2][0]+h[2][1]);
    return sum / V;
}

double S_Base::AverageN2E(const vector<double>& A, int ie) const{
    tIndex in(ie); // left bottom corner of the element
    for(int idir=0; idir<Dim; idir++) if(!IsPer[idir] && in.i[idir]==in.N[idir]-1) return 0.; // no such element

    // other corners of the element. `Neighb` function is used to treat of periodic BCs properly
    tIndex in_x = in.Neighb(0, 1);
    tIndex in_y = in.Neighb(1, 1);
    tIndex in_xy = in_x.Neighb(1, 1);

    if(Dim==2){
        return 0.25*(A[in]+A[in_x]+A[in_y]+A[in_xy]);
    }
    else{
        tIndex in_z = in.Neighb(2, 1);
        tIndex in_xz = in.Neighb(2, 1);
        tIndex in_yz = in.Neighb(2, 1);
        tIndex in_xyz = in.Neighb(2, 1);
        return 0.125*(A[in]+A[in_x]+A[in_y]+A[in_xy] + A[in_z]+A[in_xz]+A[in_yz]+A[in_xyz]);
    }
}

// Average data in elements to an edge orthogonal to `idir`. Then `in` is the lowest node of this edge
// TODO Not verified yet
double S_Base::AverageE2Edge(const vector<double>& A, const tIndex& in, int idir) const{
    double h[3][2];
    for(int jdir=0; jdir<Dim; jdir++){
        h[jdir][0] = HLeft(in, jdir);
        h[jdir][1] = HRight(in, jdir);
    }

    double sum = 0., sumV = 0.;
    for(int oz=(idir==2 || Dim==2); oz<=1; oz++){
        for(int oy=(idir==1); oy<=1; oy++)for(int ox=(idir==0); ox<=1; ox++){
            tIndex jn = in;
            if(!ox){ jn = jn.Neighb(0, 0); if(jn==-1) continue; }
            if(!oy){ jn = jn.Neighb(1, 0); if(jn==-1) continue; }
            if(!oz){ jn = jn.Neighb(2, 0); if(jn==-1) continue; }
            double dV = h[0][ox]*h[1][oy];
            if(Dim==3) dV *= h[2][oz];
            sum += A[jn]*dV;
            sumV += dV;
        }
    }
    return sum / sumV;
}


// Dual segments
double3 S_Base::GetHbar(const tIndex& ind) const{
    double3 hbar; hbar[2]=1.;
    for(int idir=0; idir<Dim; idir++){
        hbar[idir] = 0.5*(HLeft(ind,idir) + HRight(ind,idir));
    }
    return hbar;
}

// Face areas
double3 S_Base::GetSbar(const tIndex& ind) const{
    double3 hbar = GetHbar(ind);
    double3 sbar;
    sbar[0]=hbar[1]*hbar[2];
    sbar[1]=hbar[2]*hbar[0];
    sbar[2]=hbar[0]*hbar[1];
    return sbar;
}

S_Base::~S_Base(){
    if(!FFTW_initialized) return;
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(Vphys);
    fftw_free(Vspec);
    FFTW_initialized = 0;
}


void S_Base::InitBase(){
    Dim = tIndex::Dim;
    for(int idir=0; idir<3; idir++) { N[idir]=tIndex::N[idir]; IsPer[idir]=tIndex::IsPer[idir]; }
    NN = N[0]*N[1]*N[2];
}

// Thomas algorithm for a symmetric tridiagonal matrix
// Input: b = diagonal coefficients, a = coefficients to the left from b (a[0] ignored), x = RHS
// Output: x = solution (overwrites input)
// Buffer array: p
void Thomas(int n, const vector<double>& a, const vector<double>& b, vector<double>& x, vector<double>& p){
    x[0] = x[0]/b[0];
    p[0] = a[1]/b[0];
    for(int i=1; i<=n-1; i++){
        double delta = b[i] - a[i]*p[i-1];
        delta = 1./delta;
        if(i!=n-1) p[i] = a[i+1]*delta;
        x[i] = (x[i]-a[i]*x[i-1])*delta;
    }
    for(int i=n-2; i>=0; i--){
        x[i] -= x[i+1]*p[i];
    }
}

// Thomas algorithm for the 1D (in direction 'idir') mass matrix
// Matrix coefficients are h_{j-1/2}/6, (h_{j-1/2}+h_{j+1/2})*1/3, h_{j+1/2}/6
void ThomasForM(const vector<double>& c, vector<double>& x){
    const int n = c.size();
    vector<double> p(n);
    x[0] = x[0]/(C1_3*(c[1]-c[0]));
    p[0] = 0.5;
    for(int i=1; i<=n-1; i++){
        double b = (i!=n-1) ? C1_3*(c[i+1]-c[i-1]) : C1_3*(c[i]-c[i-1]);
        double delta = b - C1_6*(c[i]-c[i-1])*p[i-1];
        delta = 1./delta;
        if(i!=n-1) p[i] = C1_6*(c[i+1]-c[i])*delta;
        x[i] = (x[i]-C1_6*(c[i]-c[i-1])*x[i-1])*delta;
    }
    for(int i=n-2; i>=0; i--){
        x[i] -= x[i+1]*p[i];
    }
}

static int DLog(int n){
    if(n<=0) return -1;
    int k = 0;
    while(n>1){
        if(n&1) return -1;
        n = n/2;
        k++;
    }
    return k;
}

// Checks whether a FFT-based solver is possible. If so, sets UseFourier=1.
// If a general elliptic solver should be used, sets UseFourier=0.
void S_Base::TryInitFourier(){
    int NFourier[3] = {1,1,1};
    int NFourierDirections = 0;
    for(int idir=0; idir<Dim; idir++){
        if(IsFourier[idir]){
            if(!IsPer[idir] || DLog(N[idir])==-1) { printf("%c direction is not periodic or number of nodes is not 2^n\n", 'X'+idir); exit(0); }
            NFourier[NFourierDirections]=N[idir];
            NFourierDirections++;
        }
    }

    if(NFourierDirections<Dim-1) { UseFourier = 0; return; }

    UseFourier = 1;
    Vphys = (fftw_complex*) fftw_malloc(NN * sizeof(fftw_complex));
    Vspec = (fftw_complex*) fftw_malloc(NN * sizeof(fftw_complex));
    if(Dim==3){
        if(NFourierDirections==3){ // FFT in X,Y,Z
            p1 = fftw_plan_dft_3d(NFourier[2], NFourier[1], NFourier[0], Vphys, Vspec, FFTW_FORWARD, FFTW_MEASURE);
            p2 = fftw_plan_dft_3d(NFourier[2], NFourier[1], NFourier[0], Vspec, Vphys, FFTW_BACKWARD, FFTW_MEASURE);
        }
        else if(NFourierDirections==2){
            if(IsPer[2]) { printf("3D case with 2 Fourier directions -> non-periodic direction must be Z\n"); exit(0); }
            int n[2] = {N[1],N[0]};
            // = fftw_plan_many_dft(Rank, n, howmany, A, inembed, istride, idist, A, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
            p1 = fftw_plan_many_dft(2, n, N[2], Vphys, NULL, 1, N[0]*N[1], Vspec, NULL, 1, N[0]*N[1], FFTW_FORWARD, FFTW_MEASURE);
            p2 = fftw_plan_many_dft(2, n, N[2], Vspec, NULL, 1, N[0]*N[1], Vphys, NULL, 1, N[0]*N[1], FFTW_BACKWARD, FFTW_MEASURE);
        }
        else { printf("Unknown configuration\n"); exit(0); }
    }
    if(Dim==2){
        if(IsFourier[0] && IsFourier[1]){ // FFT in X,Y
            p1 = fftw_plan_dft_2d(NFourier[1], NFourier[0], Vphys, Vspec, FFTW_FORWARD, FFTW_MEASURE);
            p2 = fftw_plan_dft_2d(NFourier[1], NFourier[0], Vspec, Vphys, FFTW_BACKWARD, FFTW_MEASURE);
        }
        else if(IsFourier[0]){ // FFT in X
            int n[1] = {N[0]};
            p1 = fftw_plan_many_dft(1, n, N[1], Vphys, NULL, 1, N[0], Vspec, NULL, 1, N[0], FFTW_FORWARD, FFTW_MEASURE);
            p2 = fftw_plan_many_dft(1, n, N[1], Vspec, NULL, 1, N[0], Vphys, NULL, 1, N[0], FFTW_BACKWARD, FFTW_MEASURE);
        }
        else if(IsFourier[1]){ // FFT in Y
            int n[1] = {N[1]};
            p1 = fftw_plan_many_dft(1, n, N[0], Vphys, NULL, N[0], 1, Vspec, NULL, N[0], 1, FFTW_FORWARD, FFTW_MEASURE);
            p2 = fftw_plan_many_dft(1, n, N[0], Vspec, NULL, N[0], 1, Vphys, NULL, N[0], 1, FFTW_BACKWARD, FFTW_MEASURE);
        }
        else { printf("Unknown configuration\n"); exit(0); }
    }
    FFTW_initialized = 1;
}

void S_Base::LinearSystemSolveFourier(const vector<double>& f, vector<double>& XX, int FDmode){
    // FFT in all spatial dimensions
    if(IsFourier[0] && IsFourier[1] && (Dim==2 || IsFourier[2])){
        double3 MeshStep(X[0][1]-X[0][0], X[1][1]-X[0][0], Dim==2 ? 1. : X[2][1]-X[2][0]);
        double3 inv_MeshStep(1./MeshStep[0], 1./MeshStep[1], 1./MeshStep[2]);
        double V = MeshStep[0]*MeshStep[1]*(Dim==3?MeshStep[2]:1.);

        for(int in=0; in<NN; in++) { Vphys[in][0]=f[in]; Vphys[in][1]=0.; }
        fftw_execute(p1);

        vector<double> cosphi[3];
        for(int idir=0; idir<3; idir++){
            cosphi[idir].resize(N[idir]);
            double phi_over_i = 2.*Pi/N[idir];
            for(int i=0; i<N[idir]; i++) cosphi[idir][i] = cos(phi_over_i*i);
        }

        Vspec[0][0] = Vspec[0][1] = 0.; // zeroth mode, which corresponds to the average pressure
        const double mx=inv_MeshStep[0]*inv_MeshStep[0]*V;
        const double my=inv_MeshStep[1]*inv_MeshStep[1]*V;
        const double mz=Dim==2 ? 1. : inv_MeshStep[2]*inv_MeshStep[2]*V;
        for(int kx=0; kx<N[0]; kx++) for(int ky=0; ky<N[1]; ky++) for(int kz=0; kz<N[2]; kz++){
            if(kx==0 && ky==0 && kz==0) continue;
            double m = mx*(cosphi[0][kx]-1.) + my*(cosphi[1][ky]-1.) + mz*(cosphi[2][kz]-1.);
            double inv_m = 1./(2.*m);
            Vspec[kz*N[1]*N[0]+ky*N[0]+kx][0] *= inv_m;
            Vspec[kz*N[1]*N[0]+ky*N[0]+kx][1] *= inv_m;
        }

        fftw_execute(p2);

        double inv_NN = 1./NN;
        for(int in=0; in<NN; in++) XX[in]=Vphys[in][0]*inv_NN;
        return;
    }

    // FFT in one dimension in the 2D case
    // TODO: not verified for FD2
    if(Dim==2){
        if(!IsFourier[0] && !IsFourier[1]) { printf("Internal error\n"); exit(0); }
        int idir_FFT = IsFourier[0] ? 0 : 1;
        int idir_nonFFT = IsFourier[0] ? 1 : 0;
        if(IsPer[idir_nonFFT]) { printf("Not implemented\n"); exit(0); }
        const double h = X[idir_FFT][1] - X[idir_FFT][0]; // uniform mesh in this direction

        for(int in=0; in<NN; in++) { Vphys[in][0]=f[in]; Vphys[in][1]=0.; }
        fftw_execute(p1);

        // Now we have N[idir_FFT] 1D systems
        const double phi_over_i = 2.*Pi/N[idir_FFT];
        vector<double> a(N[idir_nonFFT]), b(N[idir_nonFFT]), x(N[idir_nonFFT]), buf(N[idir_nonFFT]);
        for(int i=0; i<N[idir_FFT]; i++){
            double cosphi = cos(phi_over_i*i);

            a[0]=0; // not used
            for(int j=1; j<N[idir_nonFFT]; j++) a[j]=h/(X[idir_nonFFT][j]-X[idir_nonFFT][j-1]);
            if(FDmode){ a[1]*=0.5; a[N[idir_nonFFT]-1]*=0.5; }

            for(int j=0; j<N[idir_nonFFT]; j++){
                const double _MM = FDmode&&(j==0||j==N[2]-1) ? 0. : 1.;
                b[j]=0.;
                if(j!=0) b[j] -= a[j] + 2.*(1.-cosphi)*0.5*(X[idir_nonFFT][j]-X[idir_nonFFT][j-1])*_MM/h;
                if(j!=N[idir_nonFFT]-1) b[j] -= a[j+1] + 2.*(1.-cosphi)*0.5*(X[idir_nonFFT][j+1]-X[idir_nonFFT][j])*_MM/h;
            }
            if(i==0) b[0]*=2.; // pin

            for(int iii=0; iii<2; iii++){ // real and complex parts
                if(idir_FFT==0) for(int j=0; j<N[idir_nonFFT]; j++) x[j]=Vspec[j*N[0]+i][iii];
                else for(int j=0; j<N[idir_nonFFT]; j++) x[j]=Vspec[i*N[0]+j][iii];
                Thomas(N[idir_nonFFT], a, b, x, buf);
                if(idir_FFT==0) for(int j=0; j<N[idir_nonFFT]; j++) Vspec[j*N[0]+i][iii]=x[j];
                else for(int j=0; j<N[idir_nonFFT]; j++) Vspec[i*N[0]+j][iii]=x[j];
            }
        }

        fftw_execute(p2);

        double inv_NN = 1./N[idir_FFT];
        for(int in=0; in<NN; in++) XX[in]=Vphys[in][0]*inv_NN;
        return;
    }

    // FFT in X and Y in the 3D case
    if(Dim==3 && IsFourier[0] && IsFourier[1]){
        const double hx = X[0][1] - X[0][0], hy = X[1][1] - X[1][0]; // mesh is uniform in these directions

        for(int in=0; in<NN; in++) { Vphys[in][0]=f[in]; Vphys[in][1]=0.; }
        fftw_execute(p1);

        // Now we have N[0]*N[1] 1D systems
        const double phi_over_i = 2.*Pi/N[0];
        const double psi_over_i = 2.*Pi/N[1];
        const int NXY = N[0]*N[1];
        #pragma omp parallel
        {
            vector<double> a(N[2]), b(N[2]), x(N[2]), buf(N[2]);
            #pragma omp for
            for(int ixy=0; ixy<NXY; ixy++){
                int ix=ixy%N[0], iy=ixy/N[0];
                double cosphi = cos(phi_over_i*ix);
                double cospsi = cos(psi_over_i*iy);
                double MM = (1.-cosphi)*hy/hx + (1.-cospsi)*hx/hy;

                a[0]=0; // not used
                for(int j=1; j<N[2]; j++) a[j]=hx*hy/(X[2][j]-X[2][j-1]);
                if(FDmode){ a[1]*=0.5; a[N[2]-1]*=0.5; }

                for(int j=0; j<N[2]; j++){
                    const double _MM = FDmode&&(j==0||j==N[2]-1) ? 0. : MM;
                    b[j]=0.;
                    if(j!=0) b[j] -= a[j] + (X[2][j]-X[2][j-1])*_MM;
                    if(j!=N[2]-1) b[j] -= a[j+1] + (X[2][j+1]-X[2][j])*_MM;
                }
                if(ix==0 && iy==0) b[0]*=2.; // pin

                for(int iii=0; iii<2; iii++){ // real and complex parts
                    for(int j=0; j<N[2]; j++) x[j]=Vspec[ixy + j*NXY][iii];
                    Thomas(N[2], a, b, x, buf);
                    for(int j=0; j<N[2]; j++) Vspec[ixy + j*NXY][iii]=x[j];
                }
            }
        }

        fftw_execute(p2);

        double inv_NN = 1./(N[0]*N[1]);
        for(int in=0; in<NN; in++) XX[in]=Vphys[in][0]*inv_NN;
        return;
    }

    printf("LinearSystemSolveFourier: unknown configuration\n"); exit(0);
}


// Basic approximation for the viscous term (as the discrete Laplace operator). Constant viscosity only
void S_Base::CalcViscTermCross(const vector<double3>& u, double nu_, vector<double3>& ViscTerm, int MultVolume, int Nullify) const{
    if(visc_array.size()) { printf("Standard FD discretization -- for the constant viscosity only\n"); exit(0); }

    for(int _in=0; _in<NN; ++_in){
        tIndex in(_in);

        double3 hbar = GetHbar(in);
        double3 sumflux;
        for(int idir=0; idir<tIndex::Dim; idir++){
            tIndex jn = in.Neighb(idir,0);
            tIndex kn = in.Neighb(idir,1);

            if(kn!=-1){
                double hplus = HRight(in,idir);
                sumflux += nu_ * (u[kn]-u[in])/(hplus*hbar[idir]);
            }
            if(jn!=-1){
                double hminus = HLeft(in,idir);
                sumflux -= nu_ * (u[in]-u[jn])/(hminus*hbar[idir]);
            }
        }
        if(MultVolume) sumflux *= hbar[0]*hbar[1]*hbar[2];

        if(Nullify) ViscTerm[in] = sumflux;
        else ViscTerm[in] += sumflux;
    }
}


// Viscous term approximation by the Galerkin method
void S_Base::CalcViscTermGalerkin(const vector<double3>& u, double nu_, const vector<double>& nu, vector<double3>& ViscTerm, int Nullify) const{
    #pragma omp parallel for
    for(int ic=0; ic<NN; ic++){
        tIndex in(ic);
        double3 sumflux;

        double h[3][2], invh[3][2];
        for(int idir=0; idir<tIndex::Dim; idir++){
            h[idir][0] = HLeft(in, idir);
            h[idir][1] = HRight(in, idir);
            invh[idir][0] = (h[idir][0]>0.) ? 1./h[idir][0] : 0.;
            invh[idir][1] = (h[idir][1]>0.) ? 1./h[idir][1] : 0.;
        }

        if(Dim==2){
            for(int oy=0; oy<=1; oy++){
                tIndex in_y = in.Neighb(1, oy);
                if(in_y==-1) continue;
                for(int ox=0; ox<=1; ox++){
                    tIndex in_x = in.Neighb(0, ox);
                    tIndex in_xy = in_y.Neighb(0, ox);
                    if(in_x==-1) continue;

                    double visc_coeff = nu_;
                    if(visc_array.size()) visc_coeff += nu[GetElementIndex(in, ox, oy, 0)];

                    double mx = invh[0][ox]*h[1][oy]*visc_coeff;
                    double my = invh[1][oy]*h[0][ox]*visc_coeff;

                    // nabla_j * (nu nabla_j u_i)
                    double3 g = (C1_3*(u[in_x]-u[in]) + C1_6*(u[in_xy]-u[in_y]))*mx +
                                (C1_3*(u[in_y]-u[in]) + C1_6*(u[in_xy]-u[in_x]))*my;
                    // nabla_j * (nu nabla_i u_j)
                    double gxx = C1_3*(u[in_x][0]-u[in][0]) + C1_6*(u[in_xy][0]-u[in_y][0]);
                    double gxy = 0.25*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]);
                    double gyy = C1_3*(u[in_y][1]-u[in][1]) + C1_6*(u[in_xy][1]-u[in_x][1]);
                    double gyx = 0.25*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]);
                    gxx *= mx; gyy *= my;
                    gxy *= visc_coeff*(ox^oy ? -1:1);
                    gyx *= visc_coeff*(ox^oy ? -1:1);
                    g += double3(gxx+gxy, gyx+gyy, 0.);

                    sumflux += g;
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

                        double visc_coeff = nu_;
                        if(visc_array.size()) visc_coeff += nu[GetElementIndex(in, ox, oy, oz)];

                        double mx = invh[0][ox]*h[1][oy]*h[2][oz]*visc_coeff;
                        double my = invh[1][oy]*h[0][ox]*h[2][oz]*visc_coeff;
                        double mz = invh[2][oz]*h[0][ox]*h[1][oy]*visc_coeff;

                        // nabla_j * (nu nabla_j u_i)
                        double3 g = (C1_9*(u[in_x]-u[in]) + C1_18*(u[in_xy]-u[in_y]) + C1_18*(u[in_xz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_yz]))*mx +
                                    (C1_9*(u[in_y]-u[in]) + C1_18*(u[in_xy]-u[in_x]) + C1_18*(u[in_yz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_xz]))*my +
                                    (C1_9*(u[in_z]-u[in]) + C1_18*(u[in_xz]-u[in_x]) + C1_18*(u[in_yz]-u[in_y]) + C1_36*(u[in_xyz]-u[in_xy]))*mz;

                        // nabla_j * (nu nabla_i u_j)
                        double gxx = C1_9*(u[in_x][0]-u[in][0]) + C1_18*(u[in_xy][0]-u[in_y][0]) + C1_18*(u[in_xz][0]-u[in_z][0]) + C1_36*(u[in_xyz][0]-u[in_yz][0]);
                        double gxy = 0.25*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                        double gxz = 0.25*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                        double gyy = C1_9*(u[in_y][1]-u[in][1]) + C1_18*(u[in_xy][1]-u[in_x][1]) + C1_18*(u[in_yz][1]-u[in_z][1]) + C1_36*(u[in_xyz][1]-u[in_xz][1]);
                        double gyx = 0.25*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                        double gyz = 0.25*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                        double gzz = C1_9*(u[in_z][2]-u[in][2]) + C1_18*(u[in_yz][2]-u[in_y][2]) + C1_18*(u[in_xz][2]-u[in_x][2]) + C1_36*(u[in_xyz][2]-u[in_xy][2]);
                        double gzx = 0.25*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                        double gzy = 0.25*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));

                        gxx*=mx;
                        gxy*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                        gxz*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                        gyy*=my;
                        gyx*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                        gyz*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);
                        gzz*=mz;
                        gzx*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                        gzy*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);

                        g += double3(gxx+gxy+gxz, gyy+gyx+gyz, gzz+gzx+gzy);
                        sumflux += g;
                    }
                }
            }
        }
        if(Nullify) ViscTerm[in] = sumflux;
        else ViscTerm[in] += sumflux;
    }
}


// Viscous term approximation by the AES (averaged element splittings) method
void S_Base::CalcViscTermAES(const vector<double3>& u, double nu_, const vector<double>& nu, vector<double3>& ViscTerm, int Nullify) const{
    #pragma omp parallel for
    for(int ic=0; ic<NN; ic++){
        tIndex in(ic);
        double3 sumflux;

        double h[3][2], invh[3][2];
        for(int idir=0; idir<tIndex::Dim; idir++){
            h[idir][0] = HLeft(in, idir);
            h[idir][1] = HRight(in, idir);
            invh[idir][0] = (h[idir][0]>0.) ? 1./h[idir][0] : 0.;
            invh[idir][1] = (h[idir][1]>0.) ? 1./h[idir][1] : 0.;
        }

        if(Dim==2){
            for(int oy=0; oy<=1; oy++){
                tIndex in_y = in.Neighb(1, oy);
                if(in_y==-1) continue;
                for(int ox=0; ox<=1; ox++){
                    tIndex in_x = in.Neighb(0, ox);
                    tIndex in_xy = in_y.Neighb(0, ox);
                    if(in_x==-1) continue;

                    double visc_coeff = nu_;
                    if(visc_array.size()) visc_coeff += nu[GetElementIndex(in, ox, oy, 0)];

                    double mx = invh[0][ox]*h[1][oy]*visc_coeff;
                    double my = invh[1][oy]*h[0][ox]*visc_coeff;

                    // nabla_j * (nu nabla_j u_i)
                    double3 g = 0.5*(u[in_x]-u[in])*mx +
                                0.5*(u[in_y]-u[in])*my;
                    // nabla_j * (nu nabla_i u_j)
                    double gxx = 0.5*(u[in_x][0]-u[in][0]);
                    double gxy = 0.25*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]);
                    double gyy = 0.5*(u[in_y][1]-u[in][1]);
                    double gyx = 0.25*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]);
                    gxx *= mx; gyy *= my;
                    gxy *= visc_coeff*(ox^oy ? -1:1);
                    gyx *= visc_coeff*(ox^oy ? -1:1);
                    g += double3(gxx+gxy, gyx+gyy, 0.);

                    sumflux += g;
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

                        double visc_coeff = nu_;
                        if(visc_array.size()) visc_coeff += nu[GetElementIndex(in, ox, oy, oz)];

                        double mx = invh[0][ox]*h[1][oy]*h[2][oz]*visc_coeff;
                        double my = invh[1][oy]*h[0][ox]*h[2][oz]*visc_coeff;
                        double mz = invh[2][oz]*h[0][ox]*h[1][oy]*visc_coeff;

                        #if 1
                        // nabla_j * (nu nabla_j u_i)
                        double3 g = 0.25*(u[in_x]-u[in])*mx +
                                    0.25*(u[in_y]-u[in])*my +
                                    0.25*(u[in_z]-u[in])*mz;

                        // nabla_j * (nu nabla_i u_j)
                        double gxx = 0.25*(u[in_x][0]-u[in][0]);
                        double gxy = 0.25*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                        double gxz = 0.25*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                        double gyy = 0.25*(u[in_y][1]-u[in][1]);
                        double gyx = 0.25*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                        double gyz = 0.25*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                        double gzz = 0.25*(u[in_z][2]-u[in][2]);
                        double gzx = 0.25*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                        double gzy = 0.25*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));

                        #else // test (mix of AES and Galerkin)
                        // nabla_j * (nu nabla_j u_i)
                        double3 g = 0.25*(u[in_x]-u[in])*mx +
                                    0.25*(u[in_y]-u[in])*my +
                                    (C1_9*(u[in_z]-u[in]) + C1_18*(u[in_xz]-u[in_x]) + C1_18*(u[in_yz]-u[in_y]) + C1_36*(u[in_xyz]-u[in_xy]))*mz;

                        // nabla_j * (nu nabla_i u_j)
                        double gxx = 0.25*(u[in_x][0]-u[in][0]);
                        double gxy = 0.25*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                        double gxz = 0.25*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                        double gyy = 0.25*(u[in_y][1]-u[in][1]);
                        double gyx = 0.25*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                        double gyz = 0.25*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                        double gzz = C1_9*(u[in_z][2]-u[in][2]) + C1_18*(u[in_yz][2]-u[in_y][2]) + C1_18*(u[in_xz][2]-u[in_x][2]) + C1_36*(u[in_xyz][2]-u[in_xy][2]);
                        double gzx = 0.25*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                        double gzy = 0.25*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));

                        #endif

                        gxx*=mx;
                        gxy*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                        gxz*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                        gyy*=my;
                        gyx*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                        gyz*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);
                        gzz*=mz;
                        gzx*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                        gzy*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);

                        g += double3(gxx+gxy+gxz, gyy+gyx+gyz, gzz+gzx+gzy);
                        sumflux += g;
                    }
                }
            }
        }
        if(Nullify) ViscTerm[in] = sumflux;
        else ViscTerm[in] += sumflux;
    }
}


// Matrix of the pressure equation -- basic version
void S_Base::PassDefaultPressureSystemToHYPRE(int DoPatchNearWalls){
    vector<double> L(NN*7);
    int NodeThatCanBePinned = -1;
    for(tIndex in=0; in<NN; ++in){
        double3 sbar = GetSbar(in);

        for(int idir=0; idir<tIndex::Dim; idir++){
            double phiL = 1., phiR = 1.;
            if(DoPatchNearWalls){ // in FD2, we patch the matrix near walls, and in Gal1 we do not
                phiL = 1. - 0.5*(in.IsWall() + in.Neighb(idir, 0).IsWall());
                phiR = 1. - 0.5*(in.IsWall() + in.Neighb(idir, 1).IsWall());
            }
            if(in.i[idir]>0 || tIndex::IsPer[idir]) L[in*7+idir*2+1] = phiL*sbar[idir]/HLeft(in,idir);
            if(in.i[idir]<tIndex::N[idir]-1 || tIndex::IsPer[idir])  L[in*7+idir*2+2] = phiR*sbar[idir]/HRight(in, idir);
        }
        for(int ii=1; ii<=6; ii++) L[in*7] -= L[in*7+ii]; // diagonal coefficient
        if(in.IsCornerNode()) L[in*7]=1.; // dummy corner node, pressure is not defined
        else if(NodeThatCanBePinned<0) NodeThatCanBePinned=in; // choose a node to pin
    }
    L[NodeThatCanBePinned*7] *= 1.5; // pin

    LinearSolverAlloc(0, N); // alloc memory for HYPRE data
    LinearSystemInit(0, L); // pass the matrix to HYPRE
}


// Calculate flux terms on walls (to get friction)
void S_Base::CalcViscTermOnWalls_AES(const vector<double3>& u, double3& F1, double3& F2, vector<double3>& f) const{
    if(Dim==2 || !IsPer[0] || !IsPer[1] || IsPer[2]) { printf("Not implemented\n"); exit(0); }

    const int NXY = N[0]*N[1];
    F1 = double3();
    F2 = double3();
    for(int iiz=0; iiz<2; iiz++){ // bottom and top surfaces
        for(int ixy=0; ixy<NXY; ixy++){
            tIndex in(ixy + iiz*(N[2]-1)*NXY);

            double h[3][2], invh[3][2];
            for(int idir=0; idir<tIndex::Dim; idir++){
                h[idir][0] = HLeft(in, idir);
                h[idir][1] = HRight(in, idir);
                invh[idir][0] = (h[idir][0]>0.) ? 1./h[idir][0] : 0.;
                invh[idir][1] = (h[idir][1]>0.) ? 1./h[idir][1] : 0.;
            }

            double3 sumflux;
            const int oz = 1-iiz; // No loop in oz. if iiz==0, this is the bottom, so the only way is forward in z, and vise versa
            tIndex in_z = in.Neighb(2, oz);
            for(int oy=0; oy<=1; oy++){
                tIndex in_y = in.Neighb(1, oy);
                tIndex in_yz = in_z.Neighb(1, oy);
                for(int ox=0; ox<=1; ox++){
                    tIndex in_x = in.Neighb(0, ox);
                    tIndex in_xy = in_y.Neighb(0, ox);
                    tIndex in_xz = in_z.Neighb(0, ox);
                    tIndex in_xyz = in_yz.Neighb(0, ox);

                    double visc_coeff = visc;
                    if(visc_array.size()) visc_coeff += visc_array[GetElementIndex(in, ox, oy, oz)];

                    double mx = invh[0][ox]*h[1][oy]*h[2][oz]*visc_coeff;
                    double my = invh[1][oy]*h[0][ox]*h[2][oz]*visc_coeff;
                    double mz = invh[2][oz]*h[0][ox]*h[1][oy]*visc_coeff;

                    // nabla_j * (nu nabla_j u_i)
                    double3 g = 0.25*(u[in_x]-u[in])*mx +
                                0.25*(u[in_y]-u[in])*my +
                                0.25*(u[in_z]-u[in])*mz;

                    // nabla_j * (nu nabla_i u_j)
                    double gxx = 0.25*(u[in_x][0]-u[in][0]);
                    double gxy = 0.25*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                    double gxz = 0.25*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                    double gyy = 0.25*(u[in_y][1]-u[in][1]);
                    double gyx = 0.25*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                    double gyz = 0.25*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                    double gzz = 0.25*(u[in_z][2]-u[in][2]);
                    double gzx = 0.25*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                    double gzy = 0.25*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));

                    gxx*=mx;
                    gxy*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                    gxz*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                    gyy*=my;
                    gyx*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                    gyz*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);
                    gzz*=mz;
                    gzx*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                    gzy*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);

                    g += double3(gxx+gxy+gxz, gyy+gyx+gyz, gzz+gzx+gzy);
                    sumflux += g;
                }
            }

            if(f.size()) f[in] = sumflux;
            if(iiz) F2 += sumflux;
            else F1 += sumflux;
        }
    }
}


// Experimental weird viscous terms approximation. For uniform meshes only
// Warning: `nu` are nodal (not elemental) values
void S_Base::CalcViscTermWeird(const vector<double3>& u, double nu_, const vector<double>& nu, vector<double3>& ViscTerm, int Nullify) const{
    #pragma omp parallel for
    for(int _in=0; _in<NN; ++_in){
        tIndex in(_in);
        if(in.IsWall()) continue;
        double3 hbar = GetHbar(in);
        double3 inv_MeshStep(1./hbar[0], 1./hbar[1], 1./hbar[2]);

        double3 sumflux;
        for(int idir=0; idir<Dim; idir++){
            const double invh = inv_MeshStep[idir];
            // Stencil
            tIndex jn, kn;
            jn = in.Neighb(idir,0);
            kn = in.Neighb(idir,1);

            // Viscosity
            if(nu_>0.){
                double c = nu_*invh*invh;
                sumflux += c*(u[kn]+u[jn]-2.*u[in]);
            }
            if(nu.size()){
                double viscL = std::min(nu[in],nu[jn]);
                double3 fluxL = invh*(u[in]-u[jn])*(0.5*(nu[in]+nu[jn]));
                fluxL[idir] += invh*(u[in][idir]-u[jn][idir])*viscL;
                double viscR = std::min(nu[in],nu[kn]);
                double3 fluxR = invh*(u[kn]-u[in])*(0.5*(nu[in]+nu[kn]));
                fluxR[idir] += invh*(u[kn][idir]-u[in][idir])*viscR;
                for(int idir_alpha=0; idir_alpha<Dim; idir_alpha++){
                    if(idir_alpha==idir) continue;
                    fluxL[idir_alpha] += 0.25*inv_MeshStep[idir_alpha]*viscL*(
                        u[in.Neighb(idir_alpha,1)][idir] +
                        u[jn.Neighb(idir_alpha,1)][idir] -
                        u[in.Neighb(idir_alpha,0)][idir] -
                        u[jn.Neighb(idir_alpha,0)][idir]);
                    fluxR[idir_alpha] += 0.25*inv_MeshStep[idir_alpha]*viscR*(
                        u[in.Neighb(idir_alpha,1)][idir] +
                        u[kn.Neighb(idir_alpha,1)][idir] -
                        u[in.Neighb(idir_alpha,0)][idir] -
                        u[kn.Neighb(idir_alpha,0)][idir]);
                }
                sumflux += (fluxR - fluxL)*invh;
            }
        }
        if(Nullify) ViscTerm[in] = sumflux;
        else ViscTerm[in] += sumflux;
    }
}


// Calculate AbsS at elements
// Mode: 0 - based on averaged gradients, 1 - checkerboard-supressing, 2 - based on nodal gradients (step=2*h)
static double SQR(double x) { return x*x; }
void S_Base::CalcAbsS(const vector<double3>& u, vector<double>& AbsS, int Mode) const{
    if(Mode==0 || Mode==1){
        #pragma omp parallel for
        for(int ie=0; ie<NN; ie++){
            AbsS[ie] = 0.;

            tIndex in(ie); // left bottom corner of the element
            if(!IsPer[0] && in.i[0]==N[0]-1) continue; // no such element
            if(!IsPer[1] && in.i[1]==N[1]-1) continue; // no such element
            if(!IsPer[2] && in.i[2]==N[2]-1) continue; // no such element

            const double hx = X[0][in.i[0]+1] - X[0][in.i[0]];
            const double hy = X[1][in.i[1]+1] - X[1][in.i[1]];
            const double hz = X[2][in.i[2]+1] - X[2][in.i[2]];

            // other corners of the element
            tIndex in_x = in.Neighb(0, 1);
            tIndex in_y = in.Neighb(1, 1);
            tIndex in_xy = in_x.Neighb(1, 1);
            tIndex in_z = in.Neighb(2, 1);
            tIndex in_xz = in_x.Neighb(2, 1);
            tIndex in_yz = in_y.Neighb(2, 1);
            tIndex in_xyz = in_xy.Neighb(2, 1);

            if(Mode==0){
                double3 gradu[3];
                gradu[0] = ((u[in_x]+u[in_xy]+u[in_xz]+u[in_xyz])-(u[in]+u[in_y]+u[in_z]+u[in_yz])) / (4.*hx);
                gradu[1] = ((u[in_y]+u[in_xy]+u[in_yz]+u[in_xyz])-(u[in]+u[in_x]+u[in_z]+u[in_xz])) / (4.*hy);
                gradu[2] = ((u[in_z]+u[in_xz]+u[in_yz]+u[in_xyz])-(u[in]+u[in_x]+u[in_y]+u[in_xy])) / (4.*hz);

                gradu[0][1] = 0.5*(gradu[0][1]+gradu[1][0]);
                gradu[0][2] = 0.5*(gradu[0][2]+gradu[2][0]);
                gradu[1][2] = 0.5*(gradu[1][2]+gradu[2][1]);
                double abss=SQR(gradu[0][0])+SQR(gradu[1][1])+SQR(gradu[2][2])+2.*SQR(gradu[0][1])+2.*SQR(gradu[0][2])+2.*SQR(gradu[1][2]);
                AbsS[ie]=sqrt(2.*abss);
            }
            if(Mode==1){
                double3 dudx[4] = {u[in_x]-u[in], u[in_xy]-u[in_y], u[in_xz]-u[in_z], u[in_xyz]-u[in_yz]};
                double3 dudy[4] = {u[in_y]-u[in], u[in_xy]-u[in_x], u[in_yz]-u[in_z], u[in_xyz]-u[in_xz]};
                double3 dudz[4] = {u[in_z]-u[in], u[in_xz]-u[in_x], u[in_yz]-u[in_y], u[in_xyz]-u[in_xy]};
                for(int i=0; i<4; i++){ dudx[i]/=hx; dudy[i]/=hy; dudz[i]/=hz; }

                for(int i=0; i<4; i++) for(int j=0; j<4; j++) for(int k=0; k<4; k++){
                    double gxx=dudx[i][0], gyy=dudy[j][1], gzz=dudz[k][2];
                    double gxy=0.5*(dudx[i][1]+dudy[j][0]), gxz=0.5*(dudx[i][2]+dudz[k][0]), gyz=0.5*(dudy[j][2]+dudz[k][1]);
                    double abss = SQR(gxx)+SQR(gyy)+SQR(gzz)+2.*SQR(gxy)+2.*SQR(gxz)+2.*SQR(gyz);
                    abss = sqrt(2.*abss);
                    AbsS[ie] += abss / 64.;
                }
            }
        }
    }
    if(Mode==2){
        // Calculate the nodal values of AbsS
        vector<double> AbsS_Nodal(NN);
        for(tIndex in=0; in<NN; ++in){
            double3 hbar = GetHbar(in);
            double3 gradu[3];
            for(int idir=0; idir<tIndex::Dim; idir++){
                tIndex inL = in.Neighb(idir, 0), inR = in.Neighb(idir, 1);
                if(inL==-1) inL=in;
                if(inR==-1) inR=in;
                gradu[idir] = 0.5*(u[inR]-u[inL])/hbar[idir];
            }
            // Symmetrize
            gradu[0][1] = 0.5*(gradu[0][1]+gradu[1][0]);
            gradu[0][2] = 0.5*(gradu[0][2]+gradu[2][0]);
            gradu[1][2] = 0.5*(gradu[1][2]+gradu[2][1]);
            double abss=SQR(gradu[0][0])+SQR(gradu[1][1])+SQR(gradu[2][2])+2.*SQR(gradu[0][1])+2.*SQR(gradu[0][2])+2.*SQR(gradu[1][2]);
            AbsS_Nodal[in]=sqrt(2.*abss);
        }

        // Average from nodes to elements
        #pragma omp parallel for
        for(int ie=0; ie<NN; ie++){
            AbsS[ie] = 0.;

            tIndex in(ie); // left bottom corner of the element
            if(!IsPer[0] && in.i[0]==N[0]-1) continue; // no such element
            if(!IsPer[1] && in.i[1]==N[1]-1) continue; // no such element
            if(!IsPer[2] && in.i[2]==N[2]-1) continue; // no such element

            // other corners of the element
            tIndex in_x = in.Neighb(0, 1);
            tIndex in_y = in.Neighb(1, 1);
            tIndex in_xy = in_x.Neighb(1, 1);

            if(Dim==2){
                AbsS[ie] = 0.25*(AbsS_Nodal[in] + AbsS_Nodal[in_x] + AbsS_Nodal[in_y] + AbsS_Nodal[in_xy]);
            }
            else{
                tIndex in_z = in.Neighb(2, 1);
                tIndex in_xz = in_x.Neighb(2, 1);
                tIndex in_yz = in_y.Neighb(2, 1);
                tIndex in_xyz = in_xy.Neighb(2, 1);
                AbsS[ie] = 0.125*(AbsS_Nodal[in] + AbsS_Nodal[in_x] + AbsS_Nodal[in_y] + AbsS_Nodal[in_xy] +
                                  AbsS_Nodal[in_z] + AbsS_Nodal[in_xz] + AbsS_Nodal[in_yz] + AbsS_Nodal[in_xyz]);
            }
        }
    }
}
