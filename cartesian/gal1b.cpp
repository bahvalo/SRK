// P1-Galerkin method (not lumped)
// Dirichlet (u=0) or periodic boundary conditions
#include "asrk.h"
#include "gal1.h"

#include "linsolver.h"

void S_Gal1B::ApplyGradient(const vector<double>& f, vector<double3>& R) {
    ApplyGradient_MainGalerkin(f, R);
}


// Implicit velocity stage
void S_Gal1B::ImplicitStage(double time_stage, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u){
    return ImplicitStage_MainGalerkin(time_stage, tau_stage, ustar, p, u);
}


// Explicit momentum fluxes
void S_Gal1B::ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat){
    if(TimeIntMethod==tTimeIntMethod::IMPLICIT){
        for(int in=0; in<NN; in++) kuhat[in] = double3();
    }
    if(TimeIntMethod==tTimeIntMethod::IMEX){
        CalcFluxTerm(t, u, p, true, false, true, false, kuhat);
        SetZeroNormalComponent(kuhat);
        ApplyMassMatrixInv(kuhat, tZeroAtBoundary::NORMAL);
    }
    if(TimeIntMethod==tTimeIntMethod::EXPLICIT && !DoNotApplyMInvToVisc){
        CalcFluxTerm(t, u, p, true, true, true, true, kuhat);
        SetZeroNormalComponent(kuhat);
        // If the viscous term is implicit, we can leave enforcing the Dirichlet condition to the implicit term
        // But if the viscous term is explicit, we have do to this here
        ApplyMassMatrixInv(kuhat, tZeroAtBoundary::VECTOR);
    }
    if(TimeIntMethod==tTimeIntMethod::EXPLICIT && DoNotApplyMInvToVisc){ // experimental option
        CalcFluxTerm(t, u, p, true, false, true, true, kuhat);
        SetZeroNormalComponent(kuhat);
        ApplyMassMatrixInv(kuhat, tZeroAtBoundary::VECTOR);

        vector<double3> buf(NN);
        CalcFluxTerm(t, u, p, false, true, false, false, buf);
        for(tIndex in=0; in<NN; ++in) if(!in.IsWall()) kuhat[in]+=buf[in]/GetCellVolume(in);
    }
}

// Implicit momentum fluxes - for methods of type CK (excluding ARS). Pressure should not be used
void S_Gal1B::ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku){
    bool DoConv = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoVisc = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    bool DoSourceE = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoSourceI = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    CalcFluxTerm(t, u, p, !DoConv, !DoVisc, !DoSourceE, !DoSourceI, ku);
    SetZeroNormalComponent(ku);
    ApplyMassMatrixInv(ku, tZeroAtBoundary::NORMAL);
}

// Apply the L^{-1}*Div operator
void S_Gal1B::UnapplyGradient(const vector<double3>& _f, vector<double>& InvDf){
    vector<double> Df(NN);
    vector<double3> f = _f;
    SetZeroNormalComponent(f);

    ApplyDivergence_MainGalerkin(f, Df);

    // Solve the pressure system
    if(!UseFourier){ // use HYPRE
        LinearSystemSolve(0 /*pressure system*/, InvDf, Df, AMG_Tolerance_P, AMG_MaxIters_P);
    }
    else{ // use FFT
        LinearSystemSolveFourier(Df, InvDf, false);
    }
}

void S_Gal1B::Init(){
    InitBase();


    TryInitFourier();
    if(!UseFourier) PassDefaultPressureSystemToHYPRE(false); // Can't use FFTW, so initialize HYPRE

    InitVelocitySystem();
}





void S_Gal1Bexim::ApplyGradient(const vector<double>& f, vector<double3>& R) {
    ApplyGradient_MainGalerkin(f, R);
}

// Apply the L^{-1}*Div operator
void S_Gal1Bexim::UnapplyGradient(const vector<double3>& _f, vector<double>& InvDf){
    vector<double> Df(NN);
    vector<double3> f = _f;
    SetZeroNormalComponent(f);

    ApplyDivergence_MainGalerkin(f, Df);
    LinearSystemSolveFourier(Df, InvDf, false);
}

void S_Gal1Bexim::ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat){
    printf("Not implemented\n"); exit(0);
}

void S_Gal1Bexim::ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat){
    const double C1_3 = 1./3., C1_6 = 1./6.;
    for(int in=0; in<NN; in++) kuhat[in]=double3();

    if(EnableConvection){
        CalcConvTerm(u, kuhat, false);
    }

    if(SourceE!=NULL){
        vector<double3> s(NN);
        if(SourceE!=NULL) SourceE(t, s);
        for(tIndex in=0; in<NN; ++in) kuhat[in]+=s[in]*GetCellVolume(in);
    }

    // Viscosity
    // X- and Y- stresses
    if(visc>0 || visc_array.size()){
        #pragma omp parallel for
        for(int ic=0; ic<NN; ic++){
            tIndex in(ic);
            //if(in.IsWall()) { f[in]=double3(); continue; }
            double3 sumflux;

            double h[3][2], invh[3][2];
            for(int idir=0; idir<tIndex::Dim; idir++){
                h[idir][0] = HLeft(in, idir);
                h[idir][1] = HRight(in, idir);
                invh[idir][0] = (h[idir][0]>0.) ? 1./h[idir][0] : 0.;
                invh[idir][1] = (h[idir][1]>0.) ? 1./h[idir][1] : 0.;
            }

            for(int oz=0; oz<=1; oz++){
                tIndex in_z = in.Neighb(2, oz);
                if(in_z==-1) continue; // boundary
                for(int oy=0; oy<=1; oy++){
                    //if(h[1][oy]==0.) continue;
                    tIndex in_y = in.Neighb(1, oy);
                    tIndex in_yz = in_z.Neighb(1, oy);
                    for(int ox=0; ox<=1; ox++){
                        //if(h[0][ox]==0.) continue;
                        tIndex in_x = in.Neighb(0, ox);
                        tIndex in_xy = in_y.Neighb(0, ox);
                        tIndex in_xz = in_z.Neighb(0, ox);
                        tIndex in_xyz = in_yz.Neighb(0, ox);

                        double visc_coeff = visc;
                        if(visc_array.size()) visc_coeff += visc_array[GetElementIndex(in, ox, oy, oz)];

                        double mx = invh[0][ox]*h[1][oy]*h[2][oz]*visc_coeff;
                        double my = invh[1][oy]*h[0][ox]*h[2][oz]*visc_coeff;
                        //double mz = invh[2][oz]*h[0][ox]*h[1][oy]*visc_coeff;

                        // nabla_j * (nu nabla_j u_i)
                        double3 g = 0.25*(u[in_x]-u[in])*mx +
                                    0.25*(u[in_y]-u[in])*my;
                                    // + 0.25*(u[in_z]-u[in])*mz;

                        // nabla_j * (nu nabla_i u_j)
                        double gxx = 0.25*(u[in_x][0]-u[in][0]);
                        double gxy = 0.25*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                        //double gxz = 0.25*(0.5*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]));
                        double gyy = 0.25*(u[in_y][1]-u[in][1]);
                        double gyx = 0.25*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                        //double gyz = 0.25*(0.5*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]));
                        //double gzz = 0.25*(u[in_z][2]-u[in][2]);
                        double gzx = 0.25*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                        double gzy = 0.25*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));

                        gxx*=mx;
                        gxy*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                        //gxz*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                        gyy*=my;
                        gyx*=h[2][oz]*visc_coeff*(ox^oy ? -1:1);
                        //gyz*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);
                        //gzz*=mz;
                        gzx*=h[1][oy]*visc_coeff*(ox^oz ? -1:1);
                        gzy*=h[0][ox]*visc_coeff*(oy^oz ? -1:1);

                        //g += double3(gxx+gxy+gxz, gyy+gyx+gyz, gzz+gzx+gzy);
                        g += double3(gxx+gxy, gyy+gyx, gzy+gzx);
                        sumflux += g;
                    }
                }
            }

            kuhat[in] += sumflux;
        }
    }

    SetZeroNormalComponent(kuhat);
    //ApplyMassMatrixInv(kuhat, tZeroAtBoundary::NORMAL);
    ApplyMassMatrixInv(kuhat, tZeroAtBoundary::VECTOR);
}

void S_Gal1Bexim::Init(){
    if(ViscScheme!=tViscScheme::AES) { printf("S_Gal1Bexim requires AES viscosity discretization\n"); exit(0); }
    InitBase();
    TryInitFourier();
    if(!UseFourier) { printf("Wrong configuration\n"); exit(0); }
}

// Aux
void S_Gal1Bexim::ImplicitStage_SetCoeffs(int ix, int iy, vector<double>& a, vector<double>& b, double m) const{
    const int NXY = N[0]*N[1];

    // Set the coefficients of the 3-diagonal matrix
    for(int i=1; i<N[2]; i++){ // a[0] is not used
        double h = X[2][i]-X[2][i-1];
        double nu = visc;
        if(visc_array.size()){ // adding the average of the turbulent velocity
            for(int oy=-1; oy<=0; oy++) for(int ox=-1; ox<=0; ox++){
                int jx=(ix+ox)&(N[0]-1), jy=(iy+oy)&(N[1]-1); // here we use that N[0] and N[1] are powers of 2
                int ie = (i-1)*NXY+jy*N[0]+jx;
                nu += 0.25*visc_array[ie];
            }
        }
        a[i]=-m*nu/h; // m=tau or m=tau*2
        a[i]+=h/6.; // this is the only difference from the FD method
    }

    b[0] = 0.5*(X[2][1]-X[2][0]) - a[1];
    for(int i=1; i<N[2]-1; i++) b[i] = 0.5*(X[2][i+1]-X[2][i-1]) - a[i]-a[i+1];
    b[N[2]-1] = 0.5*(X[2][N[2]-1]-X[2][N[2]-2]) - a[N[2]-1];

    a[1]=a[N[2]-1]=0.; // enforcing Dirichlet boundary conditions
}

// Aux. Must not be called for boundary nodes
double S_Gal1Bexim::CalcCrossTerm(int ix, int iy, int iz, int icomponent, const vector<double3>& u) const{
    const double C1_3 = 1./3., C1_6 = 1./6.;
    const int NXY=N[0]*N[1];
    tIndex in(iz*NXY+iy*N[0]+ix);
    double ret_val = 0.;
    for(int oz=0; oz<=1; oz++){
        tIndex in_z = in.Neighb(2, oz);
        for(int oy=0; oy<=1; oy++){
            tIndex in_y = in.Neighb(1, oy);
            tIndex in_yz = in_z.Neighb(1, oy);
            double hy = oy ? HRight(in, 1) : HLeft(in, 1);
            for(int ox=0; ox<=1; ox++){
                tIndex in_x = in.Neighb(0, ox);
                tIndex in_xy = in_y.Neighb(0, ox);
                tIndex in_xz = in_z.Neighb(0, ox);
                tIndex in_xyz = in_yz.Neighb(0, ox);
                double hx = ox ? HRight(in, 0) : HLeft(in, 0);

                double visc_coeff = visc;
                if(visc_array.size()) visc_coeff += visc_array[GetElementIndex(in, ox, oy, oz)];

                // nabla_z * (nu nabla_i u_z)
                if(icomponent==0){
                    double g = 0.25*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                    g*=hy*visc_coeff*(ox^oz ? -1:1);
                    ret_val += g;
                }
                if(icomponent==1){
                    double g = 0.25*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                    g*=hx*visc_coeff*(oy^oz ? -1:1);
                    ret_val += g;
                }
            }
        }
    }
    return ret_val;
}

// Implicit velocity stage
void S_Gal1Bexim::ImplicitStage(double time_stage, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u){
    const double C1_3 = 1./3.;
    const double C1_6 = 1./6.;
    const int NXY = N[1]*N[0];
    #pragma omp parallel for
    for(int in=NXY; in<NXY*(N[2]-1); in++) u[in]=ustar[in];
    for(int in=0; in<NXY; ++in) { u[in]=double3(); u[in+NXY*(N[2]-1)]=double3(); }

    // First update u_z
    #pragma omp parallel
    {
        vector<double> a(N[2]), b(N[2]), x(N[2]), buf(N[2]);
        #pragma omp for
        for(int oxy=0; oxy<NXY; oxy++){
            int ix=oxy%N[0], iy=oxy/N[0];
            ImplicitStage_SetCoeffs(ix, iy, a, b, tau_stage*2.); // *2 because of 2*D(u) in the governing equations
            for(int i=1; i<N[2]-1; i++) x[i] = u[oxy+i*NXY][2]*C1_3*(X[2][i+1]-X[2][i-1])+u[oxy+(i-1)*NXY][2]*C1_6*(X[2][i]-X[2][i-1])+u[oxy+(i+1)*NXY][2]*C1_6*(X[2][i+1]-X[2][i]); // keep y[0]=y[N-1]=0
            Thomas(N[2], a, b, x, buf);
            for(int i=0; i<N[2]; i++) u[oxy+i*NXY][2] = x[i];
        }
    }

    // Now update other velocity components
    const double invhx = 1./(X[0][1]-X[0][0]), invhy = 1./(X[1][1]-X[1][0]);
    #pragma omp parallel
    {
        vector<double> a(N[2]), b(N[2]), x(N[2]), buf(N[2]);
        #pragma omp for
        for(int oxy=0; oxy<NXY; oxy++){
            int ix=oxy%N[0], iy=oxy/N[0];
            ImplicitStage_SetCoeffs(ix, iy, a, b, tau_stage);
            for(int icomponent=0; icomponent<2; icomponent++){
                for(int i=1; i<N[2]-1; i++){ // keep x[0]=x[N-1]=0
                    x[i] = u[oxy+i*NXY][icomponent]*C1_3*(X[2][i+1]-X[2][i-1])+u[oxy+(i-1)*NXY][icomponent]*C1_6*(X[2][i]-X[2][i-1])+u[oxy+(i+1)*NXY][icomponent]*C1_6*(X[2][i+1]-X[2][i]); // keep y[0]=y[N-1]=0
                    x[i] += tau_stage*CalcCrossTerm(ix, iy, i, icomponent, u)*(invhx*invhy);
                }
                Thomas(N[2], a, b, x, buf);
                for(int i=0; i<N[2]; i++) u[oxy+i*NXY][icomponent] = x[i];
            }
        }
    }
}
