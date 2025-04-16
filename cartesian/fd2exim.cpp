// Second-order finite difference discretization for the channel flow (3D; periodic in X and Y; Dirichlet in Z)
// All fluxes are taken explicitly except for the z-component of stresses. TimeIntMethod parameter is ignored
// Requires AES method for the viscous terms discretization
#include "asrk.h"
#include "fd2.h"


// Gradient and divergence are inherited from S_FD2
// void S_FD2::ApplyGradient(const vector<double>& a, vector<double3>& Ga);
// void S_FD2::UnapplyGradient(const vector<double3>& v, vector<double>& InvLDv);

// Aux
void S_FD2exim::ImplicitStage_SetCoeffs(int ix, int iy, vector<double>& a, vector<double>& b, double m) const{
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
        a[i]=-m*nu/h; // m=tau or m=2*tau
    }

    b[0] = 0.5*(X[2][1]-X[2][0]) - a[1];
    for(int i=1; i<N[2]-1; i++) b[i] = 0.5*(X[2][i+1]-X[2][i-1]) - a[i]-a[i+1];
    b[N[2]-1] = 0.5*(X[2][N[2]-1]-X[2][N[2]-2]) - a[N[2]-1];

    a[1]=a[N[2]-1]=0.; // patch for the Dirichlet boundary conditions
}

// Aux. Must not be called for boundary nodes
double S_FD2exim::CalcCrossTerm(int ix, int iy, int iz, int icomponent, const vector<double3>& u) const{
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
void S_FD2exim::ImplicitStage(double time_stage, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u){
    const double invhx = 1./(X[0][1]-X[0][0]), invhy = 1./(X[1][1]-X[1][0]);
    const int NXY = N[1]*N[0];
    #pragma omp parallel for
    for(int in=0; in<NN; in++) u[in]=ustar[in];
    //#pragma omp parallel for
    for(int in=0; in<NXY; ++in) { u[in]=double3(); u[in+NXY*(N[2]-1)]=double3(); }

    // First update u_z
    #pragma omp parallel
    {
        vector<double> a(N[2]), b(N[2]), x(N[2]), buf(N[2]);
        #pragma omp for
        for(int oxy=0; oxy<NXY; oxy++){
            int ix=oxy%N[0], iy=oxy/N[0];
            ImplicitStage_SetCoeffs(ix, iy, a, b, tau_stage*2.); // *2 because of 2*D(u) in the governing equations
            for(int i=1; i<N[2]-1; i++) x[i] = u[oxy+i*NXY][2]*0.5*(X[2][i+1]-X[2][i-1]);
            x[0]=x[N[2]-1]=0.;
            Thomas(N[2], a, b, x, buf);
            for(int i=0; i<N[2]; i++) u[oxy+i*NXY][2] = x[i];
        }
    }

    // Now update other velocity components
    #pragma omp parallel
    {
        vector<double> a(N[2]), b(N[2]), x(N[2]), buf(N[2]);
        #pragma omp for
        for(int oxy=0; oxy<NXY; oxy++){
            int ix=oxy%N[0], iy=oxy/N[0];
            ImplicitStage_SetCoeffs(ix, iy, a, b, tau_stage);
            for(int icomponent=0; icomponent<2; icomponent++){
                x[0]=x[N[2]-1]=0.;
                for(int i=1; i<N[2]-1; i++){
                    x[i] = u[oxy+i*NXY][icomponent]*0.5*(X[2][i+1]-X[2][i-1]);
                    x[i] += tau_stage*CalcCrossTerm(ix, iy, i, icomponent, u)*(invhx*invhy);
                }
                Thomas(N[2], a, b, x, buf);
                for(int i=0; i<N[2]; i++) u[oxy+i*NXY][icomponent] = x[i];
            }
        }
    }
}

// Calculate terms in the momentum equation (divided by cell volume)
// Explicit velocity stage
void S_FD2exim::ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& f){
    const double C1_3 = 1./3., C1_6 = 1./6.;

    CalcConvTerm(u, p, f, true);

    if(SourceE!=NULL) SourceE(t, f);

    // X- and Y- stresses
    if(visc>0 || visc_array.size()){
        #pragma omp parallel for
        for(int ic=0; ic<NN; ic++){
            tIndex in(ic);
            if(in.IsWall()) { f[in]=double3(); continue; }
            double3 sumflux;

            double h[3][2], invh[3][2];
            for(int idir=0; idir<tIndex::Dim; idir++){
                h[idir][0] = HLeft(in, idir);
                h[idir][1] = HRight(in, idir);
                invh[idir][0] = (h[idir][0]>0.) ? 1./h[idir][0] : 0.;
                invh[idir][1] = (h[idir][1]>0.) ? 1./h[idir][1] : 0.;
            }

            for(int oz=0; oz<=1; oz++){
                //if(h[2][oz]==0.) continue;
                tIndex in_z = in.Neighb(2, oz);
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

                        #if 1 // main
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

                        #else // Try to trick the math
                        const double C1_9 = 1./9., C1_18 = 1./18., C1_36 = 1./36., Cm5_9 = -5./9.;
                        double3 g = (C1_9*(u[in_x]-u[in]) + C1_18*(u[in_xy]-u[in_y]) + C1_18*(u[in_xz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_yz]))*mx +
                                    (C1_9*(u[in_y]-u[in]) + C1_18*(u[in_xy]-u[in_x]) + C1_18*(u[in_yz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_xz]))*my;
                                    //+(Cm5_9*(u[in_z]-u[in]) + C1_18*(u[in_xz]-u[in_x]) + C1_18*(u[in_yz]-u[in_y]) + C1_36*(u[in_xyz]-u[in_xy]))*mz;

                        // nabla_j * (nu nabla_i u_j)
                        double gxx = C1_9*(u[in_x][0]-u[in][0]) + C1_18*(u[in_xy][0]-u[in_y][0]) + C1_18*(u[in_xz][0]-u[in_z][0]) + C1_36*(u[in_xyz][0]-u[in_yz][0]);
                        double gxy = 0.25*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                        //double gxz = 0.25*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                        double gyy = C1_9*(u[in_y][1]-u[in][1]) + C1_18*(u[in_xy][1]-u[in_x][1]) + C1_18*(u[in_yz][1]-u[in_z][1]) + C1_36*(u[in_xyz][1]-u[in_xz][1]);
                        double gyx = 0.25*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                        //double gyz = 0.25*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                        //double gzz = Cm5_9*(u[in_z][2]-u[in][2]) + C1_18*(u[in_yz][2]-u[in_y][2]) + C1_18*(u[in_xz][2]-u[in_x][2]) + C1_36*(u[in_xyz][2]-u[in_xy][2]);
                        double gzx = 0.25*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                        double gzy = 0.25*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));
                        #endif

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

            f[in] += sumflux / GetCellVolume(in);
        }
    }
}


// Implicit velocity term - for methods of type CK (excluding ARS). Pressure should not be used
void S_FD2exim::ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku){
    printf("Not implemented\n"); exit(0);
    //CalcFluxTerm(t, u, p, !DoConv, !DoVisc, false, !DoSourceE, !DoSourceI, ku);
    //for(tIndex in=0; in<NN; ++in) if(in.IsWall()) ku[in] = double3();
}

void S_FD2exim::Init(){
    InitBase();
    if(tIndex::Dim==2 || !IsFourier[0] || !IsFourier[1]) { printf("Wrong configuration\n"); exit(0); }
    TryInitFourier();
    if(!UseFourier) { printf("Wrong configuration\n"); exit(0); }
}
