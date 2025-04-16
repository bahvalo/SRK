// Basic ("second-order") finite difference discretization on non-uniform meshes (2D and 3D)
// with the Dirichlet or periodic boundary conditions
#include "asrk.h"
#include "fd2.h"

#include "linsolver.h"

// Implicit velocity stage. The goal is to solve
//   u = ustar + tau_stage*F(u)
// For this purpose, write the iterative process
//   u(0) = ustar
//   ( I - tau_stage * [[dF/du(u(s))]] ) (u(s+1)-u(s)) = tau_stage*F(u(s)) - (u(s) - ustar)
// where [[...]] means an approximation. If F is a linear function of u and [[dF/du]] is the exact Jacobian,
// then u(1) is the exact solution and u(s)=u(1) for each s>=1.
// Here fluxes are divided by cell volume

void S_FD2::ImplicitStage(double time_stage, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u){
    // Initial guess and specifying the velocities on boundary
    u = ustar;

    if(SourceI!=NULL){ // Even if both convective and viscous terms are taken explicitly, we may have a source term here (just for test, never used)
        vector<double3> source(NN);
        SourceI(time_stage, source);
        for(tIndex in=0; in<NN; ++in) if(!in.IsWall()) u[in] += tau_stage*source[in];
    }
    if(!DoNotUseDBoundaryValue){ // Main branch
        if(dBoundaryValue!=NULL){ // if the boundary velocity is steady, then nothing to do
            for(tIndex in=0; in<NN; ++in){
                if(!in.IsWall()) continue;
                double3 coor = GetCoor(in);
                u[in] += dBoundaryValue(time_stage, coor)*tau_stage;
            }
        }
    }
    else{ // for test
        for(tIndex in=0; in<NN; ++in){
            if(!in.IsWall()) continue;
            double3 coor = GetCoor(in);
            if(BoundaryValue!=NULL) u[in] = BoundaryValue(time_stage, coor);
            else u[in] = double3();
        }
    }

    if(TimeIntMethod==tTimeIntMethod::EXPLICIT) return; // F(u)=0

    const int ROW_SIZE = 7;
    vector<double> L(NN*ROW_SIZE); // matrix of the velocity equation
    vector<double3> R(NN); // right-hand side
    vector<double> _R(NN); // right-hand side - one component
    vector<double> sol(NN);

    int NumIters_loc = (TimeIntMethod==tTimeIntMethod::IMPLICIT) ? NumIters_Impl : NumIters_IMEX;
    for(int iter=0; iter<NumIters_loc; iter++){
        // Implicit fluxes
        bool DoConv = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
        bool DoVisc = TimeIntMethod==tTimeIntMethod::EXPLICIT;
        CalcFluxTerm(time_stage, u, p, !DoConv, !DoVisc, false, true, R);

        // Replace the equations for boundary nodes to (DeltaU)_j = 0
        for(tIndex in=0; in<NN; ++in){
            R[in] = R[in]*tau_stage - (u[in]-ustar[in]);
            if(in.IsWall()) R[in] = double3();
        }

        // debug information -- residual norm -- to check the convergence
        if(0){
            double residual_norm = 0.;
            for(int in=0; in<NN; ++in) residual_norm += DotProd(R[in], R[in]);
            printf("iter=%i resid=%e\n", iter, sqrt(residual_norm));
        }

        if(ViscScheme==tViscScheme::CROSS){
            // Fill the matrix
            for(tIndex in=0; in<NN; ++in){
                double* row = L.data() + in*ROW_SIZE;
                row[0] = 1.;
                for(int i=1; i<ROW_SIZE; i++) row[i]=0.;
                if(in.IsWall()) continue;

                //double3 sumflux;
                double3 hbar = GetHbar(in);
                for(int idir=0; idir<Dim; idir++){
                    // Convection
                    if(TimeIntMethod==tTimeIntMethod::IMPLICIT && EnableConvection){
                        // Neighboring nodes -- well defined because wall nodes are skipped
                        tIndex jn = in.Neighb(idir,0);
                        tIndex kn = in.Neighb(idir,1);
                        double un = 0.5*(u[kn][idir]+u[in][idir])*tau_stage / hbar[idir];
                        if(un>=0) row[0] += un;
                        else row[idir*2+2] += un;
                        un = -0.5*(u[jn][idir]+u[in][idir])*tau_stage / hbar[idir];
                        if(un>=0) row[0] += un;
                        else row[idir*2+1] += un;
                    }

                    // Viscosity
                    double hplus = HRight(in,idir), hminus = HLeft(in,idir);
                    double mL = visc*tau_stage/(hminus*hbar[idir]), mR = visc*tau_stage/(hplus*hbar[idir]);

                    row[idir*2+1]-= mL;
                    row[idir*2+2]-= mR;
                    row[0]       += (mL+mR);
                }
            }
            LinearSystemInit(1, L); // pass the matrix to the external linear algebra solver

            for(int jdir=0; jdir<Dim; jdir++){
                for(int in=0; in<NN; in++) { _R[in]=R[in][jdir]; sol[in]=0.; }
                LinearSystemSolve(1 /*velocity system*/, sol, _R, AMG_Tolerance_U, AMG_MaxIters_U);
                for(tIndex in=0; in<NN; ++in) if(!in.IsWall()) u[in][jdir]+=sol[in];
            }
        }
        else{
            // Here all velocity components are coupled. Whatever we write here, is an approximation (unless we write a block matrix, which we don't do here)
            // We write tree systems, each for one velocity component (in 3D, two of them actually have the same matrix)
            for(int jdir=0; jdir<Dim; jdir++){
                for(tIndex in=0; in<NN; ++in){
                    double* row = L.data() + in*ROW_SIZE;
                    row[0] = 1.;
                    for(int i=1; i<ROW_SIZE; i++) row[i]=0.;
                    if(in.IsWall()) continue;

                    double3 sumflux;
                    double3 hbar = GetHbar(in);
                    for(int idir=0; idir<Dim; idir++){
                        tIndex jn = in.Neighb(idir,0);
                        tIndex kn = in.Neighb(idir,1);

                        // Convection
                        if(TimeIntMethod==tTimeIntMethod::IMPLICIT && EnableConvection){
                            // Neighboring nodes -- well defined because wall nodes are skipped
                            double un = 0.5*(u[kn][idir]+u[in][idir])*tau_stage / hbar[idir];
                            if(un>=0) row[0] += un;
                            else row[idir*2+2] += un;
                            un = -0.5*(u[jn][idir]+u[in][idir])*tau_stage / hbar[idir];
                            if(un>=0) row[0] += un;
                            else row[idir*2+1] += un;
                        }

                        // Viscosity
                        double viscL = visc, viscR = visc;
                        if(visc_array.size()){
                            viscL += AverageE2Edge(visc_array, jn, idir);
                            viscR += AverageE2Edge(visc_array, in, idir);
                        }
                        if(idir==jdir) { viscL*=2.; viscR*=2.; }
                        double hplus = HRight(in,idir), hminus = HLeft(in,idir);
                        double mL = viscL*tau_stage/(hminus*hbar[idir]), mR = viscR*tau_stage/(hplus*hbar[idir]);

                        row[idir*2+1]-= mL;
                        row[idir*2+2]-= mR;
                        row[0]       += (mL+mR);
                    }
                }
                LinearSystemInit(1, L); // pass the matrix to the external linear algebra solver
                for(int in=0; in<NN; in++) { _R[in]=R[in][jdir]; sol[in]=0.; }
                LinearSystemSolve(1 /*velocity system*/, sol, _R, AMG_Tolerance_U, AMG_MaxIters_U);
                for(int in=0; in<NN; in++) u[in][jdir]+=sol[in]; // this affects the convection velocity for the velocity components to be processed, but nobody cares
            }
        }
    }
}


// Calculate the convection term
void S_FD2::CalcConvTerm(const vector<double3>& u, const vector<double>& p, vector<double3>& f, int Nullify){
    if(!EnableConvection) {
        if(Nullify) for(int in=0; in<NN; in++) f[in]=double3();
        return;
    }

    #pragma omp parallel for
    for(int _in=0; _in<NN; _in++){
        tIndex in(_in);
        if(in.IsWall()) { if(Nullify) f[in]=double3(); continue; }

        double3 hbar = GetHbar(in);
        double3 sumflux;
        for(int idir=0; idir<Dim; idir++){
            // Neighboring nodes -- well defined because wall nodes are skipped
            tIndex jn = in.Neighb(idir,0);
            tIndex kn = in.Neighb(idir,1);

            // Convection
            if(ConvMethod==tConvMethod::CONVECTIVE){
                double3 conv_adv = u[in][idir] * 0.5*(u[kn]-u[jn]);
                sumflux += conv_adv / hbar[idir];
            }
            else if(ConvMethod==tConvMethod::SKEWSYMMETRIC){
                double3 conv_div = 0.5*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]);
                double3 conv_adv = u[in][idir] * 0.5*(u[kn]-u[jn]);
                sumflux += 0.5*(conv_div + conv_adv) / hbar[idir];
            }
            else{
                // Velocities at center points
                double3 umidR = 0.5*(u[in]+u[kn]);
                double3 umidL = 0.5*(u[in]+u[jn]);
                // Fluxes
                double3 fluxR = umidR[idir]*umidR;
                double3 fluxL = umidL[idir]*umidL;

                sumflux += (fluxR - fluxL) / hbar[idir];

                if(ConvMethod==tConvMethod::EMAC){
                    sumflux[idir] += 0.25 * (DotProd(u[kn],u[kn]) - DotProd(u[jn],u[jn])) / hbar[idir];
                }
            }
        }
        if(Nullify) f[in] = sumflux*(-1.);
        else f[in] -= sumflux;
    }
}


// Calculate the viscous term
void S_FD2::CalcViscTerm(const vector<double3>& u, vector<double3>& f){
    if(!(visc>0. || visc_array.size())) return;

    // Standard FD discretization -- for the constant viscosity only
    if(ViscScheme==tViscScheme::CROSS){
        if(visc_array.size()) { printf("Standard FD discretization -- for the constant viscosity only\n"); exit(0); }
        return CalcViscTermCross(u, visc, f, false /*Do not multiply by CellVolume*/, false);
    }

    // Finite-element discretization
    for(tIndex in=0; in<NN; ++in) f[in]*=GetCellVolume(in);
    if(ViscScheme==tViscScheme::AES) CalcViscTermAES(u, visc, visc_array, f, false);
    if(ViscScheme==tViscScheme::GALERKIN) CalcViscTermGalerkin(u, visc, visc_array, f, false);
    for(tIndex in=0; in<NN; ++in) f[in]/=GetCellVolume(in);
}

// Calculate terms in the momentum equation (divided by cell volume)
// Behavior for nodes with the Dirichlet conditions is undefined (the result should be nullified elsewhere)
void S_FD2::CalcFluxTerm(double t, const vector<double3>& u, const vector<double>& p, bool DoConv, bool DoVisc, bool DoSourceE, bool DoSourceI, vector<double3>& f){
    if(DoConv) CalcConvTerm(u, p, f, true);
    else for(int in=0; in<NN; in++) f[in]=double3();

    if(DoSourceE && SourceE!=NULL) SourceE(t, f);
    if(DoSourceI && SourceI!=NULL) SourceI(t, f);

    if(DoVisc) CalcViscTerm(u, f);
}

// Explicit velocity stage
void S_FD2::ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat){
    bool DoConv = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoVisc = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    CalcFluxTerm(t, u, p, DoConv, DoVisc, true, false, kuhat);
    for(tIndex in=0; in<NN; ++in) if(in.IsWall()) kuhat[in] = double3();
}

// Implicit velocity term - for methods of type CK (excluding ARS). Pressure should not be used
void S_FD2::ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku){
    if(DoNotUseDBoundaryValue){ printf("ImplicitTerm: not possible if DoNotUseDBoundaryValue flag set\n"); exit(0); }
    bool DoConv = TimeIntMethod==tTimeIntMethod::EXPLICIT || TimeIntMethod==tTimeIntMethod::IMEX;
    bool DoVisc = TimeIntMethod==tTimeIntMethod::EXPLICIT;
    CalcFluxTerm(t, u, p, !DoConv, !DoVisc, false, true, ku);
    for(tIndex in=0; in<NN; ++in){
        if(!in.IsWall()) continue;
        if(dBoundaryValue) ku[in] = dBoundaryValue(t, GetCoor(in));
        else ku[in] = double3();
    }
}

// Apply the gradient operator
void S_FD2::ApplyGradient(const vector<double>& a, vector<double3>& R){
    #pragma omp parallel for
    for(int _in=0; _in<NN; _in++){
        tIndex in(_in);
        if(in.IsWall()) { R[in]=double3(); continue; }

        double3 hbar = GetHbar(in);
        for(int idir=0; idir<Dim; idir++){
            tIndex jn = in.Neighb(idir,0); if(jn==-1) jn=in;
            tIndex kn = in.Neighb(idir,1); if(kn==-1) kn=in;
            R[in][idir] = (a[kn] - a[jn]) / (2.*hbar[idir]);
        }
    }
}

// Apply the divergence operator to form the right-hand side of the pressure equation and solve it
void S_FD2::UnapplyGradient(const vector<double3>& v, vector<double>& InvLa){
    vector<double> Dv(NN);

    #pragma omp parallel for
    for(int _in=0; _in<NN; _in++){
        tIndex in(_in);
        Dv[in]=0.;

        double3 hbar = GetHbar(in);
        for(int idir=0; idir<Dim; idir++){
            tIndex jn = in.Neighb(idir,0); if(jn==-1) jn=in;
            tIndex kn = in.Neighb(idir,1); if(kn==-1) kn=in;
            Dv[in] += (v[kn][idir] - v[jn][idir]) / (2.*hbar[idir]);
        }
        Dv[in] *= (hbar[0]*hbar[1]*hbar[2]);
    }

    if(!UseFourier){
        // Solve the system with the scaled Laplace operator
        LinearSystemSolve(0 /*pressure system*/, InvLa, Dv, AMG_Tolerance_P, AMG_MaxIters_P);
    }
    else{
        // Solve the system using FFT
        LinearSystemSolveFourier(Dv, InvLa, true);
    }
}



void S_FD2::Init(){
    InitBase();

    TryInitFourier();
    if(!UseFourier) PassDefaultPressureSystemToHYPRE(true); // Can't use FFTW, will use HYPRE

    LinearSolverAlloc(1, N); // HYPRE data for the velocity system
}

// Normalize pressure
void S_FD2::NormalizePressure(vector<double>& p) const{
    double psum = 0., ssum=0.;
    for(tIndex in=0; in<NN; ++in){
        if(in.IsCornerNode()) continue;
        double V = GetCellVolume(in);
        psum += p[in]*V;
        ssum += V;
    }
    double p_shift = psum / ssum;
    for(tIndex in=0; in<NN; ++in){
        if(in.IsCornerNode()) p[in]=0.;
        else p[in] -= p_shift;
    }
}

double S_FD2::CalcKineticEnergy(const vector<double3>& u) const{
    double sum=0.;
    for(tIndex in=0; in<NN; ++in){
        double3 hbar = GetHbar(in);
        double V = hbar[0]*hbar[1]*hbar[2];
        sum+=DotProd(u[in],u[in])*V;
    }
    return 0.5*sum;
}

double3 S_FD2::CalcIntegral(const vector<double3>& u) const{
    double3 sum;
    for(tIndex in=0; in<NN; ++in){
        double3 hbar = GetHbar(in);
        double V = hbar[0]*hbar[1]*hbar[2];
        sum+=u[in]*V;
    }
    return sum;
}



// Calculate flux terms on walls (to get friction)
void S_FD2::CalcFluxTermOnWalls(const vector<double3>& u, const vector<double>& p, double3& F1, double3& F2, vector<double3>& f){
    if(Dim==2 || !IsPer[0] || !IsPer[1] || IsPer[2]) { printf("Not implemented\n"); exit(0); }
    if(ConvMethod!=tConvMethod::CONSERVATIVE){ printf("Not implemented\n"); exit(0); }

    CalcViscTermOnWalls_AES(u, F1, F2, f);

    const int NXY = N[0]*N[1];
    for(int iiz=0; iiz<2; iiz++){ // bottom and top surfaces
        for(int ixy=0; ixy<NXY; ixy++){
            tIndex in(ixy + iiz*(N[2]-1)*NXY);
            tIndex in_z = in.Neighb(2, !iiz);

            double3 sumflux;
            {
                double3 hbar = GetHbar(in);
                double S = hbar[0]*hbar[1];
                if(iiz) S *= -1.;

                // Convection
                double3 umid = 0.5*(u[in]+u[in_z]);
                sumflux += umid[2]*umid * S;

                // Pressure -- yeilds the normal component only
                sumflux[2] += 0.5*(p[in_z]+p[in]) * S;
            }

            if(f.size()) f[in] += sumflux;
            if(iiz) F2 += sumflux;
            else F1 += sumflux;
        }
    }
}
