#ifndef ASRK_H
#define ASRK_H

#include <stdio.h>
#include <math.h>
#include <vector>

// Just an array of three doubles with arithmetic operations
struct double3{
    double V[3];
    inline double3(){ V[0]=V[1]=V[2]=0.; }
    inline double3(const double3& b){ V[0]=b.V[0]; V[1]=b.V[1]; V[2]=b.V[2]; }
    inline double3(double x, double y, double z){ V[0]=x; V[1]=y; V[2]=z; }

    inline operator       double*()       {return V;}
    inline operator const double*() const {return V;}

    inline       double& operator[](int i){ return V[i];}
    inline const double& operator[](int i)const{ return V[i];}
    inline double3& operator*=(const double &x){ V[0]*=x; V[1]*=x; V[2]*=x; return *this; }
    inline double3& operator/=(const double &x){ double inv_x = 1./x; V[0]*=inv_x; V[1]*=inv_x; V[2]*=inv_x; return *this; }
    inline double3& operator+=(const double3& o){ V[0]+=o.V[0]; V[1]+=o.V[1]; V[2]+=o.V[2]; return *this; }
    inline double3& operator-=(const double3& o){ V[0]-=o.V[0]; V[1]-=o.V[1]; V[2]-=o.V[2]; return *this; }
};


inline double3 operator+(const double3 &a, const double3 &b){double3 R(a); return R += b;}
inline double3 operator-(const double3 &a, const double3 &b){double3 R(a); return R -= b;}
inline double3 operator*(const double3 &a, const double &b){double3 R(a); return R *= b;}
inline double3 operator/(const double3 &a, const double &b){double3 R(a); return R /= b;}
inline double3 operator*(const double &a, const double3 &b){double3 R(b); return R *= a;}

inline double DotProd(const double3 &a, const double3 &b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline double abs(const double3 &a) { return sqrt(DotProd(a,a)); }


// Abstract class for an IMEX method (the double Butcher table and related information)
struct tIMEXMethod{
    // Type of the IMEX methods. A1 means that the method is of type A but uses the same IMEX formulation as methods of other types
    enum struct tIMEXtype{ IMEX_A, IMEX_ARS, IMEX_CK, IMEX_A1 } IMEXtype=tIMEXtype::IMEX_A;
    int NStages = 0; // number of stages of the RK method
    std::vector<double> ButcherE, ButcherI; // Butcher tables for the explicit and implicit methods (of size (NStages+1)*NStages)
    double alpha_tau_max[2] = {2.,2.}; // Stability limit in alpha_tau for StabType=0 and StabType=1

    static std::vector<double> CalcD(int NStages, const std::vector<double>& A); // Evaluate coefficients `d' for a Butcher table
    static void OrderCheck(int NumStages, const std::vector<double>& A); // Check of some order conditions for a Butcher table
protected:
    tIMEXMethod() {} // objects of the class tIMEXMethod are not allowed. Objects of derived classes only
};


// Time integrator for the segregated Runge-Kutta method
// A class S describing a spacial approximations should provide the following methods and members:
//      int NN; // Number of degrees of freedom
//      void ApplyGradient(const vector<double>& a, vector<double3>& Ga); // Calculate the gradient of a scalar field
//      void UpapplyGradient(const vector<double3>& v, vector<double>& p); // Calculate the Delta^{-1} Div of a vector field
// Velocity-related operations. Pressure should not be used (unless for approximations like of Rhie-Chow)
// Explicit velocity term
//      void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
// Implicit velocity term (for IMEX_CK)
//      void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
// Implicit velocity stage. Then ku=(u-ustar)/tau_stage
//      void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);
template<class S, typename... S_ARGS>
struct tSRKTimeIntegrator : public S, protected tIMEXMethod {
    using S::NN;
public:
    // Method parameters that may be changed by the user
    double alpha_tau = 0.; // alpha = alpha_tau / tau. A recommended value is set automatically
    int StabType = 1; // (0) Du+Sp=0 or (1) Du+Sdp/dt=0

    // Getters
    inline double get_alpha_tau_max() const{ return alpha_tau_max[StabType?1:0]; }
    inline int get_NStages() const{ return NStages; }
    inline int get_NImplStages() const { return (IMEXtype==tIMEXtype::IMEX_A) ? NStages : NStages-1; }
    inline const std::vector<double>& get_ButcherTable(int i) const{ return i ? ButcherI : ButcherE; }

    tSRKTimeIntegrator() = delete;
    tSRKTimeIntegrator(const tIMEXMethod& T, int _StabType=1, S_ARGS... r) : S(r...), tIMEXMethod(T) {
        StabType = _StabType ? 1:0;
        alpha_tau = 0.5*alpha_tau_max[StabType];
        if(IMEXtype==tIMEXtype::IMEX_CK) d = tIMEXMethod::CalcD(NStages, ButcherI);
    }

private:
    std::vector<double> d; // Coefficients used in methods of type CK (of size NStages)
    // Data on internal stages
    std::vector<std::vector<double3>> Ku, Kuhat, H; // velocity fluxes -- for each Runge-Kutta stage
    std::vector<std::vector<double>> mujtilde, qj, qdot; // scalar data  -- for each Runge-Kutta stage
    std::vector<double3> nu1tilde; // initial momentum residual (for methods of type CK only)
    std::vector<double> buf_scalar, pj, muj, pjhat, pjstar, qjstar;
    std::vector<double3> buf_vector, buf_vector2, gradp, u;

public:
    // This will be called automatically at the first Step() call unless called before
    void AllocDataArraysCK(){
        mujtilde.resize(NStages); for(int istage=0; istage<NStages; istage++) mujtilde[istage].resize(NN);
        qj.resize(NStages);       for(int istage=0; istage<NStages; istage++) qj[istage].resize(NN);
        Ku.resize(NStages);       for(int istage=0; istage<NStages; istage++) Ku[istage].resize(NN);
        Kuhat.resize(NStages);    for(int istage=0; istage<NStages; istage++) Kuhat[istage].resize(NN);
        if(IMEXtype==tIMEXtype::IMEX_CK) nu1tilde.resize(NN);
        buf_scalar.resize(NN);
        pj.resize(NN);
        muj.resize(NN);
        pjhat.resize(NN);
        buf_vector.resize(NN);
        gradp.resize(NN);
        u.resize(NN);
    }
    void AllocDataArraysA(){
        qdot.resize(NStages);     for(int istage=0; istage<NStages; istage++) qdot[istage].resize(NN);
        qj.resize(NStages);       for(int istage=0; istage<NStages; istage++) qj[istage].resize(NN);
        H.resize(NStages);        for(int istage=0; istage<NStages; istage++) H[istage].resize(NN);
        buf_scalar.resize(NN);
        pj.resize(NN);
        pjstar.resize(NN);
        qjstar.resize(NN);
        buf_vector.resize(NN);
        buf_vector2.resize(NN);
        gradp.resize(NN);
        u.resize(NN);
    }
    void AllocDataArrays(){
        if(IMEXtype==tIMEXtype::IMEX_A) return AllocDataArraysA();
        AllocDataArraysCK();
    }

    // Main function -- make one timestep. Input: solution at t_start, output: solution at t_start+tau (overwrites input)
    // For StabType=0, dpdt is ignored (may be not allocated)
    void Step(double t_start, double tau, std::vector<double3>& velocity, std::vector<double>& pressure, std::vector<double>& dpdt){
        if(IMEXtype==tIMEXtype::IMEX_A) return StepA(t_start, tau, velocity, pressure, dpdt);
        if(!buf_scalar.size()) AllocDataArrays();

        const double ass = ButcherI[NStages*NStages-1]; // a_{ss}
        const double inv_ass = 1./ass;
        const double tau_stage = tau*ass;
        const double inv_tau_stage = 1./tau_stage;
        const double alpha = alpha_tau/tau;

        if(IMEXtype==tIMEXtype::IMEX_CK || IMEXtype==tIMEXtype::IMEX_ARS){
            S::ExplicitTerm(t_start, velocity, pressure, Kuhat[0]);
            S::ApplyGradient(pressure, gradp);
            for(int in=0; in<NN; in++) { Ku[0][in] = double3(); Kuhat[0][in] -= gradp[in]; }
        }
        if(IMEXtype==tIMEXtype::IMEX_CK){
            S::ImplicitTerm(t_start, velocity, pressure, Ku[0]);
            for(int in=0; in<NN; in++) nu1tilde[in]=Ku[0][in]+Kuhat[0][in]+alpha*velocity[in];
        }

        const int j0 = IMEXtype==tIMEXtype::IMEX_A1 ? 0 : 1;
        for(int istage=j0; istage<NStages; istage++){
            double c_i = 0.0;
            for(int jstage=0; jstage<=istage; jstage++) c_i += ButcherI[istage*NStages+jstage];
            const double time_stage = t_start + c_i*tau;

            // Velocity step
            for(int in=0; in<NN; in++) buf_vector[in] = velocity[in]; // here buf_vector = u_{j,*}
            for(int jstage=0; jstage<istage; jstage++){
                const double qi = tau*ButcherI[istage*NStages+jstage];
                const double qe = tau*ButcherE[istage*NStages+jstage];
                for(int in=0; in<NN; in++) buf_vector[in] += qi*Ku[jstage][in] + qe*Kuhat[jstage][in];
            }

            S::ImplicitStage(time_stage, tau_stage, buf_vector, pressure, u);
            for(int in=0; in<NN; in++) Ku[istage][in] = (u[in] - buf_vector[in]) * inv_tau_stage;

            // Explicit velocity term. Pressure gradient will be substracted later
            S::ExplicitTerm(time_stage, u, pressure, Kuhat[istage]);

            // Pressure step.
            // Evaluating the right-hand side of the pressure system in buf_vector
            if(StabType==0){
                for(int in=0; in<NN; in++) {
                    pjhat[in] = 0.;
                    muj[in] = pressure[in];
                    if(IMEXtype==tIMEXtype::IMEX_CK) muj[in] *= (1. - alpha_tau*ButcherI[istage*NStages]);
                    for(int jstage=j0; jstage<istage; jstage++) muj[in] += inv_ass*ButcherI[istage*NStages+jstage]*mujtilde[jstage][in];
                    buf_scalar[in] = muj[in] - alpha*tau_stage*pressure[in];
                }
            }
            else{
                for(int in=0; in<NN; in++) {
                    muj[in] = tau_stage*dpdt[in];
                    if(IMEXtype==tIMEXtype::IMEX_CK) muj[in] *= (1. - alpha_tau*ButcherI[istage*NStages]);
                    pjhat[in] = pressure[in];
                    if(IMEXtype==tIMEXtype::IMEX_CK) pjhat[in] += tau*ButcherI[istage*NStages]*dpdt[in];
                    for(int jstage=j0; jstage<istage; jstage++){
                        muj[in] += inv_ass*ButcherI[istage*NStages+jstage]*mujtilde[jstage][in];
                        pjhat[in] += tau*ButcherI[istage*NStages+jstage]*qj[jstage][in];
                    }
                    buf_scalar[in] = muj[in] + pjhat[in] - alpha*tau_stage*tau_stage*dpdt[in];
                }
            }
            S::ApplyGradient(buf_scalar, buf_vector);

            for(int in=0; in<NN; in++){
                buf_vector[in] = Ku[istage][in] + Kuhat[istage][in] - buf_vector[in] + alpha*velocity[in];
                if(IMEXtype==tIMEXtype::IMEX_CK) buf_vector[in] -= d[istage]*nu1tilde[in];
            }
            // Here we finally solve the pressure system
            S::UnapplyGradient(buf_vector, mujtilde[istage]);

            for(int in=0; in<NN; in++){
                pj[in] = mujtilde[istage][in] + buf_scalar[in];
                mujtilde[istage][in] -= alpha*tau_stage*(StabType ? tau_stage*dpdt[in] : pressure[in]);
                if(StabType) qj[istage][in] = inv_tau_stage*(pj[in]-pjhat[in]);
            }
            S::ApplyGradient(pj, buf_vector);
            for(int in=0; in<NN; in++) Kuhat[istage][in] -= buf_vector[in];
        }

        // Final solution
        for(int jstage=0; jstage<NStages; jstage++){
            const double qi = tau*ButcherI[NStages*NStages+jstage];
            const double qe = tau*ButcherE[NStages*NStages+jstage];
            for(int in=0; in<NN; in++) velocity[in] += qi*Ku[jstage][in] + qe*Kuhat[jstage][in];
        }
        for(int in=0; in<NN; in++){
            pressure[in] = pj[in]; // take the last obtained pressure
            if(StabType) dpdt[in] = qj[NStages-1][in];
        }
    }


    // Main function -- make one timestep. Input: solution at t_start, output: solution at t_start+tau (overwrites input)
    // For StabType=0, dpdt is ignored (may be not allocated)
    void StepA(double t_start, double tau, std::vector<double3>& velocity, std::vector<double>& pressure, std::vector<double>& dpdt){
        if(IMEXtype!=tIMEXtype::IMEX_A){ printf("StepA: only for methods of type A\n"); exit(0); }
        if(!H.size()) AllocDataArraysA();

        const double ass = ButcherI[NStages*NStages-1]; // a_{ss}
        const double tau_stage = tau*ass;
        const double inv_tau_stage = 1./tau_stage;
        const double alpha = alpha_tau/tau;

        for(int istage=0; istage<NStages; istage++){
            double c_i = 0.0;
            for(int jstage=0; jstage<=istage; jstage++) c_i += ButcherI[istage*NStages+jstage];
            const double time_stage = t_start + c_i*tau;

            // Explicit velocity step
            // Evaluate buf_vector = u_{j}^{(E)} and buf_scalar = p_{j}^{(E)}
            for(int in=0; in<NN; in++){
                buf_vector[in] = velocity[in];
                buf_scalar[in] = pressure[in];
                for(int jstage=0; jstage<istage; jstage++){
                    const double qe = tau*ButcherE[istage*NStages+jstage];
                    buf_vector[in] += qe*H[jstage][in];
                    buf_scalar[in] += qe*qj[jstage][in];
                }
            }
            // Explicit fluxes
            S::ExplicitTerm(istage==0 ? t_start : time_stage, buf_vector, buf_scalar, buf_vector2);
            // Pressure gradient
            S::ApplyGradient(buf_scalar, gradp);
            for(int in=0; in<NN; in++) H[istage][in] = buf_vector2[in] - gradp[in];

            // Implicit velocity step
            // Evaluate buf_vector = u_{j,*}
            for(int in=0; in<NN; in++){
                buf_vector[in] = velocity[in];
                for(int jstage=0; jstage<=istage; jstage++){ // at jstage=istage, we use the explicit term just stored in H[istage]
                    double qi = tau*ButcherI[istage*NStages+jstage];
                    buf_vector[in] += qi*H[jstage][in];
                }
            }
            // Implicit fluxes
            S::ImplicitStage(time_stage, tau_stage, buf_vector, buf_scalar, u);
            for(int in=0; in<NN; in++){
                buf_vector[in] = (u[in] - buf_vector[in]) * inv_tau_stage; // these are implicit fluxes now
                H[istage][in] += buf_vector[in];
            }
            // Explicit fluxes for the updated velocity
            int DoRecalcExplicitFluxes = 1; // (istage==0); // recalc explicit fluxes? For istage==0 obligatory because of time shift
            if(DoRecalcExplicitFluxes) S::ExplicitTerm(time_stage, u, buf_scalar, buf_vector2);
            for(int in=0; in<NN; in++) buf_vector[in] += buf_vector2[in]; // sum of the explicit and implicit fluxes

            // Pressure step.
            // Evaluating the right-hand side of the pressure system in buf_vector
            if(StabType==0){
                for(int in=0; in<NN; in++) {
                    pjstar[in] = pressure[in];
                    for(int jstage=0; jstage<istage; jstage++){
                        double qi = tau*ButcherI[istage*NStages+jstage];
                        pjstar[in] += qi*qj[jstage][in];
                    }
                    buf_scalar[in] = pjstar[in] - alpha*tau_stage*pressure[in];
                }
            }
            else{
                for(int in=0; in<NN; in++) {
                    pjstar[in] = pressure[in];
                    qjstar[in] = dpdt[in];
                    for(int jstage=0; jstage<istage; jstage++){
                        double qi = tau*ButcherI[istage*NStages+jstage];
                        pjstar[in] += qi*qj[jstage][in];
                        qjstar[in] += qi*qdot[jstage][in];
                    }
                    buf_scalar[in] = pjstar[in] + qjstar[in] * tau_stage - alpha*tau_stage*tau_stage*dpdt[in];
                }
            }
            S::ApplyGradient(buf_scalar, gradp);
            for(int in=0; in<NN; in++) buf_vector[in] = buf_vector[in] - gradp[in] + alpha*velocity[in];

            // Finally solve the pressure system
            S::UnapplyGradient(buf_vector, pj); // Here pj is not pressure

            for(int in=0; in<NN; in++){
                pj[in] += buf_scalar[in]; // now pj contains pressure
                qj[istage][in] = inv_tau_stage*(pj[in]-pjstar[in]);
                if(StabType==1) qdot[istage][in] = inv_tau_stage*(qj[istage][in]-qjstar[in]);
            }
        }

        // Final solution
        for(int jstage=0; jstage<NStages; jstage++){
            double qi = tau*ButcherI[NStages*NStages+jstage];
            for(int in=0; in<NN; in++) velocity[in] += qi*H[jstage][in];
        }
        for(int in=0; in<NN; in++){
            pressure[in] = pj[in]; // take the last obtained pressure
            if(StabType) dpdt[in] = qj[NStages-1][in];
        }
    }

    // Evaluate the continuity residual: L^{-1}(Du+Sp) = L^{-1}D(u+Gp) - p
    // In the case of type 0 stabilization, P is pressure, and in the case of type 1 stabilization, P is dp/dt
    // To evaluate L^{-1}Du, pass non-allocated array as P
    void CalcContinuityResidual(const std::vector<double3>& velocity, const std::vector<double>& P, std::vector<double>& R){
        std::vector<double3> GradP(NN);
        if(P.size()) S::ApplyGradient(P, GradP);
        for(int in=0; in<NN; in++) GradP[in] += velocity[in];
        S::UnapplyGradient(GradP, R);
        if(P.size()) for(int in=0; in<NN; in++) R[in] -= P[in];
    }
};

// Specific IMEX methods of type ARS
// Ascher, Ruuth, Spiteri. IMEX RK method for time-dependent PDEs. 1997
struct ARS_121: tIMEXMethod{ ARS_121(); };
struct ARS_232: tIMEXMethod{ ARS_232(); };
struct ARS_343: tIMEXMethod{ ARS_343(); };
// Other methods from Ascher, Ruuth, Spiteri. IMEX RK method for time-dependent PDEs. 1997
// that satisfy \hat{b} = \hat{a}_{s,*} instead of b = \hat{b}
struct ARS_111: tIMEXMethod{ ARS_111(); };
struct ARS_222: tIMEXMethod{ ARS_222(); };
struct ARS_443: tIMEXMethod{ ARS_443(); };
// Boscarino, Russo. On the uniform accuracy of IMEX RK schemes and applications to hyperbolic systems with relaxation. 2007
struct MARS_343: tIMEXMethod{ MARS_343(); }; // MARS (=modified ARS) (3,4,3) method

// Specific IMEX methods of type CK
// Kennedy, Carpenter. Additive Runge-Kutta schemes for convection-diffusion-reaction equations. 2003
struct ARK3 : tIMEXMethod{ ARK3(); }; // ARK3(2)4L[2]SA scheme
struct ARK4 : tIMEXMethod{ ARK4(); }; // ARK4(3)6L[2]SA scheme
struct ARK5 : tIMEXMethod{ ARK5(); }; // ARK5(4)8L[2]SA scheme
// Boscarino, Russo. On the uniform accuracy of IMEX RK schemes and applications to hyperbolic systems with relaxation. 2007
struct MARK3: tIMEXMethod{ MARK3(); }; // MARK3(2)4L[2]SA scheme
// Boscarino. On an accurate third order IMEX RK method for stiff problems. 2009
struct BHR_553: tIMEXMethod{ BHR_553(); };

// Specific IMEX methods of type A
// Pareschi, Russo. High order asymptotically strong-stability-preserving methods for hyperbolic systems with stiff relaxation. 2003
struct SSP2_322: tIMEXMethod{ SSP2_322(); };
// Boscarino Qiu Russo Xiong. High Order Semi-implicit WENO Schemes for All Mach Full Euler System of Gas Dynamics. 2021
struct SI_IMEX_332: tIMEXMethod{ SI_IMEX_332(); };
struct SI_IMEX_443: tIMEXMethod{ SI_IMEX_443(); }; // this method may be in error
// Boscarino Cho. Asymptotic Analysis of IMEX-RK Methods for ES-BGK Model at Navier-Stokes level. 2024
struct SI_IMEX_433: tIMEXMethod{ SI_IMEX_433(); };

// Methods of type A -- but with the use of the same IMEX formulation as methods of other types
struct SSP2_322_A1: SSP2_322{ SSP2_322_A1(){ IMEXtype=tIMEXtype::IMEX_A1; alpha_tau_max[0] = alpha_tau_max[1] = 2.; }};
struct SI_IMEX_332_A1: SI_IMEX_332{ SI_IMEX_332_A1(){ IMEXtype=tIMEXtype::IMEX_A1; alpha_tau_max[0] = alpha_tau_max[1] = 2.; }};
struct SI_IMEX_443_A1: SI_IMEX_443{ SI_IMEX_443_A1(){ IMEXtype=tIMEXtype::IMEX_A1; alpha_tau_max[0] = alpha_tau_max[1] = 2.; }};
struct SI_IMEX_433_A1: SI_IMEX_433{ SI_IMEX_433_A1(){ IMEXtype=tIMEXtype::IMEX_A1; alpha_tau_max[0] = alpha_tau_max[1] = 2.; }};

#endif
