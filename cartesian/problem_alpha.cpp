// Evaluate the stability range (in parameter alpha*tau) for a given time integration method

#include "asrk.h"
using namespace std;
#include <complex>
static double MAX(double x, double y) { return (x>y) ? x:y; }
static double StepInBeta = 1./512.;

// Dummy spatial approximation with no flux term, to simulate a single potential velocity mode
struct S_problem_alpha{
    int NN = 1; // Number of degrees of freedom
    void ApplyGradient(const vector<double>& a, vector<double3>& R) { R[0] = double3(a[0],0.,0.); }
    void UnapplyGradient(const vector<double3>& v, vector<double>& p) { p[0] = (1.-Beta)*v[0][0]; }
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat) { kuhat[0] = double3(); }
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku) { ku[0] = double3(); }
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u) { u[0] = ustar[0]; }

    double Beta = 0.; // Beta=1 => no stabilization limit; Beta=0 => checkerboard limit
};

// Dummy spatial approximation to verify the method on the zero-pressure mode (equation du1/dt=-u2, du2/dt=u1). Explicit scheme
struct S_problem_explicit{
    int NN = 1; // Number of degrees of freedom
    void ApplyGradient(const vector<double>& a, vector<double3>& R) { R[0]=double3(); }
    void UnapplyGradient(const vector<double3>& v, vector<double>& p) { p[0]=0.0; }
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat) { kuhat[0]=double3(-u[0][1],u[0][0],0.); }
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku) { ku[0] = double3(); }
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u) { u[0] = ustar[0]; }
};


typedef tSRKTimeIntegrator<S_problem_alpha> tSRK;
typedef tSRKTimeIntegrator<S_problem_explicit> tSRK_E;

double GetMaxLambda_StabType0(tSRK& T, bool OnlyNoStabilizationLimit=false){
    vector<double3> velocity(1);
    vector<double> pressure(1);
    vector<double> qressure; // not allocated
    double& v = velocity[0][0];
    double& p = pressure[0];

    T.StabType = 0;
    double maxlambda = 0.;

    for(T.Beta=0.; T.Beta<=1.; T.Beta+=StepInBeta){
       //if(OnlyNoStabilizationLimit) T.Beta=1.;
        double M[4];
        p=1; v=0; T.Step(0,1,velocity,pressure,qressure); M[0]=p; M[1]=v;
        p=0; v=1; T.Step(0,1,velocity,pressure,qressure); M[2]=p; M[3]=v;

        double det = M[0]*M[3]-M[1]*M[2];
        double tr = M[0]+M[3];
        double discr = tr*tr - 4.*det;
        complex<double> sqrt_discr = sqrt(complex<double>(discr));
        complex<double> lambda1 = 0.5*(tr + sqrt_discr);
        complex<double> lambda2 = 0.5*(tr - sqrt_discr);
        double ret_val = MAX(abs(lambda1), abs(lambda2));
        maxlambda = MAX(maxlambda, ret_val);

        if(OnlyNoStabilizationLimit) break;
    }
    return maxlambda;
}

// Returns the maximal absolute value of the three roots of z^3 + bz^2 + cz + d = 0
double max_abs_value(double b, double c, double d){
    double Delta0 = b*b-3.*c;
    double Delta1 = 2.*b*b*b-9.*b*c+27.*d;
    complex<double> aux1 = Delta1*Delta1 - 4.*Delta0*Delta0*Delta0;
    complex<double> aux2 = 0.5*(Delta1 + sqrt(aux1));
    complex<double> C = pow(aux2, 1.0 / 3.0);
    complex<double> rotate = exp(complex<double>(0,atan(1.)*8./3.));
    double ret_val = 0.;
    for (int k = 0; k < 3; k++) {
        complex<double> root = -(b+C+Delta0/C)/3.;
        if(abs(root)>ret_val) ret_val = abs(root);
        C *= rotate;
    }
    return ret_val;
}

double GetMaxLambda_StabType1(tSRK& T, bool OnlyNoStabilizationLimit=false){
    vector<double3> velocity(1);
    vector<double> pressure(1), qressure(1);
    double& v = velocity[0][0];
    double& p = pressure[0];
    double& q = qressure[0];

    T.StabType = 1;
    double maxlambda = 0.;

    for(T.Beta=0.; T.Beta<=1.; T.Beta+=StepInBeta){
        //if(OnlyNoStabilizationLimit) T.Beta=1.;
        //T.Beta = 1.-1e-6;

        double M[9];
        q=1; p=0; v=0; T.Step(0,1,velocity,pressure,qressure); M[0]=q; M[1]=p; M[2]=v;
        q=0; p=1; v=0; T.Step(0,1,velocity,pressure,qressure); M[3]=q; M[4]=p; M[5]=v;
        q=0; p=0; v=1; T.Step(0,1,velocity,pressure,qressure); M[6]=q; M[7]=p; M[8]=v;

        double b = -(M[0]+M[4]+M[8]);
        double c = -(- (M[4]*M[8]-M[5]*M[7]) - (M[0]*M[8]-M[2]*M[6]) - (M[0]*M[4]-M[1]*M[3]));
        double d = -(M[0]*(M[4]*M[8]-M[5]*M[7]) + M[1]*(M[5]*M[6]-M[3]*M[8]) + M[2]*(M[3]*M[7]-M[4]*M[6]));

        double ret_val = max_abs_value(b,c,d);
        maxlambda = MAX(maxlambda, ret_val);

        if(OnlyNoStabilizationLimit) break;
    }
    return maxlambda;
}


double GetStabFunc(tSRK_E& T, double tau, int StabType){
    vector<double3> velocity(1);
    vector<double> pressure(1);
    vector<double> qressure(1);
    double3& v = velocity[0];

    T.StabType = StabType;

    double M[4];
    v=double3(1.,0.,0.); T.Step(0,tau,velocity,pressure,qressure); M[0]=v[0]; M[1]=v[1];
    v=double3(0.,1.,0.); T.Step(0,tau,velocity,pressure,qressure); M[2]=v[0]; M[3]=v[1];

    double det = M[0]*M[3]-M[1]*M[2];
    double tr = M[0]+M[3];
    double discr = tr*tr - 4.*det;
    complex<double> sqrt_discr = sqrt(complex<double>(discr));
    complex<double> lambda1 = 0.5*(tr + sqrt_discr);
    complex<double> lambda2 = 0.5*(tr - sqrt_discr);
    double ret_val = MAX(abs(lambda1), abs(lambda2));
    return ret_val;
}



void main_alpha_range(const tIMEXMethod& IMEX, int print_tables){
    if(print_tables){
        const int NStages = IMEX.NStages;
        // Print the Butcher tables
        printf("ButcherI:\n");
        for(int j=0; j<=NStages; j++){
            double cj = 0.; for(int k=0; k<NStages; k++) cj += IMEX.ButcherI[j*NStages+k];
            printf("%.10f | ", cj);
            for(int k=0; k<NStages; k++) printf("%.4f ", IMEX.ButcherI[j*NStages+k]);
            printf("\n");
        }
        tIMEXMethod::OrderCheck(NStages, IMEX.ButcherI);
        printf("\nButcherE:\n");
        for(int j=0; j<=NStages; j++){
            double cj = 0.; for(int k=0; k<NStages; k++) cj += IMEX.ButcherE[j*NStages+k];
            printf("%.10f | ", cj);
            for(int k=0; k<NStages; k++) printf("%.4f ", IMEX.ButcherE[j*NStages+k]);
            printf("\n");
        }
        tIMEXMethod::OrderCheck(NStages, IMEX.ButcherE);
        printf("\n");
    }

    bool OnlyNoStabilizationLimit = false; // check the case Beta=0 only

    // StabType: (0) Du+Sp=0 or (1) Du+Sdp/dt=0
    // T.alpha_tau: pressure stabilization parameter (alpha = alpha_tau/tau)
    for(int StabType = 0; StabType<=1; StabType++){
        tSRK T(IMEX, StabType);
        tSRK_E TE(IMEX, StabType);

        if(0){
            for(T.alpha_tau=0.; T.alpha_tau<=3.; T.alpha_tau+=0.04){
                double maxlambda = StabType ? GetMaxLambda_StabType1(T, OnlyNoStabilizationLimit) : GetMaxLambda_StabType0(T, OnlyNoStabilizationLimit);
                printf("alpha=%f, max=%f\n", T.alpha_tau, maxlambda);
            }
        }
        if(StabType==0){ // does not depend on stabtype, so do int once
            double tau_max = -1.;
            for(double tau=5.; tau>0.; tau-=0.001){
                double R = GetStabFunc(TE, tau, StabType);
                if(R>1.000000001) tau_max = tau;
            }
            printf("Stability range (imag. axis)=%f\n", tau_max);
        }

        double alphaL = 0., alphaR = 5.;
        for(int i=0; i<50; i++){
            T.alpha_tau = 0.5*(alphaL + alphaR);
            double maxlambda = StabType ? GetMaxLambda_StabType1(T, OnlyNoStabilizationLimit) : GetMaxLambda_StabType0(T, OnlyNoStabilizationLimit);
            if(maxlambda > 1+1e-6) alphaR = T.alpha_tau;
            else alphaL = T.alpha_tau;
        }
        printf("StabType %i: alpha*tau critical = %.10f\n", T.StabType, T.alpha_tau);
    }
}

void main_alpha_range(){
    tIMEXMethod T[20] = {ARS_121(), ARS_232(), ARS_343(), ARS_111(), ARS_222(), ARS_443(), MARS_343(),
                         ARK3(), ARK4(), ARK5(), MARK3(), BHR_553(),
                         SSP2_322(), SI_IMEX_332(), SI_IMEX_443(), SI_IMEX_433(),
                         SSP2_322_A1(), SI_IMEX_332_A1(), SI_IMEX_443_A1(), SI_IMEX_433_A1()};
    char tIMEXMethodNames[20][20] = {"ARS_121", "ARS_232", "ARS_343", "ARS_111", "ARS_222", "ARS_443", "MARS_343",
                         "ARK3", "ARK4", "ARK5", "MARK3", "BHR_553",
                         "SSP2_322", "SI_IMEX_332", "SI_IMEX_443", "SI_IMEX_433",
                         "SSP2_322_A1", "SI_IMEX_332_A1", "SI_IMEX_443_A1", "SI_IMEX_433_A1"};
    for(int i=0; i<20; i++){
        printf("\n%s\n", tIMEXMethodNames[i]);
        main_alpha_range(T[i], false);
    }
}
