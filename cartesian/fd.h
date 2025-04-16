#include "asrk.h"
#include "base.h"

struct S_FD : S_Base{
    // Obligatory methods
    void ApplyGradient(const vector<double>& a, vector<double3>& R);
    void UnapplyGradient(const vector<double3>& v, vector<double>& p);
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);

    void NormalizePressure(vector<double>& p) const override;
    double CalcKineticEnergy(const vector<double3>& u) const override;
    double3 CalcIntegral(const vector<double3>& u) const override;

    // Specific members for this discretization
    int GradDivOrder = 0;
    int ConvOrder = 0;
    int ViscOrder = 0;
    double3 MeshStep, inv_MeshStep;

    // Specific methods for this discretization
    void Init();
    void CalcFluxTerm_FD(double t, const vector<double3>& u, const vector<double>&, bool DoConv, bool DoVisc, vector<double3>& f);
    void LinearSystemSolveFFT_Init(); // when tIndex::N is known
    void LinearSystemSolveFFT(vector<double>& X, const vector<double>& f); // solve the pressure equation using FFTW

    S_FD() : MeshStep(1.,1.,1.), inv_MeshStep(1.,1.,1.) {}
};
