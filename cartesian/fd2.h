#include "base.h"

// Basic central difference method
struct S_FD2 : S_Base {
    // Obligatory members and methods
    void ApplyGradient(const vector<double>& a, vector<double3>& R);
    void UnapplyGradient(const vector<double3>& v, vector<double>& p);
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);

    void NormalizePressure(vector<double>& p) const override;
    double CalcKineticEnergy(const vector<double3>& u) const override;
    double3 CalcIntegral(const vector<double3>& u) const override;

    // Specific members for this discretization
    int NumIters_IMEX = 1; // number of iterations at the implicit stage for an IMEX SRK. If we use the exact matrix, then we do not need an iterative process
    int NumIters_Impl = 20; // number of iterations at the implicit stage for an implicit SRK
    int DoNotUseDBoundaryValue = 0; // do not use dBoundaryValue in the implicit stage

    // Specific methods for this discretization
    void CalcConvTerm(const vector<double3>& u, const vector<double>& p, vector<double3>& f, int Nullify);
    void CalcViscTerm(const vector<double3>& u, vector<double3>& f); // Nullify=0 always
    void CalcFluxTerm(double t, const vector<double3>& u, const vector<double>& p, bool DoConv, bool DoVisc, bool DoSourceE, bool DoSourceI, vector<double3>& f);
    void CalcFluxTermOnWalls(const vector<double3>& u, const vector<double>& p, double3& F1, double3& F2, vector<double3>& f);
    void Init();
};

// Special IMEX version where stresses across horizontal surfaces are taken implicitly and everything else is taken explicitly
struct S_FD2exim : S_FD2 {
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);
    void Init();
private:
    void ImplicitStage_SetCoeffs(int ix, int iy, vector<double>& a, vector<double>& b, double m) const;
    double CalcCrossTerm(int ix, int iy, int iz, int icomponent, const vector<double3>& u) const;
};
