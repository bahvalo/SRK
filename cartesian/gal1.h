// P1-Galerkin method (not mass-lumped) on Cartesian meshes
// Dirichlet or periodic boundary conditions
#include "asrk.h"
#include "base.h"

enum struct tZeroAtBoundary{
    NOTHING = 0,
    NORMAL = 1,
    VECTOR = 2
};

// General methods for the P1-Galerkin method
struct S_Gal1 : S_Base{
    // Shift pressure so that is has zero average
    void NormalizePressure(vector<double>& p) const override;
    // Evaluate the integral of a vector field
    double3 CalcIntegral(const vector<double3>& u) const override;
    // Evaluate the total kinetic energy
    double CalcKineticEnergy(const vector<double3>& u) const override;
    // Evaluate the convective fluxes. Not multiplied by M^{-1}. Set Nullify=0 to increment (nor override) the output
    void CalcConvTerm(const vector<double3>& u, vector<double3>& ConvTerm, int Nullify=1) const;

    // Store the mass matrix in `_L` array
    void FillMassMatrix(vector<double>& _L) const;
    // Apply the mass matrix to a given vector field
    void ApplyMassMatrix(const vector<double3>& f, vector<double3>& Mf) const;
    // Apply the inverse mass matrix to a given scalar or vector field (output overrides input)
    void ApplyMassMatrixInv(vector<double>& f) const;
    void ApplyMassMatrixInv(vector<double3>& f, tZeroAtBoundary Flag) const;

    // Sets zero normal component on boundaries for a vector field
    void SetZeroNormalComponent(vector<double3>& R) const;
    // Gradient - with zero normal component on boundary
    void ApplyGradient_MainGalerkin(const vector<double>& a, vector<double3>& Ga);
    // Divergence
    void ApplyDivergence_MainGalerkin(const vector<double3>& v, vector<double>& DivV);

    // Kinematic pressure to effective pressure conversion (p += m*u^2)
    void PressureToEffectivePressure(const vector<double3>& u, vector<double>& p, double m) const;

    // General wrapper
    void CalcFluxTerm(double t, const vector<double3>& u, const vector<double>& p, bool DoConv, bool DoVisc, bool DoSourceE, bool DoSourceI, vector<double3>& f);

    // General solver for the implicit velocity step
    int SimplifiedImplSystem = 0;
    void InitVelocitySystem();
    int NumIters_Impl = 20;
    int NumIters_IMEX = 1; // if we use the true mass matrix, then we do not need an iterative process
    void ImplicitStage_MainGalerkin(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);
};

// Version A [experimental]: velocity space does not impose any boundary conditions
struct S_Gal1A : S_Gal1{
    void ApplyGradient(const vector<double>& a, vector<double3>& R);
    void UnapplyGradient(const vector<double3>& v, vector<double>& p);
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);

    void Init();
    void ApplyGradientMatrix(const vector<double>& a, vector<double3>& Ga) const; // without applying M^{-1}

    int DoNotApplyMInvToVisc = 0; // [experimental] slightly relax timestep restriction for tTimeIntMethod::EXPLICIT
};

// Version B [main]: normal component of the velocity is always zero on boundary
struct S_Gal1B : S_Gal1{
    void ApplyGradient(const vector<double>& a, vector<double3>& R);
    void UnapplyGradient(const vector<double3>& v, vector<double>& p);
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);

    void Init();

    int DoNotApplyMInvToVisc = 0; // [experimental] slightly relax timestep restriction for tTimeIntMethod::EXPLICIT
};

// Special IMEX version where stresses across horizontal surfaces are taken implicitly and everything else is taken explicitly
struct S_Gal1Bexim : S_Gal1{
    void ApplyGradient(const vector<double>& a, vector<double3>& R);
    void UnapplyGradient(const vector<double3>& v, vector<double>& p);
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);

    void Init();

private:
    void ImplicitStage_SetCoeffs(int ix, int iy, vector<double>& a, vector<double>& b, double m) const;
    double CalcCrossTerm(int ix, int iy, int iz, int icomponent, const vector<double3>& u) const;
};

// 1D inverse mass matrix
// Matrix coefficients are h_{j-1/2}/6 (h_{j-1/2}+h_{j+1/2})*1/3 h_{j+1/2}/6
// Version for the Dirichlet boundary conditions (boundary values of x are zero)
void ThomasForM_Dirichlet(const vector<double>& c, vector<double>& x);
void ThomasForM_Periodic(const vector<double>& c, const vector<double>& y, vector<double>& x);
