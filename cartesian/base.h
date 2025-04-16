#ifndef BASE_H
#define BASE_H

#include "asrk.h"
#include "fftw3.h"
using namespace std;

// General data structures and functions for a Cartesian (uniform or non-uniform) mesh

// Mesh node index
struct tIndex{
    int i[3], in;
    inline tIndex() { i[0]=N[0]-1; i[1]=N[1]-1; i[2]=-1; in=-1; } // error node - by default
    inline tIndex(int index) { in=index; i[0]=in%N[0]; i[1]=(in/N[0])%N[1]; i[2]=in/(N[0]*N[1]); }
    inline operator int() const { return in; }

    inline bool operator == (const tIndex &c){ return in==c.in; }
    inline bool operator == (int c){ return in==c; }

    inline tIndex& operator++() {
        in++;
        i[0]++;
        if(i[0]==N[0]){ i[0]=0; i[1]++; }
        if(i[1]==N[1]){ i[1]=0; i[2]++; }
        return *this;
    }

    // Returns the neighboring node or -1 if error. idir = coordinate direction, iLR = left or right
    inline tIndex Neighb(int idir, int iLR) const{
        if(iLR==0){
            if(i[idir]>0) { tIndex ind = *this; ind.i[idir]--; ind.in-=shift[idir]; return ind; }
            if(IsPer[idir]) { tIndex ind = *this; ind.i[idir]=N[idir]-1; ind.in+=Shift[idir]; return ind; }
            return tIndex();
        }
        else {
            if(i[idir]<N[idir]-1) { tIndex ind = *this; ind.i[idir]++; ind.in+=shift[idir]; return ind; }
            if(IsPer[idir]) { tIndex ind = *this; ind.i[idir]=0; ind.in-=Shift[idir]; return ind; }
            return tIndex();
        }
    }

    // Returns 1 if the node has the boundary condition
    inline int IsWall() const{
        for(int idir=0; idir<Dim; idir++){
            if(IsPer[idir]) continue;
            if(i[idir]==0 || i[idir]==N[idir]-1) return 1; // boundary
        }
        return 0; // not boundary
    }

    // Returns 1 if this is a corner node and thus pressure is not defined at it
    inline int IsCornerNode() const{
        if(!IsWall()) return 0;
        for(int idir=0; idir<Dim; idir++){
            if(!Neighb(idir, 0).IsWall()) return 0;
            if(!Neighb(idir, 1).IsWall()) return 0;
        }
        return 1;
    }

    // Static data: mesh size and aux data
    static int Dim;
    static int N[3];
    static int IsPer[3];
    static int shift[3];
    static int Shift[3];
    static void Init(const int* n, const int* isper) {
        N[0]=n[0]; N[1]=n[1]; N[2]=n[2];
        Dim = (N[2]>1) ? 3:2;
        IsPer[0]=isper[0]; IsPer[1]=isper[1]; IsPer[2]=isper[2];
        shift[0]=1; shift[1]=N[0]; shift[2]=N[0]*N[1];
        Shift[0]=N[0]-1; Shift[1]=N[0]*(N[1]-1); Shift[2]=N[0]*N[1]*(N[2]-1);
    }
};

int GetElementIndex(const tIndex& ind, int ox, int oy, int oz);

// Frequently used enums
enum struct tTimeIntMethod{
    EXPLICIT = 0,
    IMEX     = 1,
    IMPLICIT = 2
};

enum struct tConvMethod{
    CONSERVATIVE  = 0,
    SKEWSYMMETRIC = 1,
    CONVECTIVE    = 2,
    EMAC          = 3
};

enum struct tViscScheme{
    CROSS         = 0, // direct-cross discretization of the Laplace operator (constant viscosity only)
    GALERKIN      = 1, // standard Galerkin method
    AES           = 2  // averaged element splittings method
};


// Abstract base class for numerical schemes on structured meshes (FD, FD2, Gal1)
// Some parameters may be ignored (or not allowed) by certain schemes
struct S_Base {
    // Mesh
    int Dim = 2; // 2 (2D) or 3 (3D)
    int IsPer[3] = {0,0,0}; // 1 if there are periodic conditions for the corresponding direction
    int N[3] = {1,1,1}; // number of nodes for each direction (for non-periodic BCs, includes boundary nodes; for periodic BCs, no images)
    int NN = 0; // N[0]*N[1]*N[2]
    std::vector<double> X[3]; // nodal coordinates

    // Mesh methods
    inline double HLeft(const tIndex& ind, int idir) const{
        const int i = ind.i[idir];
        if(i>0) return X[idir][i] - X[idir][i-1];
        if(tIndex::IsPer[idir]) return X[idir][N[idir]] - X[idir][N[idir]-1];
        return 0.;
    }
    inline double HRight(const tIndex& ind, int idir) const{
        const int i = ind.i[idir];
        if(i<N[idir]-1) return X[idir][i+1] - X[idir][i];
        if(IsPer[idir]) return X[idir][1] - X[idir][0];
        return 0.;
    }
    double3 GetHbar(const tIndex& ind) const;
    double3 GetSbar(const tIndex& ind) const;
    inline double GetCellVolume(const tIndex& ind) const{ double3 h=GetHbar(ind); return h[0]*h[1]*h[2]; }
    inline double3 GetCoor(const tIndex& ind) const{ return double3(X[0][ind.i[0]], X[1][ind.i[1]], X[2][ind.i[2]]); }

    // Parameters of governing equations (including source terms and boundary values)
    double visc = 0.; // constant viscosity coefficient
    vector<double> visc_array; // viscosity coefficient at elements (may be not allocated)
    int EnableConvection = 1; // allows to disable convection
    void (*SourceE)(double t, vector<double3> &f) = NULL; // source term with the explicit fluxes
    void (*SourceI)(double t, vector<double3> &f) = NULL; // source term with the implicit fluxes
    double3 (*BoundaryValue)(double t, const double3 &coor) = NULL; // value for Dirichlet conditions
    double3 (*dBoundaryValue)(double t, const double3 &coor) = NULL; // its time derivative (for methods of type CK only)

    // Discretization parameters
    tTimeIntMethod TimeIntMethod = tTimeIntMethod::EXPLICIT;
    tConvMethod ConvMethod = tConvMethod::SKEWSYMMETRIC;
    tViscScheme ViscScheme = tViscScheme::GALERKIN;
    int IsFourier[3] = {0,0,0}; // Use FFT for X,Y,Z directions for the pressure equation (only for uniform mesh with 2^n nodes and periodic conditions)

    // Linear algebra solver parameters (if HYPRE is used)
    double AMG_Tolerance_P = 1e-4, AMG_Tolerance_U = 1e-4;
    int AMG_MaxIters_P = 20, AMG_MaxIters_U = 20;
    void PassDefaultPressureSystemToHYPRE(int DoPatchNearWalls);

    // FFTW data -- for uniform meshes with periodic conditions
    int UseFourier = 0; // Use FFT-based solver for pressure (either FFT in all dimensions, or FFT in all but one dimension)
    fftw_plan p1, p2;
    fftw_complex *Vphys=NULL, *Vspec=NULL;
    int FFTW_initialized = 0; // flag that p1,p2,Vphys,Vspec is allocated
    void TryInitFourier();
    void LinearSystemSolveFourier(const vector<double>& f, vector<double>& XX, int FDmode);

    virtual void NormalizePressure(vector<double>& p) const = 0;
    virtual double CalcKineticEnergy(const vector<double3>& u) const = 0;
    virtual double3 CalcIntegral(const vector<double3>& u) const = 0;

    // Set Nullify=0 to increment. Not multiplied by M^{-1}
    void CalcViscTermCross(const vector<double3>& u, double nu_, vector<double3>& ViscTerm, int MultVolume, int Nullify=1) const; // constant viscosity only
    void CalcViscTermGalerkin(const vector<double3>& u, double nu_, const vector<double>& nu, vector<double3>& ViscTerm, int Nullify=1) const;
    void CalcViscTermAES(const vector<double3>& u, double nu_, const vector<double>& nu, vector<double3>& ViscTerm, int Nullify=1) const;
    void CalcViscTermWeird(const vector<double3>& u, double nu_, const vector<double>& nu, vector<double3>& f, int Nullify=1) const;
    void CalcViscTermOnWalls_AES(const vector<double3>& u, double3& F1, double3& F2, vector<double3>& f) const;

    // Calculate AbsS at elements
    // Mode: 0 - based on averaged element gradients, 1 - checkerboard-supressing, 2 - based on nodal gradients (step=2*h)
    void CalcAbsS(const vector<double3>& u, vector<double>& AbsS, int Mode) const;

    double AverageE2N(const vector<double>& A, const tIndex& in) const;
    double AverageN2E(const vector<double>& A, int ie) const;
    double AverageE2Edge(const vector<double>& A, const tIndex& in, int idir) const;

    void InitBase();
    ~S_Base();
};

// Thomas algorithm for a symmetric tridiagonal matrix with coefficients `a` and `b`
void Thomas(int n, const vector<double>& a, const vector<double>& b, vector<double>& x, vector<double>& buf);
// Same with the 1D mass matrix constructed by coordinates `c`
void ThomasForM(const vector<double>& c, vector<double>& x);

// Spalart - Allmaras model. If tau>0, advances NuTilde and recalculates NuT. If tau=0, only recalculates NuT
void TurbSolver(const S_Base& S, const vector<double3>& u, vector<double>& NuTilde, vector<double>& NuT, double tau, int IsImplicit);

#endif
