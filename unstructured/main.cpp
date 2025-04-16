// 2D Taylor-Green vortex

#include "navier_solver.hpp"
#include "general/forall.hpp"
#include "asrk.h"
#include <fstream>

static double visc = 0.0; // kinematic viscosity coefficient
static double3 Uinf(1., 0., 0.); // background velocity

using namespace mfem;
using namespace navier;
using namespace std;

struct tFE_Discretization : public NavierSolver {
    int NN, dim; // number of DOFs, number of velocity components

    void ApplyGradient(const vector<double>& a, vector<double3>& R);
    void UnapplyGradient(const vector<double3>& v, vector<double>& p);
    void ExplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& kuhat);
    void ImplicitTerm(double t, const vector<double3>& u, const vector<double>& p, vector<double3>& ku);
    void ImplicitStage(double t, double tau_stage, const vector<double3>& ustar, const vector<double>& p, vector<double3>& u);

    tFE_Discretization(ParMesh *mesh, int order) : NavierSolver(mesh, order, visc){
        NN = pfes->GetTrueVSize();
        dim = pmesh->Dimension();
        if(dim!=2) { printf("Not implemented\n"); exit(0); }
    }
};

void tFE_Discretization::ApplyGradient(const vector<double>& a, vector<double3>& R){
    Vector gp, mmgp;
    gp.SetSize(NN*dim);
    mmgp.SetSize(NN*dim);
    for(int i=0; i<NN; i++) pn[i] = a[i];
    G->Mult(pn, gp);
    MvInv->Mult(gp, mmgp);
    for(int i=0; i<NN; i++) R[i] = double3(mmgp[i], mmgp[i+NN], 0.);
}

void tFE_Discretization::UnapplyGradient(const vector<double3>& v, vector<double>& p){
    for(int i=0; i<NN; i++){ un[i] = v[i][0]; un[i+NN] = v[i][1]; }
    D->Mult(un, resp);
    Orthogonalize(resp);
    pfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);

    Vector X1, B1;
    Sp_form->FormLinearSystem(pres_ess_tdof, pn_gf, resp_gf, Sp, X1, B1, 1);
    SpInv->Mult(B1, X1);
    iter_spsolve = SpInv->GetNumIterations();
    res_spsolve = SpInv->GetFinalNorm();
    Sp_form->RecoverFEMSolution(X1, resp_gf, pn_gf);

    MeanZero(pn_gf);
    pn_gf.Neg();
    pn_gf.GetTrueDofs(pn);
    for(int i=0; i<NN; i++) p[i] = pn(i);
}

void tFE_Discretization::ExplicitTerm(double, const vector<double3>& v, const vector<double>&, vector<double3>& kuhat){
    Vector mmgp;
    mmgp.SetSize(NN*dim);
    for(int i=0; i<NN; i++){ un[i] = v[i][0]; un[i+NN] = v[i][1]; }
    N->Mult(un, Nun); // already with the minus sigh
    MvInv->Mult(Nun, mmgp);
    for(int i=0; i<NN; i++){ kuhat[i][0] = mmgp[i]; kuhat[i][1] = mmgp[i+NN]; }
}

void tFE_Discretization::ImplicitTerm(double, const vector<double3>& v, const vector<double>&, vector<double3>& ku){
    if(visc==0.0) { for(int i=0; i<NN; i++) ku[i] = double3(); return; }

    for(int i=0; i<NN; i++){ un[i] = v[i][0]; un[i+NN] = v[i][1]; }
    Lext_gf.SetFromTrueDofs(un);
    ComputeCurl2D(Lext_gf, curlu_gf);
    ComputeCurl2D(curlu_gf, curlcurlu_gf, true);
    curlcurlu_gf.GetTrueDofs(un);
    un *= (-visc);
    for(int i=0; i<NN; i++) ku[i] = double3(un[i], un[i+NN], 0.);
}

void tFE_Discretization::ImplicitStage(double, double tau_stage, const vector<double3>& ustar, const vector<double>&, vector<double3>& u){
    if(visc==0.0) { u = ustar; return; }

    for(int i=0; i<NN; i++){ tmp1[i] = ustar[i][0] / tau_stage; tmp1[i+NN] = ustar[i][1] / tau_stage; }
    Mv->Mult(tmp1, resu);

    H_bdfcoeff.constant = 1./tau_stage;
    H_form->Update();
    H_form->Assemble();
    H_form->FormSystemMatrix(vel_ess_tdof, H);
    HInv->SetOperator(*H);

    vfes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

    Vector X2, B2;
    H_form->FormLinearSystem(vel_ess_tdof, un_next_gf, resu_gf, H, X2, B2, 1);
    HInv->Mult(B2, X2);
    iter_hsolve = HInv->GetNumIterations();
    res_hsolve = HInv->GetFinalNorm();
    H_form->RecoverFEMSolution(X2, resu_gf, un_next_gf);

    un_next_gf.GetTrueDofs(un_next);
    for(int i=0; i<NN; i++){ u[i][0]=un_next[i]; u[i][1]=un_next[i+NN]; }
}

// This class is copied from the MFEM Navier miniapp.
// Modification: 1) adapted for 2D; 2) pass here the polynomial order
class QuantitiesOfInterest{
public:
   QuantitiesOfInterest(ParMesh *pmesh, int FE_order){
      H1_FECollection h1fec(FE_order);
      ParFiniteElementSpace h1fes(pmesh, &h1fec);

      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(&h1fes);
      mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      mass_lf->Assemble();

      ParGridFunction one_gf(&h1fes);
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   };

   real_t ComputeKineticEnergy(ParGridFunction &v)
   {
      Vector velx, vely, velz;
      real_t integ = 0.0;
      const FiniteElement *fe;
      ElementTransformation *T;
      FiniteElementSpace *fes = v.FESpace();

      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         int intorder = 2 * fe->GetOrder();
         const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

         v.GetValues(i, *ir, velx, 1);
         v.GetValues(i, *ir, vely, 2);
         //v.GetValues(i, *ir, velz, 3);

         T = fes->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);

            real_t vel2 = velx(j) * velx(j) + vely(j) * vely(j);
                          //+ velz(j) * velz(j);

            integ += ip.weight * T->Weight() * vel2;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&integ,
                    &global_integral,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      return 0.5 * global_integral / volume;
   };

   ~QuantitiesOfInterest() { delete mass_lf; };

private:
   ConstantCoefficient onecoeff;
   ParLinearForm *mass_lf;
   real_t volume;
};


void vel_tgv_ic(const Vector &coor, real_t t, Vector &u){
    double ee = exp(-2.*visc*t); // decay due to viscosity
    double vx = Uinf[0], vy = Uinf[1];
    double x = coor(0)-vx*t, y = coor(1)-vy*t;
    u(0) = sin(x)*cos(y)*ee + Uinf[0];
    u(1) = -cos(x)*sin(y)*ee + Uinf[1];
}

real_t pres_tgv_ic(const Vector &coor, real_t t){
    double ee = exp(-2.*visc*t); // decay due to viscosity
    double vx = Uinf[0], vy = Uinf[1];
    double x = coor(0)-vx*t, y = coor(1)-vy*t;
    return (cos(2*y)+cos(2*x))/4. * ee*ee; // p exact
}

void oneoneone(double& dt_per_stage, double& erru, double& errp, double& ke, const tIMEXMethod& IMEX, int StabType, int order, double t_final){
    Mesh *mesh = new Mesh("mesh.msh");
    mesh->EnsureNodes();
    GridFunction *nodes = mesh->GetNodes();
    *nodes *= 8.*atan(1.0); // scale to [0,2\pi]^2

    int serial_refinements = 0; // we can uniformly refine the original mesh if necessary
    for (int i = 0; i < serial_refinements; ++i) mesh->UniformRefinement();
    if (Mpi::Root()) std::cout << "Number of elements: " << mesh->GetNE() << std::endl;

    auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    // Create the flow solver
    #define USE_ORIGINAL_SOLVER 1
    #if USE_ORIGINAL_SOLVER // Original solver from the Navier miniapp of MFEM
        NavierSolver flowsolver(pmesh, order, visc);
        double dt = dt_per_stage;
    #else // Segregated Runge-Kutta time integration
        tSRKTimeIntegrator<tFE_Discretization, ParMesh*, int> flowsolver(IMEX, StabType, pmesh, order);
        double dt = dt_per_stage * flowsolver.get_NImplStages();
    #endif

    // Correct the timestep size to get an integer number of timesteps
    int NumTimeSteps = int(t_final / dt + 0.5);
    dt_per_stage *= (t_final / NumTimeSteps)/dt;
    dt = t_final / NumTimeSteps;

    // Set the initial condition.
    ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
    VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv_ic);
    u_ic->ProjectCoefficient(u_excoeff);

    ParGridFunction *p_ic = flowsolver.GetCurrentPressure();
    FunctionCoefficient p_excoeff(pres_tgv_ic);
    p_ic->ProjectCoefficient(p_excoeff);

    flowsolver.Setup(dt);

    ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
    ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

    ParGridFunction w_gf(*u_gf);
    flowsolver.ComputeCurl2D(*u_gf, w_gf);

    QuantitiesOfInterest kin_energy(pmesh, order);
    double ke0 = kin_energy.ComputeKineticEnergy(*u_gf);

    ParaViewDataCollection pvdc("tgv2d_output", pmesh);
    pvdc.SetDataFormat(VTKFormat::BINARY32);
    pvdc.SetHighOrderOutput(true);
    pvdc.SetLevelsOfDetail(order);
    pvdc.SetCycle(0);
    pvdc.SetTime(t_final);
    pvdc.RegisterField("velocity", u_gf);
    pvdc.RegisterField("pressure", p_gf);
    pvdc.RegisterField("vorticity", &w_gf);
    pvdc.Save();

    #if USE_ORIGINAL_SOLVER
        real_t t = 0.0;
        for (int step = 0; step<NumTimeSteps; ++step){
            // For a fair comparison, for the first two steps, use the exact solution
            if(step<=1){
                u_excoeff.SetTime((step+1)*dt);
                p_excoeff.SetTime((step+1)*dt);
                u_gf->ProjectCoefficient(u_excoeff);
                p_gf->ProjectCoefficient(p_excoeff);
                flowsolver.GetProvisionalVelocity()->ProjectCoefficient(u_excoeff);
                flowsolver.UpdateTimestepHistory(dt);
                continue;
            }

            flowsolver.Step(t, dt, step);

            if (Mpi::Root()){
                printf("%11s %11s\n", "Time", "dt");
                printf("%.5E %.5E\n", t, dt);
                fflush(stdout);
            }
        }
    #else
        vector<double3> velocity(flowsolver.NN);
        vector<double> pressure(flowsolver.NN);
        vector<double> qressure(flowsolver.NN);
        Vector _u, _p;
        _u.SetSize(flowsolver.NN * flowsolver.dim);
        _p.SetSize(flowsolver.NN);
        u_gf->GetTrueDofs(_u);
        p_gf->GetTrueDofs(_p);
        for(int in=0; in<flowsolver.NN; in++){
            velocity[in] = double3(_u(in), _u(in+flowsolver.NN), 0.);
            pressure[in] =_p(in);
            qressure[in] = 0.;
        }

        for (int step = 0; step<NumTimeSteps; ++step){
            flowsolver.sw_step.Start();
            real_t t = step*dt;
            flowsolver.Step(t, dt, velocity, pressure, qressure);
            flowsolver.sw_step.Stop();

            if(1){//if(!(step%10)){
                u_excoeff.SetTime(t+dt);
                for(int in=0; in<flowsolver.NN; in++){ _u(in)=velocity[in][0]; _u(in+flowsolver.NN)=velocity[in][1]; }
                for(int in=0; in<flowsolver.NN; in++){ _p(in)=pressure[in]; }
                u_gf->SetFromTrueDofs(_u);
                p_gf->SetFromTrueDofs(_p);
                double erru = flowsolver.GetCurrentVelocity()->ComputeL2Error(u_excoeff);
                if (Mpi::Root()) printf("Step %i of %i, erru = %e\n", step, NumTimeSteps, erru);
            }
        }

        // flowsolver returns values to velocity and pressure arrays, so we need to put them back to MFEM structures
        for(int in=0; in<flowsolver.NN; in++){ _u(in)=velocity[in][0]; _u(in+flowsolver.NN)=velocity[in][1]; }
        for(int in=0; in<flowsolver.NN; in++){ _p(in)=pressure[in]; }
        u_gf->SetFromTrueDofs(_u);
        p_gf->SetFromTrueDofs(_p);
    #endif

    u_excoeff.SetTime(t_final);
    p_excoeff.SetTime(t_final);
    erru = flowsolver.GetCurrentVelocity()->ComputeL2Error(u_excoeff);
    errp = flowsolver.GetCurrentPressure()->ComputeL2Error(p_excoeff);
    ke = 1. - kin_energy.ComputeKineticEnergy(*(flowsolver.GetCurrentVelocity())) / ke0;
    if (Mpi::Root()){
        std::cout << "\ndt = " << dt_per_stage << '\n' << std::endl;
        std::cout << "\n|| u_h - u ||_{L^2} = " << erru << '\n' << std::endl;
        std::cout << "\n|| p_h - p ||_{L^2} = " << errp << '\n' << std::endl;
        std::cout << "\n ke = " << ke << "\n" << std::endl;
    }

    flowsolver.ComputeCurl2D(*u_gf, w_gf);
    pvdc.SetCycle(NumTimeSteps);
    pvdc.SetTime(t_final);
    pvdc.Save();

    flowsolver.PrintTimingData();

    delete pmesh;
}

void main_novisc(){
    const double t_final = 20.; // maximal integration time
    const int order = 6; // order of basic functions in the Galerkin method

    double dt_per_stage=0.0025;
    double actual_dt_per_stage = dt_per_stage;
    double erru, errp, ke;
    oneoneone(actual_dt_per_stage, erru, errp, ke, ARK4(), 0 /*StabType*/, order, t_final);
    if (Mpi::Root()){
        FILE* f = fopen("log.dat", "at");
        fprintf(f, "ARK4 %e %e %e %e\n", actual_dt_per_stage, erru, errp, ke);
        fclose(f);
    }
}


void main_visc(){
    visc = 0.5;
    const double t_final = 2.; // maximal integration time
    const int order = 4; // order of basic functions in the Galerkin method

    FILE* f = fopen("orig_nocorr.dat", "wt");
    for(double dt_per_stage=0.01; dt_per_stage<=0.5; dt_per_stage*=2.){
        double actual_dt_per_stage = dt_per_stage;
        double erru, errp, ke;
        oneoneone(actual_dt_per_stage, erru, errp, ke, SI_IMEX_443(), 0 /*StabType*/, order, t_final);
        fprintf(f, "%e %e %e\n", actual_dt_per_stage, erru, errp);
    }
    fclose(f);
}

int main(int argc, char *argv[]){
    Mpi::Init(argc, argv);
    Hypre::Init();
    //main_novisc();
    main_visc();
    return 0;
}
