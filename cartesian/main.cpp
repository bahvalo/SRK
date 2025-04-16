#include "asrk.h"

void main_alpha_range(); // check the stability range in <alpha> parameter
void main_ctest(); // test with the manufactured solution
void main_tgv2d(); // 2D viscous-dominant Taylor-Green vortex
void main_tgv3d_novisc(); // 3D no-visc Taylor-Green vortex
void main_tgv3d(); // 3D Taylor-Green vortex with Re=800
void main_turbch_180(); // turbulent channel with Re_tau=180

int main(int, char**){
    main_tgv2d();
    return 0;
}
