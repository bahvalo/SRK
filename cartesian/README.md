## What is this

This is a code (1 of 2) used to generate the numerical data for the following paper:
  Segregated Runge--Kutta schemes for the time integration of the incompressible
  Navier--Stokes equations in presence of pressure stabilization
  (to be submitted)

The capability of the code is mainly restricted to the cases considered in the paper:

* Incompressible Navier -- Stokes equations (no turbulence models)
* Rectangular domain in 2D or 3D
* Periodic or Dirichlet (generally time-dependent) boundary conditions
* Cartesian meshes (generally non-uniform)
* For a general case: second-order finite-difference method or P1-Galerkin method
* For a periodic case on uniform meshes: also high-order FD methods
* Time integration: segregated Runge -- Kutta methods (that's why this code exists)
* No parameters reading. Everything is specified in the code.

This is NOT a general-purpose CFD code. And it will not be.


## External dependencies

This code relies on
* HYPRE (general-purpose solver for linear systems)
* FFTW3 (used in pressure solver in the case of a uniform mesh and periodic BCs)


## Compilation

Just take all .cpp files to the IDE you use and specify pathes to external
libraries. Or, for Linux, use the command

g++ -I \<path_to_hypre\>/include \`ls *.cpp\` \<path\>/libHYPRE.a \<path\>/libfftw3.a -lgomp


## License

This is distributed under the terms of the MIT license.

