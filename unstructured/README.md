## What is this

This is a code (2 of 2) used to generate the numerical data for the following paper:
  Segregated Runge--Kutta schemes for the time integration of the incompressible
  Navier--Stokes equations in presence of pressure stabilization
  (to be submitted)

It demostrates the SRK schemes together with a high-order finite-element solver
provided by MFEM (mfem.org).

## External dependencies

* MFEM
* MPI
* HYPRE
* Metis

This code requires MFEM compiled in the parallel mode, which requires MPI, HYPRE and Metis.

## File structure

* navier_solver.cpp, navier_solver.hpp -- the Navier-Stokes solver provided by MFEM
* asrk.h, srk.cpp -- code responsible for SRK time integration
* main.cpp -- 2D Taylor-Green vortex test
* mesh.msh -- trianglular mesh used in the test (GMSH format)

## License

This is distributed under the terms of the 3-clause BSD license.

