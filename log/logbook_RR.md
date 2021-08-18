<!-----------------------------------------------------------------------------
This document should be written based on the Github flavored markdown specs:
https://github.github.com/gfm/
It can be converted to html or pdf with pandoc:
pandoc -s -o logbook.html  -f gfm -t html logbook.md
pandoc test.txt -o test.pdf
or with the kramdown converter:
kramdown --template document  -i GFM  -o html logbook.md

If checked in as part of a github project html is automatically generated if
using the github web interface.

Optional: Document how much time was spent. A simple python command line tool
for time tracking is [Watson](http://tailordev.github.io/Watson/).
------------------------------------------------------------------------------>

<!-----------------------------------------------------------------------------
The Agenda section is a scratchpad area for planning and Todo list
------------------------------------------------------------------------------>
# Agenda

* Use variables to accumulate forces (reduction)
* Express simulation specific kernels (cell lists, PBC, neighbor lists, comm) in a cleaner way with new syntax
* MPI support
* GPU support
* LLVM support
* OpenMP support
* Common subexpression elimination (CSE)
    * Analyze generated code by backend compilers to see which cases we really need to manage
* waLBerla as backend (Load-Balancing)
* Provide test cases (LJ, EAM, DEM, Configurational Forces + Energy Minimization, ...)
* Separate performance strategies from code (parallelism, gathering), allow to experiment different strategies
* Coupling interfaces
* Maybes:
    * LAMMPS as backend
    * Many-body potentials code generation
    * Long-range forces

<!-- ![Plot title](figures/example.png "ALT Text") -->

<!-----------------------------------------------------------------------------
START BLOCK PREAMBLE -  Global information required in all steps: Add all
information required to build and benchmark the application. Should be extended
and maintained during the project.
------------------------------------------------------------------------------>
# Project Description

* Start date: DD/MM/YYYY
* Ticket ID:
* Home HPC center:
* Contact HPC center:
   * Name: Rafael Ravedutti Lucio Machado
   * Fon: +49 9131 85 67296
   * E-Mail: rafael.r.ravedutti@fau.de

<!-----------------------------------------------------------------------------
Formulate a clear and specific performance target
------------------------------------------------------------------------------>
## Target

Code generation tool and intermediate representation for particle interactions simulations.

<!-----------------------------------------------------------------------------
## Customer Info

* Name: <CUSTOMERNAME>
* E-Mail: john.doe@foo.bar
* Fon: <PHONENUMBER>
* Web: <URL>
------------------------------------------------------------------------------>

## Application Info

* Name: PAIRS
* Domain: Particle Simulations
* Version: 0.0.1

<!-----------------------------------------------------------------------------
All steps required to build the software including dependencies
------------------------------------------------------------------------------>
## How to build software

<!-----------------------------------------------------------------------------
All steps required to run the testcase and control affinity for application
------------------------------------------------------------------------------>
## How to run software


<!-----------------------------------------------------------------------------
END BLOCK PREAMBLE
------------------------------------------------------------------------------>
