# StokesMFS
_StokesMFS_ is three-dimensional Stokes-flow solver that uses the Method of Fundamental Solutions (MFS). It can simulate almost any arbitrary particle with vanishing Reynolds number. This work is a Python implementation of some of the methodology published in [1], which is a continuation of the work in [2]. The main focus of this repository is flow visualisation. 

![Image](https://github.com/user-attachments/assets/addfc926-54e0-4230-806c-ca8c20cf7540)

## Required software
This code requires numpy and matplotlib, to install these with Linux:\
`sudo pip install numpy`\
`sudo pip install matplotlib`

## Quick Guide
The guide document 'pythonMFS Guide.pdf' contains all the background and explanation of how to run the code. All you need for a MFS simulation is contained within the file `pythonMFS.py`. The file `examples.py` contains 5 examples which cover: calculation of force and torque, two flow visualisation methods, and an investigation into numerics. The examples are written to be readable and editable as much as possible, it is quite a sandbox.

The main functions in `pythonMFS.py` of note are:
1. `constructMatrix(rb,rs)` constructs the matrix of linear equations to be solved.
2. `constructRHS(rb,v,om)` constructs the super-vector of boundary conditions.

These are the only functions required to simulate a particle, whose boundary is discretised into a number of nodes given by `rb`, with internal sites given by `rs`, which has translational velocity `v` and angular velocity `om`. You can easily simulate any particle if you can create `rb` and `rs`, by using `import pythonMFS as mfs` and using these two functions.

## Future Additions
1. Dynamic simulation of particles. For example, animation of a sphere settling under gravity.
2. Improved speed for multi-particle simulations. This has been implemented in MATLAB for [1], and the methodology can be seen in that paper.
3. Inclusion of rarefaction effects, starting with the G13 equations, possibly moving on to the R13 equations, or even the R26.

## Acknowledgments
This work is based on work supported by the Engineering and Physical Sciences Research Council: [EP/N016602/1] and [EP/V01207X/1].

## Refernces 
[1]. Josiah J.P. Jordan and Duncan A. Lockerby. “The method of fundamental solutions for multi-particle Stokes flows: Application to a ring-like array of spheres”. In: _Journal of Computational Physics_ 520 (2025), p. 113487. doi.org/10.1016/j.jcp.2024.113487\
[2]. Duncan A. Lockerby and Benjamin Collyer. “Fundamental solutions to moment equations for the simulation of microscale gas flows”. In: _Journal of Fluid Mechanics_ 806 (2016), p. 413-436. doi.org/10.1017/jfm.2016.606 
