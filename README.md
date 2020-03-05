# MatrixProductStates.jl
Julia implementation of algorithms for matrix product states

Install using `Pkg.clone("https://github.com/jdnz/MatrixProductStates.jl")

This package was developed in Julia 0.6 and has not been tested in more recent versions of Julia.

### Spin Model
The code in this package was originally developed to model the propagation of light in atom ensembles, the details of which can be found in [Marco T. Manzoni, Darrick E. Chang and James S. Douglas, Nature Communications 8, 1743 (2017)](https://www.nature.com/articles/s41467-017-01416-4). There it is described how light propagation can be reduced to a spin model with long range interactions that for one-dimensional problems can be efficiently solved using the matrix product states ansatz. Code that performs the quantum trajectory simulations performed in Manzoni et al. can be found in the [examples folder](examples/).
