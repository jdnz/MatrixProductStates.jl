# MatrixProductStates.jl
Julia implementation of algorithms for matrix product states

Install using `Pkg.clone("https://github.com/jdnz/MatrixProductStates.jl")`

This package was developed in Julia 0.6 and has not been tested in more recent versions of Julia.

### Spin Model
The code in this package was originally developed to model the propagation of light in atom ensembles, the details of which can be found in [Marco T. Manzoni, Darrick E. Chang and James S. Douglas, Nature Communications 8, 1743 (2017)](https://www.nature.com/articles/s41467-017-01416-4). There it is described how light propagation can be reduced to a spin model with long range interactions that for one-dimensional problems can be efficiently solved using the matrix product states ansatz. Code that performs the quantum trajectory simulations performed in Manzoni et al. can be found in the [examples folder](examples/).

The code was then extended to allow for full density matrix simulations of the spin model as described in [Przemyslaw Bienias et al.](https://arxiv.org/abs/1807.07586). The density matrix treatment enables the efficient simulation of photon propagation through Rydberg gases for large input probe intensities, in the presence of Rydberg blockade. Examples of these density matrix simulations are also found in the [examples folder](examples/).

### Running on CUDA enabled GPUs
This package can run on CUDA enabled GPUs to significantly reduce simulation time. Example GPU codes are in the [examples folder](examples/). To use the GPU, Julia should be built from source. An example setup for an Ubuntu system that has CUDA libraries installed would be:
```
sudo apt-get install git build-essential hdf5-tools curl gfortran patch cmake m4 pkg-config -y
git clone git://github.com/JuliaLang/julia.git
cd julia/
git checkout v0.6.4
make
julia 'Pkg.add("MAT")'
julia 'Pkg.add("CuArrays")'
julia 'Pkg.pin("CuArrays", v"0.6.2")'
julia 'Pkg.clone("https://github.com/jdnz/MatrixProductStates.jl")'
```
Then one of the example codes could be run as `julia -p2 rydberg_pollution_density_matrix_gpu.jl`.
