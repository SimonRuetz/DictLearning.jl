# module DictLearning

# export run_tests_brain_data, run_tests, run_tests_mnist
using LinearAlgebra
using Random
using BenchmarkTools
using Base.Threads
using Base.Sort
using StatsBase
using Makie
using ProgressMeter
using FFTW
using Hadamard
using Infiltrator
using MLDatasets
using Flux
using FileIO, Images, ProgressMeter, StatsBase, LinearAlgebra, Infiltrator, NPZ
using Distributions
# Write your package code here.
include("utils.jl")
include("mod_dl.jl")
include("mod_dl_copy.jl")
include("ksvd.jl")
include("ksvd_copy.jl")
include("itkrm.jl")
include("itkrm_copy.jl")
include("itkrm_metal.jl")
include("main.jl")
include("makesparsesig.jl")


# end