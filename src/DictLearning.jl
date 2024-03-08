#module DictLearning

#export mod, ksvd, itkrm, run_tests

using LinearAlgebra
using Random
using BenchmarkTools
using Base.Threads
using Base.Sort
using StatsBase
using Makie
using GLMakie
using ProgressMeter
using FFTW
using Hadamard
using Infiltrator
using Metal
# Write your package code here.
include("utils.jl")
include("mod.jl")
include("ksvd.jl")
include("itkrm.jl")
include("itkrm_metal.jl")
include("main.jl")
include("makesparsesig.jl")