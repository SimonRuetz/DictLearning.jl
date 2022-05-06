module DictLearning

export mod, ksvd, itkrm, main

using LinearAlgebra
using Plots
using LinearAlgebra
using Random
using BenchmarkTools
using Base.Threads
using Base.Sort
using StatsBase

# Write your package code here.
include("utils.jl")
include("mod.jl")
include("ksvd.jl")
include("itkrm.jl")
include("main.jl")
include("makesparsesig.jl")

# using StaticArrays
# using TimerOutputs
# const to = TimerOutput()
# using Infiltrator
# using OrthoMatchingPursuit

end