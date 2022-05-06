module DictLearning

export mod, ksvd, itkrm

using LinearAlgebra
using Plots
using LinearAlgebra
using Random
using BenchmarkTools
using Base.Threads
using Base.Sort


# Write your package code here.
include("utils.jl")
include("mod.jl")
include("ksvd.jl")
include("itkrm.jl")


# using StatsBase
# using StaticArrays
# using TimerOutputs
# const to = TimerOutput()
# using Infiltrator
# using OrthoMatchingPursuit

### initialisation of dictionary
dinit = randn(d,K)
normalise!(dinit)

##### start trials #################################

# create perturbation orthogonal to each dico atom
Z = randn(d,K)
for k = 1:K
    Z[:,k] = Z[:,k]-(Z[:,k]'*dico[:,k])*dico[:,k]
    Z[:,k] = Z[:,k]/norm(Z[:,k])
end
# perturbed dictionary
if eps == 0
else
    dinit = (1-eps^2/2)*dico + (eps^2-eps^4/4)^(1/2)*Z
end

normalise!(dinit)

#sqrt(max(abs(Diagonal((-dinit+dico)'*(-dinit+dico)))))
rtdico = dinit

x1toS=sqrt(1/S).*(1-b).^(1:S)
x1toS = x1toS./norm(x1toS)
#ip =  [zeros(K) for t in 1:Threads.nthreads()] 
#X = [zeros(S) for t in 1:Threads.nthreads()] 
ip = zeros(K,N)
absip = zeros(K,N)

Y =  zeros(d,N)
p = [collect(1:S) for t in 1:Threads.nthreads()] 
gram = zeros(K,K)
ix = [collect(1:K) for t in 1:Threads.nthreads()]
ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]



function generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)
    @inbounds Threads.@threads for n = 1:N
        sample!(1:K,w, p[Threads.threadid()]; replace=false, ordered=false)
        rand!(@view(Y[:,n]))
        x1toS .= x1toS.*rand([-1, 1],(S))
        mul!(@view(Y[:,n]),@view(Y[:,n]),rho)
        mul!(@view(Y[:,n]),@view(dico[:,p[Threads.threadid()]]),x1toS)
        #@timeit to "Y1"  sample!(1:K,w, p[Threads.threadid()]; replace=false, ordered=false)
        #@timeit to "Y2" rand!(@view(Y[:,n]))
        #@timeit to "Y3" x1toS .= x1toS.*rand([-1, 1],(S))
        #@timeit to "Y4" mul!(@view(Y[:,n]),@view(Y[:,n]),rho)
        #@timeit to "Y5" mul!(@view(Y[:,n]),@view(dico[:,p[Threads.threadid()]]),x1toS) #@view(Y[:,n]) .+= @view(dico[:,p[Threads.threadid()]])*x1toS
        
        #Y[:,n] = Y[:,n]./norm(Y[:,n])
    end
    return Y
end

function  main()
    reset_timer!(to::TimerOutput)
    @timeit to "nest 2" begin
    #### signal parameters #####################################
    dicotype = "rd";        # 'rr' random dictionary, 'dd" dirac, dct [K<=2d]
                            # "rd' dirac random dictionary [K<=2d], 'd3" dirac dct random dictionary [K=3d]
    d = 128;                 # signal dimension
    K = 2*d;                # dictionary size; set K = 3*d for dicotype "d3"
    S = 20;                  # sparsity level   
    b = 0;#0.1;                # max. decay of sparse coefficients c_i=(1-b)^i
    snr = 8
    rho = 0;#1/(sqrt(d)*sqrt(snr));                # noiselevel, e.g. rho = 0.25/sqrt(d) for SNR = 16
    weights = ones(K,1)#0.3:1.2/(K-1):1.5; # weights for non-uniform sampling without replacement
    eps = 1
    #p = randperm(K)
    ##weights = (1:K).^(-0.8)
    #weights = reverse(weights, dims = 2)
    w = aweights(weights/sum(weights)*S)
    #@infiltrate

    #### algorithm parameters #################################
    N = 200;         # number of training signals per iteration
    its = 100;        # number of iterations per trial
    runs = 1;         # number of trials

    # savefile noisy signals
    # SNR = 1/(d*rho^2)
    # savefile = strcat("comp_synth_data_its_d',num2str(d),'_K',num2str(K),'_S',num2str(S),'_N',num2str(N),'_maxit',num2str(maxit),'_b',num2str(b),'_SNR',num2str(SNR),'_',dicotype,'.mat")
    # savefile noiseless signals
    #savefile = strcat("comp_synth_data_its_d',num2str(d),'_K',num2str(K),'_S',num2str(S),'_N',num2str(N),'_maxit',num2str(maxit),'_b',num2str(b),'_',dicotype,'.mat")

    ### create signal generating dictionary and generate signals
    dico = randn(d,K)
    normalise!(dico)
    Y = generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)



    ### initialisation counter doubles
        
        for it = 1:its
            @timeit to "genY" 
            ### dictionary learning
            X = zeros(K,N)
            #@infiltrate
            #@infiltrate
            @timeit to "dicoupdate" rtdico = itkrm_update!(X::Matrix{Float64} ,Y::Matrix{Float64} ,Y::Matrix{Float64} ,K::Int64,S::Int64,1::Int64,rtdico::Matrix{Float64} ,rtdico::Matrix{Float64} ,ip::Matrix{Float64} ,gram::Matrix{Float64},ix::Vector{Vector{Int64}},ind::Vector{Vector{Int64}},absip::Matrix{Float64})
            #rtdico = itkrm_update!(X ,Y ,K,S,1,rtdico,ip ,gram,ix,ind)
            #@infiltrate
            #print(mean(maximum(abs,rtdico'*dico, dims = 1)))
            #print("   ")
            # ITKrM using OMP instead of thresholding
            #rodico = itkrm_batch_omp(Y,K,S,1,rodico)
            # K-SVD using thresholding instead of OMP
            #ktdico = ksvd_threshold[Y,K,S,1,ktdico]
            # K-SVD [original]
            #kodico = ksvd_omp[Y,K,S,1,kodico]
            
            # MOD using thresholding instead of OMP
            #mtdico = mod_dico_threshold[Y,K,S,1,mtdico]
            # MOD [original]
            #modico = mod_dico[Y,K,S,1,modico]
            
            ### doubles #0.90 #0.85
            #rtdico = aligndico[rtdico,dico]
        end
end
end

end


function normalise!(dico)
    @inbounds for k=1:size(dico)[2] 
        try
            dico[:, k] =  @view(dico[:,k])./norm(@view(dico[:,k]))
        catch e
            print(k)
        end
    end
end


function maxk!(ix, a, k; initialized=false ,reversed = false)
    partialsortperm!(ix, a, 1:k, rev=reversed, initialized=initialized)
    @views collect(ix[1:k])
end
