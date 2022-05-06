
function run_tests(;d::Int64 = 128,K::Int64 = 256,S::Int64 = 20,b::Int64 = 0,snr::Float64 = 0.0,rho::Float64 = 0.,eps::Float64 = 1.,N::Int64 = 2000,iter::Int64 = 100)
    weights = ones(K,1)#0.3:1.2/(K-1):1.5; # weights for non-uniform sampling without replacement
    #p = randperm(K)
    ##weights = (1:K).^(-0.8)
    #weights = reverse(weights, dist = 2)
    w = aweights(weights/sum(weights)*S)

    #### algorithm parameters #################################
    runs = 1;         # number of trials

    ### initialisation of dictionary
    dico = randn(d,K)
    normalise!(dico)

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
        dico = (1-eps^2/2)*dico + (eps^2-eps^4/4)^(1/2)*Z
    end

    normalise!(dico) 
    dico
    x1toS=sqrt(1/S).*(1-b).^(1:S)
    x1toS = x1toS./norm(x1toS)
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
            Y[:,n] = Y[:,n]./norm(Y[:,n])
        end
        return Y
    end

    # savefile noisy signals
    # SNR = 1/(d*rho^2)
    # savefile = strcat("comp_synth_data_its_d',num2str(d),'_K',num2str(K),'_S',num2str(S),'_N',num2str(N),'_maxit',num2str(maxit),'_b',num2str(b),'_SNR',num2str(SNR),'_',dicotype,'.mat")
    # savefile noiseless signals
    #savefile = strcat("comp_synth_data_its_d',num2str(d),'_K',num2str(K),'_S',num2str(S),'_N',num2str(N),'_maxit',num2str(maxit),'_b',num2str(b),'_',dicotype,'.mat")

    ### create signal generating dictionary and generate signals
    dico = randn(d,K)
    normalise!(dico)
    Y = generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)
    
    rtdico = itkrm(Y,S,K,dico,iter)
    #rtdico = itkrm_update!(X ,Y ,K,S,1,rtdico,ip ,gram,ix,ind)
    
    print(mean(maximum(abs,rtdico'*dico, dims = 1)))
    print("  asdf ")
    
    # K-SVD using thresholding instead of OMP
    #ktdico = ksvd_threshold[Y,K,S,1,ktdico]

    # MOD using thresholding instead of OMP
    #mtdico = mod_dico_threshold[Y,K,S,1,mtdico]
end

