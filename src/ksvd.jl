
function ksvd(Y,S,K,dico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
    #### MOD algorithm
    # one iteration of the mod dictionary learning algorithm with thresholding.

    # Y ..... Data
    # S ..... Sparsity
    # K ..... Number of dictionary elements
    # dico ..... initial dictionary

    #### 2022 Simon Ruetz

    ip .= 0
    absip .= 0
    X .= 0
    gram .= 0
    weights .= 0
    est_weights= [zeros(K) for t in 1:Threads.nthreads()]
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    mul!(ip,dico',Y)

    dicos = [zeros(d,K) for t in 1:Threads.nthreads() ]
    #### algorithm
    # calculate inner products
    mul!(ip,dico',Y)
    absip .= abs.(ip)
    signip=sign.(ip)
    mul!(gram,dico',dico)

    @inbounds Threads.@threads for n = 1:N  
            #### thresholding
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        #try
            X[ind[Threads.threadid()], n] = (@view(gram[ind[Threads.threadid()],ind[Threads.threadid()]] ))\(@view(ip[ind[Threads.threadid()],n]))
        #catch e
        #end

        ### dictionary update step
        dicos[Threads.threadid()][:,ind[Threads.threadid()]] += (Y[:,n] - dico[:,ind[Threads.threadid()]]*X[ind[Threads.threadid()], n])* X[ind[Threads.threadid()],n]';
        dicos[Threads.threadid()][:,ind[Threads.threadid()]] += dico[:,ind[Threads.threadid()]].*abs.(X[ind[Threads.threadid()],n]').^2;
    end 
    
    #sum over the different Threads (combine the different dictionaries)
    dico = sum(dicos);
    #normalisation of all atoms to norm 1
    normalise!(dico)
    mul!(ip,dico',Y)
    
    absip .= abs.(ip)
    signip=sign.(ip)
    X .= 0
    mul!(gram,dico',dico)
    @inbounds Threads.@threads for n = 1:N  
        #### thresholding
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
    #try
        X[ind[Threads.threadid()], n] = (@view(gram[ind[Threads.threadid()],ind[Threads.threadid()]] ))\(@view(ip[ind[Threads.threadid()],n]))
    #catch e
    #end

    ### dictionary update step
    
    end 
    return dico, X
end

