
function ksvd_copy(Y,S,K,dico)
    #### Approximative K-SVD algorithm
    # one iteration of the approximative K-SVD dictionary learning algorithm with thresholding.

    # Y ..... Data
    # S ..... Sparsity
    # K ..... Number of dictionary elements
    # dico ..... initial dictionary
 
    #### 2022 Simon Ruetz


    ### Allocations for more speed
    d,N = size(Y)
    ip = zeros(K,N)
    absip = zeros(K,N)
    signip = zeros(K,N)
    X = zeros(K,N)
    gram = zeros(K,K)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    dicos = [zeros(d,K) for t in 1:Threads.nthreads() ]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    est_weights= [zeros(K) for t in 1:Threads.nthreads()]
    # calculate inner products
    mul!(ip,dico',Y)

    absip .= abs.(ip)

    @inbounds Threads.@threads for n = 1:N  
        #### thresholding 
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        est_weights[Threads.threadid()][ind[Threads.threadid()]] .+= 1
    end
    weights = sum(est_weights)/sum(sum(est_weights))*S
    dico = (dico * diagm(weights) * dico')^(-1/2) * dico * diagm(weights)
    #normalisation of all atoms to norm 1
    normalise!(dico)

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
    
    return dico
end

