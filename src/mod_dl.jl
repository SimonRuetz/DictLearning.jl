


function mod_dl(Y,S,K,dico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
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

    #### algorithm
    mul!(ip,dico',Y)
    mul!(gram,dico',dico)
    absip .= abs.(ip)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    #### thresholding on all signals
    @inbounds Threads.@threads for n = 1:N  
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        X[ind[Threads.threadid()], n] = pinv(dico[:, ind[Threads.threadid()]]) * Y[:,n]
    end 
    ### dictionary update step
    # Compute eigenvalue decomposition of X'X
    XtX = X*X'
    eigvals, eigenvecs = eigen(XtX)
    #println(eigvals)
    # Clip the smallest eigenvalues
    eigvals = max.(eigvals, 0)
    eigvals[eigvals .< 1e-8] .= 0

    d = 1 ./ eigvals
    d[eigvals .< 1e-8] .= 0
    # Compute the inverse of X'X with clipped eigenvalues
    inv_XtX = eigenvecs * diagm(vec(d)) * eigenvecs'

    # Compute the updated dictionary
    dico = Y * X' * inv_XtX
    #normalisation of all atoms to norm 1
    normalise!(dico)
    X .= 0
    mul!(ip,dico',Y)
    mul!(gram,dico',dico)
    absip .= abs.(ip)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    #### thresholding on all signals
    @inbounds Threads.@threads for n = 1:N  
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        X[ind[Threads.threadid()], n] = pinv(dico[:, ind[Threads.threadid()]]) * Y[:,n]
    end 

    
    # print(X)
    return dico, X 
end