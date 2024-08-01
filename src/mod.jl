function mod_dl(Y,S,dico)
    #### MOD algorithm
    # one iteration of the mod dictionary learning algorithm with thresholding.

    # Y ..... Data
    # S ..... Sparsity
    # dico ..... current dictionary

    #### 2024 Simon Ruetz
    N = size(Y,2)
    K = size(dico,2)
    ip = zeros(Float64,K,N)
    absip = zeros(Float64,K,N)
    X = zeros(Float64,K,N)
    gram = zeros(Float64,K,K)
    weights = zeros(Float64,K)

    ip .= 0
    absip .= 0
    X .= 0
    gram .= 0
    
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
    weights .= 0
    est_weights= [zeros(K) for t in 1:Threads.nthreads()]
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    mul!(ip,dico',Y)

    absip .= abs.(ip)

    @inbounds Threads.@threads for n = 1:N
        #### thresholding 
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        est_weights[Threads.threadid()][ind[Threads.threadid()]] .+= 1
    end
    weights = sum(est_weights)/sum(sum(est_weights))*S 

    return dico, weights
end