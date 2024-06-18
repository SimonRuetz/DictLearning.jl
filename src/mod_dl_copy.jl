



function mod_dl_copy(Y,S,K,dico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
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

    absip .= abs.(ip)

    @inbounds Threads.@threads for n = 1:N
        #### thresholding 
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        est_weights[Threads.threadid()][ind[Threads.threadid()]] .+= 1
    end
    weights = sum(est_weights)/sum(sum(est_weights))*S 
    #weights[weights .< 1e-6] .= 0
    diag_sqrt_weights = diagm(sqrt.(weights))

    dico_diag_sqrt_weights = dico * diag_sqrt_weights
    eigvals, eigvecs = eigen(dico_diag_sqrt_weights * dico_diag_sqrt_weights')
    # print(minimum(eigvals))
    # set the negative eigenvalues to 0
    eigvals = max.(eigvals, 0)
    # eigvals below 1e-6 are set to 0
    eigvals[eigvals .< 1e-8] .= 0

    diag_inv_sqrt_eigvals = 1 ./ sqrt.(eigvals)

    diag_inv_sqrt_eigvals[eigvals .< 1e-8] .= 0

    dico_stable = eigvecs * diagm(vec(diag_inv_sqrt_eigvals)) * eigvecs'

    dico = dico_stable * dico*  diag_sqrt_weights
    normalise!(dico)

    #### algorithm
    mul!(ip,dico',Y)
    mul!(gram,dico',dico)
    absip .= abs.(ip)

    #### thresholding on all signals
    @inbounds Threads.@threads for n = 1:N  
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        X[ind[Threads.threadid()], n] = pinv(dico[:, ind[Threads.threadid()]]) * Y[:,n]
    end 

    ### dictionary update step
    XtX = X*X'
    eigvals, eigenvecs = eigen(XtX)

    eigvals = max.(eigvals, 0)
    # eigvals below 1e-6 are set to 0
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
    return dico, X
end