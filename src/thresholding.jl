



function thresholding(Y,S,dico)
    #### Thresholding algorithm
    # one iteration of the mod dictionary learning algorithm with thresholding.

    # Y ..... Data
    # S ..... Sparsity
    # dico ..... initial dictionary

    #### 2024 Simon Ruetz

    ip = zeros(size(dico,2),size(Y,2))
    absip = zeros(size(dico,2),size(Y,2))
    X = zeros(size(dico,2),size(Y,2))
    ix = [collect(1:size(dico,2)) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    
    mul!(ip,dico',Y)
    absip .= abs.(ip)
    ix = [collect(1:size(dico,2)) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]

    #### thresholding on all signals
    @inbounds Threads.@threads for n = 1:size(Y,2)
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        X[ind[Threads.threadid()], n] = pinv(dico[:, ind[Threads.threadid()]]) * Y[:,n]
    end 
    return X
end