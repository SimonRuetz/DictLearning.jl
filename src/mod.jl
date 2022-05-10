


function mod(Y,S,K,dico,iter)
    ### Allocations for more speed
    d,N = size(Y)
    ip = zeros(K,N)
    absip = zeros(K,N)
    
    
    X = zeros(K,N)
    gram = zeros(K,K)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]

    

    #### algorithm
    mul!(ip,dico',Y)
    mul!(gram,dico',dico)
    absip .= abs.(ip)
    #### thresholding on all signals
    @inbounds Threads.@threads for n = 1:N  
        ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
        X[ind[Threads.threadid()], n] = (@view(gram[ind[Threads.threadid()],ind[Threads.threadid()]] ))\(@view(ip[ind[Threads.threadid()],n]))
    end 

    ### dictionary update step
    dico = Y/X;
    #print(size(Y*pinv(X)))
    normalise!(dico)
    return dico
end



