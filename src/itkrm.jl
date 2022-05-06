
function itkrm(Y,S,K,dico,iter)
    ### Allocations for more speed
    ip = zeros(K,N)
    absip = zeros(K,N)
    signip = zeros(K,N)
    X = zeros(K,N)
    gram = zeros(K,K)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]

    for i = 1:iter
        #### algorithm
        d,N = size(Y)
        mul!(ip,dico',Y)
        
        absip .= abs.(ip)
        signip=sign.(ip)
        mul!(gram,dico',dico)

        #### thresholding on all signals
        @inbounds Threads.@threads for n = 1:N  
            ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(ip[:,n]),S,initialized = true, reversed = true)
            X[ind[Threads.threadid()], n] = (@view(gram[ind[Threads.threadid()],ind[Threads.threadid()]] ))\(@view(ip[ind[Threads.threadid()],n]))
        end 
        
        ### dictionary update step
        mul!(data2,dico,X)
        broadcast!(+, data, data, data2)
        mul!(dico,dico,Diagonal(vec(sum(abs.(X),dims=2))))
        broadcast!(sign, X, X)#X .= sign.(X)
        mul!(dico2, data, X')
        broadcast!(+, dico, dico, dico2)  

        normalise!(dico)
    end

    return dico
end

