
function itkrm(Y,S,K,dico)
    #### ITkrM algorithm
    # one iteration of the ITkrM dictionary learning algorithm with thresholding.

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

    for i = 1:iter
        #### algorithm
        mul!(ip,dico',Y)
        
        absip .= abs.(ip)
        signip=sign.(ip)
        mul!(gram,dico',dico)

        
        @inbounds Threads.@threads for n = 1:N  
            #### thresholding 
            ind[Threads.threadid()] = maxk!(ix[Threads.threadid()],@view(absip[:,n]),S,initialized = true, reversed = true)
            try
                X[ind[Threads.threadid()], n] = (@view(gram[ind[Threads.threadid()],ind[Threads.threadid()]] ))\(@view(ip[ind[Threads.threadid()],n]))
            catch e
            end

            ### dictionary update step
            dicos[Threads.threadid()][:,ind[Threads.threadid()]] += (Y[:,n] - dico[:,ind[Threads.threadid()]]*X[ind[Threads.threadid()], n])* signip[ind[Threads.threadid()],n]';
            dicos[Threads.threadid()][:,ind[Threads.threadid()]] += dico[:,ind[Threads.threadid()]].*absip[ind[Threads.threadid()],n]';
        end 
        
        #sum over the different Threads (combine the different dictionaries)
        dico = sum(dicos);

        #normalisation of all atoms to norm 1
        normalise!(dico)
    end

    return dico
end

