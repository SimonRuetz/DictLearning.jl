
function itkrm_metal(d,N,ip,absip,diag_zero,Y,S,K,dico,diag_one,ind)
    #### ITkrM algorithm
    # one iteration of the ITkrM dictionary learning algorithm with thresholding.

    # Y ..... Data
    # S ..... Sparsity
    # K ..... Number of dictionary elements
    # dico ..... initial dictionary
 
    #### 2022 Simon Ruetz

    ### Allocations for more speed
    
    #### algorithm
    ip = dico'*Y
    absip = abs.(ip)
    

    #gram=dico'*dico

    ind = absip.> Float32(0.15)
    ip = ip.*ind

    dico = Y*ip'-dico*ip*ip'*diag_zero
    #normalisation of all atoms to norm 1
    dico = dico*((dico'*dico).^(Float32(-1/2)).*diag_one)
    return dico
end

