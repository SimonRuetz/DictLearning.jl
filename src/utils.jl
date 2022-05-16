function normalise!(dico)
    # Function to normalise the columns of dico

    @inbounds for k=1:size(dico)[2] 
        try
            dico[:, k] =  @view(dico[:,k])./norm(@view(dico[:,k]))
        catch e
            print("fail")
        end
    end
end


function maxk!(ix, a, k; initialized=false ,reversed = false)
    ## Function that returns the indices of the largest k entries of the vector a and saves it in ix
    partialsortperm!(ix, a, 1:k, rev=reversed, initialized=initialized)
    @views collect(ix[1:k])
end
