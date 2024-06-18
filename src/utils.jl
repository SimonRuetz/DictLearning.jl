function normalise!(dico)
    # Function to normalise the columns of dico
    #@infiltrate
    @inbounds for k=1:size(dico)[2] 
        try
            
            dico[:, k] =  @view(dico[:,k])./norm(@view(dico[:,k]))
        catch e
            print("fail 22")
            
            dico[:, k] = randn(size(dico, 1))
            dico[:, k] =  @view(dico[:,k])./norm(@view(dico[:,k]))
        end
        if norm(dico[:, k]) <= 1e-4 || any(isnan.(dico[:, k])) || norm(dico[:, k]) >= 1e6 || any(isinf.(dico[:, k]))
            print("fail")
            dico[:, k] = randn(size(dico, 1))
            dico[:, k] =  @view(dico[:,k])./norm(@view(dico[:,k]))
        end
    end
end

function normalise_metal!(dico,diag_metal)
   
end


function maxk!(ix, a, k; initialized=false ,reversed = false)
    ## Function that returns the indices of the largest k entries of the vector a and saves it in ix
    partialsortperm!(ix, a, 1:k, rev=reversed, initialized=initialized)
    @views collect(ix[1:k])
end
