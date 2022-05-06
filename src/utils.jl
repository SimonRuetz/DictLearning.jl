
function f(x,y)
    2x + y
end
function normalise!(dico)
    @inbounds for k=1:size(dico)[2] 
        try
            dico[:, k] =  @view(dico[:,k])./norm(@view(dico[:,k]))
        catch e
            print(k)
        end
    end
end


function maxk!(ix, a, k; initialized=false ,reversed = false)
    partialsortperm!(ix, a, 1:k, rev=reversed, initialized=initialized)
    @views collect(ix[1:k])
end
