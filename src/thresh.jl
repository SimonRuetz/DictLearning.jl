include("utils.jl")

function tresh(Y,Dict,S)
    d,N = size(Y);

    for i = 1:N
        ip = @view Y[:,i]'*Dict;
    
    
end