function makesparsesig(dico,N,S,T,b,rho,bfix)
    for n=1:N
        if bfix .== 1
            beta=1-b
            
        else()
            beta=1-b *rand(1,1)
        end
        x1toS=sqrt(1/S)*beta.^[1:S]'
        x1toSsign=2*round(rand(S,1))-1
        x1toS=x1toS.*x1toSsign
        #norm(x1toS)
        if T .> S
            xSp1toT=randn(T-S,1)
            xSp1toT=xSp1toT* sqrt(1-norm(x1toS)^2)/norm(xSp1toT)
            x1toT=[x1toS xSp1toT]'
        else()
            x1toT=x1toS/norm(x1toS)
        end
        p=randperm(K)
        sig=dico[:,p[1:T]]*x1toT
        if (nargin .== 6)
            noise = rho*randn(d,1)
            sig=(sig+noise)/sqrt(1+noise'*noise)
        end     
        sigN=[sigN,sig];  
    end

    return(sigN)
end