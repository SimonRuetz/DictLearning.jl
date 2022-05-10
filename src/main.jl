
function run_tests(;d::Int64 = 64,K::Int64 = 128,S::Int64 = 2,b::Int64 = 0,snr::Float64 = 0.0,rho::Float64 = 0.,eps::Float64 = 1.1 ,N::Int64 = 100000,iter::Int64 = 10)
    weights = ones(K,1)#0.3:1.2/(K-1):1.5; # weights for non-uniform sampling without replacement
    #p = randperm(K)
    ##weights = (1:K).^(-0.8)
    #weights = reverse(weights, dist = 2)
    w = aweights(weights/sum(weights)*S)

    #### algorithm parameters #################################
    runs = 1;         # number of trials

    ### initialisation of dictionary
    dico = randn(d,K)
    normalise!(dico)
    org_dico = copy(dico)
    ##### start trials #################################
    # create perturbation orthogonal to each dico atom
    Z = randn(d,K)
    for k = 1:K
        Z[:,k] = Z[:,k]-(Z[:,k]'*dico[:,k])*dico[:,k]
        Z[:,k] = Z[:,k]/norm(Z[:,k])
    end 
    # perturbed dictionary
    if eps == 0
        dico_init = randn(d,K);
    else
        dico_init = (1-eps^2/2)*dico + (eps^2-eps^4/4)^(1/2)*Z;
    end

    normalise!(dico_init) 
    x1toS=sqrt(1/S).*(1-b).^(1:S)
    x1toS = x1toS./norm(x1toS)
    Y =  zeros(d,N)
    p = [collect(1:S) for t in 1:Threads.nthreads()] 

    function generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)
        @inbounds Threads.@threads for n = 1:N
            sample!(1:K,w, p[Threads.threadid()]; replace=false, ordered=false)
            #rand!(@view(Y[:,n]))
            x1toS .= x1toS.*rand([-1, 1],(S))
            #mul!(@view(Y[:,n]),@view(Y[:,n]),rho)
            mul!(@view(Y[:,n]),@view(dico[:,p[Threads.threadid()]]),x1toS)
            Y[:,n] = Y[:,n]./norm(Y[:,n])
        end
        return Y
    end

    # savefile noisy signals
    # SNR = 1/(d*rho^2)
    # savefile = strcat("comp_synth_data_its_d',num2str(d),'_K',num2str(K),'_S',num2str(S),'_N',num2str(N),'_maxit',num2str(maxit),'_b',num2str(b),'_SNR',num2str(SNR),'_',dicotype,'.mat")
    # savefile noiseless signals
    #savefile = strcat("comp_synth_data_its_d',num2str(d),'_K',num2str(K),'_S',num2str(S),'_N',num2str(N),'_maxit',num2str(maxit),'_b',num2str(b),'_',dicotype,'.mat")

    ### create signal generating dictionary and generate signals
    rtdico = copy(dico_init);
    mtdico = copy(dico_init);
    ktdico = copy(dico_init);

    #print(rtdico)
    var = zeros(3,iter+1,4);
    var[:,1,1] .= mean(maximum(abs,dico_init'*org_dico, dims = 1))
    var[:,1,3] .= mean(sqrt.(2*ones(1,K) -2*maximum(abs,dico_init'*org_dico, dims = 1)))
    var[:,1,4] .= maximum(sqrt.(2*ones(1,K) -2*maximum(abs,dico_init'*org_dico, dims = 1)))
    var[:,1,2] .= sum(maximum(abs,dico_init'*org_dico, dims = 1).>0.9)/K

    for i = 1:iter
        Y = generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)
        rtdico = itkrm(Y,S,K,rtdico,1)
        mtdico = mod(Y,S,K,rtdico,1)
        ktdico = ksvd(Y,S,K,rtdico,1)
        
        
        #rtdico = itkrm_update!(X ,Y ,K,S,1,rtdico,ip ,gram,ix,ind)
        
        #print(mean(maximum(abs,rtdico'*org_dico, dims = 1)))

        var[1,i+1,1] = mean(maximum(abs,rtdico'*org_dico, dims = 1))
        var[2,i+1,1] = mean(maximum(abs,mtdico'*org_dico, dims = 1))
        var[3,i+1,1] = mean(maximum(abs,ktdico'*org_dico, dims = 1))
        var[1,i+1,3] = mean(sqrt.(2*ones(1,K) - 2*maximum(abs,rtdico'*org_dico, dims = 1)))
        var[2,i+1,3] = mean(sqrt.(2*ones(1,K) - 2*maximum(abs,mtdico'*org_dico, dims = 1)))
        var[3,i+1,3] = mean(sqrt.(2*ones(1,K) -2*maximum(abs,ktdico'*org_dico, dims = 1)))
        var[1,i+1,4] = maximum(sqrt.(2*ones(1,K) -2*maximum(abs,rtdico'*org_dico, dims = 1)))
        var[2,i+1,4] = maximum(sqrt.(2*ones(1,K) -2*maximum(abs,mtdico'*org_dico, dims = 1)))
        var[3,i+1,4] = maximum(sqrt.(2*ones(1,K) -2*maximum(abs,ktdico'*org_dico, dims = 1)))
        var[1,i+1,2] = sum(maximum(abs,rtdico'*org_dico, dims = 1).>0.9)/K
        var[2,i+1,2] = sum(maximum(abs,mtdico'*org_dico, dims = 1).>0.9)/K
        var[3,i+1,2] = sum(maximum(abs,ktdico'*org_dico, dims = 1).>0.9)/K
    end
    f = Figure()

    axes = [Axis(f[i, j]) for i in 1:2, j in 1:2]

    xs = 1:iter+1;
    for (i, ax) in enumerate(axes)
        lines!(ax,xs,var[1,:,i],label = "ITkrM")
        lines!(ax,xs,var[2,:,i],label = "MOD")
        lines!(ax,xs,var[3,:,i],label = "K-SVD")
        #axislegend(ax, position = :rb)
        ax.xlabel = "Iterations"
    end

    axes[1].title = "Average Inner Product"
    axislegend(axes[1],position = :rb)
    axes[2].title = "Number of Atoms with ip > 0.9"
    axislegend(axes[2],position = :rb)
    axes[3].title = "Average Distance"
    axislegend(axes[3],position = :rt)
    axes[4].title = "Biggest Distance" 
    axislegend(axes[4],position = :rt)
    # axes[1].xticks = 0:6

    # axes[2].xticks = 0:pi:2pi
    # axes[2].xtickformat = xs -> ["$(x/pi)π" for x in xs]

    # axes[3].xticks = (0:pi:2pi, ["start", "middle", "end"])

    # axes[4].xticks = 0:pi:2pi
    # axes[4].xtickformat = "{:.2f}ms"
    # 
    #xes[4].xlabel = "Time"
    f
end
