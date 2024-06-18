


using LinearAlgebra
using Random
using BenchmarkTools
using Base.Threads
using Base.Sort
using StatsBase
using Makie
using ProgressMeter
using CairoMakie
using FFTW
using Hadamard
using Infiltrator
using Distributions

function create_2d_plot()
    # a function that calls run_tests for 10 values each of alpha and beta in 0,1 and then plots the number of recovered dict elements in a 2d heatmap
    d = 128
    K = 256
    S = 4
    b = 0.0
    rho = 0.0
    eps = 1.1
    N = 10000
    iter = 10
    recovered = zeros(3,3,3) .+ 0.00000000001
    runs = 100

    for i in 0:0.5:1
        for j in 0:0.5:1
            for l in 1:runs
                println(i,j,l)
                recovered[:,round(Int,i*2)+1,round(Int,j*2)+1] += run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=N,iter=iter,alpha=i,beta=j)[1:3,end,2]/runs
            end
            # recovered[round(Int,i*10)+1,round(Int,j*10)+1] = run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=N,iter=iter,alpha=i,beta=j)[1,end,2]
        end
    end
    println(recovered)
    fig = Figure(resolution = (1200, 1200))
    ax = CairoMakie.Axis(fig[1, 1], yscale = log, title = string(scale),xminorticksvisible = true, xminorgridvisible = true)#,xminorticks = IntervalsBetween(5)) 
    lines!(ax, 0:0.5:1, recovered[1,:, 1], color = :blue, linewidth = 2, markersize = 6, marker = :circle, label = "beta = 0.0")
    # lines!(ax, 0:0.5:1, recovered[1,:, 2], color = :green, linewidth = 2, markersize = 6, marker = :circle, label = "beta = 0.5")
    # lines!(ax, 0:0.5:1, recovered[1,:, 3], color = :red, linewidth = 2, markersize = 6, marker = :circle, label = "beta = 1.0")

    lines!(ax, 0:0.5:1, recovered[2,:, 1], color = :orange, linewidth = 2, markersize = 6, marker = :square, label = "one preconditioning, beta = 0.0")
    # lines!(ax, 0:0.5:1, recovered[2,:, 2], color = :purple, linewidth = 2, markersize = 6, marker = :square, label = "one preconditioning, beta = 0.5")
    # lines!(ax, 0:0.5:1, recovered[2,:, 3], color = :pink, linewidth = 2, markersize = 6, marker = :square, label = "one preconditioning, beta = 1.0")

    lines!(ax, 0:0.5:1, recovered[3,:, 1], color = :cyan, linewidth = 2, markersize = 6, marker = :diamond, label = "all preconditioning, beta = 0.0")
    # lines!(ax, 0:0.5:1, recovered[3,:, 2], color = :magenta, linewidth = 2, markersize = 6, marker = :diamond, label = "all preconditioning, beta = 0.5")
    # lines!(ax, 0:0.5:1, recovered[3,:, 3], color = :yellow, linewidth = 2, markersize = 6, marker = :diamond, label = "all preconditioning, beta = 1.0")
    axislegend(ax,position = :rb)
    display(fig)

    # fig = Figure(resolution = (1200, 1200))
    # ax = CairoMakie.Axis(fig[1, 1])#, yscale = log, title = "Number of Recovered Dict Elements", xminorticksvisible = true, xminorgridvisible = true)
    fig, ax, hm = heatmap(recovered[1,:,:], colormap = :grays)
    Colorbar(fig[:, end+1], hm)
    display(fig)

    # fig = Figure(resolution = (1200, 1200))
    #, yscale = log, title = "Number of Recovered Dict Elements", xminorticksvisible = true, xminorgridvisible = true)
    fig, ax, hm = heatmap(recovered[2,:,:], colormap = :grays)
    Colorbar(fig[:, end+1], hm)
    display(fig)

end


function run_tests(;d::Int64 = 128,K::Int64 = 256,S::Int64 = 4, b::Float64 = 0.0,
    rho::Float64 = 0., eps::Float64 = 1.1 ,N::Int64 = 1000,iter::Int64 = 50, alpha::Float64 = 1., beta::Float64 = 0.)
    #### Testfile to reproduce plots in the paper. 

    # weights = 0.3:10.2/(K-1):10.5; # weights for non-uniform sampling without replacement
    #p = randperm(K)
    weights = (1:K).^(-0.8)
    # weights[1:Int(K/2)] .= 4
    # weights[Int(K/2)+1:end] .= 1
    weights .= 1.:9.0/(K-1):10.0;
    # weights = abs.(randn(K))
    shuffle!(weights)
    #weights .= 1
    # weights[1:Int(K/4)] .= 3
    # weights[Int(K/2)+1:Int(K/2)+Int(K/4)] .= 3
    #weights = reverse(weights, dist = 2)
    # println(weights)
    w = aweights(weights/sum(weights)*S)

    ### initialisation of dictionary
    dico = randn(d,K)   

    # dico = [Matrix(1.0I, d, d) idct(Matrix(1.0I, d, d),1) ]#ifwht(Matrix(1.0I, d, d)) randn(d,d)]
    normalise!(dico)

    diag_sqrt_weights = diagm(sqrt.(weights))

    dico_diag_sqrt_weights = dico * diag_sqrt_weights
    eigvals, eigvecs = eigen(dico_diag_sqrt_weights * dico_diag_sqrt_weights')

    # set the negative eigenvalues to 0
    eigvals = max.(eigvals, 0)
    # eigvals below 1e-6 are set to 0
    eigvals[eigvals .< 1e-6] .= 0

    diag_inv_sqrt_eigvals = 1 ./ sqrt.(eigvals)
    diag_inv_sqrt_eigvals[diag_inv_sqrt_eigvals .== Inf] .= 0

    dico_stable = eigvecs * diagm(vec(diag_inv_sqrt_eigvals)) * eigvecs'

    dico_2 = dico_stable * dico*  diag_sqrt_weights
    normalise!(dico_2)
    dico = dico * (1-alpha) + dico_2 * (alpha)
    normalise!(dico)
    org_dico = copy(dico)
    # plot abs(org_dico'*org_dico - diagm(ones(K)))
    fig = Figure(size = (1200, 1200))
    ax = CairoMakie.Axis(fig[1, 1])
    heatmap!(ax, abs.(org_dico'*org_dico - diagm(ones(K))), colormap = :grays)
    ax = CairoMakie.Axis(fig[1, 2])
    # display(fig)

    println("Norm of dict:"*string(norm(diagm(sqrt.(w))*abs.(org_dico'*org_dico - diagm(ones(K)))*diagm(sqrt.(w)))))

    Z = randn(d,K) #
    normalise!(Z)
    #@Z += 0.2*dico;
    #bad = dico[:,1];
    #@infiltrate
    for k = 1:K
        Z[:,k] =  Z[:,k]-(Z[:,k]'*dico[:,k])*dico[:,k]
        Z[:,k] = Z[:,k]/norm(Z[:,k])
     end 
    # # # perturbed dictionary
    if eps == 0
        dico_init = Z;
    else
        dico_init = (1-eps^2/2)*dico + (eps^2-eps^4/4)^(1/2)*Z;
    end
    Z = randn(d,K) #

    Z_sorted = zeros(size(Z))
    selected_idx = []
    for k = 1:K
        inner_products = abs.(Z' * dico_init[:, k])
        max_idx = argmax(inner_products)
        while max_idx in selected_idx
            inner_products[max_idx] = 0
            max_idx = argmax(inner_products)
        end
        selected_idx = [selected_idx; max_idx]
        Z_sorted[:, k] = Z[:, max_idx]
    end
    Z = Z_sorted


    normalise!(Z)
    dico_init = Z;

    #dico_init = dico;
    #dico_init[:,1] = dico[:,1] + dico[:,2];
    #dico_init[:,2] = dico[:,3]
    #dico_init[:,5] = dico[:,4] + dico[:,5];
    #dico_init[:,4] = dico[:,6]
    # dico_init[:,3] = (1-eps^2/2)*dico[:,3] + (eps^2-eps^4/4)^(1/2)*dico[:,1] ;
    # dico_init[:,3] = (1-eps^2/2)*dico[:,3] + (eps^2-eps^4/4)^(1/2)*dico[:,1] ;
    #dico_init[:,1] += randn(d,1)
    #dico_init = (dico_init * diagm(w.^2) * dico_init')^(-1) * dico_init * diagm(w)
    normalise!(dico_init) 

    dico_diag_sqrt_weights = dico_init * diag_sqrt_weights
    eigvals, eigvecs = eigen(dico_diag_sqrt_weights * dico_diag_sqrt_weights')

    # set the negative eigenvalues to 0
    eigvals = max.(eigvals, 0)
    # eigvals below 1e-6 are set to 0
    eigvals[eigvals .< 1e-6] .= 0

    diag_inv_sqrt_eigvals = 1 ./ sqrt.(eigvals)
    diag_inv_sqrt_eigvals[diag_inv_sqrt_eigvals .== Inf] .= 0

    dico_stable = eigvecs * diagm(vec(diag_inv_sqrt_eigvals)) * eigvecs'
    dico_init_2 = dico_stable * dico_init*  diag_sqrt_weights
    normalise!(dico_init_2)
    dico_init = dico_init * (1-beta) + dico_init_2 * (beta)
    normalise!(dico_init)
    org_dico = copy(dico)
    x1toS=sqrt(1/S).*(1-b).^(1:S)
    x1toS = x1toS./norm(x1toS)
    Y =  zeros(d,N)
    p = [collect(1:S) for t in 1:Threads.nthreads()] 

    function generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)
        @inbounds Threads.@threads for n = 1:N
            # sample!(1:K,w, p[Threads.threadid()]; replace=false, ordered=false)
            p[Threads.threadid()]=bernoulli_sample(w, S)
            shuffle!(p[Threads.threadid()])
            x1toS .= x1toS.*rand([-1, 1],(S))
            
            mul!(@view(Y[:,n]),@view(dico[:,p[Threads.threadid()]]),x1toS)
            Y[:,n] += randn(d,1)*rho;
            Y[:,n] = Y[:,n]./norm(Y[:,n])
        end
        return Y
    end
    
    function load_brain(d)
        train_path::String=joinpath(pwd(), "images", "Brain4")
        # load("brain_data.mat")
        # d = 64
        list = readdir(train_path)[2:end]
        Y = zeros(d,len(list))
        for j in eachindex(list)
            img = imresize( Gray.(load(joinpath(path_vec,list[j]))),(sqrt(d),sqrt(d)))
            img = convert(Matrix{Float64},img)
            img = img./norm(img);
            Y[:,j] = reshape(img,d)
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
    var[:,1,3] .= mean(sqrt.(2*ones(1,K).+ 0.0000001 -2*maximum(abs,dico_init'*org_dico, dims = 1)))
    var[:,1,4] .= maximum(sqrt.(2*ones(1,K).+ 0.0000001 -2*maximum(abs,dico_init'*org_dico, dims = 1)))
    var[:,1,2] .= sum(maximum(abs,dico_init'*org_dico, dims = 1).>0.9)/K
    println("size of dictionary:"*string(K))
    println(var[1,1,4])
    prog = Progress(iter, dt=0.5,desc="Learning los dictionarios...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    i = 1
    


    # ip_metal = MtlArray(Float32.(zeros(K,N)))
    # absip_metal = MtlArray(Float32.(zeros(K,N)))
    # ind_metal = MtlArray(Float32.(zeros(K,N)))
    # ma_metal = ones(K,K)-diagm(ones(K))
    # diag_zero = MtlArray(Float32.(ma_metal))
    # ma_metal = zeros(K,K)+diagm(ones(K))
    # diag_one = MtlArray(Float32.(ma_metal))
    # rtdico_metal = MtlArray(Float32.(rtdico))
    
    Y = generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)
    
    # Y_metal = MtlArray(Float32.(Y))

    #@infiltrate
    
    d,N = size(Y)
    ip = zeros(K,N)
    absip = zeros(K,N)
    X = zeros(K,N)
    gram = zeros(K,K)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    
    est_weights= [zeros(K) for t in 1:Threads.nthreads()]
    weights = zeros(K)
    N_org = N
    while i <= iter
        Y = generate!(Y,w,x1toS,rho,N,K,p,S,dico,d)
    
        start_idx = Int((i-1) *  round(N_org/iter) + 1)
        end_idx = Int(min(i * round(N_org/iter), size(Y, 2)))
        Y_new = Y#Y[:, start_idx:end_idx]
        d,N = size(Y_new)
        ip = zeros(K,N)
        absip = zeros(K,N)
        X = zeros(K,N)
        gram = zeros(K,K)
        ix = [collect(1:K) for t in 1:Threads.nthreads()]
        ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
        
        est_weights= [zeros(K) for t in 1:Threads.nthreads()]
        weights = zeros(K)
        rtdico, _ = mod_dl(Y_new, S, K, rtdico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
        
        # if i == 1
        #     mtdico, x_mdico = mod_dl_copy(Y_new, S, K, mtdico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
        # else
        #     mtdico, _ = mod_dl(Y_new, S, K, mtdico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
        # end
        
        #rtdico = itkrm_update!(X ,Y ,K,S,1,rtdico,ip ,gram,ix,ind)
        
        #print(mean(maximum(abs,rtdico'*org_dico, dims = 1)))

        var[1,i+1,1] = mean(maximum(abs,rtdico'*org_dico, dims = 1))
        var[2,i+1,1] = mean(maximum(abs,mtdico'*org_dico, dims = 1))
        var[3,i+1,1] = mean(maximum(abs,ktdico'*org_dico, dims = 1))
        var[1,i+1,3] = mean(sqrt.(2*ones(1,K) .+ 0.0000001 - 2*maximum(abs,rtdico'*org_dico, dims = 1)))
        var[2,i+1,3] = mean(sqrt.(2*ones(1,K) .+ 0.0000001 - 2*maximum(abs,mtdico'*org_dico, dims = 1)))
        var[3,i+1,3] = mean(sqrt.(2*ones(1,K) .+ 0.0000001 -2*maximum(abs,ktdico'*org_dico, dims = 1)))
        var[1,i+1,4] = maximum(sqrt.(2*ones(1,K) .+ 0.0000001  -2*maximum(abs,rtdico'*org_dico, dims = 1)))
        var[2,i+1,4] = maximum(sqrt.(2*ones(1,K) .+ 0.0000001 -2*maximum(abs,mtdico'*org_dico, dims = 1)))
        var[3,i+1,4] = maximum(sqrt.(2*ones(1,K) .+ 0.0000001 -2*maximum(abs,ktdico'*org_dico, dims = 1)))
        var[1,i+1,2] = sum(maximum(abs,rtdico'*org_dico, dims = 1).>0.95)/K
        var[2,i+1,2] = sum(maximum(abs,mtdico'*org_dico, dims = 1).>0.95)/K
        var[3,i+1,2] = sum(maximum(abs,ktdico'*org_dico, dims = 1).>0.95)/K
        # next!(prog; showvalues = [(:iter,i), (:found,round(var[1,i+1,2]*K))])
        i += 1
    end
    f = Figure(resolution = (1200, 1200))

    axes = [CairoMakie.Axis(f[i, j], yscale = log,xminorticksvisible = true, xminorgridvisible = true,xminorticks = IntervalsBetween(2)) for i in 1:2, j in 1:2]

    xs = 1:iter+1;
    for (i, ax) in enumerate(axes)
        lines!(ax,xs,var[1,:,i].+ 0.00000001,label = "MOD")
        lines!(ax,xs,var[2,:,i].+ 0.00000001,label = "MOD mit preconditioning")
        #lines!(ax,xs,var[3,:,i],label = "K-SVD")
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
    display(f)
    println(var[1,end,2])
    if var[1,iter+1,2] == 1
        println("success")
    else
        println("try again")
    end
    return var
end


function load_mnist(d)
    # Load MNIST dataset
    dataset = MNIST(; Tx=Float32) # FashionMNIST(; Tx=Float32)  # Load training data
    train_x, _ = dataset[:] 
    # Reshape images to vectors of length d
    N = size(train_x)[3]  # Number of samples
    # resize images to sqrt(d) x sqrt(d)
    Y = zeros(d,N)
    for i = 1:N
        img = imresize(train_x[:, :, i], (Int(sqrt(d)),Int(sqrt(d))))
        img = (img-mean(img)*ones(size(img)))./norm(img-mean(img)*ones(size(img)));
        Y[:,i] = reshape(img,d)
    end
   
    return Y
end


function run_tests_brain_data(;d::Int64 = 28^2,K::Int64 = Int(round(28^2)), S::Int64 = Integer(round(28^2*0.05)),iter::Int64 = 1)
    #### Testfile to reproduce plots in the paper. 

    #### algorithm parameters #################################
    runs = 1;         # number of trials

    Z = randn(d,K) #
    normalise!(Z)

    dico_init = Z;

    normalise!(dico_init)

    function load_brain(d)
        train_path::String=joinpath(pwd(), "images", "Brain4")
        # load("brain_data.mat")
        # d = 64
        list = readdir(train_path)[2:end]
        Y = zeros(d,length(list))
        for j in eachindex(list)
            img = imresize( Gray.(load(joinpath(train_path,list[j]))),(Integer(sqrt(d)),Integer(sqrt(d))))
            img = convert(Matrix{Float64},img)
            img = img./norm(img);
            Y[:,j] = reshape(img,d)
        end
        return Y
    end
   
    ### create signal generating dictionary and generate signals
    rtdico = copy(dico_init);
    mtdico = copy(dico_init);
    var = zeros(3,iter+1,4);

    prog = Progress(iter, dt=0.5,desc="Learning los dictionarios...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    i = 1
    
    
    Y = load_brain(d)
    replace!(Y, Inf=>0)
    replace!(Y, NaN=>0)


    d,N = size(Y)
    ip = zeros(K,N)
    absip = zeros(K,N)
    X = zeros(K,N)
    gram = zeros(K,K)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    
    est_weights= [zeros(K) for t in 1:Threads.nthreads()]
    weights = zeros(K)
    N_org = N
    while i <= iter
        start_idx = Int((i-1) *  round(N_org/iter) + 1)
        end_idx = Int(min(i * round(N_org/iter), size(Y, 2)))
        Y_new = Y#Y[:, start_idx:end_idx]
        d,N = size(Y_new)
        ip = zeros(K,N)
        absip = zeros(K,N)
        X = zeros(K,N)
        gram = zeros(K,K)
        ix = [collect(1:K) for t in 1:Threads.nthreads()]
        ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
        
        est_weights= [zeros(K) for t in 1:Threads.nthreads()]
        weights = zeros(K)
        rtdico, x_rtdico = mod_dl(Y_new, S, K, rtdico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
        mtdico, x_mdico = mod_dl_copy(Y_new, S, K, mtdico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)

        var[1,i+1,1] = norm(Y - rtdico*x_rtdico)
        var[2,i+1,1] = norm(Y - mtdico*x_mdico)
        var[1,i+1,2] = 0
        var[2,i+1,2] = 0
        var[1,i+1,3] = 0
        var[2,i+1,3] = 0
        var[1,i+1,4] = 0
        var[2,i+1,4] = 0
        
        next!(prog; showvalues = [(:iter,i), (:norm,round(var[1,i+1,1]))])
        i += 1
    end
    print(var[1,:,1])
    print(var[2,:,1])
    f = Figure(resolution = (1200, 1200))

    var[:,1,1] .= 0
    var[:,1,3] .= 0
    var[:,1,4] .= 0
    var[:,1,2] .= 0

    axes = [CairoMakie.Axis(f[i, j]) for i in 1:2, j in 1:2]

    xs = 1:iter+1;
    for (i, ax) in enumerate(axes)
        lines!(ax,xs,var[1,:,i],label = "MOD")
        lines!(ax,xs,var[2,:,i],label = "MOD mit preconditioning")
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

    display(f)
    function plot_dictionary(dico_init, K, name)
        
        f = Figure(resolution = (3200, 3200))
        axes = [CairoMakie.Axis(f[i,j]) for i in 1:Integer(round(sqrt(K)-1)), j in 1:Integer(round(sqrt(K)-1))]

        for (i, ax) in enumerate(axes)
            atom = reshape(dico_init[:, i], Integer(sqrt(d)), Integer(sqrt(d)))
            if i == 1
                heatmap!(ax, atom, colormap = :grays, text = text("Dictionary", color = :black, fontsize = 20, pos = (0.5, 0.5), halign = :center, valign = :center))
            else
                heatmap!(ax, atom, colormap = :grays)
            end
            
        end
        display(f)
    end

    # Call the function to plot the dictionary
    plot_dictionary(rtdico, K, "rtdico")
    plot_dictionary(mtdico, K, "mtdico")

end


function run_tests_mnist(;d::Int64 = 8^2,K::Int64 = Int(round(8^2*2)), S::Int64 = Integer(round(8^2*0.05)) ,iter::Int64 = 10)
    #### Testfile to reproduce plots in the paper. 

    #### algorithm parameters #################################
    runs = 1;         # number of trials

    Z = randn(d,K) #
    normalise!(Z)

    dico_init = Z;

    normalise!(dico_init)

   
    ### create signal generating dictionary and generate signals
    rtdico = copy(dico_init);
    mtdico = copy(dico_init);
    var = zeros(3,iter+1,4);

    prog = Progress(iter, dt=0.5,desc="Learning los dictionarios...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    i = 1
    
    
    Y = load_mnist(d)

    ### Allocations for more speed
    d,N = size(Y)
    ip = zeros(K,N)
    absip = zeros(K,N)
    X = zeros(K,N)
    gram = zeros(K,K)
    ix = [collect(1:K) for t in 1:Threads.nthreads()]
    ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
    
    est_weights= [zeros(K) for t in 1:Threads.nthreads()]
    weights = zeros(K)
    N_org = N
    while i <= iter
        start_idx = Int((i-1) *  round(N_org/iter) + 1)
        end_idx = Int(min(i * round(N_org/iter), size(Y, 2)))
        Y_new = Y[:, start_idx:end_idx]
        d,N = size(Y_new)
        ip = zeros(K,N)
        absip = zeros(K,N)
        X = zeros(K,N)
        gram = zeros(K,K)
        ix = [collect(1:K) for t in 1:Threads.nthreads()]
        ind= [Vector{Int}(undef,S) for t in 1:Threads.nthreads()]
        
        est_weights= [zeros(K) for t in 1:Threads.nthreads()]
        weights = zeros(K)
        rtdico, x_rtdico = mod_dl(Y_new, S, K, rtdico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
        mtdico, x_mdico = mod_dl_copy(Y_new, S, K, mtdico, d, N, ip, absip, X, gram, ix, ind, est_weights, weights)
        #println(size(Y_new))
        #println(Y[:,1])
        #println(norm(x_rtdico))
        var[1,i+1,1] = norm(Y_new - rtdico*x_rtdico)
        var[2,i+1,1] = norm(Y_new - mtdico*x_mdico)
        var[1,i+1,2] = 0
        var[2,i+1,2] = 0
        var[1,i+1,3] = 0
        var[2,i+1,3] = 0
        var[1,i+1,4] = 0
        var[2,i+1,4] = 0
        
        next!(prog; showvalues = [(:iter,i), (:norm,round(var[1,i+1,1]))])
        i += 1
    end
    print(var[1,:,1])
    print(var[2,:,1])
    f = Figure(resolution = (1200, 1200))

    var[:,1,1] .= 0
    var[:,1,3] .= 0
    var[:,1,4] .= 0
    var[:,1,2] .= 0

    axes = [CairoMakie.Axis(f[i, j]) for i in 1:2, j in 1:2]

    xs = 1:iter+1;
    for (i, ax) in enumerate(axes)
        lines!(ax,xs,var[1,:,i],label = "MOD")
        lines!(ax,xs,var[2,:,i],label = "MOD mit preconditioning")
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

    display(f)
    # function plot_dictionary(dico_init, K, name)
        
    #     f = Figure(resolution = (3200, 3200))
    #     axes = [CairoMakie.Axis(f[i,j]) for i in 1:Integer(round(sqrt(K)-1)), j in 1:Integer(round(sqrt(K)-1))]

    #     for (i, ax) in enumerate(axes)
    #         atom = reshape(dico_init[:, i], Integer(sqrt(d)), Integer(sqrt(d)))
    #         heatmap!(ax, atom, colormap = :grays)     
    #     end
    #     axes[1].title = name
    #     display(f)
    # end
    # #@infiltrate
    # # Call the function to plot the dictionary
    # plot_dictionary(rtdico, K, "rtdico")
    # plot_dictionary(mtdico, K, "mtdico")

end
