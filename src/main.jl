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
using DelimitedFiles


function create_convergence_plot()
    d = 64
    K = 128
    S = 4
    b = 0.0
    rho = 0.0
    eps = 0.9
    runs = 100
    alpha = 1.0
    beta = 0.0
    iter_max = 8
    recovered = zeros(6, iter_max+1, 5)
    for i in 1:runs
        recovered[1, :,:] += run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=2500,iter=iter_max,alpha=alpha,beta=beta,decay_factor=2.)[1,:,:]/runs
        recovered[2, 1:6,:] += run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=2500,iter=5,alpha=alpha,beta=beta,decay_factor=4.)[1,:,:]/runs
        recovered[3, :,:] += run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=10000,iter=iter_max,alpha=alpha,beta=beta,decay_factor=1.)[1,:,:]/runs
        recovered[4, :,:] += run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=40000,iter=iter_max,alpha=alpha,beta=beta,decay_factor=1.)[1,:,:]/runs
        recovered[5, :,:] += run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=160000,iter=iter_max,alpha=alpha,beta=beta,decay_factor=1.)[1,:,:]/runs
        println(i)
    end
    fig = Figure(size = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1], yscale = log10, xminorticksvisible = true, xminorgridvisible = true, xminorticksize = 6,
        yminorticksize = 6, xlabel = "iteration", ylabel = L"$\delta(\Phi, \Psi)$",
        titlesize = 23, ylabelsize = 21, xlabelsize = 18)
    j = 1
    for N in [1250, 2500, 10000, 40000, 160000]
        if N == 1250
             scatterlines!(ax, 0:iter_max, max(recovered[j,:, 5],recovered[j,:, 4]), linewidth = 4, marker = :xcross, markersize = 13, label = "N adapted, scaling factor 2", linestyle = :dashdot, colormap = :broc)
        elseif N == 2500
            scatterlines!(ax, 0:5, max(recovered[j,1:6, 5],recovered[j,1:6, 4]), linewidth = 4, marker = :star5, markersize = 13, label = "N adapted, scaling factor 4", linestyle = :dashdot, colormap = :broc)
        elseif N == 10000
            scatterlines!(ax, 0:iter_max, max(recovered[j,:, 5],recovered[j,:, 4]), linewidth = 2,  marker = :circle, markersize = 13, label = "N = $N", colormap = :broc)
        elseif N == 40000
            scatterlines!(ax, 0:iter_max, max(recovered[j,:, 5],recovered[j,:, 4]), linewidth = 2, marker = :utriangle, markersize = 13, label = "N = $N", colormap = :broc)
        elseif N == 160000
            scatterlines!(ax, 0:iter_max, max(recovered[j,:, 5],recovered[j,:, 4]), linewidth = 2, marker = :diamond, markersize = 13, label = "N = $N", colormap = :broc)
        end
        j = j+1
    end
    lines!(ax, collect(0:iter_max), eps .* ((1/2) .^ collect(0:iter_max)),
       linewidth = 2, linestyle = :dash, color = :black, label = "Reference line")

    scatter!(ax, collect(0:iter_max), eps .* ((1/2) .^ collect(0:iter_max)),
         markersize = 6, color = :black)
    axislegend(ax)
    display(fig)
    save("convergence_plot_05022025.pdf", fig)
    writedlm( "convergence_plot_new_expo_05022025.csv",  recovered, ',')
end



function create_2d_plot()
    # a function that calls run_tests for 10 values each of alpha and beta in 0,1 and then plots the number of recovered dict elements in a 2d heatmap
    d = 128
    K = 256
    S = 4
    b = 0.0
    rho = 0.0
    eps = 0.0
    N = 10000
    iter = 5
    recovered = zeros(3,6,6) .+ 0.00000000001
    runs = 10

    for i in 0:1.0:1
        for j in 0:1.0:1
           for l in 1:runs
                println(i,j,l)
                recovered[:,round(Int,i*5)+1,round(Int,j*5)+1] += run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=N,iter=iter,alpha=i,beta=j)[1:3,end,2]/runs
            end
            # recovered[round(Int,i*10)+1,round(Int,j*10)+1] = run_tests(d=d,K=K,S=S,b=b,rho=rho,eps=eps,N=N,iter=iter,alpha=i,beta=j)[1,end,2]
        end
    end
    println(recovered)
    fig = Figure(size = (1200, 1200))
    # ax = CairoMakie.Axis(fig[1, 1], yscale = log, title = string(scale),xminorticksvisible = true, xminorgridvisible = true)#,xminorticks = IntervalsBetween(5)) 
    # lines!(ax, 0:0.5:1, recovered[1,:, 1], color = :blue, linewidth = 2, markersize = 6, marker = :circle, label = "beta = 0.0")
    # # lines!(ax, 0:0.5:1, recovered[1,:, 2], color = :green, linewidth = 2, markersize = 6, marker = :circle, label = "beta = 0.5")
    # # lines!(ax, 0:0.5:1, recovered[1,:, 3], color = :red, linewidth = 2, markersize = 6, marker = :circle, label = "beta = 1.0")

    # lines!(ax, 0:0.5:1, recovered[2,:, 1], color = :orange, linewidth = 2, markersize = 6, marker = :square, label = "one preconditioning, beta = 0.0")
    # # lines!(ax, 0:0.5:1, recovered[2,:, 2], color = :purple, linewidth = 2, markersize = 6, marker = :square, label = "one preconditioning, beta = 0.5")
    # # lines!(ax, 0:0.5:1, recovered[2,:, 3], color = :pink, linewidth = 2, markersize = 6, marker = :square, label = "one preconditioning, beta = 1.0")

    # lines!(ax, 0:0.5:1, recovered[3,:, 1], color = :cyan, linewidth = 2, markersize = 6, marker = :diamond, label = "all preconditioning, beta = 0.0")
    # # lines!(ax, 0:0.5:1, recovered[3,:, 2], color = :magenta, linewidth = 2, markersize = 6, marker = :diamond, label = "all preconditioning, beta = 0.5")
    # # lines!(ax, 0:0.5:1, recovered[3,:, 3], color = :yellow, linewidth = 2, markersize = 6, marker = :diamond, label = "all preconditioning, beta = 1.0")
    # axislegend(ax,position = :rb)
    # display(fig)

    # fig = Figure(size = (1200, 1200))
    # ax = CairoMakie.Axis(fig[1, 1])#, yscale = log, title = "Number of Recovered Dict Elements", xminorticksvisible = true, xminorgridvisible = true)
    fig, ax, hm = heatmap(recovered[1,:,:], colormap = :grays)
    Colorbar(fig[:, end+1], hm)
    display(fig)
    save("recovered_dict_elements_2606.pdf", fig)
    # # fig = Figure(size = (1200, 1200))
    # #, yscale = log, title = "Number of Recovered Dict Elements", xminorticksvisible = true, xminorgridvisible = true)
    # fig, ax, hm = heatmap(recovered[2,:,:], colormap = :grays)
    # Colorbar(fig[:, end+1], hm)
    # display(fig)
    writedlm( "2d_plot_2606.csv",  recovered, ',')
end


function run_tests(;d::Int64 = 128,K::Int64 = 256,S::Int64 = 4, b::Float64 = 0.0,
    rho::Float64 = 0., eps::Float64 = 1.1 ,N::Int64 = 1000,iter::Int64 = 50, alpha::Float64 = 1., beta::Float64 = 0., decay_factor::Float64 = 0.)
    weights = (1:K).^(-0.5)
    shuffle!(weights)
    weights_sorted = sort(weights, rev = true)
    
    w = aweights(weights/sum(weights)*S)
    inclusion_probs = zeros(K)
    l = 0
    while l < 100000
        inclusion_probs[bernoulli_sample(w, S)] .+=1/100000
        l = l +1
    end
    weights_sorted = sort(inclusion_probs, rev = true)
    # fig = Figure(size = (800, 600))
    # ax = CairoMakie.Axis(fig[1, 1],xminorticksvisible = true, xminorgridvisible = true, xminorticksize = 6,
        # yminorticksize = 6, xlabel = "index", ylabel = L"$\pi$",
        # titlesize = 23, ylabelsize = 21, xlabelsize = 18)
    # t = "inclusion probabilities"
    # CairoMakie.lines!(ax, weights_sorted, color = :black, label = L"%$(t) $\pi$")
    # axislegend(ax)
    # display(fig)
    # save("distribution_of_weights.pdf", fig)

    ### initialisation of dictionary
    dico = randn(d,K)
    normalise!(dico)

    diag_sqrt_weights = diagm(sqrt.(inclusion_probs))

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
    # fig = Figure(size = (1200, 1200))
    # ax = CairoMakie.Axis(fig[1, 1])
    # heatmap!(ax, abs.(org_dico'*org_dico - diagm(ones(K))), colormap = :grays)
    # ax = CairoMakie.Axis(fig[1, 2])
    # display(fig)

    Z = randn(d,K) #
    normalise!(Z)
    if eps <= 0.1
        Z_sorted = zeros(size(Z))
        selected_idx = []
        for k = 1:K
            inner_products = abs.(Z' * dico[:, k])
            max_idx = argmax(inner_products)
            while max_idx in selected_idx
                inner_products[max_idx] = 0
                max_idx = argmax(inner_products)
            end
            selected_idx = [selected_idx; max_idx]
            Z_sorted[:, k] = Z[:, max_idx]
        end
        Z = Z_sorted
        dico_init = Z
    else
        for k = 1:K
            Z[:,k] =  Z[:,k]-(Z[:,k]'*dico[:,k])*dico[:,k]
            Z[:,k] = Z[:,k]/norm(Z[:,k])
        end 
        dico_init = (1-eps^2/2)*dico + (eps^2-eps^4/4)^(1/2)*Z;
    end
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

    function generate!(Y,w,x1toS,rho,N,p,S,dico,d)
        @inbounds Threads.@threads for n = 1:N
            p[Threads.threadid()]=bernoulli_sample(w, S)
            shuffle!(p[Threads.threadid()])
            x1toS .= x1toS.*rand([-1, 1],(S))
            
            mul!(@view(Y[:,n]),@view(dico[:,p[Threads.threadid()]]),x1toS)
            Y[:,n] += randn(d,1)*rho;
            Y[:,n] = Y[:,n]./norm(Y[:,n])
        end
        return Y
    end
    
    ### create signal generating dictionary and generate signals
    rtdico = copy(dico_init);

    var = zeros(3,iter+1,5);
    var[:,1,1] .= mean(maximum(abs,dico_init'*org_dico, dims = 1))
    var[:,1,3] .= mean(sqrt.(2*ones(1,K).+ 0.0000001 -2*maximum(abs,dico_init'*org_dico, dims = 1)))
    var[:,1,4] .= maximum(sqrt.(2*ones(1,K).+ 0.0000001 -2*maximum(abs,dico_init'*org_dico, dims = 1)))
    var[:,1,2] .= sum(maximum(abs,dico_init'*org_dico, dims = 1).>0.9)/K
    var[:,1,5] .= opnorm((rtdico -org_dico)*diagm(sqrt.(inclusion_probs)))

    i = 0
    while i < iter
        Y_new = generate!(zeros(d,Int(round(N*(decay_factor)^(i)))),w,x1toS,rho,Int(round(N*(decay_factor)^(i))),p,S,dico,d)

        rtdico, _ = mod_dl(Y_new, S,rtdico)

        var[1,i+2,1] = mean(maximum(abs,rtdico'*org_dico, dims = 1))
        var[1,i+2,3] = mean(sqrt.(2*ones(1,K) .+ 0.0000001 - 2*maximum(abs,rtdico'*org_dico, dims = 1)))
        var[1,i+2,4] = maximum(sqrt.(2*ones(1,K) .+ 0.0000001  -2*maximum(abs,rtdico'*org_dico, dims = 1)))
        var[1,i+2,2] = sum(maximum(abs,rtdico'*org_dico, dims = 1).>0.95)/K
        var[1,i+2,5] = opnorm((rtdico -org_dico)*diagm(sqrt.(inclusion_probs)))
        
        i += 1
    end
    f = Figure(size = (1200, 1200))

    axes = [CairoMakie.Axis(f[i, j], yscale = log,xminorticksvisible = true, xminorgridvisible = true,xminorticks = IntervalsBetween(2)) for i in 1:2, j in 1:2]

    xs = 1:iter+1;
    for (i, ax) in enumerate(axes)
        lines!(ax,xs,var[1,:,i].+ 0.00000001,label = "MOD")
        lines!(ax,xs,var[2,:,i].+ 0.00000001,label = "MOD mit preconditioning")
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


function run_tests_mnist(;d::Int64 =16^2,K::Int64 = Int(round(16^2)), S::Int64 = Integer(round(5)) ,iter::Int64 = 5)
    pdico = randn(d,K)
    normalise!(pdico)
   
    ### create signal generating dictionary and generate signals
    var = zeros(3,iter+1,4);

    prog = Progress(iter, dt=0.5,desc="Learning los dictionarios...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    i = 1
    
    # Load MNIST dataset
    Y = load_mnist(d)
    Y = Y *-1
    d,N = size(Y)
    
    weights_estimated = zeros(K)
    while i <= iter
        start_idx = Int((i-1) *  round(N/iter) + 1)
        end_idx = Int(min(i * round(N/iter), size(Y, 2)))
        Y_curr = Y[:, start_idx:end_idx]
        
        pdico , weights_estimated = mod_dl(Y_curr,S,pdico)
        indices = sortperm(weights_estimated, rev=true)
        pdico = pdico[:,indices]
        X = thresholding(Y_curr, S, pdico)
        
        var[1,i+1,1] = norm(Y_curr - pdico*X)
    
        next!(prog; showvalues = [(:iter,i), (:norm,round(var[1,i+1,1]))])
        i += 1
    end
    _, weights_estimated = mod_dl(Y,S,pdico)
    indices = sortperm(weights_estimated, rev=true)
    pdico = pdico[:,indices]

    weights_sorted = sort(weights_estimated, rev= true)
    fig = Figure(size = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1],xminorticksvisible = true, xminorgridvisible = true, xminorticksize = 6,
        yminorticksize = 6, xlabel = "index", ylabel = L"$\pi$",
        titlesize = 23, ylabelsize = 21, xlabelsize = 18)
    t = "inclusion probabilities"
    CairoMakie.lines!(ax, weights_sorted, color = :black, label = L"%$(t) $\pi$")
    axislegend(ax)
    display(fig)
    save("estimated_weights_mnist.pdf", fig)
    print(var[1,:,1])
    print(var[2,:,1])
    f = Figure(size = (1200, 1200))

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
        
    function plot_dictionary(dico_init, Y)
        d = size(dico_init, 1)
        f = Figure(size = (800, 500))

        # Original signals
        for i in 1:6
            ax = CairoMakie.Axis(f[1, i],yticklabelsvisible=false,xticklabelsvisible=false, xticksvisible=false, yticksvisible=false)
            atom = reshape(Y[:, i], Integer(sqrt(d)), Integer(sqrt(d)))
            heatmap!(ax, atom, colormap = :grays, clims = (0, 1))
            
            ax = CairoMakie.Axis(f[2, i],yticklabelsvisible=false,xticklabelsvisible=false, xticksvisible=false, yticksvisible=false)
            atom = reshape(dico_init[:, i], Integer(sqrt(d)), Integer(sqrt(d)))
            if atom[1,1] < 0
                atom *= -1
            end
            heatmap!(ax, atom, colormap = :grays, clims = (0, 1))
            
            ax = CairoMakie.Axis(f[3, i],yticklabelsvisible=false,xticklabelsvisible=false, xticksvisible=false, yticksvisible=false)
            atom_index = 100 + (i - 1)
            atom = reshape(dico_init[:, atom_index], Integer(sqrt(d)), Integer(sqrt(d)))
            if atom[1,1] < 0
                atom *= -1
            end
            heatmap!(ax, atom, colormap = :grays, clims = (0, 1))

            ax = CairoMakie.Axis(f[4, i],yticklabelsvisible=false,xticklabelsvisible=false, xticksvisible=false, yticksvisible=false)
            atom_index = 200 + (i - 1)
            atom = reshape(dico_init[:, atom_index], Integer(sqrt(d)), Integer(sqrt(d)))
            if atom[1,1] < 0
                atom *= -1
            end
            heatmap!(ax, atom, colormap = :grays, clims = (0, 1))
        end

        display(f)
        save("mnist_elements.pdf", f)
    end
    plot_dictionary(pdico, Y)

end
