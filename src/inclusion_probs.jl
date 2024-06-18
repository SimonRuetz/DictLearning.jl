

using Random
using Plots
using StatsBase

S = 5
weights = rand(100);
α = weights/sum(weights);

p = α*S;

N = 100000
inc = zeros(size(p))

for i = 1:N
    inc += (p .>= rand(100))/N
end


inc_rej = zeros(size(p))
counter = 0

for i = 1:N
    random_vector_1 = rand(100)
    if sum(p .>= random_vector_1) == S
        counter += 1
        inc_rej += p .>= random_vector_1
    end
end
inc_rej = inc_rej/counter



inc_suc = zeros(size(p))


weights = aweights(α)



for i = 1:N
    ind = sample( 1:100, weights, Int64(S); replace=false, ordered=false)
    inc_suc[ind] .+= 1/N
end

