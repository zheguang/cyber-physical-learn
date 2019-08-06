module Iris
using RDatasets, Random, LinearAlgebra

include("SVM.jl")
using .SVM

rng = MersenneTwister(1234)

iris = dataset("datasets", "iris")

X = transpose(convert(Matrix, iris[:, 1:4])) # transpose to make outer dimension the intenral dimension, and the inner dimension the number of training examples
n, m = size(X)

Y = [species == "setosa" ? 1.0 : -1.0 for species in iris[:, :Species]]

train = bitrand(rng, m)
test = .!train

@show model = svm(X[:,train], Y[train], max_passes = 1000)

@show train_cost = cost(model, X[:,train], Y[train])

@show train_accuracy = accuracy(model, X[:,train], Y[train])
@show test_accuracy = accuracy(model, X[:,test], Y[test])

end # module Iris
