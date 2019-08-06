module RoomTemperatureLearn

using StatsBase, Random, LinearAlgebra, Gadfly

include("RoomTemperature.jl")
using .RoomTemperature

include("SVM.jl")
using .SVM

t_max = 10 # seconds
data = RoomTemperature.simulate(t_max)

# show size of data
@show size(data)

# plot a window of data
plot(data[1:100, :], x=:t, y=:T, color=:du)

# try to feed data as is
rng = MersenneTwister(1234)

X = transpose(convert(Matrix, data[!, [:u, :T]]))

n, m = size(X)

Y = [du == 1.0 ? 1.0 : -1.0 for du in data[!, :du]]

train = bitrand(rng, m)
test = .!train

train = bitrand(rng, m)
test = .!train

@show model = svm(X[:,train], Y[train], max_passes = 1000)

@show train_cost = cost(model, X[:,train], Y[train])

@show train_accuracy = accuracy(model, X[:,train], Y[train])
@show test_accuracy = accuracy(model, X[:,test], Y[test])

end # RoomTemperaturerLearn
