#module RoomTemperatureLearn

using StatsBase, Random, LinearAlgebra, Gadfly

include("RoomTemperature.jl")
using .RoomTemperature

include("SVM.jl")
using .SVM

t_max = 10 # seconds
data = RoomTemperature.simulate(t_max)

# learn rules to turn on heater
u_0_condition = 0
if u_0_condition != undef
    filter!(row -> row[:u_0] == u_0_condition, data)
end

# plot a window of data
#plot(data[data[!, :u_0] .== 0, :], x=:t, y=:T_0, color=:du)

# Learn the joint of heater state and room temperature
rng = MersenneTwister(1234)

du_on = 1.0

# make the points separable
data_transit = filter(row -> row[:du] == du_on, data)
@show size(data_transit)
@show T_transit_max, T_transit_min = maximum(data_transit[!, :T]), minimum(data_transit[!, :T])
#within_transit_noise = (T_transit_min - 0.5 .≤ data[!, :T_0]) .& (data[!, :T_0] .≤ T_transit_max + 0.5)
within_transit_noise = (T_transit_min - 0.5 .≤ data[!, :T]) .& (data[!, :T] .≤ T_transit_max + 0.5) .& (data[!, :du] .!= du_on)
#data[within_transit_noise, :du] .= du_on)
data = data[.!within_transit_noise, :]

X = transpose(convert(Matrix, data[!, [:T]]))
Y = [du == du_on ? 1.0 : -1.0 for du in data[!, :du]]
n, m = size(X)

# Add bias term 1 for each example
X = cat(X, ones(m)'; dims=(1))
n, m = size(X)

# bad: this probably leaves out those rare transition points!
#train = bitrand(rng, m)
#test = .!train
train = ones(m) .== 1
test = train

λ = 0.01
max_passes = 1000

@show model = svm(X[:,train], Y[train], max_passes = max_passes, λ = λ)

@show train_cost = cost(model, X[:,train], Y[train])

@show train_accuracy = accuracy(model, X[:,train], Y[train])
@show test_accuracy = accuracy(model, X[:,test], Y[test])

# more info
# show an example of a training example...
@show u_condition
@show X[:,train][:, 1]
@show size(X)

@show model.w
@show model.λ

# show size of filtered data
@show size(data)

#end # RoomTemperaturerLearn
