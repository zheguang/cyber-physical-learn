module SVM

using StatsBase
import StatsBase: predict # Import to override

using Printf, LinearAlgebra, Random, Distributions

export svm, predict, cost, accuracy, pegasos

struct SVMFit
    w::Vector{Float64}
    λ::Float32
    passes::Int32
    converged::Bool
end


function Base.show(io::IO, fit::SVMFit)
    @printf(io, "Fitted linear SVM\n")
    @printf(io, " * Non-zero weights: %d\n", count(w_i -> w_i ≠ 0, fit.w))
    @printf(io, " * Iterations: %d\n", fit.passes)
    @printf(io, " * Converged: %s\n", fit.converged)
end


function predict(fit::SVMFit, X::AbstractMatrix{<:Real})
    n, m = size(X)
    preds = fill(0, m)
    for i in 1:m
        dot_prod = 0.0
        for j in 1:n
            dot_prod += fit.w[j] * X[j, i]
        end
        preds[i] = sign(dot_prod)
    end
    return preds
end


function svm(X::AbstractMatrix{<:Real},
             Y::AbstractVector{<:Real};
             λ::Real = 0.1,
             ϵ::Real = 1e-6,
             max_passes::Integer = 100)
    (w, t_converge) = pegasos_batch(X, Y, lambda = λ, maxpasses = max_passes)
    if t_converge < Inf
        SVMFit(w, Float32(λ), t_converge, true)
    else
        SVMFit(w, Float32(λ), max_passes, false)
    end
end


function cost(fit::SVMFit,
              X::AbstractMatrix{<:Real},
              Y::AbstractVector{<:Real})
    w = fit.w
    λ = fit.λ
    n, m = size(X)
    risk = λ / 2 * norm(w, 2)^2
    for i in 1:m
        p = 0.0
        l(w, x, y) = max(0, 1 - y * (w ⋅ x))
        risk += l(w, X[:, i], Y[i])
    end
    return risk / m
end


function accuracy(fit::SVMFit,
                  X::AbstractMatrix{<:Real},
                  Y::AbstractVector{<:Real})
    n, m = size(X)
    return count(predict(fit, X) .== Y) / m
end


function pegasos(X::AbstractMatrix{<:Real},
                 Y::AbstractVector{<:Real};
                 λ::Real = 0.1,
                 ϵ::Real = 1e-3,
                 T::Integer = 100,
                 seed::Integer = 123)
    rng = MersenneTwister(seed)

    n, m = size(X)

    # Initialize weights to 0
    w = zeros(n)
    I = DiscreteUniform(1, m)

    t_converge = Inf

    # Iterations
    for t in 1:T
        i = rand(rng, I)
        η = 1 / (λ * t)
        (x_i, y_i) = (X[:, i], Y[i])
        p = y_i * (w ⋅ x_i)
        ∇_w = nothing
        if p ≥ 1
           ∇_w = λ * w
        else # p < 1
           ∇_w = λ * w - y_i * x_i
        end
        w = w - η * ∇_w
        if t % 10 == 0
            @show norm(η * ∇_w, 2)
            @show cost(SVMFit(w, λ, T, t_converge < Inf), X, Y)
        end

#        if norm(η * ∇_w, 2) < ϵ
#            t_converge = t
#            break
#        end
    end

    return (w, t_converge)
end

# S is X,Y
# T is maxpasses
# p: # of features
# n: # of data points
# k: size of minibatch
function pegasos_batch(X::AbstractMatrix{<:Real},
                          Y::AbstractVector{<:Real};
                          k::Integer = 5,
                          lambda::Real = 0.1,
                          maxpasses::Integer = 100)
    # p features, n observations
    p, n = size(X)

    # Initialize weights so norm(w) <= 1 / sqrt(lambda)
    w = randn(p)
    sqrtlambda = sqrt(lambda)
    normalizer = sqrtlambda * norm(w)
    for j in 1:p
        w[j] /= normalizer
    end

    # Allocate storage for repeated used arrays
    deltaw = Array{Float64,1}(p)
    w_tmp = Array{Float64,1}(p)

    # Loop
    for t in 1:maxpasses
        # Calculate stepsize parameters
        alpha = 1.0 / t
        eta_t = 1.0 / (lambda * t)

        # Calculate scaled sum over misclassified examples
        # Subgradient over minibatch of size k
        fill!(deltaw, 0.0)
        for i in 1:k
            # Select a random item from X
            # This is one element of At of S
            index = rand(1:n)

            # Test if prediction isn't sufficiently good
            # If so, current item is element of At+
            pred = Y[index] * dot(w, X[:, index])
            if pred < 1.0
                # Update subgradient
                for j in 1:p
                    deltaw[j] += Y[index] * X[j, index]
                end
            end
        end

        # Rescale subgradient
        for j in 1:p
            deltaw[j] *= (eta_t / k)
        end

        # Calculate tentative weight-update
        for j in 1:p
            w_tmp[j] = (1.0 - alpha) * w[j] + deltaw[j]
        end

        # Find projection of weights into L2 ball
        proj = min(1.0, 1.0 / (sqrtlambda * norm(w_tmp)))
        for j in 1:p
            w[j] = proj * w_tmp[j]
        end
    end

    return (w, maxpasses)
end


end # module SVM
