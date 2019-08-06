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
    @printf(io, " *Non-zero weights: %d\n", count(w_i -> w_i ≠ 0, fit.w))
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
    (w, t_converge) = pegasos(X, Y, λ = λ, T = max_passes)
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
    return risk
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

end # module SVM
