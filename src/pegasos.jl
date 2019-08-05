using Random, Distributions, LinearAlgebra

# S is X, Y
# T is max_passes
# n: # of features
# m: # of data points
# k: size of minibatch

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
        end

        if norm(η * ∇_w, 2) < ϵ
            t_converge = t
            break
        end
    end

    return (w, t_converge)
end
