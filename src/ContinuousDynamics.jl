using Distributions, Random, LinearAlgebra

ẋ(t; x, u, A_d, ϵ) = A_d * [x; u] + ϵ

x̂(t; t_0, x_0, u_0, A_d, ϵ) = x_0 + ẋ(t_0; x=x_0, u=u_0, A_d=A_d, ϵ=ϵ) * (t - t_0)

function simulate(Δt, num_t)
    σ = 1
    rng = MersenneTwister(123)

    p_ϵ = Normal(0, σ^2)

    c = 0.25
    u = 1

    x_ini = 10
    t_ini = 1

    t_end = t_ini + (num_t - 1)

    w = [-0.5; c]
    A_d = Diagonal(w)

    m = num_t
    n = 1

    X = Matrix{Float32}(undef, m, n)

    X[1, :] = x_ini
    for (i, t) in enumerate(t_ini : Δt : t_end - Δt)
        X[i + 1, :] = x̂(t + Δt; t_0=t, x_0=X[i, :], A_d=A_d, ϵ=rand(rng, p_ϵ, n))
    end

    return X
end

function estimate_parameter(X, Δt)
    # (x_{t+1} - x_t) / Δt ≈ A_d * x_t + ϵ
    # x_{t+1} ≈ (A_d .* Δt + I) * x_t + Δt .* ϵ
    # x_{t+1} = Ax_t + Bϵ where A = A_d .* Δt + I and B = Δt
    m, n = size(X)
    Σ_0, Σ_1 = zeros(n, n), zeros(n, n)
    for i in 1:m-1
        Σ_0 += X[i, :] * X[i, :]'
        Σ_1 += X[i, :] * X[i+1, :]'
    end

    @show Σ_0, Σ_1

    @show Â = Σ_1 / Σ_0

    @show Â_d = (Â - I) ./ Δt
    B = Δt

    return (Â_d, B)
end

