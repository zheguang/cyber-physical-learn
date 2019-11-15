using JuMP, Ipopt, LinearAlgebra

# case 1: 1-bit logical control, 1 transition condition, 1 process (done)
# case 2: 2-bit logical control (ind), 1 transition condition (1 proc-dep), 2 processes (ind) (done)
# case 3: 2-bit logical control (ind), 1 transition codnition (1 proc-dep), 2 processes (rel)
# case 4: 2-bit logical control (rel), 1 transition condition (1 proc-dep), 2 processes (ind)
# case 5: 2-bit logical control (rel), 1 transition condition (1 proc-dep), 2 processes (rel)

# case 6: 1-bit logical control, 2 transition conditions, 1 process

# For transition conditions
# h_1: 1 transition condition for 1-bit control,  based on single process each
# h_1_mp: 1 transition condition for 1-bit control, based on multiple processes
# h_2: 2 transition condition for 2-bit control, based on single process each


function f(A, B, y, u)
    A * y + B * u
end


function example_f_case_3()
    # 2-bit control (ind), 2 processes (ind)
    A = Diagonal([0.9; 0.85])
    B = Diagonal([4; 5])
    y = [20; 20]
    u = [0; 0]
    A, B, y, u
end


function example_f_case_4()
    # 2-bit control (rel), 2 processes (ind)
    A = Diagonal([0.9; 0.85])
    B = [4  1;
         2  5]
    y = [20; 20]
    u = [0; 0]
    A, B, y, u
end


function example_f_case_5()
    # 2-bit control (rel), 2 processes (rel)
    A = [0.9  0.3;
         0.2  0.85]
    B = [4  1;
         2  5]
    y = [20; 20]
    u = [0; 0]
    A, B, y, u
end

function H_1(y, u, C, D)
    u_next = (u + ([(1 .- u)'; u']' * (C * [y'; y'] + D) .< 0)' ) .% 2
    return u_next
end

# For n_u bit control, there are 2n_u step functions, 2 for each bit.
# C is a 2n_u by n_y matrix with ±1 and 0. C * y means relate to each control bit the relevant processes with positive relation (<) or negative relation (>).
# y is the process
# D is a 2n_u vector
function H_n(y, u, C, D)
    s = rand(size(u)[1] * 2)
    for i in 1:size(u)[1]
        s[2*i-1] = 1 - u[i]
        s[2*i] = u[i]
    end

    step_funs = C * y + D .< 0
    u_next = (u + step_funs[s .== 1]) .% 2
    return u_next
end


function h_1(y, u)
    # 1-bit control transition, each with single process dependence
    u′ = zeros(size(u))

    # heater: 1st bit
    b = 1
    if u[b] == 0
        if y[b] < 20  # y[b] - 20 < 0
            u′[b] = 1
        else
            u′[b] = u[b]
        end
    else
        # u == 1
        if y[b] > 25 # 25 - y[b] < 0
            u′[b] = 0
        else
            u′[b] = u[b]
        end
    end
    u′
end


function h_2(y, u)
    # 2-bit control transition, each with singel process dependence
    u′ = zeros(size(u))

    # heater: 1st bit
    b = 1
    if u[b] == 0
        if y[b] < 20
            u′[b] = 1
        else
            u′[b] = u[b]
        end
    else
        # u == 1
        if y[b] > 25
            u′[b] = 0
        else
            u′[b] = u[b]
        end
    end

    # heater: 2nd bit
    b = 2
    if u[b] == 0
        if y[b] < 20
            u′[b] = 1
        else
            u′[b] = u[b]
        end
    else
        # u == 1
        if y[b] > 25
            u′[b] = 0
        else
            u′[b] = u[b]
        end
    end

    u′
end


function trajectory(T, A, B, y₀, u₀, h)
    (y, u) = (y₀, u₀)
    n_y = size(y)[1]
    n_u = size(u)[1]
    Y = zeros(n_y, T)
    U = zeros(n_y, T)
    for t = 1:T
        u = h(y, u)
        y = f(A, B, y, u)
        U[:, t] = u
        Y[:, t] = y
    end
    (Y, U)
end

function trajectory_with_H(T, A, B, C, D, y_0, u_0, H)
    h(y, u) = H(y, u, C, D)
    trajectory(T, A, B, y_0, u_0, h)
end



function ols(Y_out, Y_in, U_in)
    @assert size(Y_out)[1] == size(Y_in)[1]
    @assert size(Y_out)[2] == size(Y_in)[2] == size(U_in)[2]

    n_y, n_t = size(Y_in)
    n_u = size(U_in)[1]
    @show n_y, n_t, n_u

    model = Model(with_optimizer(Ipopt.Optimizer))

    @variable(model, Â[1:n_y, 1:n_y])
    @variable(model, B̂[1:n_y, 1:n_u])
    @objective(model, Min, sum((Y_out - (Â * Y_in + B̂ * U_in)) .^ 2))

    optimize!(model)
    @show value.(Â)
    @show value.(B̂)

    value.(Â), value.(B̂)
end


function format_ols(Y, U)
    Y_in = Y[:, 1:end-1]
    Y_out = Y[:, 2:end]
    U_in = U[:, 2:end]  # u(t) is first updated with y(t-1) and then used to calculate y(t)

    Y_out, Y_in, U_in
end


function example()
    A, B, y, u = example_f_case_3()
    Y, U = trajectory(100, A, B, y, u, h_2)
    Y_out, Y_in, U_in = format_ols(Y, U)
    ols(Y_out, Y_in, U_in)
end
