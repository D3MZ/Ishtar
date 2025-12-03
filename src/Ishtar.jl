module Ishtar

using Dates

mutable struct Environment
    times::Vector{DateTime}        # length T
    prices::Matrix{Float64}        # size (n, T)
    features::Matrix{Float64}      # size (d, T)
    index::Int                     # current time index, 1 ≤ index ≤ T
    positions::Vector{Float64}     # q_t, length n
    cash::Float64                  # c_t
end

struct State
    time::DateTime
    prices::Vector{Float64}      # p_t
    features::Vector{Float64}    # x_t
    positions::Vector{Float64}   # q_t
    cash::Float64                # c_t
end

struct Action
    target_positions::Vector{Float64}  # q_t^*
end

struct Transition{S,A}
    state::S
    action::A
    reward::Float64
    next_state::S
    done::Bool
end


Base.@kwdef mutable struct ReplayBuffer{S,A}
    states::Vector{S} = S[]
    actions::Vector{A} = A[]
    rewards::Vector{Float64} = Float64[]
    next_states::Vector{S} = S[]
    dones::Vector{Bool} = Bool[]
end

# Additional functions and constructors
function Environment(times::Vector{DateTime},
                     prices::Matrix{Float64},
                     features::Matrix{Float64};
                     initial_cash::Float64 = 0.0)
    n, T = size(prices)
    @assert length(times) == T "times length must match number of columns in prices"
    @assert size(features, 2) == T "features must have same number of columns as prices"
    positions = zeros(Float64, n)
    Environment(times, prices, features, 1, positions, initial_cash)
end

function current_state(env::Environment)::State
    t = env.index
    State(env.times[t], env.prices[:, t], env.features[:, t], copy(env.positions), env.cash)
end

function step!(env::Environment, action::Action)
    T = length(env.times)
    if env.index >= T
        return current_state(env), 0.0, true
    end

    t = env.index

    prices_t = env.prices[:, t]
    value_t = env.cash + dot(env.positions, prices_t)

    target = action.target_positions
    @assert length(target) == length(env.positions) "action.target_positions length must match positions length"

    delta_q = target .- env.positions
    trade_cost = dot(delta_q, prices_t)

    env.cash -= trade_cost
    env.positions .= target

    if env.index < T
        env.index += 1
    end

    prices_tp1 = env.prices[:, env.index]
    value_tp1 = env.cash + dot(env.positions, prices_tp1)

    reward = log(value_tp1 / value_t)
    next_state = current_state(env)
    done = env.index >= T

    return next_state, reward, done
end


function Base.push!(buffer::ReplayBuffer{S,A}, transition::Transition{S,A}) where {S,A}
    push!(buffer.states, transition.state)
    push!(buffer.actions, transition.action)
    push!(buffer.rewards, transition.reward)
    push!(buffer.next_states, transition.next_state)
    push!(buffer.dones, transition.done)
    return buffer
end


end
