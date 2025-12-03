module Ishtar

using Dates, LinearAlgebra, CSV

mutable struct Environment
    times::Vector{DateTime}        # length T
    prices::Matrix{Float64}        # size (T, n)
    features::Matrix{Float64}      # size (T, d)
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


mutable struct ReplayBuffer{S,A}
    states::Vector{S}
    actions::Vector{A}
    rewards::Vector{Float64}
    next_states::Vector{S}
    dones::Vector{Bool}
end

# Additional functions and constructors
function Environment(times::Vector{DateTime},
                     prices::Matrix{Float64},
                     features::Matrix{Float64};
                     initial_cash::Float64 = 0.0)
    T, n = size(prices)
    @assert length(times) == T
    @assert size(features, 1) == T
    positions = zeros(Float64, n)
    Environment(times, prices, features, 1, positions, initial_cash)
end

function current_state(env::Environment)::State
    t = env.index
    State(env.times[t], env.prices[t, :], env.features[t, :], copy(env.positions), env.cash)
end

function step!(env::Environment, action::Action)
    T = length(env.times)
    if env.index >= T
        return current_state(env), 0.0, true
    end

    t = env.index

    prices = env.prices[t, :]
    value = env.cash + env.positions ⋅ prices

    target = action.target_positions

    delta_q = target .- env.positions
    trade_cost = delta_q ⋅ prices

    env.cash -= trade_cost
    env.positions .= target

    env.index += 1

    nextprices = env.prices[env.index, :]
    nextvalue = env.cash + env.positions ⋅ nextprices

    reward = log(nextvalue / value)
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
