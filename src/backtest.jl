mutable struct Environment{P,F}
    times::Vector{DateTime}        # length T
    prices::Matrix{P}              # size (T, n)
    features::Matrix{F}            # size (T, d)
    index::Int                     # current time index, 1 ≤ index ≤ T
    positions::Vector{P}           # q_t, length n
    cash::P                        # c_t
end

struct State{P,F}
    time::DateTime
    prices::Vector{P}              # p_t
    features::Vector{F}            # x_t
    positions::Vector{P}           # q_t
    cash::P                       # c_t
end

struct Action{P}
    target_positions::Vector{P}   # q_t^*
end

struct Transition{S,A,R}
    state::S
    action::A
    reward::R
    next_state::S
    done::Bool
end

mutable struct ReplayBuffer{S,A,R}
    states::Vector{S}
    actions::Vector{A}
    rewards::Vector{R}
    next_states::Vector{S}
    dones::Vector{Bool}
end

# Additional functions and constructors
function Environment(times::Vector{DateTime},
                     prices::Matrix{P},
                     features::Matrix{F};
                     initial_cash::P = zero(P)) where {P,F}
    T, n = size(prices)
    @assert length(times) == T
    @assert size(features, 1) == T
    positions = zeros(P, n)
    Environment{P,F}(times, prices, features, 1, positions, initial_cash)
end

function current_state(env::Environment)::State
    t = env.index
    State{eltype(env.prices), eltype(env.features)}(
        env.times[t],
        env.prices[t, :],
        env.features[t, :],
        copy(env.positions),
        env.cash
    )
end

function step!(env::Environment, action::Action)
    T = length(env.times)
    if env.index >= T
        return current_state(env), zero(eltype(env.prices)), true
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

    reward = log(nextvalue / value)::eltype(env.prices)
    next_state = current_state(env)
    done = env.index >= T

    return next_state, reward, done
end

abstract type AbstractStrategy end

function act(strategy::AbstractStrategy, env::Environment)::Action
    state = current_state(env)
    return act(strategy, state)
end

function Base.push!(buffer::ReplayBuffer{S,A,R}, transition::Transition{S,A,R}) where {S,A,R}
    push!(buffer.states, transition.state)
    push!(buffer.actions, transition.action)
    push!(buffer.rewards, transition.reward)
    push!(buffer.next_states, transition.next_state)
    push!(buffer.dones, transition.done)
    return buffer
end

function run_strategy(env::Environment, strategy::AbstractStrategy)
    rewards = Float64[]
    while true
        action = act(strategy, env)
        _, reward, done = step!(env, action)
        push!(rewards, reward)
        done && break
    end
    return rewards
end