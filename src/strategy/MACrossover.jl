struct MACrossover{P} <: AbstractStrategy
    short::Int
    long::Int
    asset::Int
    position_size::P
end

function act(strategy::MACrossover{P}, env::Environment{P,F})::Action{P} where {P,F}
    t = env.index
    if t < strategy.long
        return Action(copy(env.positions))
    end

    prices_col = @view env.prices[1:t, strategy.asset]

    short_ma = mean(@view prices_col[end - strategy.short + 1:end])
    long_ma  = mean(@view prices_col[end - strategy.long + 1:end])

    target = zeros(P, length(env.positions))

    if short_ma > long_ma
        target[strategy.asset] = strategy.position_size
    elseif short_ma < long_ma
        target[strategy.asset] = zero(P)
    else
        target[strategy.asset] = env.positions[strategy.asset]
    end

    return Action(target)
end