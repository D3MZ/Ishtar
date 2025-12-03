module Ishtar

using Dates, LinearAlgebra, CSV

include("backtest.jl")
include("strategy/MACrossover.jl")

export Environment, State, Action, Transition, step!, current_state
export AbstractStrategy, act
export MACrossoverRaw, MACrossoverFeatures

end
