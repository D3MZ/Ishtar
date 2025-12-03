module Ishtar

using Dates, LinearAlgebra, CSV, Statistics

include("backtest.jl")
include("strategy/MACrossover.jl")

export Environment, State, Action, Transition, step!, current_state
export AbstractStrategy, act, run_strategy
export MACrossover

end
