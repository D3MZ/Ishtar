using Test, Dates, CSV, Statistics, Ishtar

@testset "MA crossover single instrument" begin
    data_path = joinpath(@__DIR__, "..", "data", "EURGBP_1min.csv")
    file = CSV.File(data_path)

    times = DateTime[]
    prices = Float64[]

    for row in file
        push!(times, DateTime(row.timestamp))
        push!(prices, Float64(row.close))
    end

    prices_matrix = reshape(prices, :, 1)
    features = zeros(Float64, length(times), 0)

    strategy = MACrossoverRaw{Float64}(5, 20, 1, 1.0)
    env = Environment(times, prices_matrix, features; initial_cash = 100.0)

    rewards = run_strategy(env, strategy)

    @test length(rewards) == length(times) - 1
    @test env.index == length(times)
    @test all(isfinite, rewards)

    env_check = Environment(times, prices_matrix, features; initial_cash = 100.0)
    env_check.index = strategy.long + 5

    action = act(strategy, env_check)
    prices_window = @view prices[1:env_check.index]

    short_ma = mean(@view prices_window[end - strategy.short + 1:end])
    long_ma = mean(@view prices_window[end - strategy.long + 1:end])

    expected = short_ma > long_ma ? strategy.position_size :
               short_ma < long_ma ? zero(strategy.position_size) :
               env_check.positions[1]

    @test length(action.target_positions) == 1
    @test action.target_positions[1] == expected
end
