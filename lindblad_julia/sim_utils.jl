using OpenQuantumTools, OrdinaryDiffEq


function lindblad_sim(
    omegaZ::Float64,
    gammaAD::Float64,
    gammaZ::Float64,
    f::Float64,
    u0::Array,
    time::Int,
    length::Int,
)

    σp = [0.0 0.0; 1.0 0.0]
    σm = [0.0 1.0; 0.0 0.0]

    H = DenseHamiltonian([(s) -> omegaZ], [σz])

    lindRe = Lindblad(gammaAD * f, σp)
    lindEx = Lindblad(gammaAD * (1 - f), σm)
    lindZ = Lindblad(gammaZ, σz)

    annealing = Annealing(H, u0, interactions = InteractionSet(lindRe, lindEx, lindZ))

    sol = solve_lindblad(annealing, time, alg = Tsit5())

    t_axis = range(0, time, length = length)
    bloch_vector = []
    for t in t_axis
        push!(bloch_vector, 2 * real.(matrix_decompose(sol(t), [σx, σy, σz])))
    end

    off_diag = []
    for t in t_axis
        push!(off_diag, abs(sol(t)[1, 2]))
    end

    return t_axis, bloch_vector
end

function dm_to_bloch(dm::Array)
    return 2 * real.(matrix_decompose(dm, [σx, σy, σz]))
end


function plot_bloch(bloch_vector)
    plot(t_axis, [c[1] for c in bloch_vector], label="X", linewidth=2)
    plot!(t_axis, [c[2] for c in bloch_vector], label="Y", linewidth=2)
    plot!(t_axis, [c[3] for c in bloch_vector], label="Z", linewidth=2)
    xlabel!("t (ns)")
    ylabel!("Bloch Vector")
end


function chi2(sim_bloch::Array, exp_bloch::Array)

    if length(sim_bloch) != length(exp_bloch)
        error("Simulation and experiment have different length.")
    end

    total_diff = 0
    for i = 1:length(sim_bloch)
        curr_diff = 0

        for j = 1:3
            curr_diff += (exp_bloch[i][j] - sim_bloch[i][j])^2
        end

        total_diff += curr_diff
    end

    return total_diff
end


function optimize_params(
    omegaZ_ls::Array,
    gammaAD_ls::Array,
    gammaZ_ls::Array,
    f_ls::Array,
    u0::Array,
    time::Int,
    length::Int,
    exp_data::Array,
)

    sol0 =
        lindblad_sim(omegaZ_ls[1], gammaAD_ls[1], gammaZ_ls[1], f_ls[1], u0, time, length)[2]
    min_diff = chi2(exp_data, sol0)
    opt_params = [omegaZ_ls[1] gammaAD_ls[1] gammaZ_ls[1] f_ls[1]]

    for curr_omegaZ in omegaZ_ls
        for curr_gammaAD in gammaAD_ls
            for curr_gammaZ in gammaZ_ls
                for curr_f in f_ls

                    curr_sol = lindblad_sim(
                        curr_omegaZ,
                        curr_gammaAD,
                        curr_gammaZ,
                        curr_f,
                        u0,
                        time,
                        length,
                    )[2]
                    curr_diff = chi2(exp_data, curr_sol)

                    if curr_diff < min_diff
                        min_diff = curr_diff
                        opt_params = [curr_omegaZ curr_gammaAD curr_gammaZ curr_f]
                    end

                end
            end
        end
    end

    return opt_params
end


# sol1 = lindblad_sim(0.06, 0.01, 0.03, 0.1, 1 / 2 * [1.0 1.0; 1.0 1.0], 100, 200);
# t_axis = sol1[1];
# bloch_v1 = sol1[2];

# sol2 = lindblad_sim(0.01, 0.06, 0.02, 0.1, 1 / 2 * [1.0 1.0; 1.0 1.0], 100, 200);
# bloch_v2 = sol2[2];

# diff = chi2(bloch_v1, bloch_v2)
