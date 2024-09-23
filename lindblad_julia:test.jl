using OpenQuantumTools, OrdinaryDiffEq, Plots
@time begin

tf = 10

omegaZ = 0.00001
gammaAD = 0.00002
gammaZ = 0.00003
f = 0.2

σp = [0.0 0.0; 1.0 0.0]
σm = [0.0 1.0; 0.0 0.0]

u0 = 1 / 2 * [1.0 1.0; 1.0 1.0]
# u0 = PauliVec[1][1]*PauliVec[1][1]'

H = DenseHamiltonian([(s)->omegaZ], [σz])

lindRe = Lindblad(gammaAD * f, σp)
lindEx = Lindblad(gammaAD * (1 - f), σm)
lindZ = Lindblad(gammaZ, σz)

annealing = Annealing(H, u0, interactions = InteractionSet(lindRe, lindEx, lindZ))

sol = solve_lindblad(annealing, 127000, alg=Tsit5())

t_axis = range(0, 127000, length=127)
bloch_vector = []
for t in t_axis
    # matrix_decompose projects a matrix onto a list of basis elements
    push!(bloch_vector, 2*real.(matrix_decompose(sol(t), [σx, σy, σz])))
end

off_diag = []
for t in t_axis
    push!(off_diag, abs(sol(t)[1,2]))
end

end

plot(t_axis, [c[1] for c in bloch_vector], label="X", linewidth=2)
plot!(t_axis, [c[2] for c in bloch_vector], label="Y", linewidth=2)
plot!(t_axis, [c[3] for c in bloch_vector], label="Z", linewidth=2)
xlabel!("t (ns)")
ylabel!("Bloch Vector")