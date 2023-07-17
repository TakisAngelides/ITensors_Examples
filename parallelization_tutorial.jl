# --------------------------------------------------------------------------------------------------------------------

# Simple Julia parallelization

# using Base.Threads

# # Print the number of threads to be used
# println("Number of threads: ", Threads.nthreads())

# # Simple example of parallelizing a for loop
# a = zeros(9)
# @threads for i in eachindex(a)
#     a[i] = Threads.threadid()
# end
# println(a)

# # Use atomic operations when the threads will access and modify a given variable to avoid race problems
# b = Ref(0)
# @threads for i in 1:1000
#     b[] += 1
# end
# println(b)

# c = Atomic{Int}(0);
# @threads for id in 1:1000
#     atomic_add!(c, 1)
# end
# println(c)

# --------------------------------------------------------------------------------------------------------------------

using ITensors
using ITensorGPU

# Set to identity to run on CPU
gpu = cu

N = 50
sites = siteinds("S=1", N)

ampo = AutoMPO()
for j in 1:(N - 1)
  ampo .+= 0.5, "S+", j, "S-", j + 1
  ampo .+= 0.5, "S-", j, "S+", j + 1
  ampo .+= "Sz", j, "Sz", j + 1
end
H = gpu(MPO(ampo, sites))

ψ₀ = gpu(randomMPS(sites))

sweeps = Sweeps(6)
maxdim!(sweeps, 10, 20, 40, 100)
mindim!(sweeps, 1, 10)
cutoff!(sweeps, 1e-11)
noise!(sweeps, 1e-10)
energy, ψ = @time dmrg(H, ψ₀, sweeps)
@show energy

# --------------------------------------------------------------------------------------------------------------------

