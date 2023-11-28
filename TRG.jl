using LinearAlgebra
using Plots
using QuadGK
using ITensors

function get_TRG_initial_tensor(beta, J, h)

    M = zeros(2, 2)
    M[1, 1] = exp(-beta*(-J*(1)*(1) + h*(1)))
    M[2, 1] = exp(-beta*(-J*(-1)*(1) + h*(-1)))
    M[1, 2] = exp(-beta*(-J*(1)*(-1) + h*(1)))
    M[2, 2] = exp(-beta*(-J*(-1)*(-1) + h*(-1)))

    evals, U = eigen(M)
    U = U*sqrt(diagm(evals))

    sigma, k = size(U)

    A = zeros(k, k, k, k)

    for i in 1:sigma
    
        tmp_vec = U[i, :]
        
        tmp_A = kron(kron(kron(tmp_vec, tmp_vec), tmp_vec), tmp_vec)

        tmp_A = reshape(tmp_A, (k, k, k, k))

        A += tmp_A

    end

    return ITensor(A, siteinds(k, length(size(A))))

end


function coarse_grain(A, nsv, norms)

    """

    The index convention for tensor A is:

         1
         |
    4 -- A --- 2 
         |
         3

    where 1 = t, 2 = l, 3 = b, 4 = r

    Inputs:

    A = the 4-tensor to be coarse grained

    nsv = max_dim = the maximum bond dimension allowed on the A

    norms = the list tracking the norms we accumulate during the TRG, we normalize A for stable numerics by the highest value in A i.e. divide all its elements by that value
    
    Outputs:

    A_final = the coarse grained 4-tensor

    """

    # Normalize by the highest value in the A_final tensor
    norm = argmax(abs2, A)
    append!(norms, norm)
    A = A/norm

    # Prepare two tensors into matrices such that svd can be performed on them
    A_1 = permutedims(A, (1, 4, 3, 2)) # t, l, b, r (this comment and subsequent ones keep track of the indices that the final tensor ends up with i.e. in this case A_1)
    t, l, b, r = size(A_1) # t is the dimension of the index t and similarly for l, b, r
    A_1 = reshape(A_1, (t*l, b*r)) # (t, l), (b, r)
    A_2 = reshape(A, (t*r, b*l)) # (t, r), (b, l)

    # Perform the svd
    svd_1 = svd(A_1)
    svd_2 = svd(A_2)

    # These tensors from svd_1 will produce tensors F_1, F_3
    diag_S_1 = svd_1.S # s1
    s_1 = min(nsv, length(diag_S_1))
    U_1 = svd_1.U[:, 1:s_1] # (t, l), s1 
    sqrt_S_1 = Diagonal(sqrt.(diag_S_1[1:s_1])) # s1, s1
    Vt_1 = svd_1.Vt[1:s_1, :] # s1, (b, r)

    # These tensors from svd_1 will produce tensors F_2, F_4
    diag_S_2 = svd_2.S # s2
    s_2 = min(nsv, length(diag_S_2))
    U_2 = svd_2.U[:, 1:s_2] # (t, r), s2
    sqrt_S_2 = Diagonal(sqrt.(diag_S_2[1:s_2])) # s2, s2
    Vt_2 = svd_2.Vt[1:s_2, :] # s2, (b, l)

    # Multiply the square root of the singular value matrix on U and V respectively to get F_1, F_2, F_3, F_4
    F_1 = U_1*sqrt_S_1 # (t, l), s1
    F_3 = sqrt_S_1*Vt_1 # s1, (b, r)
    F_2 = U_2*sqrt_S_2 # (t, r), s2
    F_4 = sqrt_S_2*Vt_2 # s2, (b, l)

    # Reshape F_1, F_2, F_3, F_4, to open up the original indices merged together to perform svd
    F_1 = reshape(F_1, (t, l, s_1)) # t, l, s1
    F_3 = reshape(F_3, (s_1, b, r)) # s1', b, r
    F_2 = reshape(F_2, (t, r, s_2)) # t, r, s2
    F_4 = reshape(F_4, (s_2, b, l)) # s2', b, l

    # Contract the F_1, F_2, F_3, F_4 to obtain the new coarse grained A tensor
    tmp_1 = contraction(F_1, (2,), F_4, (3,)) # t, s1, s2', b 
    tmp_2 = contraction(F_2, (2,), F_3, (3,)) # t, s2, s1', b
    tmp_3 = contraction(tmp_1, (1, 4), tmp_2, (1, 4)) # s1, s2', s2, s1' : this is the biggest cost of the algorithm's subroutine (which is performed log(N_tensors) times) and has cost order(bond dimension ^ 6) - see page 26 of https://libstore.ugent.be/fulltxt/RUG01/002/836/279/RUG01-002836279_2020_0001_AC.pdf

    # Permute the indices into the order of the convention
    A_final = permutedims(tmp_3, (2, 1, 3, 4)) # s2', s1, s2, s1' (here we are effectively tilting the lattice back into its original orientation but the tilting rotation direction does not matter i.e. if its anti-clockwise or clockwise)

    return A_final

end

function TRG(A, N, max_dim; verbose = false)

    """

    The TRG wants to calculate the partition function of the 2D Ising model in this particular script example by contracting a 2 dimension network of tensors.
    Given the fact that all tensors in the network are the same, we can just play with 1 tensor and any action we do on it will be the same for all others so 
    we don't have to actually do it for all others and only keep in memory a single A tensor. For the description of the algorithm on which this script example
    is based on see https://tensornetwork.org/trg/ which describes plain TRG. 

    Inputs:

    A = initial 4-tensor to be input in TRG, it can be non-normalized because the coarse grain function normalizes before anything else

    N = the side length of the square spin lattice

    max_dim = the maximum bond dimension allowed in the TRG

    verbose = boolean to check whether we want the dimensions of the tensor we are doing TRG on to be printed out

    Ouputs:

    Z, norms = partition function given by the trace of the last A/A_norm tensor we end up with and
               a list of floats used to normalize the A tensor in the coarse graining iterations to keep stable numerics, 
               the length of this vector is number_of_TRG_iterations-1 where the -1 comes from the last norm being in the Z accounted for

    """

    N_spins = N*N # Total number of spins
    N_tensors = N_spins # Each tensor has 2 unique legs and since spins live on the legs of the tensors for every tensor we have 2 spins
    norms = [] # Initialize the norm list and put in it the norm of the initial tensor
    for i in 1:Int(log2(N_tensors)) # This amount of iterations will effectively allow us to end up with 1 single A tensor which we finally trace its legs over
        if verbose 
            println("Iteration number: ", i, ", Size of A tensor: ", size(A))
        end
        A = coarse_grain(A, max_dim, norms) 
    end

    t, l = size(A)[1:2]
    tmp_1 = contraction(A, (1, 3), I(t), (1, 2)) # l, r (this comment keeps track of the result indices i.e. the indices of tmp_1)
    Z = abs(contraction(tmp_1, (1, 2), I(l), (1, 2))[1]) # contract the legs of the final A/A_norm tensor where the A_norm is attached in the coarse grain function

    return Z, norms

end

function get_free_energy_density(Z, beta, norms, N_total)

    """

    Inputs:

    Z = The value of the partition function which is given by tr(A_last/A_last_norm), float
    
    beta = 1/T (inverse temperature, float)

    norms = a list of floats used to normalize the A tensor in the coarse graining iterations to keep stable numerics, 
            the length of this vector is number_of_TRG_iterations-1 where the -1 comes from the last norm being in the Z accounted for

    N_total = total number of spins which is equal to N*N where N is the side length of the square lattice

    Output:

    The free energy density f based on the numerical results of TRG where f is defined overall as f = -(T/Vol)*ln(Z) = -(1/beta*Vol)*ln(Z) where Vol = volume = number of spins = N*N = N_total = 2*N_tensors

    """

    N_tensors = N_total 
    f = 0
    for i in 1:Int(log2(N_tensors))
        f += -((1)/(beta*(2^(i-1))))*log(abs(norms[i])) # As the TRG is being carried out in coarse graining iterations, we normalize the A tensor for stable numerics and these norm factors end up in the formula for f
    end
    f += -((1)/(N_tensors*beta))*log(Z) # This takes care of the last tensor we trace out into Z

    return f

end

function get_free_energy_density_exact(beta)

    """

    Sources for the exact Onsager's result for the 2D Ising model: 

    https://gandhiviswanathan.wordpress.com/2015/01/09/onsagers-solution-of-the-2-d-ising-model-the-combinatorial-method/ or https://itensor.org/docs.cgi?page=book/trg

    Inputs:

    beta = 1/T (inverse temperature, float)

    Output:

    Onsager's analytical solution to the 2D Ising model at a given beta

    """
    
    inner1(theta1, theta2) = log(cosh(2 * beta)^2 - sinh(2 * beta) * cos(theta1) - sinh(2 * beta) * cos(theta2))

    inner2(theta2) = quadgk(theta1 -> inner1(theta1, theta2), 0, 2 * π)[1]

    I = quadgk(inner2, 0, 2*π)[1]
    
    return -(log(2) + I / (8 * pi^2)) / beta

end

# Define the length of the side of the square spin lattice
pow = 8
N = 2^(pow)

# Define the maximum bond dimension allowed for the A tensor to be coarse grained
max_dim_list = [8]

# Define the temperature = 1/beta, interaction J and applied magnetic field h
temp_list = LinRange(2, 3, 2)
J = 1.0
h = 0.0

# Some plotting stuff
available_markers = [:+, :x, :diamond, :hexagon, :square, :circle, :star4]
plot_1 = plot()
plot_2 = plot()

# Do TRG for different bond dimensions and temperatures and plot results
for (max_dim_idx, max_dim) in enumerate(max_dim_list)
    
    # These local lists keep trace of the free energy density f and its exact solution
    local f_list = []
    local f_exact_list = []

    println("-----------------------------------------")
    for temp in temp_list
        
        println("Starting temperature: ", temp)
        
        beta = 1/temp
        A_initial = get_TRG_initial_tensor(beta, J, h) # Remeber this A_initial can be unnormalized since coarse grain will normalize it
        Z, norms = TRG(A_initial, N, max_dim, verbose = false) # Perform the actual algorithm to get the partition function Z and the norms used for stable numerics
        f = get_free_energy_density(Z, beta, norms, N*N)
        f_exact = get_free_energy_density_exact(beta)
        append!(f_list, f)
        append!(f_exact_list, f_exact)
        
        println("Temperature: ", temp, ", Partition function: ", Z, ", Free energy density: ", f, ", Exact free energy density: ", f_exact)
        println("-----------------------------------------")
    end
    

    scatter!(plot_1, temp_list, f_list, marker = rand(available_markers), label = "Max bond dim = $(max_dim)")
   
    delta_f_list = []
    for i in 1:length(temp_list)
        append!(delta_f_list, abs(f_list[i] - f_exact_list[i])/abs(f_exact_list[i])) # Fractional difference between f and f_exact for all temperatures at fixed maximum bond dimension simulation
    end
    display(delta_f_list)
    scatter!(plot_2, temp_list, delta_f_list, marker = rand(available_markers), label = "Max bond dim = $(max_dim)")

end

# # Plot free energy density f vs temperature for different maximum bond dimensions
# title!(plot_1, "Lattice size $(N)x$(N)")
# ylabel!(plot_1, "Free Energy Density")
# xlabel!(plot_1, "Temperature")
# plot!(plot_1, legendrows = length(max_dim_list), size=(900, 600))
# savefig(plot_1, "f_vs_t.pdf")

# # Plot the fractional error of f with respect to f_exact for different maximum bond dimensions
# ylabel!(plot_2, "Fractional f Error")
# xlabel!(plot_2, "Temperature T")
# title!(plot_2, "Lattice size $(N)x$(N)")
# plot!(plot_2, legendrows = length(max_dim_list), size=(900, 600))
# savefig(plot_2, "fractional_error.pdf")
