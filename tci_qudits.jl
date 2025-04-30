using LinearAlgebra
using ITensors
using Plots
using Printf
import Random
Random.seed!(1235)

function contraction(A, c_A::Tuple, B, c_B::Tuple)::Array{ComplexF64}

    """
    The contraction function takes 2 tensors A, B and 2 tuples c_A, c_B and returns
    another tensor after contracting A and B

    A: first tensor
    c_A: indices of A to contract (Tuple of Int64)
    B: second tensor
    c_B: indices of B to contract (Tuple of Int64)

    Note 1: c_A and c_B should be the same length and the first index from c_A should
    have the same dimension as the first index of c_B, the second index from c_A
    should have the same dimension as the second index of c_B and so on.

    Note 2: It is assumed that the first index in c_A is to be contracted with the
    first index in c_B and so on.

    Note 3: If we were instead to use vectors for c_A and c_B, the memory allocation 
    sky rockets and the run time is 10 times slower. Vectors require more memory than
    tuples and run time since tuples are immutable and only store a certain type each time etc.

    Example: If A is a 4-tensor, B is a 3-tensor and I want to contract the first
    index of A with the second index of B and the fourth index of A with the first
    index of B, then the input to the contraction function should be:

    contraction(A, (1, 4), B, (2, 1))

    This will result in a 3-tensor since we have 3 open indices left after the
    contraction, namely second and third indices of A and third index of B

    Code Example:
    # @time begin
    # A = cat([1 2; 3 4], [5 6; 7 8], dims = 3)
    # B = cat([9 11; 11 12], [13 14; 15 16], dims = 3)
    # c_A = (1, 2)
    # c_B = (2, 1)
    # display(contraction(A, c_A, B, c_B))
    # end
    """

    # Get the dimensions of each index in tuple form for A and B

    A_indices_dimensions = size(A) # returns tuple(dimension of index 1 of A, ...)
    B_indices_dimensions = size(B)

    # Get the uncontracted indices of A and B named u_A and u_B. The setdiff
    # returns the elements which are in the first argument and which are not
    # in the second argument.

    u_A = setdiff(1:ndims(A), c_A)
    u_B = setdiff(1:ndims(B), c_B)

    # Check that c_A and c_B agree in length and in each of their entry they
    # have the same index dimension using the macro @assert. Below we also find
    # the dimensions of each index of the uncontracted indices as well as for the
    # contracted ones.

    dimensions_c_A = A_indices_dimensions[collect(c_A)]
    dimensions_u_A = A_indices_dimensions[collect(u_A)]
    dimensions_c_B = B_indices_dimensions[collect(c_B)]
    dimensions_u_B = B_indices_dimensions[collect(u_B)]

    @assert(dimensions_c_A == dimensions_c_B, "Note 1 in the function
    contraction docstring is not satisfied: indices of tensors to be contracted
    should have the same dimensions. Input received: indices of first tensor A
    to be contracted have dimensions $(dimensions_c_A) and indices of second
    tensor B to be contracted have dimensions $(dimensions_c_B).")

    # Permute the indices of A and B so that A has all the contracted indices
    # to the right and B has all the contracted indices to the left.

    # NOTE: The order in which we give the uncontracted indices (in this case
    # they are in increasing order) affects the result of the final tensor. The
    # final tensor will have indices starting from A's indices in increasing
    # ordera and then B's indices in increasing order. In addition c_A and c_B
    # are expected to be given in such a way so that the first index of c_A is
    # to be contracted with the first index of c_B and so on. This assumption is
    # crucial for below, since we need the aforementioned specific order for
    # c_A, c_B in order for the vectorisation below to work.

    A = permutedims(A, (u_A..., c_A...)) # Splat (...) unpacks a tuple in the argument of a function
    B = permutedims(B, (c_B..., u_B...))

    # Reshape tensors A and B so that for A the u_A are merged into 1 index and
    # the c_A are merged into a second index, making A essentially a matrix.
    # The same goes with B, so that A*B will be a vectorised implementation of
    # a contraction. Remember that c_A will form the columns of A and c_B will
    # form the rows of B and since in A*B we are looping over the columns of A
    # with the rows of B it is seen from this fact why the vectorisation works.

    # To see the index dimension of the merged u_A for example you have to think
    # how many different combinations I can have of the individual indices in
    # u_A. For example if u_A = (2, 4) this means the uncontracted indices of A
    # are its second and fourth index. Let us name them alpha and beta
    # respectively and assume that alpha ranges from 1 to 2 and beta from
    # 1 to 3. The possible combinations are 1,1 and 1,2 and 1,3 and 2,1 and 2,2
    # and 2,3 making 6 in total. In general the total dimension of u_A will be
    # the product of the dimensions of its indivual indices (in the above
    # example the individual indices are alpha and beta with dimensions 2 and
    # 3 respectively so the total dimension of the merged index for u_A will
    # be 2x3=6).

    A = reshape(A, (prod(dimensions_u_A), prod(dimensions_c_A)))
    B = reshape(B, (prod(dimensions_c_B), prod(dimensions_u_B)))

    # Perform the vectorised contraction of the indices

    C = A*B

    # Reshape the resulting tensor back to the individual indices in u_A and u_B
    # which we previously merged. This is the unmerging step.

    C = reshape(C, (dimensions_u_A..., dimensions_u_B...))

    return C

end

function lu_full_pivoting(A, max_pivots, tolerance, all_rows, all_cols)

    """
    Perform the LDU factorization of A, at the end we should have A = transpose(P)*L*D*U*transpose(Q), where 
    P and Q are the permutation matrices for the required row and column swaps during the Gaussian elimination 
    of the LU decomposition. The inverse of P and Q is their transpose; formally the decomposition yields
    P*A*Q = LU, and then we extra D from U.

    Inputs:
    A = matrix to be LDU factorized
    max_pivots = stopping condition on the factorization if max_pivots < rank of A
    tolerance = stopping condition on the factorization if the absolute value of a given pivot is below this tolerance
    all_rows, all_cols = specifies the row and column basis of A in multi-index notation

    Return: 
    P = permutation matrix for the row swaps during Gaussian elimination of the LU factorization
    L_truncated, D_truncated, U_truncated = the decompoisition matrices truncated according to max_pivots or tolerance
    Q = permutation matrix for the column swaps during Gaussian elimination of the LU factorization
    new_row_pivots = row multi-indices for the new pivots
    new_col_pivots = column multi-indices for the new pivots
    pivot_error = an approximation of the error of the LDU factorization
    """

    n, m = size(A) # row and column size of A

    L = Matrix{ComplexF64}(I, n, n) # We will mutate this to an lower triangular matrix via Gaussian elimination
    U = copy(A) # We will mutate this to an upper triangular matrix via Gaussian elimination
    P = Matrix{ComplexF64}(I, n, n) # Keeping track of the row swaps we need to do, hence this is the permutation matrix for the row swaps
    Q = Matrix{ComplexF64}(I, m, m) # Keeping track of the column swaps we need to do, hence this is the permutation matrix for the column swaps
    new_row_pivots = [] # We will store the multi-indices corresponding to the rows of the newly found pivots
    new_col_pivots = [] # We will store the multi-indices corresponding to the columns of the newly found pivots
    
    pivot_error = 0 # The estimated error of the LU factorization which we update in the while loop
    col = 1 # The column to perform Gaussian elimination for (updated in the while loop)
    pivots = 0 # Number of pivots found (updated in the while loop)
    max_rank = min(max_pivots, size(A, 1), size(A, 2)) # The Gaussian elimination should stop if we found the specified number of pivots or reached the actual maximum rank of A which is min(size(A, 1), size(A, 2))
    
    # While loop to perform Gaussian elimination for the U matrix which builds up the LU factorization of A
    while pivots < max_rank

        # Find the position of the pivot in the submatrix specified by U[col:end, col:end]
        largest_element_row, largest_element_col = Tuple(argmax(abs2.(U[col:end, col:end]))) # find position of largest element in column col     
        largest_element_row, largest_element_col = largest_element_row + col - 1, largest_element_col + col - 1

        # Stopping condition on the magnitude of the pivot compared to the input tolerance
        pivot_error = abs2(U[largest_element_row, largest_element_col]) # Estimate the error of the LU factorization as the last pivot we searched for
        if pivot_error <= tolerance # Note if this is not satisfied, we will use the pivot that we assigned to pivot_error (this is why pivot_error is an estimate, because the actual error would not be what we account for)
            break
        end
        
        # Insert the multi-index of the row of the pivot found to the new row pivots
        push!(new_row_pivots, all_rows[largest_element_row])

        # Peform a row swap on U, P and L matrices if the new pivot is not on row = col, here we also update the all_rows list which has an order of elements specifying the row basis of U
        if largest_element_row != col # case of row swap needed
            all_rows[largest_element_row], all_rows[col] = all_rows[col], all_rows[largest_element_row] 
            U[largest_element_row, :], U[col, :] = U[col, :], U[largest_element_row, :] # swaps rows largest_element_row with col in U
            P[largest_element_row, :], P[col, :] = P[col, :], P[largest_element_row, :] # swaps rows largest_element_row with col in P
            if col > 1
                L[largest_element_row, 1:col-1], L[col, 1:col-1] = L[col, 1:col-1], L[largest_element_row, 1:col-1] # swaps rows (only under diagonal) largest_element_row with col in L
            end
        end

        # Insert the multi-index of the row of the pivot found to the new row pivots
        push!(new_col_pivots, all_cols[largest_element_col])

        # Peform a row swap on the U and Q matrices if the new pivot is not on column = col, here we also update the all_cols list which has an order of elements specifying the column basis of U
        if largest_element_col != col # case of col swap needed
            all_cols[largest_element_col], all_cols[col] = all_cols[col], all_cols[largest_element_col]
            U[:, largest_element_col], U[:, col] = U[:, col], U[:, largest_element_col] # swaps cols largest_element_col with col in U
            Q[:, largest_element_col], Q[:, col] = Q[:, col], Q[:, largest_element_col] # swaps cols largest_element_col with col in Q
        end

        # Perform the Gaussian elimination step that transitions the L and U matrices towards lower and upper triangular
        multiplicative_factors = U[col+1:end, col] ./ U[col, col] # compute the vector holding the multiplicative factors for each row below row = col
        L[col+1:end, col] = multiplicative_factors # update L
        U[col+1:end, :] = U[col+1:end, :] .- multiplicative_factors * transpose(U[col, :]) # update U by subtracting from all rows the multiplicative factors time row = col

        # Increment the number of pivots found
        pivots += 1

        # In the next while loop iteration we perform Gaussian elimination on the next column
        col += 1

    end

    # Truncate the L and U matrices to the given bond dimension
    L_truncated = L[:, 1:col-1] # here it is col-1 and not col because at the end of every loop of the while loop above we increase col by 1
    U_truncated = U[1:col-1, :]

    # In the case of no truncation there is no error in the LU factorization
    if pivots == min(n, m)
        pivot_error = 0
    end

    # Extract the D matrix from the U matrix and change the U matrix accordingly such that we have LU -> LDU
    D_truncated = Diagonal(vcat(diag(U_truncated), ones(max(0, size(U_truncated, 1) - length(diag(U_truncated))))))
    Dinv = Diagonal(vcat(diag(U_truncated).^-1, ones(max(0, size(U_truncated, 1) - length(diag(U_truncated))))))
    U_truncated = Dinv * U_truncated

    return P, L_truncated, D_truncated, U_truncated, Q, new_row_pivots, new_col_pivots, pivot_error

end

function fixed_dinary_to_decimal(dinary::Vector{Int}, d::Int)
    if d < 2
        error("Base d must be at least 2.")
    end
    
    decimal = 0
    power = length(dinary) - 1
    dinary = dinary .- 1
    
    for digit in dinary
        if digit < 0 || digit >= d
            error("Digits must be in the range 0 to $(d-1).")
        end
        decimal += digit * d^power
        power -= 1
    end

    return decimal
end

function func(dinary, min_grid, max_grid, d)

    """
    Convert a dinary vector to a grid value and then evaluate a given function with that input value

    Inputs:
    dinary = dinary vector of 1s and 2s e.g. [1, 2, 2, 1]
    min_grid, max_grid = the grid domain over which values are mapped

    Return:
    x, y = grid value, function value at that grid value input
    """

    N = length(dinary)

    delta = (max_grid - min_grid)/(d^N - 1)

    x = min_grid + fixed_dinary_to_decimal(dinary, d) * delta

    return x, (sin(5x) + 0.5cos(3x^2)) * exp(-0.1x^2) + atan(x)/5 + 0.1 * x * cos(x)
    
    # return x, sin(x)^2 + exp(-x^2) * cos(3x) + log(1 + x^2)

    # return x, tanh(sin(x) + cosh(0.5x)) + atan(x^3) - exp(-abs(x))

    # return x, log1p(sin(x)^2 + x^2) + exp(-x^2/5) * cos(5x)

    # return x, cos(x^2) * exp(-abs(x)) + sinh(x)/cosh(x + 1)

    # return x, sin(x^3) / sqrt(x^2 + 1) + log(1 + x^4)

    # return x, x

    # return x, sin(x)

    # return x, sin(1/x)

    # return x, 1

    # return x, 10*sin(2π*x/365 - π/2) + 15

end

function get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, min_grid, max_grid, max_pivots, tolerance, func, func_cache, d)

    """
    Constructs the matrix Pi for a given level `l`, performs LDU decomposition, and retrieves updated pivot info.

    Inputs:
    l = level in the tensor contraction algorithm
    row_pivots, col_pivots = current list of multi-index row/col pivots
    min_grid, max_grid = domain boundaries for function evaluation
    max_pivots = truncation parameter
    tolerance = stopping tolerance for pivot selection
    func = function used to evaluate the matrix entries
    func_cache = dictionary to cache results of func

    Return:
    P, L, L11, L21, D, U, U11, U12, Q = decomposed components
    new_row_pivots, new_col_pivots = updated row and column pivots
    pivot_error = estimated decomposition error
    func_cache = updated cache dictionary
    """

    # We select the row and column pivots corresponding to optimizing sites l and l+1
    row_pivots_l = row_pivots[l]
    col_pivots_l = col_pivots[l+1]

    # We build the Pi tensor to be LDU decomposed, the important thing to ensure is that the all_rows, all_cols lists have the same ordering of elements as the basis of Pi when reshaped into a matrix
    all_rows = []
    all_cols = []
    Pi = zeros(ComplexF64, length(row_pivots_l), d, d, length(col_pivots_l))
    flag = true
    for row_s_idx in 1:d # The order of the for loops for the rows here is important to match the basis order stored in all_rows, all_cols with the reshape of the Pi tensor done below
        for (row_idx, row) in enumerate(row_pivots_l)
            push!(all_rows, vcat(row, row_s_idx))
            for (col_idx, col) in enumerate(col_pivots_l) # The order of the for loops for the columns here is important to match the basis order stored in all_rows, all_cols with the reshape of the Pi tensor done below
                for col_s_idx in 1:d
                    if flag # Flag is set to true only for the first round of looping over the columns, once the second row starts we set this to false as we have inserted all columns in the all_cols list
                        push!(all_cols, vcat(col_s_idx, col))
                    end
                    dinary = vcat(row, row_s_idx, col_s_idx, col)
                    if haskey(func_cache, dinary) # If the function has already been evaluated at a given dinary we get the value from a dictionary cache
                        x, y = func_cache[dinary]
                    else
                        x, y = func(dinary, min_grid, max_grid, d)
                        func_cache[dinary] = (x, y)
                    end
                    Pi[row_idx, row_s_idx, col_s_idx, col_idx] = y # Build the Pi tensor to be LDU decomposed
                end
            end
            flag = false
        end
    end
    # Reshape uses column major order so the left most index changes faster hence the ordering of the for loops above for the rows and columns 
    Pi = reshape(Pi, length(row_pivots_l)*d, d*length(col_pivots_l))
    
    P, L, D, U, Q, new_row_pivots, new_col_pivots, pivot_error = lu_full_pivoting(Pi, max_pivots, tolerance, all_rows, all_cols)

    # These matrices are needed for the assignment of tensors into the MPS in CI canonical form
    L11, L21 = UnitLowerTriangular(L[1:size(L, 2), :]), L[size(L, 2)+1:end, :] # L11 = chi x chi, L21 = n - chi x chi
    U11, U12 = UnitUpperTriangular(U[:, 1:size(U, 1)]), U[:, size(U, 1)+1:end] # U11 = chi x chi, U12 = chi x m - chi

    return P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache

end

function tci(N, func, min_grid, max_grid, tolerance, max_pivots, sweeps, d)

    """
    Performs the Tensor Cross Interpolation (TCI) algorithm to approximate a high-dimensional tensor 
    using matrix product states (MPS).

    Inputs:
    N = number of dimensions (sites)
    func = function to sample the tensor
    min_grid, max_grid = bounds of the domain
    tolerance = stopping criterion for pivot values
    max_pivots = maximum number of pivots allowed per decomposition
    sweeps = number of left-right sweeps for refinement

    Return:
    mps_list = list of MPS tensors approximating the original tensor
    pivot_error_list = per-site pivot errors during final sweep
    func_cache = dictionary of evaluated tensor entries
    """

    # Each link will be associated with an error coming from LDU decomposing the corresponding Pi tensor
    pivot_error_list = zeros(N-1)

    # We will store the MPS tensors in this list and its bond dimensions
    mps_list = []
    for i in 1:N
        if i == 1
            push!(mps_list, zeros(ComplexF64, d, 1))
        elseif i == N
            push!(mps_list, zeros(ComplexF64, 1, d))
        else
            push!(mps_list, zeros(ComplexF64, 1, d, 1))
        end
    end
    bonddims = [1 for _ in 1:N-1]
    bonddims_previous = copy(bonddims)

    # Initialize the pivots with a random dinary
    initial_dinary = [rand(1:d) for _ in 1:N]
    row_pivots = [[initial_dinary[1:l]] for l in 0:N-1]
    col_pivots = [[initial_dinary[l+1:N]] for l in 1:N]

    # Initialize a dictionary to hold the function evaluations
    func_cache = Dict{Vector{Int}, Tuple{Any, ComplexF64}}()
    
    for sweep in 1:sweeps
        
        for dir in (true, false)  # true = forward, false = backward
        
            if dir
                range_l = 1:N-1
            else
                range_l = N-1:-1:1
            end

            for l in range_l

                # Perform the LU decomposition on site l, l+1 and get the new tensors and pivots
                P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache = get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, min_grid, max_grid, max_pivots, tolerance, func, func_cache, d)

                if dir
                    left_tensor = transpose(P) * vcat(Matrix(I, size(L, 2), size(L, 2)), L21 * inv(L11)) # T_l * P^-1
                    right_tensor = L11 * D * U * transpose(Q) # T_{l+1}
                else
                    left_tensor = transpose(P) * L * D * U11 # T_l
                    right_tensor = hcat(Matrix(I, size(U, 1), size(U, 1)), inv(U11) * U12) * transpose(Q) # P^-1 * T_{l+1}
                end

                # Reshape tensors into their MPS form
                if l == 1
                    left_tensor = reshape(left_tensor, d, size(left_tensor, 2)) # Open boundary conditions have no left index for the first MPS tensor
                else
                    left_tensor = reshape(left_tensor, div(size(left_tensor, 1), d), d, size(left_tensor, 2)) # Indexing is: left bond index, physical index, right bond index
                end
                if l == N-1
                    right_tensor = reshape(right_tensor, size(right_tensor, 1), d) # Open boundary conditions have no right index for the last MPS tensor
                else
                    right_tensor = reshape(right_tensor, size(right_tensor, 1), d, div(size(right_tensor, 2), d))
                end

                # Update the MPS tensors
                mps_list[l] = left_tensor
                mps_list[l + 1] = right_tensor

                # Update the pivots
                row_pivots[l + 1] = new_row_pivots
                col_pivots[l] = new_col_pivots

                # Update the error
                pivot_error_list[l] = pivot_error

                # Update bond dimensions
                bonddims[l] = size(right_tensor, 1)
    
            end
        end

        # Stopping condition if the bond dimensions remain the same after a full sweep
        if bonddims == bonddims_previous
            println("TCI stopped by bond dimension convergence on sweep $(sweep) with bond dimensions $(bonddims).")
            return mps_list, pivot_error_list, func_cache, sweep, bonddims
        else
            bonddims_previous = bonddims
        end

    end

    println("TCI stopped by maximum number of sweeps")
    return mps_list, pivot_error_list, func_cache, sweep, bonddims

end

let

# Define parameters and perform TCI
N = 14
d = 2
min_grid = -10
max_grid = 10
tolerance = 1e-16
max_pivots = 256
sweeps = 200
mps_list, pivot_error_list, func_cache, sweep, bonddims = tci(N, func, min_grid, max_grid, tolerance, max_pivots, sweeps, d)

# Plot the results
x_val = [v[1] for v in values(func_cache)]
y_val = [real(v[2]) for v in values(func_cache)]
function decimal_to_fixed_dinary(n::Int, d::Int, N::Int)
    if d < 2
        error("Base d must be at least 2.")
    end
    if n < 0
        error("Decimal number must be non-negative.")
    end

    result = fill(0, N)
    i = N

    while n > 0 && i > 0
        result[i] = n % d
        n ÷= d
        i -= 1
    end

    if n > 0
        error("Number too large to fit in $N digits for base $d.")
    end

    return result .+ 1
end
contracted_mps = mps_list[1]
for i in 2:N
    contracted_mps = contraction(contracted_mps, (i,), mps_list[i], (1,))
end
mps_vec = []
for i in 0:d^N-1
    dinary_list = decimal_to_fixed_dinary(i, d, N) 
    push!(mps_vec, contracted_mps[dinary_list...])
end
p = plot(title = "N = $(N), Sweeps = $(sweep), Tolerance = $(tolerance), Max Pivots = $(max_pivots)\nBond dimensions: $(bonddims)")
delta = (max_grid-min_grid)/(d^N-1)
x_values = [min_grid + delta*i for i in 0:d^N-1]
y_values = [func(decimal_to_fixed_dinary(i, d, N), min_grid, max_grid, d)[2] for i in 0:d^N-1] 
scatter!(x_val, y_val, label = "Function evaluations, $( @sprintf("%.1f", length(x_val) * 100 / d^N) )% of domain", m = (2, :white, stroke(1, :black)))
plot!(p, x_values, y_values, color = :red, label = "Analytical")
plot!(p, x_values, mps_vec, color = :black, label = "TCI, Infidelity = $( @sprintf("%.16f", abs(1 - (y_values' * mps_vec)/(norm(y_values)*norm(mps_vec)))) )", linestyle = :dash, legend = :outertop, titlefont = font(10))
display(p)

# Convert mps list to ITensors MPS
sites = siteinds(d, N)
links = [Index(size(mps_list[n], 1), "Link,i=$n") for n in 2:N]
mps_tensors = Vector{ITensor}(undef, N)
mps_tensors[1] = ITensor(mps_list[1], sites[1], links[1])
for n in 2:(N-1)
    mps_tensors[n] = ITensor(mps_list[n], links[n-1], sites[n], links[n])
end
mps_tensors[N] = ITensor(mps_list[N], links[N-1], sites[N])
mps = MPS(mps_tensors)

println("\nFinished.")

end
