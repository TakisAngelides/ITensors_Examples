using ITensors
using KrylovKit
using LinearAlgebra
import ITensors.svd as ITensors_SVD


# Number of sites, sweeps of DMRG, cut-off error, max dim of mps
N = 8
max_sweeps = 10
cut_off = 1e-10
max_dim = 10

# Define sites object
sites = siteinds("S=1/2", N)

# Define the Hamiltonian
function get_Hamiltonian(sites)
    os = OpSum()
    for j=1:N-1
        os += "Sz",j,"Sz",j+1
        os += 0.5,"S+",j,"S-",j+1
        os += 0.5,"S-",j,"S+",j+1
    end
    return MPO(os, sites)
end
mpo = get_Hamiltonian(sites)

# Initialize the MPS and put it in normalized canonical form
mps = randomMPS(ComplexF64, sites)
if !isortho(mps) || ortho_lims(mps)[1] != 1
    ITensors.orthogonalize!(mps, 1)
end

# Initialize an array which will hold the left and right parts of the effective Hamiltonian
H_eff_parts = Array{ITensor}(undef, N)
for i in 3:N
    H_eff_parts[i] = prime(dag(mps[i]))*mpo[i]*mps[i]
end

function DMRG()

    # E_curr will change after a full sweep and E will change within the sweep
    E_curr = 0.0
    E = 0.0

    # Perform the DMRG sweeps
    for sweep in 1:max_sweeps

        # Forward part of sweep from left to right
        for i in 1:N-1 # This is the left index of the two sites being updated
    
            # Get the effective Hamiltonian
            if i == 1
                H_eff = mpo[i]*mpo[i+1]*prod(H_eff_parts[i+2:N])
            elseif i == N-1
                H_eff = prod(H_eff_parts[1:i-1])*mpo[i]*mpo[i+1]
            else
                H_eff = prod(H_eff_parts[1:i-1])*mpo[i]*mpo[i+1]*prod(H_eff_parts[i+2:N])
            end

            # Get the two site part of the mps
            mps_two_site = mps[i]*mps[i+1]

            # Now we call the eigensolver with H_eff and mps_two_site
            vals, vecs = eigsolve(H_eff, mps_two_site, 1, :SR; ishermitian = true) # ; ishermitian = ishermitian)
            
            # Update current energy with the lowest we found from optimizing sites i, i+1
            E = vals[1]

            # Define the lowest-eigenvalue eigenvector
            gs_evec = vecs[1]

            # Perform the SVD decomposition on it to get two mps tensors
            gs_even_site_idx = inds(gs_evec; :tags => "n=$i")
            if i == 1
                U, S, V = ITensors_SVD(gs_evec, (gs_even_site_idx); maxdim = max_dim, lefttags = "l=$(i)", righttags = "l=$(i)")
                V = S*V
            else
                U, S, V = ITensors_SVD(gs_evec, (gs_even_site_idx, inds(gs_evec; :tags => "l=$(i-1)")); maxdim = max_dim, lefttags = "l=$(i)", righttags = "l=$(i)")
                V = S*V
            end
            
            # Update the mps
            mps[i], mps[i+1] = U, V

            # Update the H_eff_parts
            H_eff_parts[i] = prime(dag(mps[i]))*mpo[i]*mps[i]
            H_eff_parts[i+1] = prime(dag(mps[i+1]))*mpo[i+1]*mps[i+1]

            println("Sweep: ", sweep, ", Left to right: ", i, ", norm: ", norm(mps), ", Max D: ", maxlinkdim(mps), ", E: ", E)
            
        end

        # Backward part of sweep from right to left
        for i in N-1:-1:1 # This is the left index of the two sites being updated
    
            # Get the effective Hamiltonian
            if i == 1
                H_eff = mpo[i]*mpo[i+1]*prod(H_eff_parts[i+2:N])
            elseif i == N-1
                H_eff = prod(H_eff_parts[1:i-1])*mpo[i]*mpo[i+1]
            else
                H_eff = prod(H_eff_parts[1:i-1])*mpo[i]*mpo[i+1]*prod(H_eff_parts[i+2:N])
            end

            # Get the two site part of the mps
            mps_two_site = mps[i]*mps[i+1]

            # Now we call the eigensolver with H_eff and mps_two_site
            vals, vecs = eigsolve(H_eff, mps_two_site, 1, :SR; ishermitian = true) # ; ishermitian = ishermitian)
            
            # Update current energy with the lowest we found from optimizing sites i, i+1
            E = vals[1]

            # Define the lowest-eigenvalue eigenvector
            gs_evec = vecs[1]

            # Perform the SVD decomposition on it to get two mps tensors
            gs_even_site_idx = inds(gs_evec; :tags => "n=$i")
            if i == 1
                U, S, V = ITensors_SVD(gs_evec, (gs_even_site_idx), maxdim = max_dim, lefttags = "l=$(i)", righttags = "l=$(i)")
                U = U*S
            else
                U, S, V = ITensors_SVD(gs_evec, (gs_even_site_idx, inds(gs_evec; :tags => "l=$(i-1)")), maxdim = max_dim, lefttags = "l=$(i)", righttags = "l=$(i)")
                U = U*S
            end
            
            # Update the mps
            mps[i], mps[i+1] = U, V

            # Update the H_eff_parts
            H_eff_parts[i] = prime(dag(mps[i]))*mpo[i]*mps[i]
            H_eff_parts[i+1] = prime(dag(mps[i+1]))*mpo[i+1]*mps[i+1]

            println("Sweep: ", sweep, ", Right to left: ", i, ", norm: ", norm(mps), ", Max D: ", maxlinkdim(mps), ", E: ", E)
            
        end

        # Compute the fractional energy change after a full sweep
        fractional_energy_change = abs((E - E_curr)/E_curr)

        # Set the current energy variable E_curr to the final decreased energy E after a full sweep
        E_curr = E
        
        # Check for stopping conditions
        if (fractional_energy_change < cut_off)
            println("Energy accuracy reached.")
            break
        elseif (max_sweeps == sweep)
            println("Maximum sweeps reached before reaching desired energy accuracy.")
            break
        end

    end

    return E_curr

end

println("Energy before DMRG: ", real(inner(mps', mpo, mps)))
gs_energy = DMRG()
println("Energy after DMRG: ", real(inner(mps', mpo, mps)))


