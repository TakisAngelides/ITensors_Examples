using ITensors
using KrylovKit
using LinearAlgebra
import ITensors.svd as ITensors_SVD
include("heat_equation.jl")

function get_inverse(mpo, sites, cutoff, max_sweeps)

    """
    This function finds the inverse MPO of the mpo input
    
    We are essentially solving M v = N_tilde where M and N_tilde are tensor networks and v is the solution to our problem for a given pair of sites and we optimize sweeping left and right the
    trial solution until it reaches the maximum number of sweeps

    SVDs below are affected by the cutoff input

    What the function essentially does is SVD on M to get M = USV and then v is given by v = V^-1 S^-1 U^-1 N_tilde but we know U^-1 and V^-1 because U and V are unitary
    hence U^-1, V^-1 = U^\dagger, V^\dagger

    We initialize arrays left_right_parts_M, left_right_parts_N_tilde to help us compute M and N_tilde and we update them during the optimization
    """

    N = length(sites)
    trial = MPO(sites, "Id")

    # Initialize left_right_parts_M 
    left_right_parts_M = []
    for i in 1:N
        # We are careful here and below wherever we are priming and unpriming indices to contract the right indices together 
        push!(left_right_parts_M, prime(trial[i]; :tags => "Site")*mpo[i]*prime(dag(mpo[i]); :tags => "Link")*dag(trial[i]'))
    end

    # Initialize left_right_parts_N_tilde 
    left_right_parts_N_tilde = []
    for i in 1:N
        push!(left_right_parts_N_tilde, setprime(mpo[i]', 0; :plev => 2)*setprime(dag(trial[i]), 1; :tags => "Link"))
    end

    # Start optimization
    for sw in 1:max_sweeps

        # Optimize by sweeping the trial MPO from left to right
        for i in 1:N-1
            # Compute M
            if N == 2
                M = setprime(setprime(prime(mpo[i]*mpo[i+1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i+1]'), 1; :plev => 2)
            else
                M = setprime(setprime(prime(mpo[i]*mpo[i+1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i+1]'), 1; :plev => 2)*prod(left_right_parts_M[setdiff(1:N, [i, i+1])])
            end
            
            # Compute the indices of M to be on U of SVD's M = USV
            bot = inds(M; :plev => 1) 
            
            # Computer N_tilde
            if N == 2
                N_tilde = dag(mpo[i]')*dag(mpo[i+1]')
            else
                N_tilde = dag(mpo[i]')*dag(mpo[i+1]')*prod(left_right_parts_N_tilde[setdiff(1:end, [i, i+1])])
            end
            N_tilde = setprime(prime(N_tilde; :tags => "Site"), 1; :plev => 3)

            # Do SVD on M
            U, S, V = ITensors.svd(M, bot..., cutoff = cutoff)

            # Get v named here res
            S_inv = ITensor(diagm(diag(Array(S, inds(S))).^(-1)), inds(S))
            res = dag(V)*S_inv*dag(U)*N_tilde
            res = setprime(res, 1; :plev => 2)

            # SVD on v a.k.a. res so as to update the two sites of the trial MPO
            U, S, V = ITensors.svd(res, commoninds(res, trial[i]), lefttags = "Link,l=$(i)", righttags = "Link,l=$(i)", cutoff = cutoff)
            trial[i], trial[i+1] = U, S*V
        
            # Update the left_right_parts_M
            for j in [i, i+1]                
                left_right_parts_M[j] = prime(trial[j]; :tags => "Site")*mpo[j]*prime(dag(mpo[j]); :tags => "Link")*dag(trial[j]')
            end

            # Update the left_right_parts_N_tilde
            for j in [i, i+1]
                left_right_parts_N_tilde[j] = setprime(mpo[j]', 0; :plev => 2)*setprime(dag(trial[j]), 1; :tags => "Link")
            end
        end

        # Same as above but now sweeping from right to left
        for i in N:-1:2

            if N == 2
                M = setprime(setprime(prime(mpo[i]*mpo[i-1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i-1]'), 1; :plev => 2)
            else
                M = setprime(setprime(prime(mpo[i]*mpo[i-1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i-1]'), 1; :plev => 2)*prod(left_right_parts_M[setdiff(1:N, [i, i-1])])
            end
            
            bot = inds(M; :plev => 1) 
            
            if N == 2
                N_tilde = dag(mpo[i]')*dag(mpo[i-1]')
            else
                N_tilde = dag(mpo[i]')*dag(mpo[i-1]')*prod(left_right_parts_N_tilde[setdiff(1:end, [i, i-1])])
            end
            N_tilde = setprime(prime(N_tilde; :tags => "Site"), 1; :plev => 3)

            U, S, V = ITensors.svd(M, bot..., cutoff = cutoff)

            S_inv = ITensor(diagm(diag(Array(S, inds(S))).^(-1)), inds(S))
            res = dag(V)*S_inv*dag(U)*N_tilde
            res = setprime(res, 1; :plev => 2)

            U, S, V = ITensors.svd(res, commoninds(res, trial[i-1]), lefttags = "Link,l=$(i-1)", righttags = "Link,l=$(i-1)", cutoff = cutoff)
            
            trial[i-1], trial[i] = U*S, V

            for j in [i, i-1]                
                left_right_parts_M[j] = prime(trial[j]; :tags => "Site")*mpo[j]*prime(dag(mpo[j]); :tags => "Link")*dag(trial[j]')
            end

            for j in [i, i-1]                
                left_right_parts_N_tilde[j] = setprime(mpo[j]', 0; :plev => 2)*setprime(dag(trial[j]), 1; :tags => "Link")
            end
        end
    end
    return trial
        
end


N = 4
sites = siteinds("S=1/2", N)
mpo = get_mpo(sites)
max_sweeps = 10
cutoff = 1e-12
trial = get_inverse(mpo, sites, cutoff, max_sweeps)

let
    t = trial'*mpo
    tmp = t[1]
    for i in 2:length(t)
        tmp*=t[i]
    end
    tmp = Array(tmp, inds(tmp; :plev => 2)..., inds(tmp; :plev => 0)...)
    tmp = reshape(tmp, (2^N, 2^N))
    println(isapprox(tmp, I))
end
