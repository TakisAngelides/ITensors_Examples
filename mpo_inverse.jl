using ITensors
using KrylovKit
using LinearAlgebra
import ITensors.svd as ITensors_SVD
include("heat_equation.jl")

N = 5
sites = siteinds("S=1/2", N)
mpo = get_mpo(sites)

function get_inverse(mpo, sites)

    # Follows fig 4 of https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.3.040313

    N = length(sites)
    trial = MPO(sites, "Id")

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

    left_right_parts_A = []
    for i in 1:N
        
        tmp6 = prime(trial[i]; :tags => "Site")
        
        tmp7 = tmp6*mpo[i]
        
        tmp8 = prime(dag(mpo[i]); :tags => "Link")
        
        tmp9 = tmp7*tmp8
        
        tmp10 = dag(trial[i]')
        
        push!(left_right_parts_A, tmp9*tmp10)
    end

    left_right_parts_B = []
    for i in 1:N
        
        tmp9 = setprime(mpo[i]', 0; :plev => 2)
        
        # tmp10 = setprime(trial[i]', 0; :plev => 2)

        tmp10 = setprime(dag(trial[i]), 1; :tags => "Link")
        
        push!(left_right_parts_B, tmp9*tmp10)
    end

    max_sweeps = 10

    for sw in 1:max_sweeps

        # println(sw)

        for i in 1:N-1

            println(sw, " ", i)

            tmp1 = prime(mpo[i]*mpo[i+1]; :tags => "Site")

            tmp2 = setprime(tmp1, 0; :plev => 2)

            tmp3 = tmp2*dag(mpo[i]')*dag(mpo[i+1]')

            tmp4 = setprime(tmp3, 1; :plev => 2)

            if N == 2
                M = tmp4
            else
                tmp5 = prod(left_right_parts_A[setdiff(1:N, [i, i+1])])
                M = tmp4*tmp5
            end
            
            bot = inds(M; :plev => 1) 
            
            if N == 2
                N_tilde = dag(mpo[i]')*dag(mpo[i+1]')
            else
                N_tilde = dag(mpo[i]')*dag(mpo[i+1]')*prod(left_right_parts_B[setdiff(1:end, [i, i+1])])
            end
            N_tilde = setprime(prime(N_tilde; :tags => "Site"), 1; :plev => 3)

            # println(inds(N_tilde))

            U, S, V = ITensors.svd(M, bot...)

            S_inv = ITensor(diagm(diag(Array(S, inds(S))).^(-1)), inds(S))

            res = dag(V)*S_inv*dag(U)*N_tilde

            res = setprime(res, 1; :plev => 2)

            # println(i)
            # println(inds(res))
            # println(inds(trial[i]))

            U, S, V = ITensors.svd(res, commoninds(res, trial[i]), lefttags = "Link,l=$(i)", righttags = "Link,l=$(i)")
            
            trial[i] = U
            
            trial[i+1] = S*V

            for j in [i, i+1]
            
                tmp6 = prime(trial[j]; :tags => "Site")
        
                tmp7 = tmp6*mpo[j]
                
                tmp8 = prime(dag(mpo[j]); :tags => "Link")
                
                tmp9 = tmp7*tmp8
                
                tmp10 = dag(trial[j]')
                
                left_right_parts_A[j] = tmp9*tmp10
            end

            for j in [i, i+1]
                
                tmp14 = setprime(mpo[j]', 0; :plev => 2)
                
                # tmp15 = setprime(trial[j]', 0; :plev => 2)

                tmp15 = setprime(dag(trial[j]), 1; :tags => "Link")
                
                left_right_parts_B[j] = tmp14*tmp15
            end
            
        end

        for i in N:-1:2

            println(sw, " ", i)

            tmp1 = prime(mpo[i]*mpo[i-1]; :tags => "Site")

            tmp2 = setprime(tmp1, 0; :plev => 2)

            tmp3 = tmp2*dag(mpo[i]')*dag(mpo[i-1]')

            tmp4 = setprime(tmp3, 1; :plev => 2)

            if N == 2
                M = tmp4
            else
                tmp5 = prod(left_right_parts_A[setdiff(1:N, [i, i-1])])
                M = tmp4*tmp5
            end
            
            bot = inds(M; :plev => 1) 
            
            if N == 2
                N_tilde = dag(mpo[i]')*dag(mpo[i-1]')
            else
                N_tilde = dag(mpo[i]')*dag(mpo[i-1]')*prod(left_right_parts_B[setdiff(1:end, [i, i-1])])
            end
            N_tilde = setprime(prime(N_tilde; :tags => "Site"), 1; :plev => 3)

            U, S, V = ITensors.svd(M, bot...)

            S_inv = ITensor(diagm(diag(Array(S, inds(S))).^(-1)), inds(S))

            res = dag(V)*S_inv*dag(U)*N_tilde

            res = setprime(res, 1; :plev => 2)

            # println(i)
            # println(inds(res))
            # println(inds(trial[i-1]))

            U, S, V = ITensors.svd(res, commoninds(res, trial[i-1]), lefttags = "Link,l=$(i-1)", righttags = "Link,l=$(i-1)")
            
            trial[i] = V
            
            trial[i-1] = U*S

            for j in [i, i-1]
            
                tmp6 = prime(trial[j]; :tags => "Site")
        
                tmp7 = tmp6*mpo[j]
                
                tmp8 = prime(dag(mpo[j]); :tags => "Link")
                
                tmp9 = tmp7*tmp8
                
                tmp10 = dag(trial[j]')
                
                left_right_parts_A[j] = tmp9*tmp10
            end

            for j in [i, i-1]
                
                tmp14 = setprime(mpo[j]', 0; :plev => 2)
                
                # tmp15 = setprime(trial[j]', 0; :plev => 2)

                tmp15 = setprime(dag(trial[j]), 1; :tags => "Link")
                
                left_right_parts_B[j] = tmp14*tmp15
            end
            
        end
    end

    return trial
        
end


trial = get_inverse(mpo, sites)

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
