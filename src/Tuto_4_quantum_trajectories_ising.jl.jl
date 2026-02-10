using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit
using Random

# --- Générateurs d'Opérateurs ---

"""
Génère les opérateurs many-body (Sx, Sz, et Sm) pour une chaîne de N spins.
"""
function build_spin_operators(N)
    sm_local = sparse([0 1; 0 0])
    sz_local = sparse([1 0; 0 -1]) # sm'*sm - sm*sm'
    sx_local = sparse([0 1; 1 0]) # sm + sm'

    sms = Vector{SparseMatrixCSC{ComplexF64, Int}}()
    sxs = Vector{SparseMatrixCSC{ComplexF64, Int}}()
    szs = Vector{SparseMatrixCSC{ComplexF64, Int}}()

    for i in 1:N
        left_id = i == 1 ? 1 : iidentity(2^(i-1))
        right_id = i == N ? 1 : iidentity(2^(N-i))
        
        push!(sms, kron(left_id, sm_local, right_id))
        push!(sxs, kron(left_id, sx_local, right_id))
        push!(szs, kron(left_id, sz_local, right_id))
    end
    return sms, sxs, szs
end

iidentity(d) = sparse(I, d, d)

"""
Construit le Hamiltonien transverse d'Ising avec interactions en loi de puissance.
"""
function build_hamiltonian(N, J, alpha, hx, sxs, szs)
    H = spzeros(ComplexF64, 2^N, 2^N)
    
    # Champ transverse
    for i in 1:N
        H += hx * sxs[i]
    end
    
    # Interactions ZZ
    for i in 1:N
        for j in (i+1):N
            dist = (j - i)^alpha
            H += (J / dist) * szs[i] * szs[j]
        end
    end
    return H
end

# --- Fonctions de Simulation ---

"""
Évolution via Trajectoires Quantiques (Monte Carlo Wave Function).
"""
function simulate_trajectories(N, H, sms, szs, gamma, dt, steps, nt, m)
    rng = MersenneTwister()
    
    # Hamiltonien effectif non-hermitien
    Heff = copy(H)
    for i in 1:N
        Heff -= 1im * (gamma/2) * (sms[i]' * sms[i])
    end
    
    jump_ops = sqrt(gamma) .* sms
    
    # État initial (Néel)
    idx_neel = div(4^cld(N,2)-1, 3) + 1
    psi_init = zeros(ComplexF64, 2^N)
    psi_init[idx_neel] = 1.0

    out_sz = zeros(steps, N)

    for q in 1:nt
        println("Trajectoire $q/$nt")
        psi = copy(psi_init)
        
        for t in 1:steps
            # Mesure Sz
            for i in 1:N
                out_sz[t, i] += real(dot(psi, szs[i], psi))
            end
            
            # Un pas d'évolution
            psi = trajectory_step!(psi, Heff, jump_ops, dt, rng, m)
        end
    end
    
    return out_sz ./ nt
end

function trajectory_step!(psi, Heff, Ls, dt, rng, m)
    # Évolution déterministe (Heff)
    psi_new, _ = exponentiate(Heff, -1im * dt, psi; krylovdim=m, tol=1e-15)
    
    p_no_jump = norm(psi_new)^2
    r = rand(rng)

    if r > p_no_jump
        # Un saut a eu lieu
        probs = [norm(L * psi)^2 for L in Ls]
        # Normalisation grossière pour le choix du canal de saut
        cum_probs = cumsum(probs)
        target = rand(rng) * cum_probs[end]
        idx = findfirst(x -> x >= target, cum_probs)
        
        psi = Ls[idx] * psi
        return psi ./ norm(psi)
    else
        # Pas de saut, on normalise l'état évolué par Heff
        return psi_new ./ norm(psi_new)
    end
end

"""
Évolution exacte via l'équation de Lindblad (Espace de Liouville).
"""
function simulate_liouvillian(N, H, sms, szs, gamma, dt, steps, m)
    Id = iidentity(2^N)
    
    # Hamiltonien effectif pour le Liouvillien
    Heff = H - 0.5im * gamma * sum(L' * L for L in sms)
    
    # Construction du super-opérateur Lindbladien A
    # A * rho = -i[H, rho] + gamma * sum(L rho L' - 0.5{L'L, rho})
    A = -1im * (kron(Id, Heff) - kron(conj(Heff), Id))
    for L in sms
        A += gamma * kron(conj(L), L)
    end

    # État initial (Néel) en format vecteur (densité)
    idx_neel = div(4^cld(N,2)-1, 3) + 1
    psi = zeros(ComplexF64, 2^N)
    psi[idx_neel] = 1.0
    rho_vec = vec(psi * psi')

    out_sz = zeros(steps, N)

    for t in 1:steps
        rho = reshape(rho_vec, 2^N, 2^N)
        for i in 1:N
            out_sz[t, i] = real(tr(rho * szs[i]))
        end
        rho_vec, _ = exponentiate(A, dt, rho_vec; krylovdim=m, tol=1e-12)
    end

    return out_sz
end

# --- Main ---

function main()
    # Paramètres
    N, m = 7, 20
    J, alpha, hx = 1.0, 1.36, 1.0
    gamma, dt = 0.2, 0.01
    steps, nt = 100, 500  # Réduit pour l'exemple

    # Initialisation des opérateurs
    println("Construction des opérateurs...")
    sms, sxs, szs = build_spin_operators(N)
    H = build_hamiltonian(N, J, alpha, hx, sxs, szs)

    # Simulation Trajectoires
    println("Début simulation trajectoires quantiques...")
    @time sz_traj = simulate_trajectories(N, H, sms, szs, gamma, dt, steps, nt, m)

    # Simulation Idéale (Liouvillian)
    println("Début simulation Liouvillien...")
    @time sz_ideal = simulate_liouvillian(N, H, sms, szs, gamma, dt, steps, m)

    # Plotting
    t_axis = (0:steps-1) .* dt
    p = plot(title="Dynamics of Transverse Ising Model (N=$N)", xlabel="Time", ylabel="<Sz>")
    
    plot!(p, t_axis, sz_traj[:, 3], label="Spin 3 (MCWF)", lw=2)
    plot!(p, t_axis, sz_traj[:, 4], label="Spin 4 (MCWF)", lw=2)
    plot!(p, t_axis, sz_ideal[:, 3], label="Spin 3 (Exact)", ls=:dash, color=:black)
    plot!(p, t_axis, sz_ideal[:, 4], label="Spin 4 (Exact)", ls=:dash, color=:grey)

    display(p)
end

main()