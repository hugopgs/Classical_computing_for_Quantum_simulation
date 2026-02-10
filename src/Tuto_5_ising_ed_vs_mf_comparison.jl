using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit
using Random

# --- 1. GÉNÉRATEURS D'OPÉRATEURS ---

"""
Génère les opérateurs de spin (Sx, Sz, Sm) pour une chaîne de N spins.
"""
function build_operators(N)
    # Bases locales
    sm_loc = sparse([0 1; 0 0])
    sx_loc = sparse([0 1; 1 0])
    sz_loc = sparse([1 0; 0 -1])
    id_loc = sparse(I, 2, 2)

    sms = [kron(sparse(I, 2^(i-1), 2^(i-1)), sm_loc, sparse(I, 2^(N-i), 2^(N-i))) for i in 1:N]
    sxs = [kron(sparse(I, 2^(i-1), 2^(i-1)), sx_loc, sparse(I, 2^(N-i), 2^(N-i))) for i in 1:N]
    szs = [kron(sparse(I, 2^(i-1), 2^(i-1)), sz_loc, sparse(I, 2^(N-i), 2^(N-i))) for i in 1:N]

    return (sm=sms, sx=sxs, sz=szs)
end

"""
Construit le Hamiltonien transverse d'Ising.
"""
function build_hamiltonian(N, J, alpha, hx, ops)
    H = spzeros(ComplexF64, 2^N, 2^N)
    # Champ transverse
    for i in 1:N
        H += hx * ops.sx[i]
    end
    # Interactions long-range
    for i in 1:N, j in (i+1):N
        dist = abs(i - j)^alpha
        H += (J / dist) * ops.sz[i] * ops.sz[j]
    end
    return H
end

# --- 2. SIMULATION EXACTE (ED - LIOUVILLIEN) ---

function simulate_ed(N, J, alpha, hx, dt, steps, m)
    ops = build_operators(N)
    H = build_hamiltonian(N, J, alpha, hx, ops)
    
    # Construction du super-opérateur Liouvillien (sans dissipation ici, gamma=0)
    # L(rho) = -i[H, rho] -> A = -i(I ⊗ H - H* ⊗ I)
    Id = sparse(I, 2^N, 2^N)
    A = -1im * (kron(Id, H) - kron(conj(H), Id))

    # État initial : tout en bas |00...0>
    psi = zeros(ComplexF64, 2^N); psi[1] = 1.0
    rho_vec = vec(psi * psi')

    sz_history = zeros(steps, N)

    for t in 1:steps
        rho = reshape(rho_vec, 2^N, 2^N)
        for i in 1:N
            sz_history[t, i] = real(tr(rho * ops.sz[i]))
        end
        rho_vec, _ = exponentiate(A, dt, rho_vec; krylovdim=m)
    end
    return sz_history
end

# --- 3. SIMULATION CHAMP MOYEN (MF - RK4) ---

"""
Dérivée temporelle pour les équations de Bloch (Mean Field).
Y contient [sx_1..sx_N, sy_1..sy_N, sz_1..sz_N]
"""
function mf_dynamics(Y, Vij, hx, N)
    dY = zeros(eltype(Y), 3*N)
    sx = @view Y[1:N]
    sy = @view Y[N+1:2*N]
    sz = @view Y[2*N+1:3*N]

    dsx = @view dY[1:N]
    dsy = @view dY[N+1:2*N]
    dsz = @view dY[2*N+1:3*N]

    for i in 1:N
        # Champ moyen local selon Z dû aux autres spins
        field_z = 2.0 * sum(Vij[i, :] .* sz)
        
        # dS/dt = B_eff × S
        dsx[i] = -sy[i] * field_z
        dsy[i] =  sx[i] * field_z - 2.0 * sz[i] * hx
        dsz[i] =  2.0 * sy[i] * hx
    end
    return dY
end

function rk4_step(f, Y, Vij, hx, N, dt)
    k1 = f(Y,          Vij, hx, N)
    k2 = f(Y + dt/2*k1, Vij, hx, N)
    k3 = f(Y + dt/2*k2, Vij, hx, N)
    k4 = f(Y + dt*k3,   Vij, hx, N)
    return Y + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

# --- 4. MAIN ---

function main()
    # Paramètres
    N, m = 8, 20
    J, alpha, hx = 1.0, 1.0, 2.0
    dt, steps = 0.1, 101
    t_axis = (0:steps-1) * dt

    println("Calcul ED (Exact)...")
    @time sz_ed = simulate_ed(N, J, alpha, hx, dt, steps, m)

    println("Calcul MF (Champ Moyen)...")
    # Initialisation MF : tous les spins en bas (sz = -1)
    Y = zeros(3*N)
    Y[2*N+1:3*N] .= -1.0
    
    # Matrice d'interaction
    Vij = [i == j ? 0.0 : J / abs(i-j)^alpha for i in 1:N, j in 1:N]
    
    sz_mf = zeros(steps, N)
    for t in 1:steps
        sz_mf[t, :] .= Y[2*N+1:3*N]
        Y = rk4_step(mf_dynamics, Y, Vij, hx, N, dt)
    end

    # Plotting
    mag_ed = sum(sz_ed, dims=2)
    mag_mf = sum(sz_mf, dims=2)

    p = plot(t_axis, mag_ed, label="ED (Exact)", lw=2, color=:black, title="Magnetization Dynamics (N=$N)")
    scatter!(p, t_axis[1:5:end], mag_mf[1:5:end], label="Mean Field (RK4)", markershape=:circle, markercolor=:red)
    
    xlabel!("Time (tJ)")
    ylabel!("Total Magnetization <Σ Sz>")
    
    display(p)
end

main()