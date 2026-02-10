using ITensors
using LinearAlgebra

# --- 1. STRUCTURE MPS PERSONNALISÉE ---

mutable struct VidalMPS
    N::Int
    chi_max::Int
    
    gammas::Vector{ITensor}  # Tenseurs de sites
    lambdas::Vector{ITensor} # Matrices de liens (valeurs singulières)
    
    qs::Vector{Index}        # Indices physiques (sites)
    links::Vector{Index}     # Indices de liens (bonds)

    function VidalMPS(N::Int, chi_max::Int)
        # Création des indices physiques
        qs = [Index(2, "q$i") for i in 1:N]
        # Création des indices de liens (1 de plus que de sites)
        links = [Index(1, "link$i") for i in 1:(N + 1)]
        
        gammas = Vector{ITensor}(undef, N)
        lambdas = Vector{ITensor}(undef, N + 1)

        # État initial |00...0>
        for i in 1:N
            gammas[i] = ITensor(links[i], qs[i], links[i+1])
            gammas[i][links[i]=>1, qs[i]=>1, links[i+1]=>1] = 1.0
        end

        for i in 1:N + 1
            lambdas[i] = diagITensor([1.0], links[i])
        end

        new(N, chi_max, gammas, lambdas, qs, links)
    end
end

# --- 2. FONCTIONS DE MISE À JOUR (GATES) ---

""" Applique une porte à 1 site sur le site n """
function apply_1s_gate!(psi::VidalMPS, gate::ITensor, n::Int)
    # Contraction simple : la porte possède prime(qs[n]) et qs[n]
    psi.gammas[n] = noprime(gate * psi.gammas[n])
    return nothing
end

""" Applique une porte à 2 sites (TEBD update) entre n et n+1 """
function apply_2s_gate!(psi::VidalMPS, gate::ITensor, n::Int)
    # 1. On forme le tenseur d'onde local (Theta)
    # Formule : Λ[n] * Γ[n] * Λ[n+1] * Γ[n+1] * Λ[n+2]
    theta = psi.lambdas[n] * psi.gammas[n] * psi.lambdas[n+1] * psi.gammas[n+1] * psi.lambdas[n+2]
    
    # 2. Application de la porte
    theta = noprime(gate * theta)

    # 3. SVD pour séparer les sites et tronquer la dimension
    # On coupe entre (link[n], q[n]) et (q[n+1], link[n+2])
    U, S, V = svd(theta, (psi.links[n], psi.qs[n]); 
                  maxdim=psi.chi_max, 
                  cutoff=1e-12)

    # 4. Normalisation et mise à jour de Λ[n+1]
    psi.lambdas[n+1] = S / norm(S)
    
    # 5. Extraction des nouveaux Γ par division (multiplication par l'inverse)
    # On doit retirer l'influence des Λ extérieurs
    inv_L_left = inv_diag(psi.lambdas[n])
    inv_L_right = inv_diag(psi.lambdas[n+2])
    
    psi.gammas[n] = noprime(U * inv_L_left)
    psi.gammas[n+1] = noprime(V * inv_L_right)
    
    # Mise à jour de l'index de lien interne
    psi.links[n+1] = commonind(U, S)
    
    return nothing
end

# Utilitaire pour inverser les matrices diagonales Λ
function inv_diag(L::ITensor)
    res = copy(L)
    for i in 1:dim(inds(L)[1])
        val = res[i, i]
        res[i, i] = abs(val) > 1e-14 ? 1.0 / val : 0.0
    end
    return res
end

# --- 3. CONSTRUCTEURS DE CIRCUITS ---

function build_trotter_ising(psi::VidalMPS, J, hx, dt)
    N = psi.N
    circ = []
    
    # Portes 2-sites (Interaction ZZ)
    sz = [1 0; 0 -1]
    gate_zz = exp(-1im * (dt/2) * J * kron(sz, sz))
    
    # Portes 1-site (Champ Transverse X)
    sx = [0 1; 1 0]
    gate_x = exp(-1im * (dt/2) * hx * sx)

    # On pré-construit les ITensors pour chaque paire/site
    for i in 1:N-1
        # Gate 2-sites
        U2 = ITensor(gate_zz, prime(psi.qs[i]), prime(psi.qs[i+1]), psi.qs[i], psi.qs[i+1])
        push!(circ, ("2S", U2, i))
    end
    
    for i in 1:N
        # Gate 1-site
        U1 = ITensor(gate_x, prime(psi.qs[i]), psi.qs[i])
        push!(circ, ("1S", U1, i))
    end
    
    return vcat(circ, reverse(circ)) # Trotter 2nd ordre
end

# --- 4. MAIN ---

function main()
    N = 6
    chi_max = 32
    dt = 0.1
    J, hx = 1.0, 0.5

    # Initialisation
    psi = VidalMPS(N, chi_max)
    
    # Construction du circuit de Trotter
    println("Construction du circuit Trotter...")
    circuit = build_trotter_ising(psi, J, hx, dt)

    # Simulation : application du circuit
    println("Exécution de la simulation...")
    for (type, gate, pos) in circuit
        if type == "1S"
            apply_1s_gate!(psi, gate, pos)
        else
            apply_2s_gate!(psi, gate, pos)
        end
    end

    # Mesure : Calcul de <Sz> sur le site 1
    # On forme la fonction d'onde locale : Λ1 * Γ1 * Λ2
    rho1 = psi.lambdas[1] * psi.gammas[1] * psi.lambdas[2]
    sz_op = ITensor([1 0; 0 -1], prime(psi.qs[1]), psi.qs[1])
    val = scalar(dag(prime(rho1, psi.qs[1])) * sz_op * rho1)
    
    println("Résultat <Sz> au site 1 : ", real(val))
end

main()