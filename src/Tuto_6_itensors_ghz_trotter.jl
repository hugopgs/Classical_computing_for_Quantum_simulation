using ITensors
using LinearAlgebra

# --- 1. GÉNÉRATION D'ÉTAT GHZ ---

"""
Construit un circuit pour générer un état GHZ : H sur le premier spin, 
puis une cascade de CNOT(i, i+1).
"""
function build_ghz_circuit(s::Vector{Index{Int64}})
    N = length(s)
    circuit = ITensor[]

    # Porte Hadamard sur le premier site
    # 'op' est la méthode recommandée pour obtenir des opérateurs standards
    push!(circuit, op("H", s[1]))

    # Cascade de CNOT
    for j in 1:(N - 1)
        push!(circuit, op("CNOT", s[j], s[j+1]))
    end
    
    return circuit
end

# --- 2. DYNAMIQUE DE TROTTER (TRANSVERSE ISING) ---

"""
Construit les portes de Trotter pour le modèle d'Ising Transverse.
H = J * Σ Sz_i Sz_{i+1} + hx * Σ Sx_i
"""
function build_trotter_circuit(s::Vector{Index{Int64}}, J, hx, dt)
    N = length(s)
    circuit = ITensor[]
    
    # Portes à deux sites (Lien j, j+1)
    # H_ij = J*Sz*Sz + hx*Sx*Id
    for j in 1:(N - 1)
        # On définit le Hamiltonien local sur deux sites
        # Note : On divise hx par (N-1) ou on l'applique différemment 
        # selon la convention de Trotter choisie.
        h_ij = J * op("Sz", s[j]) * op("Sz", s[j+1]) +
               hx * op("Sx", s[j]) * op("Id", s[s[j+1]])
        
        # Exponentielle : exp(-im * dt * h_ij)
        G_ij = exp(-1im * dt * h_ij)
        push!(circuit, G_ij)
    end
    
    return circuit
end

# --- 3. EXÉCUTION DU CIRCUIT ---

function apply_circuit!(psi::ITensor, circuit::Vector{ITensor})
    for gate in circuit
        # Contraction de la porte avec l'état
        psi = gate * psi
        # On ramène les indices primés au niveau 0 pour la porte suivante
        psi = noprime(psi)
    end
    return psi
end

# --- 4. MAIN ---

function main()
    N = 4
    J = 1.0
    hx = 0.5
    dt = 0.1

    # Initialisation des indices (sites)
    s = siteinds("S=1/2", N)

    # 1. Création de l'état initial |00...0>
    # On crée un produit d'états (MPS simple ou ITensor dense)
    psi = ITensor(s)
    # On initialise l'état |1,1,1,1> (représentant spin Up dans ITensors)
    initial_indices = [s[i] => 1 for i in 1:N]
    psi[initial_indices...] = 1.0

    println("Norme initiale : ", norm(psi))

    # 2. GHZ Circuit
    println("Application du circuit GHZ...")
    ghz_gates = build_ghz_circuit(s)
    psi_ghz = apply_circuit!(copy(psi), ghz_gates)
    
    # Calcul de la fidélité GHZ (optionnel)
    println("Fidélité de l'état GHZ : ", norm(psi_ghz)^2)

    # 3. Trotter Evolution
    println("Application d'un pas de Trotter (Ising)...")
    trotter_gates = build_trotter_circuit(s, J, hx, dt)
    psi_evolved = apply_circuit!(copy(psi), trotter_gates)

    # 4. Mesure simple : <Sz> sur le premier site
    sz1_op = op("Sz", s[1])
    sz1_val = scalar(dag(prime(psi_evolved, s[1])) * sz1_op * psi_evolved)
    println("<Sz> sur le site 1 après Trotter : ", real(sz1_val))

end

main()