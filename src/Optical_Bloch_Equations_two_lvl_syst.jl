# Hugo PAGES
# Quantum Technologies and application, 19/09/2025
# Write a code for simulating the OBEs for a two-lvl system

# Function that accepts, rho, parameters and return rho_dot
# Function which updates the density matrix rho
# function which runs a simulation with fixed time steps

function rho_dot(rho, delta, omega, L)
    # Define the Hamiltonian (in the rotating frame)
    H = [0 omega/2; omega/2 -delta]

    # Compute the commutator [H, rho]
    commutator = H * rho - rho * H
    dissipator = zeros(ComplexF64, size(rho))
    for l in L
        dissipator += l * rho * l' - 0.5 * (l' * l * rho + rho * l' * l)
    end
    
    # Return the time derivative of the density matrix
    return -1im * commutator + dissipator
end 


function Euler_rho_update(rho, delta, omega, L, dt)
    rho_dot_val = rho_dot(rho, delta, omega, L)
    return rho + rho_dot_val * dt
end

function run_simulation(rho0, delta, omega, L, dt, t_final, RK4)
    num_steps = Int(t_final / dt)
    rho = rho0
    results = Vector{Matrix{ComplexF64}}(undef, num_steps+1)
    time_list = collect(0:dt:(num_steps*dt))
    results[1] = rho
    for step in 1:num_steps
        if RK4 == true
            rho = runge_kutta_4(rho, delta, omega, L, dt)
        else
            rho = Euler_rho_update(rho, delta, omega, L, dt)
        end
        results[step+1] = rho
    end
    return time_list, results
end

function runge_kutta_4(rho, delta, omega, L, dt)
    k1 = rho_dot(rho, delta, omega, L)
    k2 = rho_dot(rho + 0.5 * dt * k1, delta, omega, L)
    k3 = rho_dot(rho + 0.5 * dt * k2, delta, omega, L)
    k4 = rho_dot(rho + dt * k3, delta, omega, L)
    return rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
end

function exp_(x, alpha)
    return exp.(-alpha * x)
end

gamma=0.8
alpha=0
delta = [1]
omega= 4
L1 = sqrt(gamma) * [0 1; 0 0]
L2 = sqrt(alpha/2) * [0 1; 0 -1]
L = [L1, L2]
rho0=[1/2 1im/2; -1im/2 1/2]
dt=0.01
t_final=10.0



using Plots
Plots.closeall()   # start clean

plot()    
for d in delta
    time_list, results= run_simulation(rho0, d, omega, L, dt, t_final, false)
    rho11 = [M[1,1] for M in results]
    rho11 = [real(z) for z in rho11]
    plot!(time_list, rho11, label="rho_11, δ=$d, Euler",
            xlabel="Time", ylabel="Population",
            title="Two-level System Dynamics")
    # plot!(time_list, exp_(time_list, gamma), label="exp-gammat", color=:red)


    time_list, results= run_simulation(rho0, d, omega, L, dt, t_final, true)
    rho11 = [M[1,1] for M in results]
    rho11 = [real(z) for z in rho11]
    plot!(time_list, rho11, label="rho_11, δ=$d, RK4",
            xlabel="Time", ylabel="Population",
            title="Two-level System Dynamics")
end
















# delta =0.5
# omega= [0,0.5,2]

# for omeg in omega
#     time_list, results= run_simulation(rho0, delta, omeg, L, 0.01, 10.0)
#     rho11 = [M[1,1] for M in results]
#     # Extraire uniquement la partie réelle
#     rho11 = [real(z) for z in rho11]
#     plot!(time_list, rho11, label="rho_11, omega=$omeg",
#             xlabel="Time", ylabel="Population",
#             title="Two-level System Dynamics")
# end


display(current())

