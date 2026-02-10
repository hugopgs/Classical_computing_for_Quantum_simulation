using LinearAlgebra
using Plots




# Step 1: Buld a series of lowering operators
function build_lowering_ops(sm, N)

    sms = Vector{}()
    # this is an empty vector
    # ... it will be filled with matrices as elements

    for ii = 1:N
        ldim = 2 ^ (ii - 1) # the Hilbert space dimention left of spin ii
        rdim = 2 ^ (N - ii) # the Hilbert space dimention right of spin ii
        left_id = Matrix(I, ldim, ldim) # identity matrix for left Hilbert space
        right_id = Matrix(I, rdim, rdim) # identity matrix for right Hilbert space
        push!(sms, kron(left_id, sm, right_id))
    end

    return sms

end

# Build spin operators
function build_spin_ops_bad(sms)

    sxs = Vector{}()
    szs = Vector{}()
    for ii = 1:length(sms)
        push!(sxs, sms[ii] + sms[ii]')
        push!(szs, sms[ii]' * sms[ii] - sms[ii] * sms[ii]')
    end 

    return sxs, szs
end


# Step 1: Buld the necessary many-body spin operators
function build_spin_ops(sm, N)

    sxs = Vector{}()
    szs = Vector{}()

    for ii = 1:N
        ldim = 2 ^ (ii - 1) # the Hilbert space dimention left of spin ii
        rdim = 2 ^ (N - ii) # the Hilbert space dimention right of spin ii
        left_id = Matrix(I, ldim, ldim) # identity matrix for left Hilbert space
        right_id = Matrix(I, rdim, rdim) # identity matrix for right Hilbert space
        push!(sxs, kron(left_id, sm + sm', right_id))
        push!(szs, kron(left_id, sm'*sm - sm*sm', right_id))
    end

    return sxs, szs

end




# Step 2: Build Hamiltonian
function build_hamiltonian(J, alpha, hx, sxs, szs)

    N = length(sxs)
    H = zeros(Float64, 2^N, 2^N)

    for ii = 1:N
        H += hx .* sxs[ii]
        for jj = (ii+1):N
            H += (J / (jj-ii)^alpha) .* szs[ii] * szs[jj]
        end
    end

    return H

end


# Step 3: Function doing the time evolution after center spin-flip
function ti_simulation(N, J, alpha, hx, dt, steps)

    sm = [0 1; 0 0]
    sxs, szs = build_spin_ops(sm, N)
 
    H = build_hamiltonian(J, alpha, hx, sxs, szs)

    # initial state
    psi = zeros(ComplexF64, 2^N)
    psi[1] = 1.0 # the all zero state 
    psi = sxs[ div(N,2) + 1 ] * psi # flip spin in center
    #mneel = div(4^cld(N,2) - 1, 3) + 1
    #psi[mneel] = 1.0 # the Néel state

    # build evolution operator for a single time-step
    U = exp(-1im .* dt .* H)

    out_sz = zeros(Float64, steps, N)

    for tt = 1:steps
        #println("Step $tt/$steps - norm(psi) = $(norm(psi))")

        for ii = 1:N
            out_sz[tt, ii] = real(psi' * szs[ii] * psi)
        end

        psi = U * psi
    end

    return out_sz

end


# simulating and plotting
function main()
    

    N = 7 # number of spins

    J = 1 # defines energy/time units
    alpha = 3 # interaction range
    hx = 1 # transverse field

    dt = 0.1 # time step for plotting
    steps = 101
 
    @time out_sz = ti_simulation(N, J, alpha, hx, dt, steps)
    tran= 0:dt:(steps-1)*dt
    p=plot(tran, out_sz[:,3], label="⟨Sz⟩ of center spin", title="Transverse Ising model dynamics")
    display(p)
    # # nice plot
    # cmap = cgrad(:RdBu)

    # h = heatmap(1:N, 0:dt:((steps-1) * dt),  out_sz)
    # xlims!((0.5, N+0.5))
    # display(h)


    return nothing

end




# # Compute approximation to exp(A)*psi0
# # ... with m Krylov basis states
function arnoldi_exp(A, psi0, m)

    D = length(psi0)

    Q = similar(A, D, m) # the projection matrix
    h = zeros(eltype(A), m, m) # Krylov projection of A
    
    Q[:,1] = psi0   # assumed normalized
    for ii = 1:(m-1)
        psi_i = A*Q[:,ii]   
        for jj = 1:ii
            h[jj,ii] = Q[:,jj]' * psi_i
            psi_i -= h[jj,ii] .* Q[:,jj]
        end
        h[ii+1, ii] = norm(psi_i)
        Q[:,ii+1] = psi_i ./ h[ii+1, ii]
    end
 
    # now return the matrix exponential
    return Q * exp(h)[:,1]

end



main()