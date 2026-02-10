using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit

# Step 1: Buld a series of lowering operators
function build_lowering_ops(sm, N)

    sms = Vector{}()
    # this is an empty vector
    # ... it will be filled with matrices as elements

    for ii = 1:N
        ldim = 2 ^ (ii - 1) # the Hilbert space dimention left of spin ii
        rdim = 2 ^ (N - ii) # the Hilbert space dimention right of spin ii
        left_id = sparse(I, ldim, ldim) # identity matrix for left Hilbert space
        right_id = sparse(I, rdim, rdim) # identity matrix for right Hilbert space
        push!(sms, kron(left_id, sm, right_id))
    end

    return sms

end


# Step 1: Buld the necessary many-body spin operators
function build_sparse_spin_ops(sm, N)

    sxs = Vector{}()
    szs = Vector{}()

    for ii = 1:N
        ldim = 2 ^ (ii - 1) # the Hilbert space dimention left of spin ii
        rdim = 2 ^ (N - ii) # the Hilbert space dimention right of spin ii
        left_id = sparse(I, ldim, ldim) # identity matrix for left Hilbert space
        right_id = sparse(I, rdim, rdim) # identity matrix for right Hilbert space
        push!(sxs, kron(left_id, sm + sm', right_id))
        push!(szs, kron(left_id, sm'*sm - sm*sm', right_id))
    end

    return sxs, szs

end




# Step 2: Build Hamiltonian
function build_sparse_hamiltonian(J, alpha, hx, sxs, szs)

    N = length(sxs)
    H = spzeros(Float64, 2^N, 2^N)

    for ii = 1:N
        H += hx .* sxs[ii]
        for jj = (ii+1):N
            H += (J / (jj-ii)^alpha) .* szs[ii] * szs[jj]
        end
    end

    return H

end

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




function sparse_ti_simulation(N, J,alpha,hx,dt, steps, m)

    sm =[0 1; 0 0]
    sxs, szs = build_sparse_spin_ops(sm, N)

    H = build_sparse_hamiltonian(J, alpha, hx, sxs, szs)
    display(H)
    psi = zeros(ComplexF64, 2^N)
    mneel= div(4^cld(N,2)-1,3)+1
    psi[mneel] = 1.0


    out_sz = zeros(steps, N)
    out_szsz = zeros(steps, N-3)



    A=-1im*dt*H
    for tt =1:steps
        for ii =1:N
            out_sz[tt, ii] = real(psi' * szs[ii] * psi)
        end
        for  cc =4:N
            out_szsz[tt, cc-3] = real(psi' * szs[3] * szs[cc] * psi)
            out_szsz[tt, cc-3] -= out_sz[tt,3]*out_sz[tt,cc]
        end
            # psi= arnoldi_exp(A, psi, m)
        psi, _ = exponentiate(A,1.0,psi; krylovdim=m, tol=1e-12)
    end 
    return out_sz, out_szsz

end




N = 20 # number of spins
m=20# dimension of Krylov space
J = 1 # defines energy/time units
alpha = 3 # interaction range
hx = 2# transverse field

dt = 0.1 # time step for plotting
steps =101

@time out_sz, out_szsz =  sparse_ti_simulation(N, J, alpha, hx, dt, steps, m)
# display(out_sz)
cmap = cgrad(:RdBu)
    # default(
    #     tickfontsize = 10, 
    #     labelfontsize = 12, 
    #     fontfamily="times",
    #     colorbar_ticks=-1:0.5:1,
    #     color = cmap,
    #     aspect_ratio=1.2,
    #     dpi=200)

    h = heatmap(1:N-3, 0:dt:((steps-1) * dt),  out_szsz)
    xlims!((0.5, N+0.5))

display(h)
    #savefig("ti_ising.png")
