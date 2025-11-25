using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit
using Random

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

    return complex.(sms)

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

    return complex.(sxs), complex.(szs)

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


function sparse_ti_simulation(N, J,alpha,hx,gamma,dt, steps,nt, m)
    rng = MersenneTwister()
    sm =[0 1; 0 0]
    sms=build_lowering_ops(sm, N)
    sxs, szs = build_sparse_spin_ops(sm, N)

    psi = zeros(ComplexF64, 2^N)
    mneel= div(4^cld(N,2)-1,3)+1
    psi[mneel] = 1.0000
    y = (psi*psi')[:]

    H = build_sparse_hamiltonian(J, alpha, hx, sxs, szs)
    Heff= H
    Heff = complex.(Heff)
    # sm .*sqrt(gamma/2)
    for ii = 1: N 
        Heff -= 1im .*(gamma/2) .* (sms[ii]' * sms[ii])
    end
     
    Ls= sqrt(gamma/2) .* sms
    out_sz=zeros(steps,N)
    mneel= div(4^cld(N,2)-1,3)+1
    for qq= 1:nt
        psi = zeros(ComplexF64, 2^N)
        psi[mneel] = 1.0

        traj_sz = zeros(steps, N)
        for tt =1:steps
            for ii =1:N
                traj_sz[tt, ii] = real(psi' * szs[ii] * psi)
            end 
            #QT evolution
            psi = ti_trajectory_step(psi, Heff, Ls, dt, rng, m)
        end

        out_sz += traj_sz 

    end
    return out_sz ./= nt
end


function ti_trajectory_step(psi, Heff, Ls, dt, rng, m)
    psi0, _ = exponentiate(Heff, -1im .* dt, psi; krylovdim=m, tol=1e-15)
    nrm_psi0 = norm(psi0)
    pc = nrm_psi0^2

    pt = rand(rng)
    if pt > real(pc)
        s = 0
        for ee = 1:(length(Ls)-1)
            p = norm(Ls[ee] * psi)
            @show p
            s += p
            if s > pt
                psi0 = Ls[ee] * psi
                return psi0 ./= sqrt(norm(psi0))
            end
        end
        psi0 = Ls[end] * psi
        return psi0 ./= sqrt(norm(psi0))
    else
        return psi0 ./= nrm_psi0
    end
end


#     # A = spzeros(Float64, 4^N, 4^N)
#     Id = sparse(I, 2^N, 2^N)
#     Id = complex.(Id)
#     A = -1im .*(kron(Id, Heff) - kron(conj.(Heff), Id))
#     for ii = 1:N
#         A += gamma .* kron(conj.(sms[ii]), sms[ii])
#     end

#     out_sz=zeros(steps, N )

#     for tt= 1: steps
#         #evaluation 
#         rho= reshape(y, 2^N, 2^N)
#         for ii = 1:N
#             out_sz[tt, ii] = real(tr(rho * szs[ii]))
#         end
#         y, _ = exponentiate(A,dt, y; krylovdim=m, tol=1e-12)
#         println("Step $tt/$steps - norm(rho) = $(real(tr(reshape(y, 2^N, 2^N))))")
#     end

#     return out_sz
# end




function main()
    Plots.closeall()  
    N = 7 # number of spins
    m=20# dimension of Krylov space
    J = 1 # defines energy/time units
    alpha = 1.36 # interaction range
    hx = 1# transverse field
    gamma= 0.2
    dt = 0.01 # time step for plotting
    steps =1001
    nt=1
    @time out_sz =  sparse_ti_simulation(N, J, alpha, hx, gamma, dt, steps,nt, m)
    tran= 0:dt:(steps-1)*dt
    plot(tran, out_sz[:,3], label="⟨Sz⟩ of spin 3", title="Transverse Ising model dynamics")
    plot!(tran, out_sz[:,4], label="⟨Sz⟩ of spin 4", title="Transverse Ising model dynamics")
    display(current())
    cmap = cgrad(:RdBu)
        # default(
        #     tickfontsize = 10, 
        #     labelfontsize = 12, 
        #     fontfamily="times",
        #     colorbar_ticks=-1:0.5:1,
        #     color = cmap,
        #     aspect_ratio=1.2,
        #     dpi=200)

        h = heatmap(1:N, 0:dt:((steps-1) * dt),  out_sz)
        xlims!((0.5, N+0.5))
        # xlabel!(L"i")
        # ylabel!(L"tJ")

        display(h)

    return nothing
end

main()


