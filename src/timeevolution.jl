
"""
function to find MPS approximation of exp(A)v using the Arnoldi method with 
numv Krylov space vectors
"""

function Arnoldi_expm(A, v, numv)

    rhov = Array{Array{Array{Complex{Float64}, 3}, 1}, 1}(numv)
    vprod = Array{Array{Array{Complex{Float64}, 3}, 1}, 1}(numv)
    N = complex(zeros(numv, numv))
    H = complex(zeros(numv, numv))

    rhov[1] = v
    vprod[1] = applyMPOtoMPS(rhov[1], A)
    H[1] = scal_prod(rhov[1], vprod[1])
    N[1] = scal_prod(rhov[1], rhov[1])

    coef = complex(zeros(numv))

    for jj = 2:numv

        coef[1:jj - 1] = [-H[ii, jj - 1]/N[ii, ii] for ii in 1:jj - 1]
        rhov[jj] = compress_sum_var(v, [1.0, coef[1:jj - 1]...],
                                                  [vprod[jj - 1], rhov[1:jj - 1]...], 4)
        vprod[jj] = applyMPOtoMPS(rhov[jj], A)
        H[jj, jj] = scal_prod(rhov[jj], vprod[jj])
        N[jj, jj] = scal_prod(rhov[jj], rhov[jj])

        for ll = 1:jj - 1

            H[jj, ll] = scal_prod(rhov[jj], vprod[ll])
            H[ll, jj] = scal_prod(rhov[ll], vprod[jj])
            N[jj, ll] = scal_prod(rhov[jj], rhov[ll])
            N[ll, jj] = conj(N[jj, ll])

        end

    end
    println(H)
    println(diag(N))
    coef = expm(N\H)[:, 1]
    println(coef)
    return compress_sum_var(v, coef, rhov, 4)

end

function Arnoldi_expm_norm(A, v, numv)

    rhov = Array{Array{Array{Complex{Float64}, 3}, 1}, 1}(numv)
    vprod = Array{Array{Array{Complex{Float64}, 3}, 1}, 1}(numv)
    N = complex(eye(numv))
    H = complex(zeros(numv, numv))

    rhov[1] = deepcopy(v)
    v_norm = normalize_lo!(rhov[1])
    vprod[1] = applyMPOtoMPS(rhov[1], A)
    H[1] = scal_prod(rhov[1], vprod[1])

    coef = complex(zeros(numv))

    for jj = 2:numv

        coef[1:jj - 1] = [-H[ii, jj - 1]/N[ii, ii] for ii in 1:jj - 1]
        rhov[jj] = compress_sum_var(v, [1.0, coef[1:jj - 1]...],
                                                  [vprod[jj - 1], rhov[1:jj - 1]...], 1)
        no = normalize_lo!(rhov[jj])
        println(no)
        vprod[jj] = applyMPOtoMPS(rhov[jj], A)
        H[jj, jj] = scal_prod(rhov[jj], vprod[jj])

        for ll = 1:jj - 1

            H[jj, ll] = scal_prod(rhov[jj], vprod[ll])
            H[ll, jj] = scal_prod(rhov[ll], vprod[jj])
            N[jj, ll] = scal_prod(rhov[jj], rhov[ll])
            N[ll, jj] = conj(N[jj, ll])

        end

    end
    coef = sqrt(v_norm)*expm(N\H)[:, 1]
    println(coef)
    return compress_sum_var(v, coef, rhov, 1)

end

function Arnoldi_expm!(A, rhov, numv, env, envop)

    N = complex(eye(numv))
    H = complex(zeros(numv, numv))

    H[1, 1] = scal_op_prod(rhov[1], A, rhov[1])

    for jj = 2:numv

        compress_sum_var_apply_H!(rhov[jj], envop, 
                                    [- H[1:(jj - 1), jj - 1]..., 1.0],
                                        [rhov[1:(jj - 1)]..., rhov[jj - 1]], A, 1)
        normalize_lo!(rhov[jj])
        H[jj, jj] = scal_op_prod(rhov[jj], A, rhov[jj])

        for ll = 1:jj - 1

            H[jj, ll] = scal_op_prod(rhov[jj], A, rhov[ll])
            H[ll, jj] = scal_op_prod(rhov[ll], A, rhov[jj])
            N[jj, ll] = scal_prod(rhov[jj], rhov[ll])
            N[ll, jj] = conj(N[jj, ll])

        end

    end
    coef = expm(N\H)[:, 1]
    println(abs.(coef))
    rhov[1] = compress_sum_var(rhov[1], coef, rhov, 1)
    normalize_lo!(rhov[1])
    nothing

end
"""
function to evolve MPS using the 4th order Runga Kutta method
"""

function RK4stp(dt,H,psi,k0)

k1 = compress_var(k0,applyMPOtoMPS(psi,H[1]),1)
k2 = compress_var(k1,applyMPOtoMPS(
    compress_sum_var(psi,[1,dt/2],[psi,k1],1),H[2]),1)
k3 = compress_var(k2,applyMPOtoMPS(
    compress_sum_var(psi,[1,dt/2],[psi,k2],1),H[2]),1)
k4 = compress_var(k3,applyMPOtoMPS(
    compress_sum_var(psi,[1,dt],[psi,k3],1),H[3]),1)

psi = compress_sum_var(psi,[1,dt/6,dt/3,dt/3,dt/6],[psi,k1,k2,k3,k4],1)
return psi,k4

end

"""
function to evolve MPS using the 4th order Runga Kutta method
"""

function RK4stp_alt(dt,H,psi)

psi1 = compress_sum_var(psi,[1,dt/2],[psi,applyMPOtoMPS(psi,H[1])],1)
psi2 = compress_sum_var(psi1,[1,dt/2],[psi,applyMPOtoMPS(psi1,H[2])],1)
psi3 = compress_sum_var(psi2,[1,dt],[psi,applyMPOtoMPS(psi2,H[2])],1)
psi = compress_sum_var(psi3,[-1/3,1/3,2/3,1/3,dt/6],[psi,psi1,psi2,psi3,
                       applyMPOtoMPS(psi3,H[3])],1)

end


"""
function to evolve MPS using the 4th order Runga Kutta method
"""

function RK4stp_apply_H(dt,H,psi)

psi1 = compress_var_apply_H(psi,psi,H[1],1)
psi2 = compress_sum_var_apply_H(psi1,[1,dt/2],[psi,psi1],H[2],1)
psi3 = compress_sum_var_apply_H(psi2,[1,dt],[psi,psi2],H[2],1)
psi = compress_sum_var_apply_H(psi3,[-1/3,1/3,2/3,1/3],[psi,psi1,psi2,
                               psi3],H[3],1)

end

function RK4stp_apply_H_half(dt, H, psi)

psi1 = compress_var_apply_H_left(psi, psi, H[1])
psi2 = compress_sum_var_apply_H_right(psi1, [1, dt/2], [psi, psi1], H[2])
psi3 = compress_sum_var_apply_H_left(psi2, [1, dt], [psi, psi2], H[2])
compress_sum_var_apply_H_right(psi3, [-1/3, 1/3, 2/3, 1/3], 
                                           [psi, psi1, psi2, psi3], H[3])

end

function RK4stp_apply_H_half!(dt, H, psi, psi1, psi2, psi3, env1, env2, env3, envop)

compress_var_apply_H_left!(psi1, envop, psi, psi, H[1])
compress_sum_var_apply_H_right!(psi2, [env1], envop, psi1, [1, dt/2], [psi, psi1], H[2])
compress_sum_var_apply_H_left!(psi3, [env1], envop, psi2, [1, dt], [psi, psi2], H[2])
compress_sum_var_apply_H_right!(psi, [env1, env2, env3], envop, psi3, [-1/3, 1/3, 2/3, 1/3],
                                [psi, psi1, psi2, psi3], H[3])

end

function RK4stp_apply_H_half!(dt, H, psi0, psi, psi1, psi2, psi3, env1, env2, env3, envop)

compress_var_apply_H_left!(psi1, envop, psi, psi, H[1])
compress_sum_var_apply_H_right!(psi2, [env1], envop, psi1, [1, dt/2], [psi, psi1], H[2])
compress_sum_var_apply_H_left!(psi3, [env1], envop, psi2, [1, dt], [psi, psi2], H[2])
compress_sum_var_apply_H_right!(psi0, [env1, env2, env3], envop, psi3, [-1/3, 1/3, 2/3, 1/3],
                                [psi, psi1, psi2, psi3], H[3])

end

function RK4stp_apply_H_half_two_site(dt,H,psi,svd_tol,dims,dlim)

psi1 = compress_var_apply_H_two_site_left(psi,psi,H[1],1,svd_tol,dims,dlim)
psi2 = compress_sum_var_apply_H_right(psi1,[1,dt/2],[psi,psi1],H[2],1)
psi3 = compress_sum_var_apply_H_left(psi2,[1,dt],[psi,psi2],H[2],1)
psi = compress_sum_var_apply_H_right(psi3,[-1/3,1/3,2/3,1/3],[psi,psi1,psi2,
                               psi3],H[3],1)

end

function RK4stp_apply_H_half_full_two_site(dt,H,psi,svd_tol,dims,dlim)

psi1 = compress_var_apply_H_two_site_left(psi,psi,H[1],1,svd_tol,dims,dlim)
psi2 = compress_sum_var_apply_H_two_site_right(psi1,[1,dt/2],[psi,psi1],H[2],1,svd_tol,dims,dlim)
psi3 = compress_sum_var_apply_H_two_site_left(psi2,[1,dt],[psi,psi2],H[2],1,svd_tol,dims,dlim)
psi = compress_sum_var_apply_H_two_site_right(psi3,[-1/3,1/3,2/3,1/3],[psi,psi1,psi2,
                               psi3],H[3],1,svd_tol,dims,dlim)

end

function RK4stp_apply_H_half_two_site_2(dt,H,psi,svd_tol,dims,dlim)

psi1 = compress_var_apply_H_left(psi,psi,H[1],1)
psi2 = compress_sum_var_apply_H_right(psi1,[1,dt/2],[psi,psi1],H[2],1)
psi3 = compress_sum_var_apply_H_left(psi2,[1,dt],[psi,psi2],H[2],1)
psi = compress_sum_var_apply_H_two_site_right(psi3,[-1/3,1/3,2/3,1/3],[psi,psi1,psi2,
                               psi3],H[3],1,svd_tol,dims,dlim)

end



"""
function to evolve MPS A (must be in left canonical form)
in time acording to the trapezoidal method
"""

function Evolve_trap(A,LF,LB,maxsweeps,nor)

	na = length(A)
	At = deepcopy(A)
	FL = Array{Array}(na)
	FR = Array{Array}(na)
	SL = Array{Array}(na)
	SR = Array{Array}(na)
	XY = applyMPOtoMPO(LF,Conj_MPO(LB))
	XX = applyMPOtoMPO(LB,Conj_MPO(LB))

	FL[1] = ones(1,1,1)
	SL[1] = ones(1,1,1)
	for jj = 1:na-1
		FL[jj+1] = update_le(At[jj],XY[jj],A[jj],FL[jj])
		SL[jj+1] = update_le(At[jj],XX[jj],At[jj],SL[jj])
	end
	FR[na] = ones(1,1,1)
	SR[na] = ones(1,1,1)

	for sweep = 1:maxsweeps

		for jj = na:-1:2
		    HH = reshape(O_prod_LR(SL[jj],XX[jj],SR[jj]),
		        size(SL[jj],1)*size(XX[jj],2)*size(SR[jj],1),
		        size(SL[jj],1)*size(XX[jj],2)*size(SR[jj],1))
		    HE = reshape(A_prod_LOR(FL[jj],A[jj],XY[jj],FR[jj]),
		        size(SL[jj],1)*size(XX[jj],2)*size(SR[jj],1),1)
		    At[jj] = reshape(HH\HE,size(SL[jj],1),size(XX[jj],2),size(SR[jj],1))
		    Q,R = qr(reshape(At[jj],size(At[jj],1),size(At[jj],3)*size(At[jj],2))')
		    At[jj] = reshape(Q',size(Q,2),size(At[jj],2),size(At[jj],3))
		    FR[jj-1] = update_re(At[jj],XY[jj],A[jj],FR[jj])
		    SR[jj-1] = update_re(At[jj],XX[jj],At[jj],SR[jj])
		end
	    HH = reshape(O_prod_LR(SL[1],XX[1],SR[1]),
	        size(SL[1],1)*size(XX[1],2)*size(SR[1],1),
	        size(SL[1],1)*size(XX[1],2)*size(SR[1],1))
	    HE = reshape(A_prod_LOR(FL[1],A[1],XY[1],FR[1]),
	        size(SL[1],1)*size(XX[1],2)*size(SR[1],1),1)
	    At[1] = reshape(HH\HE,size(SL[1],1),size(XX[1],2),size(SR[1],1));


		for jj = 1:na-1;
		    HH = reshape(O_prod_LR(SL[jj],XX[jj],SR[jj]),
		        size(SL[jj],1)*size(XX[jj],2)*size(SR[jj],1),
		        size(SL[jj],1)*size(XX[jj],2)*size(SR[jj],1))
		    HE = reshape(A_prod_LOR(FL[jj],A[jj],XY[jj],FR[jj]),
		        size(SL[jj],1)*size(XX[jj],2)*size(SR[jj],1),1)
		    At[jj] = reshape(HH\HE,size(SL[jj],1),size(XX[jj],2),size(SR[jj],1));
		    Q,R = qr(reshape(At[jj],size(At[jj],1)*size(At[jj],2),size(At[jj],3)));
		    At[jj] = reshape(Q,size(At[jj],1),size(At[jj],2),size(Q,2))
		    FL[jj+1] = update_le(At[jj],XY[jj],A[jj],FL[jj]);
		    SL[jj+1] = update_le(At[jj],XX[jj],At[jj],SL[jj]);
		end
	    HH = reshape(O_prod_LR(SL[na],XX[na],SR[na]),
	        size(SL[na],1)*size(XX[na],2)*size(SR[na],1),
	        size(SL[na],1)*size(XX[na],2)*size(SR[na],1))
	    HE = reshape(A_prod_LOR(FL[na],A[na],XY[na],FR[na]),
	        size(SL[na],1)*size(XX[na],2)*size(SR[na],1),1)
	    At[na] = reshape(HH\HE,size(SL[na],1),size(XX[na],2),size(SR[na],1))

	end

	no = sum(abs2(At[na][:]))

	if nor
	    At[na] = At[na]./sqrt(no)
	end

	return At, no

end