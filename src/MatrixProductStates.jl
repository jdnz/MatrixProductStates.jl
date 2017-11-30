module MatrixProductStates

export mpsgroundstate, mpsproductstate, mpsrandom
export makempo, makemps, addsiteMPS, steady_state_search!, steady_state_search_two_site!
export applyMPOtoMPO, apply_site_MPOtoMPO, applyMPOtoMPS, build_env
export mps_to_mpo, mpo_to_mps, mps_to_mpo_dens, mpo_to_mps_dens, mpo_to_mps_site
export leftorthonormalizationQR!, leftorthogonalizeQR!, compressionSVD!
export compress_var, compress_var_apply_H, compress_sum_var, RK4stp_apply_H_half, RK4stp_apply_H_half!
export compress_sum_var_apply_H, RK4stp_apply_H, ground_state_search!, compress_sum_var_apply_H!
export Arnoldi_expm!, normalize_lo, compress_var_apply_H!, compress_mpo!, compress_var!
export compress_var_two_site, compress_var_apply_H_two_site, compress_sum_var_two_site
export compress_sum_var_apply_H_two_site, RK4stp_apply_H_half_two_site, RK4stp_apply_H_half_full_two_site
export ground_state_search_full, expandMPS, mpsdims, normMPS
export normalize_lo!, norm_lo, conj_site_mpo, conj_mpo, scal_prod_no_conj, scal_op_prod

using TensorOperations
using LinearMaps
#using CuArrays

import Base.randn
#import CuArrays.CUSOLVER.qrq!

Base.@irrational SQRT_HALF 0.7071067811865475244008  sqrt(big(0.5))

randn(rng::AbstractRNG, ::Type{Complex{T}}) where {T<:AbstractFloat} =
    Complex{T}(SQRT_HALF*randn(rng, T), SQRT_HALF*randn(rng, T))

include("mps_base.jl")
include("constructors.jl")
include("compression.jl")
include("steadystate.jl")
include("groundstate.jl")
include("timeevolution.jl")
include("orthonormalisation.jl")

"""
return a vec of the bond dimensions
"""

function mpsdims(mps)

    na = length(mps)
    dims = ones(Int64, na + 1)
    for jj = 1:na
        dims[jj] = size(mps[jj], 1)
    end

    return dims

end

function diffMP(mps1,mps2)

    n = length(mps2);
    diffm = 0
    for i = 1:n
        diffm += sum(abs2.(mps1[i][:] - mps2[i][:]))
    end
    return diffm
end

"""
make operator product mpo2*mpo1
"""

function applyMPOtoMPO(mpo1,mpo2)

        n = length(mpo2)
        mpoout = similar(mpo1)

        for i = 1:n
            mpoout[i] = apply_site_MPOtoMPO(mpo1[i],mpo2[i])
        end
        return mpoout
end

function applyMPOtoMPO_to(mpo1,mpo2)

        n = length(mpo2)
        mpoout = similar(mpo1)

        for i = 1:n
            mpoout[i] = apply_site_MPOtoMPO_to(mpo1[i],mpo2[i])
        end
        return mpoout
end

function apply_site_MPOtoMPO(mpo1,mpo2)

        (dwl1,d,dwr1,_) = size(mpo1)
        (dwl2,_,dwr2,_) = size(mpo2)
        ai = reshape(mpo2, dwl2*d*dwr2, d)*reshape(permutedims(mpo1, [2, 1, 3, 4]), d, dwl1*dwr1*d)
        ai = reshape(ai, dwl2, d, dwr2, dwl1, dwr1, d)
        ai = permutedims(ai, [4, 1, 2, 5, 3, 6])
        reshape(ai,(dwl1*dwl2,d,dwr1*dwr2,d))

end

function apply_site_MPOtoMPO_to(mpo1,mpo2)

        (dwl1,d,dwr1,_) = size(mpo1)
        (dwl2,_,dwr2,_) = size(mpo2)
        @tensor ai[-1,-2,-3,-5,-4,-6] := mpo1[-1,1,-5,-6]*mpo2[-2,-3,-4,1]
        mpoout = reshape(ai,(dwl1*dwl2,d,dwr1*dwr2,d))

end

"""
applyMPOtoMPS(mpsin,mpo);

Multiplies the MPS A and the MPO U
"""

function applyMPOtoMPS(mpsin,mpo)

    n = length(mpsin)
    mpsout = similar(mpsin)

    for i = 1:n
        mpsout[i] = apply_site_MPOtoMPS(mpsin[i], mpo[i])
    end
    return mpsout
end

function apply_site_MPOtoMPS(ms,mo)

    (dl,_,dr) = size(ms)
    (dwl,d,dwr,_) = size(mo)
    @tensor ai[-1,-2,-3,-5,-4] := ms[-1,1,-5]*mo[-2,-3,-4,1]
    mpsout = reshape(ai,(dl*dwl,d,dr*dwr))

end

"""
product of state and local operator with left and right environments operators
"""

function O_prod_LR(L,O,R)

    @tensor A[-1,-2,-3,-4,-5,-6] := L[-4,1,-1]*O[1,-2,2,-5]*R[-6,2,-3]

end

"""
product of two sites states and local operators with left and right environments operators
"""

function A_prod_LOR_two_site(L,A1,O1,A2,O2,R)

    @tensor A[-1,-2,-3,-4] := L[1,2,-1]*A1[1,3,4]*O1[2,-2,5,3]*A2[4,6,7]*O2[5,-3,8,6]*R[7,8,-4]

end

function prod_AOR_direct(A,O,R)

    a1,a2,a3 = size(A)
    _,r2,r3 = size(R)
    o1 = size(O, 1)
    AOR = complex(zeros(a1, o1, a2, r3))

    @inbounds for r3i in 1:r3, o2i in 1:a2, o1i in 1:o1, a1i in 1:a1, a3i in 1:a3, a2i in 1:a2, r2i in 1:r2
        @fastmath AOR[a1i, o1i, o2i, r3i] += A[a1i, a2i, a3i]*R[a3i, r2i, r3i]*
                                                O[o1i, o2i, r2i, a2i]
    end

    return AOR

end

"""
update right environment for operator
"""

function update_re_large(A0,O,A,FR)

    @tensor FR[-1,-2,-3,-4] := FR[1,2,3,4]*conj(A0[-4,3,4])*A[-1,5,1]*O[-2,-3,2,5]

end


function update_le_large(A0,O,A,FL)

    @tensor FL[-1,-2,-3,-4] := FL[1,2,3,4]*conj(A0[4,3,-4])*A[1,5,-1]*O[2,-3,-2,5]

end


function A_prod_LOR(L,A,O,R)

	@tensor A[-1,-2,-3] := L[1,2,-1]*A[1,3,4]*O[2,-2,5,3]*R[4,5,-3]

end

function A_prod_LOR_test(L,A,O,R)

    al,d,ar = size(A)
    rl,dwr,rr = size(R)
    ll,dwl,lr = size(L)

    A = reshape(A,al*d,ar)*reshape(R,ar,dwr*rr)
    A = permutedims(reshape(A,al,d,dwr,rr),[3,2,1,4])
    A = reshape(O,dwl*d,d*dwr)*reshape(A,d*dwr,rr*al)
    A = permutedims(reshape(A,dwl,d,al,rr),[3,1,2,4])
    A = reshape(L,ll*dwl,lr).'*reshape(A,ll*dwl,rr*d)
    A = reshape(A,lr,d,rr)

end

function A_prod_LOR_test_2(L,A,O,R)

    al,d,ar = size(A)
    rl,dwr,rr = size(R)
    ll,dwl,lr = size(L)

    A = reshape(A,al*d,ar)*reshape(R,ar,dwr*rr)
    A = reshape(L,ll,dwl*lr).'*reshape(A,ll,d*dwr*rr)
    A = permutedims(reshape(A,dwl,lr,d*dwr,rr),[1,3,2,4])
    O = reshape(permutedims(O,[2,1,4,3]),d,dwr*d*dwl)
    A = O*reshape(A,dwr*d*dwl,lr*rr)
    A = permutedims(reshape(A,d,lr,rr),[2,1,3])

end

"""
product of state and local operator with left and right environments operators
designed for use with eigs
"""

function A_prod_LOR_eigs(L,Ain,O,R)

	A = reshape(Ain[:],size(L,1),size(O,4),size(R,1))
	@tensor A[-1,-2,-3] := L[1,2,-1]*A[1,3,4]*O[2,-2,5,3]*R[4,5,-3]
	A[:]

end

"""
product of state and two local operators with left and right environments operators
designed for use with eigs
"""

function A_prod_L_OO_R_eigs(L,Ain,O1,O2,R)

    A = reshape(Ain[:],size(L,1),size(O1,4),size(R,1))
    @tensor A[-1,-2,-3] := L[1,2,3,-1]*A[1,4,6]*O1[2,5,7,4]*O2[3,-2,8,5]*R[6,7,8,-3]
    A[:]

end


function neighbour_contract(A1,A2)

    @tensor A[-1,-2,-3,-4] := A1[-1,-2,1]*A2[1,-3,-4]
    A[:]

end

function A_prod_L_OO_R_two_site(L,Ain,L1,cL1,L2,cL2,R)

    A = reshape(Ain[:],size(L,1),size(L1,4),size(L2,4),size(R,1))
    @tensor A[-1,-2,-3,-4] := L[1,2,3,-1]*A[1,4,5,10]*L1[2,7,6,4]*L2[6,8,11,5]*
                                    cL1[3,-2,9,7]*cL2[9,-3,12,8]*R[10,11,12,-4]
    A[:]

end

"""
product of two local A tensors with left and right environments
"""

function A_prod_LR_two_site(L,A1,A2,R)

    @tensor A[-1,-2,-3,-4] := L[1,-1]*A1[1,-2,2]*A2[2,-3,3]*R[3,-4]

end

"""
update left environment for two operators
"""

function update_lenv_two_op(A0,O1,O2,A,FL)

    @tensor FL[-1,-2,-3,-4] := FL[1,2,3,4]*A[1,5,-1]*O1[2,6,-2,5]*O2[3,7,-3,6]*conj(A0[4,7,-4])

end

"""
update right environment for two operators
"""

function update_renv_two_op(A0,O1,O2,A,FR)

    @tensor FR[-1,-2,-3,-4] := FR[1,2,3,4]*A[-1,5,1]*O1[-2,6,2,5]*O2[-3,7,3,6]*conj(A0[-4,7,4])

end

"""
Compute norm of arbitrary MPS A, no left or right orthogonality assumed
"""
function normMPS(mps)

    abs(scal_prod(mps, mps))

end


"""
Compute sandwcich of operator by mps, <mps1|O|mps2>
"""
function scal_op_prod(mps1, op, mps2)

    n = length(mps1)
    lenv = similar(mps1[1], 1, 1, 1)
    lenv[1] = 1.0

    for i = 1:n
        #Update lenv from A[i], conj(A[i]) and the previous lenv
        lenv = update_le(mps1[i], op[i], mps2[i], lenv)
    end

    return lenv[1]
    
end

"""
Compute scalar product of two mps, <mps1|mps2>
"""
function scal_prod(mps1, mps2)

    na = length(mps1)
    lenv = similar(mps1[1], 1, 1)
    lenv[1] = 1.0

    for n = 1:na
        #Update lenv from A[i], conj(A[i]) and the previous lenv
        lenv = update_lenv(mps1[n], mps2[n], lenv)
    end
    
    return lenv[1]

end

"""
scal_prod(mps1,mps2)

Compute scalar product of two mps, <mps1|mps2> without conjugating mps1
"""

function scal_prod_no_conj(mps1, mps2)

    na = length(mps1)
    lenv = similar(mps1[1], 1, 1)
    lenv[1] = 1.0
    
    for n = 1:na
        #Update lenv from A[i], conj(A[i]) and the previous lenv
        lenv = update_lenv_nc(mps1[n], mps2[n], lenv)
    end
    
    return lenv[1]

end


"""
diff_norm(mps1,mps2)

Compute norm of difference between two mps,<mps1|mps1> + <mps2|mps2> - <mps1|mps2> - <mps2|mps1>
"""

function diff_norm(mps1,mps2)

    normMPS(mps1) + normMPS(mps2) - 2*real(scal_prod(mps1,mps2))

end


function conj_mpo(mpo)

	na = length(mpo)
	mpoconj = similar(mpo)

	for jj = 1:na
    	mpoconj[jj] = conj_site_mpo(mpo[jj])
	end

	return mpoconj

end


function conj_site_mpo(mpo_site)

    mpoconj = permutedims(conj(mpo_site),[1,4,3,2])

end

end
