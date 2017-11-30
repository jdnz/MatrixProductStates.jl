"""
Base routines used in higher level algorithms implemeted using the 
TensorOperations package

product of site tensor A with local operator O and right environment R
"""
function prod_AOR(A::AbstractArray{<:Number, 3}, O::AbstractArray{<:Number, 4},
    R::AbstractArray{<:Number, 3})

    a1, a2, a3 = size(A)
    _, r2, r3 = size(R)
    o1 = size(O, 1)

    AOR = reshape(A, a1*a2, a3)*reshape(R, a3, r2*r3)
    AOR = permutedims(reshape(AOR, a1, a2, r2, r3), [3, 2, 1, 4])
    AOR = reshape(O, o1*a2, r2*a2)*reshape(AOR, r2*a2, a1*r3)
    permutedims(reshape(AOR, o1, a2, a1, r3), [3, 1, 2, 4])

end

"""
product of site tensor A with local operator O and left environment L
"""


function prod_AOL(A::AbstractArray{<:Number, 3}, O::AbstractArray{<:Number, 4},
    L::AbstractArray{<:Number, 3})

    a1, d, a3 = size(A)
    _, l2, l3 = size(L)
    o3 = size(O, 3)

    AOL = reshape(L, a1, l2*l3)'*reshape(A, a1, d*a3)
    AOL = permutedims(reshape(AOL, l2, l3, d, a3), [2, 4, 1, 3])
    AOL = reshape(AOL, l3*a3, l2*d)
    AOL = AOL*reshape(permutedims(O, [1, 4, 2, 3]), l2*d, d*o3)
    permutedims(reshape(AOL, l3, a3, d, o3), [1, 3, 2, 4])

end


"""
product of left environment and AOR from prod_AOR
"""
function prod_LR(L::AbstractArray{<:Number, 3}, R::AbstractArray{<:Number, 4})

    r1, r2, r3, r4 = size(R)
    l3 = size(L, 3)
    reshape(reshape(L, r1*r2, l3)'*reshape(R, r1*r2, r3*r4), l3, r3, r4)

end

function prod_LR!(A0, L::AbstractArray{<:Number, 3}, R::AbstractArray{<:Number, 4})

    r1, r2, r3, r4 = size(R)
    l3 = size(L, 3)
    A0 = reshape(A0, l3, r3*r4)
    Ac_mul_B!(A0, reshape(L, r1*r2, l3), reshape(R, r1*r2, r3*r4))
    #A0 = reshape(A0, l3, r3, r4)
    #nothing

end
"""
product of right environment and AOL from prod_AOL
"""
function prod_LR(L::AbstractArray{<:Number, 4}, R::AbstractArray{<:Number, 3})

    l1, l2, l3, l4 = size(L)
    r3 = size(R, 3)
    reshape(reshape(L, l1*l2, l3*l4)*reshape(R, l3*l4, r3), l1, l2, r3)

end

function prod_LR!(A0, L::AbstractArray{<:Number, 4}, R::AbstractArray{<:Number, 3})

    l1, l2, l3, l4 = size(L)
    r3 = size(R, 3)
    A0 = reshape(A0, l1*l2, r3)
    A_mul_B!(A0, reshape(L, l1*l2, l3*l4), reshape(R, l3*l4, r3))
    #A0 = reshape(A0, l1, l2, r3)
    #nothing

end
"""
product of left environment and larger right environment or site mps
"""
function prod_LR(L::AbstractArray{<:Number, 2}, R::AbstractArray{<:Number, 3})

    r1, r2, r3 = size(R)
    reshape(L*reshape(R, r1, r2*r3), size(L, 1), r2, r3)

end

function prod_LR_co(L::AbstractArray{<:Number, 2}, R::AbstractArray{<:Number, 3})

    r1, r2, r3 = size(R)
    reshape(L.'*reshape(R, r1, r2*r3), size(L, 2), r2, r3)

end


function prod_LR!(A0, L::AbstractArray{<:Number, 2}, R::AbstractArray{<:Number, 3})

    r1, r2, r3 = size(R)
    l1 = size(L, 1)
    A0 = reshape(A0, l1, r2*r3)
    A_mul_B!(A0, L, reshape(R, r1, r2*r3))
    #A0 = reshape(A0, l1, r2, r3)
    #nothing

end

function prod_LR_co!(A0, L::AbstractArray{<:Number, 2}, R::AbstractArray{<:Number, 3})

    r1, r2, r3 = size(R)
    l2 = size(L, 2)
    A0 = reshape(A0, l2, r2*r3)
    At_mul_B!(A0, L, reshape(R, r1, r2*r3))
    #A0 = reshape(A0, l2, r2, r3)
    #nothing

end
"""
product of right environment and larger left environment or site mps
"""
function prod_LR(L::AbstractArray{<:Number, 3}, R::AbstractArray{<:Number, 2})

    l1, l2, l3 = size(L)
    reshape(reshape(L, l1*l2, l3)*R, l1, l2, size(R, 2))

end

function prod_LR!(A0, L::AbstractArray{<:Number, 3}, R::AbstractArray{<:Number, 2})

    l1, l2, l3 = size(L)
    r2 = size(R, 2)
    A0 = reshape(A0, l1*l2, r2)
    A_mul_B!(A0, reshape(L, l1*l2, l3), R)
    #A0 = reshape(A0, l1, l2, r2)
    #nothing

end

"""
update right environment from AOR and site tensor A0
"""
function update_re(A0::AbstractArray{<:Number, 3},
    AOR::AbstractArray{<:Number, 4})

    r1, r2, r3, r4 = size(AOR)
    a1 = size(A0, 1)
    reshape(reshape(AOR, r1*r2, r3*r4)*reshape(A0, a1, r3*r4)', r1, r2, a1)

end

function update_re!(env, A0::AbstractArray{<:Number, 3},
    AOR::AbstractArray{<:Number, 4})

    r1, r2, r3, r4 = size(AOR)
    a1 = size(A0, 1)
    env = reshape(env, r1*r2, a1)
    A_mul_Bc!(env, reshape(AOR, r1*r2, r3*r4), reshape(A0, a1, r3*r4))
    #env = reshape(env, r1, r2, a1)
    #nothing

end
"""
update right environment from A, O and A0
"""
function update_re(A0::AbstractArray{<:Number, 3},
    O::AbstractArray{<:Number, 4}, A::AbstractArray{<:Number, 3},
    FR::AbstractArray{<:Number, 3})

    AOR = prod_AOR(A, O, FR)
    update_re(A0, AOR)

end

function update_re!(env, A0::AbstractArray{<:Number, 3},
    O::AbstractArray{<:Number, 4}, A::AbstractArray{<:Number, 3},
    FR::AbstractArray{<:Number, 3})

    AOR = prod_AOR(A, O, FR)
    update_re!(env, A0, AOR)
    #nothing

end

"""
update left environment from AR and site tensor A0
"""
function update_renv(A0::AbstractArray{<:Number, 3},
    AR::AbstractArray{<:Number, 3})

    a1, d, a2 = size(A0)
    reshape(AR, size(AR, 1), d*a2)*reshape(A0, a1, d*a2)'

end

function update_renv!(env, A0::AbstractArray{<:Number, 3},
    AR::AbstractArray{<:Number, 3})

    a1, d, a2 = size(A0)
    A_mul_Bc!(env, reshape(AR, size(AR, 1), d*a2), reshape(A0, a1, d*a2))
    #nothing

end

"""
update right environment from A and A0
"""
function update_renv(A0::AbstractArray{<:Number, 3}, 
    A::AbstractArray{<:Number, 3}, FR::AbstractArray{<:Number, 2})

    AR = prod_LR(A, FR)
    update_renv(A0, AR)

end

function update_renv!(env, A0::AbstractArray{<:Number, 3}, 
    A::AbstractArray{<:Number, 3}, FR::AbstractArray{<:Number, 2})

    AR = prod_LR(A, FR)
    update_renv!(env, A0, AR)
    #nothing

end
"""
update left environment from AOL and site tensor A0, in the case the left
environment is conjugated
"""
function update_le(A0::AbstractArray{<:Number, 3},
    AOL::AbstractArray{<:Number, 4})

    l1, l2, l3, l4 = size(AOL)
    a3 = size(A0, 3)
    reshape(reshape(AOL, l1*l2, l3*l4)'*reshape(A0, l1*l2, a3), l3, l4, a3)

end

function update_le!(env, A0::AbstractArray{<:Number, 3},
    AOL::AbstractArray{<:Number, 4})

    l1, l2, l3, l4 = size(AOL)
    a3 = size(A0, 3)
    env = reshape(env, l3*l4, a3)
    Ac_mul_B!(env, reshape(AOL, l1*l2, l3*l4), reshape(A0, l1*l2, a3))
    #env = reshape(env, l3, l4, a3)
    #nothing

end

"""
update left environment from A, O and A0
"""
function update_le(A0::AbstractArray{<:Number, 3},
    O::AbstractArray{<:Number, 4}, A::AbstractArray{<:Number, 3},
    FL::AbstractArray{<:Number, 3})

    AOL = prod_AOL(A, O, FL)
    update_le(A0, AOL)

end

function update_le!(env, A0::AbstractArray{<:Number, 3},
    O::AbstractArray{<:Number, 4}, A::AbstractArray{<:Number, 3},
    FL::AbstractArray{<:Number, 3})

    AOL = prod_AOL(A, O, FL)
    update_le!(env, A0, AOL)
    #nothing

end

"""
update left environment from AL and site tensor A0
"""
function update_lenv(A0::AbstractArray{<:Number, 3},
    AL::AbstractArray{<:Number, 3})

    a1, d, a3 = size(A0)
    reshape(A0, a1*d, a3)'*reshape(AL, a1*d, size(AL, 3))

end

function update_lenv!(env, A0::AbstractArray{<:Number, 3},
    AL::AbstractArray{<:Number, 3})

    a1, d, a3 = size(A0)
    Ac_mul_B!(env, reshape(A0, a1*d, a3), reshape(AL, a1*d, size(AL, 3)))
    #nothing

end

function update_lenv_co!(env, A0::AbstractArray{<:Number, 3},
    AL::AbstractArray{<:Number, 3})

    a1, d, a3 = size(A0)
    At_mul_B!(env, reshape(AL, a1*d, size(AL, 3)), reshape(conj(A0), a1*d, a3))
    #nothing

end
"""
update left environment from A, and A0
"""
function update_lenv(A0::AbstractArray{<:Number, 3},
    A::AbstractArray{<:Number, 3}, L::AbstractArray{<:Number, 2})

    AL = prod_LR(L, A)
    update_lenv(A0, AL)

end

function update_lenv!(env, A0::AbstractArray{<:Number, 3},
    A::AbstractArray{<:Number, 3}, L::AbstractArray{<:Number, 2})

    AL = prod_LR(L, A)
    update_lenv!(env, A0, AL)
    #nothing

end

function update_lenv_co!(env, A0::AbstractArray{<:Number, 3},
    A::AbstractArray{<:Number, 3}, L::AbstractArray{<:Number, 2})

    AL = prod_LR_co(L, A)
    update_lenv_co!(env, A0, AL)
    #nothing

end
"""
update left environment from AL without conjugation of site tensor A0
"""
function update_lenv_nc(A0::AbstractArray{<:Number, 3},
    AL::AbstractArray{<:Number, 3})

    a1, d, a3 = size(A0)
    reshape(A0, a1*d, a3).'*reshape(AL, a1*d, size(AL, 3))

end

"""
update left environment from A without conjugation of A0
"""
function update_lenv_nc(A0::AbstractArray{<:Number, 3},
    A::AbstractArray{<:Number, 3}, L::AbstractArray{<:Number, 2})

    AL = prod_LR(L, A)
    update_lenv_nc(A0, AL)

end

"""
local operator expectation value assuming mixed conjugation
"""
function local_exp(A0, A, O)

    a1, d, a3 = size(A)
    Ap = reshape(permutedims(A, [2, 1, 3]), d, a1*a3)
    A0p = reshape(permutedims(A0, [2, 1, 3]), d, a1*a3)
    Ap = Ap*A0p'
    trace(O*Ap)

end
