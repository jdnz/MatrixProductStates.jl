"""
make product state MPS in left orthonomal form
"""

mpsproductstate(na, dmax, d, state::Array) = 
    mpsproductstate(Complex{Float64}, Array, na, dmax, d, state)

function mpsproductstate(::Type{TN}, ::Type{TA}, na, dmax, d, state::Array) where {TN, TA}

    mps = Array{TA{TN, 3}, 1}(na)
    dims = ones(Int64, na + 1)
    dims[2:na] = dmax
    r = ones(TN, 1, 1)

    for n = 1:(na - 1)
        
        dims[n + 1] = min(dims[n]*d, dims[n + 1])
        dims[na + 1 - n] = min(dims[na + 2 - n]*d, dims[na + 1 - n])
        mps_site = zeros(TN, dims[n], d, dims[n + 1])
        mps_site[1, :, 1] = state
        mps_site = prod_LR(r, mps_site)
        mps[n], r = leftorth(mps_site)

    end

    mps_site = zeros(TN, dims[na], d, dims[na + 1])
    mps_site[1, :, 1] = state
    mps[na] = prod_LR(r, mps_site)

    return mps, dims

end

mpsproductstate(na, dmax, d, state::Function) = 
    mpsproductstate(Complex{Float64}, Array, na, dmax, d, state)

function mpsproductstate(::Type{TN}, ::Type{TA}, na, dmax, d, state::Function) where {TN, TA}

    mps = Array{TA{TN, 3}, 1}(na)
    dims = ones(Int64, na + 1)
    dims[2:na] = dmax
    r = ones(TN, 1, 1)

    for n = 1:(na - 1)
        
        dims[n + 1] = min(dims[n]*d, dims[n + 1])
        dims[na + 1 - n] = min(dims[na + 2 - n]*d, dims[na + 1 - n])
        mps_site = zeros(TN, dims[n], d, dims[n + 1])
        mps_site[1, :, 1] = state(n)
        mps_site = prod_LR(r, mps_site)
        mps[n], r = leftorth(mps_site)

    end

    mps_site = zeros(TN, dims[na], d, dims[na + 1])
    mps_site[1, :, 1] = state(na)
    mps[na] = prod_LR(r, mps_site)

    return mps, dims

end

"""
make groundstate MPS in left orthonomal form (also has trace 1 for density matrix)
"""

mpsgroundstate(na, dmax, d) = mpsgroundstate(Complex{Float64}, Array, na, dmax, d)

function mpsgroundstate(::Type{TN}, ::Type{TA}, na, dmax, d) where {TN, TA}

    state = zeros(TN, d)
    state[1] = one(TN)
    mpsproductstate(TN, TA, na, dmax, d, state)

end


mpsgroundstate(na, dmax, d::Array) = 
    mpsgroundstate(Complex{Float64}, Array, na, dmax, d)

function mpsgroundstate(::Type{TN}, ::Type{TA}, na, dmax, d::Array) where {TN, TA}

    mps = Array{TA{TN, 3}, 1}(na)
    dims = ones(Int64, na + 1)
    dims[2:na] = dmax
    r = ones(TN, 1, 1)

    for n = 1:(na - 1)
        
        dims[n + 1] = min(dims[n]*d[n], dims[n + 1])
        dims[na + 1 - n] = min(dims[na + 2 - n]*d[na + 1 - n], dims[na + 1 - n])
        mps_site = zeros(TN, dims[n], d[n], dims[n + 1])
        mps_site[1, 1, 1] = one(TN)
        mps_site = prod_LR(r, mps_site)
        mps[n], r = leftorth(mps_site)

    end

    mps_site = zeros(TN, dims[na], d[na], dims[na + 1])
    mps_site[1, 1, 1] = one(TN)
    mps[na] = prod_LR(r, mps_site)

    return mps, dims

end

"""
make random left orthonormalized MPS
"""

mpsrandom(na, dmax, d) = mpsrandom(Complex{Float64}, Array, na, dmax, d)

function mpsrandom(::Type{TN}, ::Type{TA}, na, dmax, d) where {TN, TA}

    mps = Array{TA{TN, 3}, 1}(na)
    dims = ones(Int64, na + 1)
    dims[2:na] = dmax

    for n = 1:na
  
        dims[n + 1] = min(dims[n]*d, dims[n + 1])
        dims[na + 1 - n] = min(dims[na + 2 - n]*d, dims[na + 1 - n])
        mps[n] = leftorth!(randn(TN, dims[n], d, dims[n + 1]))

    end

return mps, dims

end

"""
expand MPS to larger bond dimension
"""
function expandmps(::Type{TN}, ::Type{TA}, mps, dmax) where {TN, TA}

    na = size(mps, 1)
    d = size(mps[1], 2)
    mpsout = similar(mps, na)
    dims = dmax*ones(Int16, na + 1)
    dims[1] = 1
    dims[na + 1] = 1
    r = ones(TN, 1, 1)

    for n = 1:(na - 1)
        
        dims[n + 1] = min(dims[n]*d, dims[n + 1])
        dims[na + 1 - n] = min(dims[na + 2 - n]*d, dims[na + 1 - n])
        mps_site = zeros(TN, dims[n], d, dims[n + 1])
        mps_site[1:size(mps[n], 1), :, 1:size(mps[n], 3)] = mps[n]
        mps_site = prod_LR(r, mps_site)
        mpsout[n], r = leftorth(mps_site)

    end

    mps_site = zeros(TN, dims[na], d, dims[na + 1])
    mps_site[1:size(mps[na], 1), :, 1:size(mps[na], 3)] = mps[na]
    mps_site = prod_LR(r, mps_site)
    mpsout[na] = mps_site

    return mpsout, dims

end

"""
add additional site to MPS
"""
function addsitemps(::Type{TN}, ::Type{TA}, mps, dmax) where {TN, TA}

    na = size(mps, 1) + 1
    d = size(mps[1], 2)
    mpsout = similar(mps, na)
    dims = dmax*ones(Int16, na + 1)
    dims[1] = 1
    dims[na + 1] = 1
    r = ones(TN, 1, 1)

    for n = 1:(na - 1)

        dims[n + 1] = min(dims[n]*d, dims[n + 1])
        dims[na + 1 - n] = min(dims[na + 2 - n]*d, dims[na + 1 - n])
        mps_site = zeros(T, dims[n], d, dims[n + 1])
        mps_site[1:size(mps[n], 1), :, 1:size(mps[n], 3)] = mps[n]
        mps_site = prod_LR(r, mps_site)
        mpsout[n], r = leftorth(mps_site)

    end

    mpsout[na] = leftorth!(randn(TN, dl, d, 1)+im*randn(TN, dl, d, 1))

    return mpsout, dims

end

"""
make MPS operator from input matrix
"""

makemps(mpsmat, na, d) = makemps(Complex{Float64}, Array, mpsmat, na, d) 

function makemps(::Type{TN}, ::Type{TA}, mpsmat, na, d) where {TN, TA}

    mps = Array{TA{TN, 3}, 1}(na)

    b1 = size(mpsmat, 1) 
    b2 = size(mpsmat, 2)

    for n = 1:na
        mpssite = zeros(TN, b1, d, b2)
        for m = 1:b2
            for l = 1:b1
                mpssite[l, :, m] = mpsmat[l, m](n)
            end
        end
        mps[n] = mpssite
    end

    mps[1] = mps[1][1:1, :, 1:b2]
    mps[na] = mps[na][1:b1, :, b2:b2]

    return mps

end


"""
make MPO operator from input matrix (slow, shouldn't be used in loops)
"""

makempo(mpomat, na, d) = makempo(Complex{Float64}, Array, mpomat, na, d)

function makempo(::Type{TN}, ::Type{TA}, mpomat, na, d) where {TN, TA}

    mpo = Array{TA{TN, 4}, 1}(na)

    b1 = size(mpomat, 1)
    b2 = size(mpomat, 2)

    for n = 1:na
        mposite = zeros(TN, b1, d, b2, d)
        for m = 1:b2
            for l = 1:b1
                mposite[l, :, m, :] = mpomat[l, m](n)
            end
        end
        mpo[n] = mposite
    end

    mpo[1] = mpo[1][1:1, :, 1:b2, :]
    mpo[na] = mpo[na][1:b1, :, b2:b2, :]

    return mpo

end


"""
MPS_to_MPO(mpsin)
Revert an MPS to MPO after previous conversion to MPS
"""
function mps_to_mpo(::Type{TN}, ::Type{TA}, mpsin) where {TN, TA}
    
    na = length(mpsin)
    mpoout = Array{TA{TN, 4}, 1}(na)
    for n = 1:na
        (dl, d, dr) = size(mpsin[n])
        mpoout[n] = permutedims(reshape(mpsin[n],
            (dl, Int64(sqrt(d)), Int64(sqrt(d)) , dr)), [1, 2, 4, 3])
    end
    
    return mpoout

end

"""
MPS_to_MPO_dens(mpsin)
Revert an MPS to MPO after previous conversion to MPS
especially for density matrix which has different indice order
"""

function mps_to_mpo_dens(::Type{TN}, ::Type{TA}, mpsin) where {TN, TA}
    
    na = length(mpsin)
    mpoout = Array{TA{TN, 4}, 1}(na)
    for n = 1:na
        (dl, d, dr) = size(mpsin[n])
        mpoout[n] = permutedims(reshape(mpsin[n],
            (dl, Int64(sqrt(d)), Int64(sqrt(d)), dr)), [1, 3, 4, 2])
    end
    
    return mpoout

end
"""
MPO_to_MPS(mpoin)
turn MPO into and MPS, reducing the two d dimension
physical indices of the operator into one d^2 index
"""
function mpo_to_mps(::Type{TN}, ::Type{TA}, mpoin) where {TN, TA}

    na = length(mpoin)
    mpsout = Array{TA{TN, 3}, 1}(na)
    for n = 1:na
        mpsout[n] = mpo_to_mps_site(mpoin[n])
    end
    
    return mpsout

end

function mpo_to_mps_site(mpoin)

    (dl, d, dr, ) = size(mpoin)
    mpsout = reshape(permutedims(mpoin, [1, 2, 4, 3]), (dl, d^2, dr))

end

"""
MPO_to_MPS_dens(mpoin)
turn MPO into and MPS, reducing the two d dimension
physical indices of the operator into one d^2 index
with special order for density matrix
"""

function mpo_to_mps_dens(::Type{TN}, ::Type{TA}, mpoin) where {TN, TA}

    na = length(mpoin)
    mpsout = Array{TA{TN, 3}, 1}(na)
    for n = 1:na
        mpsout[n] = mpo_to_mps_dens_site(mpoin[n])
    end
    
    return mpsout

end

function mpo_to_mps_dens_site(mpoin)

    (dl, d, dr, ) = size(mpoin)
    mpsout = reshape(permutedims(mpoin, [1, 4, 2, 3]), (dl, d^2, dr))

end



build_env(dims1, dims2, mpo_dims) = build_env(Complex{Float64}, Array, dims1, dims2, mpo_dims)

function build_env(::Type{TN}, ::Type{TA}, dims1, dims2, mpo_dims) where {TN, TA}

    env = Array{TA{TN, 3}, 1}(length(dims1))

    env[1] = ones(TN, 1, 1, 1)
    for n in 2:(length(dims1) - 1)
        env[n] = zeros(TN, dims1[n], mpo_dims[n], dims2[n])
    end
    env[end] = ones(TN, 1, 1, 1)

    return env

end

build_env(dims1, dims2) = build_env(Complex{Float64}, Array, dims1, dims2)

function build_env(::Type{TN}, ::Type{TA}, dims1, dims2) where {TN, TA}

    env = Array{TA{TN, 2}, 1}(length(dims1))

    env[1] = ones(TN, 1, 1)
    for n in 2:(length(dims1) - 1)
        env[n] = zeros(TN, dims1[n], dims2[n])
    end
    env[end] = ones(TN, 1, 1)

    return env

end

