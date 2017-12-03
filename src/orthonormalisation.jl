
function leftorthoQR_MPO!(mpoin)
    n = length(mpoin);
    mpoin[1] = permutedims(mpoin[1],[1,2,4,3])
    for i = 1:n-1
        (dl,d,_,dr) = size(mpoin[i])
        bi = reshape(mpoin[i],(dl*d^2,dr))
        q,r = qr(bi);
        mpoin[i] = permutedims(reshape(q,dl,d,d,size(q,2)),[1,2,4,3])
        mo = mpoin[i+1]
        @tensor mo[-1,-2,-4,-3] := r[-1,1]*mo[1,-2,-3,-4]
        mpoin[i+1] = mo
    end
#    (dl,d,_,dr) = size(mpoin[n])
#    mo = reshape(mpoin[n],(dl,d^2,dr))
    mpoin[n] = permutedims(mpoin[n],[1,2,4,3])
#    @tensor no = real(scalar(mo[1,2,3]*conj(mo[1,2,3])))
#    return mpoin, no;
    return mpoin
end



"""
leftorthonormalizationQR!(A)

Left-orthonormalizes the MPS A in place and returns the norm
"""
function leftorthonormalizationQR!(mpsin)
    
    no = leftorthogonalizeQR!(mpsin)
    mpsin[end] = mpsin[end]./sqrt(no)

    return no

end

"""
leftorthogonalizeQR!(A)

Left-orthogonalizes the MPS A in place and calculate the norm, but doesn't actually normalize
"""
function leftorthogonalizeQR!(mpsin)
    
    n = length(mpsin)
    
    for i = 1:(n - 1)
        
        mpsin[i], r = leftorth(mpsin[i])
        mpsin[i + 1] = prod_LR(r, mpsin[i + 1])

    end
    
    no = sum(abs2.(mpsin[n][:]))

end


"""
rightorthonormalizationQR(A)

Right-orthonormalizes the MPS A in place and returns the norm
"""
function rightorthonormalizationQR!(mpsin)
    
    no = rightorthogonalizeQR!(mpsin)
    mpsin[1] = mpsin[1]./sqrt(no)

    return no

end
"""
rightorthogonalizeQR(A)

Right-orthogonalizes the MPS A in place and returns the norm
"""
function rightorthogonalizeQR!(mpsin)

    n = length(mpsin)

    for i = n:-1:2

        mpsin[i], r = rightorth(mpsin[i])
        mpsin[i - 1] = prod_LR(mpsin[i - 1], r)

    end

    no = sum(abs2.(mpsin[1][:]))

end


"""
function to normalize left orthogonal MPS
"""

function normalize_lo!(mps)

    no = sum(abs2.(mps[end][:]))
	mps[end] = mps[end]./sqrt(no)
	return no

end

function normalize_lo(mps)

    no = sum(abs2.(mps[end][:]))
    mps[end] = mps[end]./sqrt(no)
    return mps

end

"""
function to get norm of left orthogonal MPS
"""

function norm_lo(mps)

    no = sum(abs2.(mps[end][:]))

end

"""
function to normalize left orthogonal MPS
"""

function normalize_ro!(mps)

    no = sum(abs2(mps[1][:]))
	mps[1] = mps[1]./sqrt(no)
	return no

end

"""
function to get norm of right orthogonal MPS
"""

function norm_ro(mps)

    no = sum(abs2(mps[1][:]))

end

"""
Separate a MPS tensor A into a left orthonormal tensor AL and a bond
factor C
"""

function leftorth(A)

    Dl, d, Dr = size(A)
    Q, C = qr(reshape(A, d*Dl, Dr))
    Q = reshape(Q, Dl, d, size(Q, 2))
    return Q, C

end

"""
Separate a MPS tensor A into a left orthonormal tensor AL and a bond
factor C in place
"""

function leftorth!(A)

    Dl, d, Dr = size(A)
    A = reshape(A, d*Dl, Dr)
    Q = qrq!(A)
    reshape(Q, Dl, d, size(Q, 2))

end

function qrq_julia!(A::Matrix)

    QR = qrfact!(A)
    Array(QR[:Q])

end

function qrq!(A::Matrix)

    LAPACK.orgqr!(LAPACK.geqrf!(A)...)

end

function qrq!(Agpu::AbstractMatrix)

    A = collect(Agpu)
    n = min(size(A)...)
    LAPACK.orgqr!(LAPACK.geqrf!(A)...)
    copy!(Agpu, A)
    if n < size(A, 2)
        Agpu[:, 1:n]
    else
        Agpu
    end

end

"""
Separate a MPS tensor A into a right orthonormal tensor AR and a bond
factor C
"""
function rightorth(A)

    Dl, d, Dr = size(A)
    Q, C = qr(reshape(A, Dl, d*Dr).')
    Q = reshape(Q.', size(Q, 2), d, Dr)
    return Q, C.'

end

function rightorth_lq(A)

    Dl, d, Dr = size(A)
    C, Q = lq(reshape(A, Dl, d*Dr))
    Q = reshape(Q, size(Q, 1), d, Dr)
    return Q, C

end

"""
Separate a MPS tensor A into a right orthonormal tensor AR and a bond
factor C in place
"""

function rightorth!(A)

    Dl, d, Dr = size(A)
    A = reshape(A, Dl, d*Dr).'
    Q = qrq!(A)
    reshape(Q.', size(Q, 2), d, Dr)

end

function rightorth_lq!(A::Array{<:Number, 3})

    Dl, d, Dr = size(A)
    A = reshape(A, Dl, d*Dr)
    Q = lqq!(A)
    reshape(Q, size(Q, 1), d, Dr)

end

function lqq_julia!(A::Matrix)

    LQ = lqfact!(A)
    Array(LQ[:Q])

end

function lqq!(A::Matrix)

    LAPACK.orglq!(LAPACK.gelqf!(A)...)

end

function lqq!(Agpu::AbstractMatrix)

    A = collect(Agpu)
    m = min(size(A)...)
    LAPACK.orglq!(LAPACK.gelqf!(A)...)
    if m < size(A, 1)
        Agpu = Agpu[1:m, :]
        copy!(Agpu, A)
    else
        copy!(Agpu, A)
    end

end