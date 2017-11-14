
function compress_mpo!(env, mpstarget, mpo, maxsweeps)

    mps = MPO_to_MPS(mpo)

    compress_var!(mpstarget, mps, env, maxsweeps)

    MPS_to_MPO(mpstarget)

end


"""
function to compress mps A using maxsweeps left and right of the
variational algorithm with Ain as initial guess. Ain must be in left
canonical form (normalisation not necessary)
"""

function compress_var(Ain,A,maxsweeps)

    na = length(Ain)
    FL = Array{Array{Complex{Float64}, 2}, 1}(na)
    FR = Array{Array{Complex{Float64}, 2}, 1}(na)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)

    #initializing left environment

    FL[1] = eye(1)
    for jj = 1:(na - 1)
        FL[jj + 1] = update_lenv(Ain[jj], A[jj], FL[jj])
    end
    FR[na] = eye(1)

    for sweep = 1:maxsweeps

        #sweep left

        for jj = na:-1:2
            AR = prod_LR(A[jj], FR[jj])
            A0[jj] = prod_LR(FL[jj], AR)
            A0[jj] = rightorth!(A0[jj])
            FR[jj - 1] = update_renv(A0[jj], AR)
        end
        AR = prod_LR(A[1], FR[1])
        A0[1] = prod_LR(FL[1], AR)

        #sweep right

        for jj = 1:(na - 1)
            AL = prod_LR(FL[jj], A[jj])
            A0[jj] = prod_LR(AL, FR[jj])
            A0[jj] = leftorth!(A0[jj])
            FL[jj + 1] = update_lenv(A0[jj], AL)
        end
        AL = prod_LR(FL[na], A[na])
        A0[na] = prod_LR(AL, FR[na])

    end

    return A0

end


function compress_var!(A0, A, env, maxsweeps)

    na = length(A0)

    #initializing left environment

    env[1][1] = 1.0
    for jj = 1:(na - 1)
        update_lenv_co!(env[jj + 1], A0[jj], A[jj], env[jj])
    end
    env[na + 1][1] = 1.0

    for sweep = 1:maxsweeps

        #sweep left

        for jj = na:-1:2
            AR = prod_LR(A[jj], env[jj + 1])
            prod_LR_co!(A0[jj], env[jj], AR)
            A0[jj] = rightorth!(A0[jj])
            update_renv!(env[jj], A0[jj], AR)
        end
        AR = prod_LR(A[1], env[2])
        prod_LR!(A0[1], env[1], AR)

        #sweep right

        for jj = 1:(na - 1)
            AL = prod_LR_co(env[jj], A[jj])
            prod_LR!(A0[jj], AL, env[jj + 1])
            A0[jj] = leftorth!(A0[jj])
            update_lenv_co!(env[jj + 1], A0[jj], AL)
        end
        AL = prod_LR_co(env[na], A[na])
        prod_LR!(A0[na], AL, env[na + 1])

    end

end
"""
function to compress mps A using maxsweeps left and right of the two
site variational algorithm with Ain as initial guess. Ain must be in left
canonical form (normalisation not necessary)
"""

function compress_var_two_site(Ain,A,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    FL = Array{Array{Complex{Float64},2},1}(na)
    FR = Array{Array{Complex{Float64},2},1}(na)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing left environment

    FL[1] = eye(1)
    for jj = 1:na-1
        FL[jj+1] = update_lenv(Ain[jj],A[jj],FL[jj])
    end
    FR[na] = eye(1);

    for sweep = 1:maxsweeps

        U,S,Vd = eye(1),eye(1),eye(1)
        #sweep left

        for jj = na:-1:2;

            av = A_prod_LR_two_site(FL[jj-1],A[jj-1],A[jj],FR[jj])
            U,S,Vd = svdecon(reshape(av,dims[jj-1]*size(A[jj-1],2),
                size(A[jj],2)*dims[jj+1]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(Vd[:,1:dims[jj]]',dims[jj],size(A[jj],2),dims[jj+1])
            FR[jj-1] = update_renv(A0[jj],A[jj],FR[jj])

        end
        #A0[1] = reshape(U[:,1:Dims[2]]*diagm(S[1:Dims[2]]),Dims[1],size(A[1],2),Dims[2])

        #sweep right

        for jj = 1:na-1;

            av = A_prod_LR_two_site(FL[jj],A[jj],A[jj+1],FR[jj+1])
            U,S,Vd = svdecon(reshape(av,dims[jj]*size(A[jj],2),
                size(A[jj+1],2)*dims[jj+2]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj+1] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(U[:,1:dims[jj+1]],dims[jj],size(A[jj],2),dims[jj+1])
            FL[jj+1] = update_lenv(A0[jj],A[jj],FL[jj])

        end
        A0[na] = reshape(diagm(S[1:dims[na]])*(Vd[:,1:dims[na]]'),dims[na],size(A[na],2),dims[na+1])

    end

    return A0

end


"""
function to compress mps resulting from the product of H and A using
maxsweeps left and right of the variational algorithm with Ain as
initial guess. Ain must be in left canonical form (normalisation not necessary)
"""

function compress_var_apply_H(Ain, A, H, maxsweeps)

    na = length(Ain)
    FL = Array{Array{Complex{Float64}, 3}, 1}(na)
    FR = Array{Array{Complex{Float64}, 3}, 1}(na)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)

    #initializing left environment

    FL[1] = ones(1, 1, 1)
    for jj = 1:(na - 1)
        FL[jj + 1] = update_le(Ain[jj], H[jj], A[jj], FL[jj])
    end
    FR[na] = ones(1, 1, 1)

    for sweep = 1:maxsweeps

        #sweep left

        for jj = na:-1:2;
            AOR = prod_AOR(A[jj], H[jj], FR[jj])
            A0[jj] = prod_LR(FL[jj], AOR)
            A0[jj] = rightorth!(A0[jj])
            FR[jj - 1] = update_re(A0[jj], AOR)
        end

        #sweep right

        for jj = 1:(na - 1)
            AOL = prod_AOL(A[jj], H[jj], FL[jj])
            A0[jj] = prod_LR(AOL, FR[jj])
            A0[jj] = leftorth!(A0[jj])
            FL[jj + 1] = update_le(A0[jj], AOL)
        end
        
        AOL = prod_AOL(A[na], H[na], FL[na])
        A0[na] = prod_LR(AOL, FR[na])

    end

    return A0

end

function compress_var_apply_H!(A0, envop, Ain, A, H, maxsweeps)

    na = length(Ain)

    #initializing left environment

    envop[1][1] = 1.0
    for jj = 1:(na - 1)
        update_le!(envop[jj + 1], Ain[jj], H[jj], A[jj], envop[jj])
    end
    envop[na + 1][1] = 1.0

    for sweep = 1:maxsweeps

        #sweep left

        for jj = na:-1:2

            AOR = prod_AOR(A[jj], H[jj], envop[jj + 1])
            prod_LR!(A0[jj], envop[jj], AOR)
            A0[jj] = rightorth!(A0[jj])
            update_re!(envop[jj], A0[jj], AOR)

        end

        #sweep right

        for jj = 1:(na - 1)
            
            AOL = prod_AOL(A[jj], H[jj], envop[jj])
            prod_LR!(A0[jj], AOL, envop[jj + 1])
            A0[jj] = leftorth!(A0[jj])
            update_le!(envop[jj + 1], A0[jj], AOL)

        end
        
        AOL = prod_AOL(A[na], H[na], envop[na])
        prod_LR!(A0[na], AOL, envop[na + 1])

    end

end


function compress_var_apply_H_right(Ain, A, H, maxsweeps)

    na = length(Ain)
    FR = Array{Array{Complex{Float64}, 3}, 1}(na)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)

    #initializing right environment

    FR[na] = ones(1, 1, 1)
    for jj = na:-1:2;
        FR[jj - 1] = update_re(Ain[jj], H[jj], A[jj], FR[jj])
    end
    FL = complex(ones(1, 1, 1))

    #sweep right

    for jj = 1:(na - 1)
        AOL = prod_AOL(A[jj], H[jj], FL)
        A0[jj] = prod_LR(AOL, FR[jj])
        A0[jj] = leftorth!(A0[jj])
        FL = update_le(A0[jj], AOL)
    end

    AOL = prod_AOL(A[na], H[na], FL)
    A0[na] = prod_LR(AOL, FR[na])

    return A0

end

function compress_var_apply_H_left(Ain, A, H)

    na = length(Ain)
    FL = Array{Array{Complex{Float64}, 3}, 1}(na)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)

    #initializing left environment

    FL[1] = ones(1, 1, 1)
    for jj = 1:(na - 1)
        FL[jj + 1] = update_le(Ain[jj], H[jj], A[jj], FL[jj])
    end
    FR = complex(ones(1, 1, 1))

    #sweep left

    for jj = na:-1:2;
        AOR = prod_AOR(A[jj], H[jj], FR)
        A0[jj] = prod_LR(FL[jj], AOR)
        A0[jj] = rightorth!(A0[jj])
        FR = update_re(A0[jj], AOR)
    end

    AOR = prod_AOR(A[1], H[1], FR)
    A0[1] = prod_LR(FL[1], AOR)

    return A0

end

function compress_var_apply_H_left!(A0, envop, Ain, A, H)

    na = length(Ain)

    #initializing left environment

    envop[1][1] = 1.0
    for jj = 1:(na - 1)
        update_le!(envop[jj + 1], Ain[jj], H[jj], A[jj], envop[jj])
    end
    envop[na + 1][1] = 1.0

    #sweep left

    for jj = na:-1:2
        
        AOR = prod_AOR(A[jj], H[jj], envop[jj + 1])
        prod_LR!(A0[jj], envop[jj], AOR)
        A0[jj] = rightorth!(A0[jj])
        update_re!(envop[jj], A0[jj], AOR)

    end

    AOR = prod_AOR(A[1], H[1], envop[2])
    prod_LR!(A0[1], envop[1], AOR)

end

"""
function to compress mps resulting from the product of H and A using
maxsweeps left and right of the variational two site algorithm with Ain as
initial guess. Ain must be in left canonical form (normalisation not necessary)
"""

function compress_var_apply_H_two_site(Ain,A,H,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    FL = Array{Array{Complex{Float64},3},1}(na)
    FR = Array{Array{Complex{Float64},3},1}(na)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing left environment

    FL[1] = ones(1,1,1)
    for jj = 1:na-2
        FL[jj+1] = update_le(Ain[jj],H[jj],A[jj],FL[jj])
    end
    FR[na] = ones(1,1,1);

    for sweep = 1:maxsweeps

        U,S,Vd = eye(1),eye(1),eye(1)
        #sweep left

        for jj = na:-1:2;

            av = A_prod_LOR_two_site(FL[jj-1],A[jj-1],H[jj-1],A[jj],H[jj],FR[jj])
            U,S,Vd = svdecon(reshape(av,dims[jj-1]*size(A[jj-1],2),
                size(A[jj],2)*dims[jj+1]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(Vd[:,1:dims[jj]]',dims[jj],size(A[jj],2),dims[jj+1])
            FR[jj-1] = update_re(A0[jj],H[jj],A[jj],FR[jj])

        end

        #A0[1] = reshape(U[:,1:Dims[2]]*diagm(S[1:Dims[2]]),Dims[1],size(A[1],2),Dims[2])

        #sweep right

        for jj = 1:na-1;

            av = A_prod_LOR_two_site(FL[jj],A[jj],H[jj],A[jj+1],H[jj+1],FR[jj+1])
            U,S,Vd = svdecon(reshape(av,dims[jj]*size(A[jj],2),
                size(A[jj+1],2)*dims[jj+2]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj+1] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(U[:,1:dims[jj+1]],dims[jj],size(A[jj],2),dims[jj+1])
            FL[jj+1] = update_le(A0[jj],H[jj],A[jj],FL[jj])

        end

        A0[na] = reshape(diagm(S[1:dims[na]])*(Vd[:,1:dims[na]]'),dims[na],size(A[na],2),dims[na+1])

    end

    return A0

end

function compress_var_apply_H_two_site_left(Ain,A,H,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    FL = Array{Array{Complex{Float64},3},1}(na)
    FR = Array{Array{Complex{Float64},3},1}(na)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing left environment

    FL[1] = ones(1,1,1)
    for jj = 1:na-2
        FL[jj+1] = update_le(Ain[jj],H[jj],A[jj],FL[jj])
    end
    FR[na] = ones(1,1,1);

    U,S,Vd = eye(1),eye(1),eye(1)
    #sweep left

    for jj = na:-1:2;

        av = A_prod_LOR_two_site(FL[jj-1],A[jj-1],H[jj-1],A[jj],H[jj],FR[jj])
        U,S,Vd = svdecon(reshape(av,dims[jj-1]*size(A[jj-1],2),
            size(A[jj],2)*dims[jj+1]))
        S = S[S.>0]
        stot = sum(S.^2)
        dims[jj] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
        A0[jj] = reshape(Vd[:,1:dims[jj]]',dims[jj],size(A[jj],2),dims[jj+1])
        FR[jj-1] = update_re(A0[jj],H[jj],A[jj],FR[jj])

    end

    A0[1] = reshape(U[:,1:dims[2]]*diagm(S[1:dims[2]]),dims[1],size(A[1],2),dims[2])

    return A0

end

function compress_var_apply_H_two_site_right(Ain,A,H,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    FL = Array{Array{Complex{Float64},3},1}(na)
    FR = Array{Array{Complex{Float64},3},1}(na)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing right environment

    FR[na] = ones(1,1,1);
    for jj = na:-1:2;
        FR[jj-1] = update_re(Ain[jj],H[jj],A[jj],FR[jj])
    end
    FL[1] = ones(1,1,1);

    U,S,Vd = eye(1),eye(1),eye(1)

    #sweep right

    for jj = 1:na-1;

        av = A_prod_LOR_two_site(FL[jj],A[jj],H[jj],A[jj+1],H[jj+1],FR[jj+1])
        U,S,Vd = svdecon(reshape(av,dims[jj]*size(A[jj],2),
            size(A[jj+1],2)*dims[jj+2]))
        S = S[S.>0]
        stot = sum(S.^2)
        dims[jj+1] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
        A0[jj] = reshape(U[:,1:dims[jj+1]],dims[jj],size(A[jj],2),dims[jj+1])
        FL[jj+1] = update_le(A0[jj],H[jj],A[jj],FL[jj])

    end

    A0[na] = reshape(diagm(S[1:dims[na]])*(Vd[:,1:dims[na]]'),dims[na],size(A[na],2),dims[na+1])

    return A0

end


"""
function to compress linear combination of mps stored in array A, which
have weights given by coef, using a
variational algorithm with Ain as initial guess. Ain must be in left
canonical form (normalisation not necessary)
"""

function compress_sum_var(Ain, coef, A, maxsweeps)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Array{Complex{Float64}, 2}, 1}, 1}(numv)
    FR = Array{Array{Array{Complex{Float64}, 2}, 1}, 1}(numv)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)
    AR = Array{Array{Complex{Float64}, 3}, 1}(numv)

    #initializing left environment

    for ll = 1:numv
        #A[ll][na] = coef[ll]*Av[ll][na]
        FL[ll] = Array{Array{Complex{Float64}, 2}, 1}(na)
        FL[ll][1] = eye(1)
        for jj = 1:(na - 1)
            FL[ll][jj + 1] = update_lenv(Ain[jj], A[ll][jj], FL[ll][jj])
        end
        FR[ll] = Array{Array{Complex{Float64}, 2}, 1}(na)
        FR[ll][na] = coef[ll]*eye(1)
    end

    for sweep = 1:maxsweeps

        #sweep left

        for jj = na:-1:2

            AR[1] = prod_LR(A[1][jj], FR[1][jj])
            A0[jj] = prod_LR(FL[1][jj], AR[1])
            for ll = 2:numv
                AR[ll] = prod_LR(A[ll][jj], FR[ll][jj])
                A0[jj] += prod_LR(FL[ll][jj], AR[ll])
            end
            A0[jj] = rightorth!(A0[jj])
            for ll = 1:numv
                FR[ll][jj - 1] = update_renv(A0[jj], AR[ll])
            end

        end

        #sweep right

        for jj = 1:(na - 1)

            AR[1] = prod_LR(FL[1][jj], A[1][jj])
            A0[jj] = prod_LR(AR[1], FR[1][jj])
            for ll = 2:numv
                AR[ll] = prod_LR(FL[ll][jj], A[ll][jj])
                A0[jj] += prod_LR(AR[ll], FR[ll][jj])
            end
            A0[jj] = leftorth!(A0[jj])
            for ll = 1:numv
                FL[ll][jj + 1] = update_lenv(A0[jj], AR[ll])
            end

        end

    end

    AR[1] = prod_LR(FL[1][na], A[1][na])
    A0[na] = prod_LR(AR[1], FR[1][na])
    for ll = 2:numv
        AR[ll] = prod_LR(FL[ll][na], A[ll][na])
        A0[na] += prod_LR(AR[ll], FR[ll][na])
    end

    return A0

end

"""
function to compress linear combination of mps stored array A, which
have weights given by coef, using the two site
variational algorithm with Ain as initial guess. Ain must be in left
canonical form (normalisation not necessary)
"""

function compress_sum_var_two_site(Ain,coef,A,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Array{Complex{Float64},2},1},1}(numv)
    FR = Array{Array{Array{Complex{Float64},2},1},1}(numv)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing left environment

    for ll = 1:numv

        #A[ll][na] = coef[ll]*Av[ll][na]
        FL[ll] = Array{Array{Complex{Float64},2},1}(na)
        FL[ll][1] = eye(1)
        for jj = 1:na-2
            FL[ll][jj+1] = update_lenv(Ain[jj],A[ll][jj],FL[ll][jj])
        end
        FR[ll] = Array{Array{Complex{Float64},2},1}(na)
        FR[ll][na] = coef[ll]*eye(1);

    end

    for sweep = 1:maxsweeps

        U,S,Vd = eye(1),eye(1),eye(1)
        #sweep left

        for jj = na:-1:2

            av = A_prod_LR_two_site(FL[1][jj-1],A[1][jj-1],A[1][jj],FR[1][jj])
            for ll = 2:numv
                av += A_prod_LR_two_site(FL[ll][jj-1],A[ll][jj-1],A[ll][jj],FR[ll][jj])
            end
            U,S,Vd = svdecon(reshape(av,dims[jj-1]*size(A[1][jj-1],2),
                size(A[1][jj],2)*dims[jj+1]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(Vd[:,1:dims[jj]]',dims[jj],size(A[1][jj],2),dims[jj+1])
            for ll = 1:numv
                FR[ll][jj-1] = update_renv(A0[jj],A[ll][jj],FR[ll][jj])
            end

        end

        #A0[1] = reshape(U[:,1:Dims[2]]*diagm(S[1:Dims[2]]),Dims[1],size(A[1][1],2),Dims[2])

        #sweep right

        for jj = 1:na-1;

            av = A_prod_LR_two_site(FL[1][jj],A[1][jj],A[1][jj+1],FR[1][jj+1])
            for ll = 2:numv
                av += A_prod_LR_two_site(FL[ll][jj],A[ll][jj],A[ll][jj+1],FR[ll][jj+1])
            end
            U,S,Vd = svdecon(reshape(av,dims[jj]*size(A[1][jj],2),
                size(A[1][jj+1],2)*dims[jj+2]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj+1] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(U[:,1:dims[jj+1]],dims[jj],size(A[1][jj],2),dims[jj+1])
            for ll = 1:numv
                FL[ll][jj+1] = update_lenv(A0[jj],A[ll][jj],FL[ll][jj])
            end
        end

        A0[na] = reshape(diagm(S[1:dims[na]])*(Vd[:,1:dims[na]]'),dims[na],size(A[1][na],2),dims[na+1])

    end

    return A0

end


"""
function to compress linear combination of mps stored in cell array A, which
have weights given by coef and where H is applied to the last MPS.
The compression uses the variational algorithm with A0 as initial guess.
A0 must be in left canonical form (normalisation not necessary)
"""

function compress_sum_var_apply_H(Ain, coef, A, H, maxsweeps)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Array{Complex{Float64}, 2}, 1}, 1}(numv - 1)
    FR = Array{Array{Array{Complex{Float64}, 2}, 1}, 1}(numv - 1)
    FLH = Array{Array{Complex{Float64}, 3}, 1}(na)
    FRH = Array{Array{Complex{Float64} ,3}, 1}(na)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)
    AR = Array{Array{Complex{Float64}, 3}, 1}(numv - 1)

    #initializing left environment

    for ll = 1:(numv - 1)
        #A[ll][na] = coef[ll]*Av[ll][na]
        FL[ll] = Array{Array{Complex{Float64}, 2}, 1}(na)
        FL[ll][1] = eye(1)
        for jj = 1:(na - 1)
            FL[ll][jj + 1] = update_lenv(Ain[jj], A[ll][jj], FL[ll][jj])
        end
        FR[ll] = Array{Array{Complex{Float64}, 2}, 1}(na)
        FR[ll][na] = coef[ll]*eye(1)
    end

    FLH[1] = ones(1, 1, 1)
    for jj = 1:(na - 1)
        FLH[jj + 1] = update_le(Ain[jj], H[jj], A[numv][jj], FLH[jj])
    end
    FRH[na] = coef[numv]*ones(1, 1, 1)

    for sweep = 1:maxsweeps

        #sweep left

        for jj = na:-1:2;

            AR[1] = prod_LR(A[1][jj], FR[1][jj])
            A0[jj] = prod_LR(FL[1][jj], AR[1])
            for ll = 2:(numv - 1)
                AR[ll] = prod_LR(A[ll][jj], FR[ll][jj])
                A0[jj] += prod_LR(FL[ll][jj], AR[ll])
            end
            AOR = prod_AOR(A[numv][jj], H[jj], FRH[jj])
            A0[jj] += prod_LR(FLH[jj], AOR)
            A0[jj] = rightorth!(A0[jj])
            for ll = 1:(numv - 1)
                FR[ll][jj - 1] = update_renv(A0[jj], AR[ll])
            end
            FRH[jj - 1] = update_re(A0[jj], AOR)
        end

        #sweep right

        for jj = 1:(na - 1)

            AR[1] = prod_LR(FL[1][jj], A[1][jj])
            A0[jj] = prod_LR(AR[1], FR[1][jj])
            for ll = 2:(numv - 1)
                AR[ll] = prod_LR(FL[ll][jj], A[ll][jj])
                A0[jj] += prod_LR(AR[ll], FR[ll][jj])
            end
            AOL = prod_AOL(A[numv][jj], H[jj], FLH[jj])
            A0[jj] += prod_LR(AOL, FRH[jj])
            A0[jj] = leftorth!(A0[jj])
            for ll = 1:(numv - 1)
                FL[ll][jj + 1] = update_lenv(A0[jj], AR[ll])
            end
            FLH[jj + 1] = update_le(A0[jj], AOL)
        end

    end

    AR[1] = prod_LR(FL[1][na], A[1][na])
    A0[na] = prod_LR(AR[1], FR[1][na])
    for ll = 2:numv-1
        AR[ll] = prod_LR(FL[ll][na], A[ll][na])
        A0[na] += prod_LR(AR[ll], FR[ll][na])
    end
    AOL = prod_AOL(A[numv][na], H[na], FLH[na])
    A0[na] += prod_LR(AOL, FRH[na])

    return A0

end

function compress_sum_var_apply_H_left(Ain, coef, A, H)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Array{Complex{Float64}, 2}, 1}, 1}(numv - 1)
    FR = Array{Array{Complex{Float64}, 2}, 1}(numv - 1)
    FLH = Array{Array{Complex{Float64}, 3}, 1}(na)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)
    AR = Array{Array{Complex{Float64}, 3}, 1}(numv - 1)

    #initializing left environment

    for ll = 1:(numv - 1)
        #A[ll][na] = coef[ll]*Av[ll][na]
        FL[ll] = Array{Array{Complex{Float64}, 2}, 1}(na)
        FL[ll][1] = eye(1)
        for jj = 1:(na - 1)
            FL[ll][jj + 1] = update_lenv(Ain[jj], A[ll][jj], FL[ll][jj])
        end
        FR[ll] = coef[ll]*eye(1)

    end

    FLH[1] = ones(1, 1, 1)
    for jj = 1:(na - 1)
        FLH[jj + 1] = update_le(Ain[jj], H[jj], A[numv][jj], FLH[jj])
    end

    FRH = coef[numv]*complex(ones(1, 1, 1))

    #sweep left

    for jj = na:-1:2;
        
        AR[1] = prod_LR(A[1][jj], FR[1])
        A0[jj] = prod_LR(FL[1][jj], AR[1])
        for ll = 2:(numv - 1)
            AR[ll] = prod_LR(A[ll][jj], FR[ll])
            A0[jj] += prod_LR(FL[ll][jj], AR[ll])
        end
        AOR = prod_AOR(A[numv][jj], H[jj], FRH)
        A0[jj] += prod_LR(FLH[jj], AOR)
        A0[jj] = rightorth!(A0[jj])
        for ll = 1:(numv - 1)
            FR[ll] = update_renv(A0[jj], AR[ll])
        end
        FRH = update_re(A0[jj], AOR)
    end

    AR[1] = prod_LR(A[1][1], FR[1])
    A0[1] = prod_LR(FL[1][1], AR[1])
    for ll = 2:numv-1
        AR[ll] = prod_LR(A[ll][1], FR[ll])
        A0[1] += prod_LR(FL[ll][1], AR[ll])
    end
    AOR = prod_AOR(A[numv][1], H[1], FRH)
    A0[1] += prod_LR(FLH[1], AOR)

    return A0

end


function compress_sum_var_apply_H!(A0, envop, coef, A, H, maxsweeps)

    na = length(A0)
    numv = length(coef)
    env = Array{Array{Array{Complex{Float64}, 2}, 1}, 1}(numv - 1)
    AR = Array{Array{Complex{Float64}, 3}, 1}(numv - 1)

    #initializing left environment
    
    for ll = 1:(numv - 1)
        #A[ll][na] = coef[ll]*Av[ll][na]
        env[ll] = Array{Array{Complex{Float64}, 2}, 1}(na + 1)
        env[ll][1][1] = 1.0
        for jj = 1:(na - 1)
            env[ll][jj + 1] = update_lenv(A0[jj], A[ll][jj], env[ll][jj])
        end
        env[ll][na + 1][1] = coef[ll]
    end

    envop[1][1] = 1.0
    for jj = 1:(na - 1)

        update_le!(envop[jj + 1], A0[jj], H[jj], A[numv][jj], envop[jj])

    end
    envop[na + 1][1] = coef[numv]

    for sweep = 1:maxsweeps

        #sweep left

        for jj = na:-1:2;

            AR[1] = prod_LR(A[1][jj], env[1][jj + 1])
            prod_LR!(A0[jj], env[1][jj], AR[1])
            for ll = 2:(numv - 1)
                AR[ll] = prod_LR(A[ll][jj], env[ll][jj + 1])
                A0[jj] += prod_LR(env[ll][jj], AR[ll])
            end
            AOR = prod_AOR(A[numv][jj], H[jj], envop[jj + 1])
            A0[jj] += prod_LR(envop[jj], AOR)
            A0[jj] = rightorth!(A0[jj])
            for ll = 1:(numv - 1)
                update_renv!(env[ll][jj], A0[jj], AR[ll])
            end
            update_re!(envop[jj], A0[jj], AOR)

        end

        #sweep right

        for jj = 1:(na - 1)

            AR[1] = prod_LR(env[1][jj], A[1][jj])
            prod_LR!(A0[jj], AR[1], env[1][jj + 1])
            for ll = 2:(numv - 1)
                AR[ll] = prod_LR(env[ll][jj], A[ll][jj])
                A0[jj] += prod_LR(AR[ll], env[ll][jj + 1])
            end
            AOL = prod_AOL(A[numv][jj], H[jj], envop[jj])
            A0[jj] += prod_LR(AOL, envop[jj + 1])
            A0[jj] = leftorth!(A0[jj])
            for ll = 1:(numv - 1)
                update_lenv!(env[ll][jj + 1], A0[jj], AR[ll])
            end
            update_le!(envop[jj + 1], A0[jj], AOL)

        end

    end

    AR[1] = prod_LR(env[1][na], A[1][na])
    prod_LR!(A0[na], AR[1], env[1][na + 1])
    for ll = 2:numv-1
        AR[ll] = prod_LR(env[ll][na], A[ll][na])
        A0[na] += prod_LR(AR[ll], env[ll][na + 1])
    end
    AOL = prod_AOL(A[numv][na], H[na], envop[na])
    A0[na] += prod_LR(AOL, envop[na + 1])

    nothing

end

function compress_sum_var_apply_H_left!(A0, env, envop, Ain, coef, A, H)

    na = length(Ain)
    numv = length(coef)
    AR = similar(A0, numv - 1)

    #initializing left environment

    for ll = 1:(numv - 1)

        env[ll][1][1] = 1.0
        for jj = 1:(na - 1)
            update_lenv!(env[ll][jj + 1], Ain[jj], A[ll][jj], env[ll][jj])
        end
        env[ll][na + 1][1] = coef[ll]

    end

    envop[1][1] = 1.0
    for jj = 1:(na - 1)

        update_le!(envop[jj + 1], Ain[jj], H[jj], A[numv][jj], envop[jj])

    end
    envop[na + 1][1] = coef[numv]

    #sweep left

    for jj = na:-1:2;
        
        AR[1] = prod_LR(A[1][jj], env[1][jj + 1])
        prod_LR!(A0[jj], env[1][jj], AR[1])
        for ll = 2:(numv - 1)
            AR[ll] = prod_LR(A[ll][jj], env[ll][jj + 1])
            A0[jj] += prod_LR(env[ll][jj], AR[ll])
        end
        AOR = prod_AOR(A[numv][jj], H[jj], envop[jj + 1])
        A0[jj] += prod_LR(envop[jj], AOR)
        A0[jj] = rightorth!(A0[jj])
        for ll = 1:(numv - 1)
            update_renv!(env[ll][jj], A0[jj], AR[ll])
        end
        update_re!(envop[jj], A0[jj], AOR)
    end

    AR[1] = prod_LR(A[1][1], env[1][2])
    prod_LR!(A0[1], env[1][1], AR[1])
    for ll = 2:numv-1
        AR[ll] = prod_LR(A[ll][1], env[ll][2])
        A0[1] += prod_LR(env[ll][1], AR[ll])
    end
    AOR = prod_AOR(A[numv][1], H[1], envop[2])
    A0[1] += prod_LR(envop[1], AOR)
    nothing

end

function compress_sum_var_apply_H_right(Ain, coef, A, H)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Complex{Float64}, 2}, 1}(numv - 1)
    FR = Array{Array{Array{Complex{Float64}, 2}, 1}, 1}(numv - 1)
    FRH = Array{Array{Complex{Float64}, 3}, 1}(na)
    A0 = Array{Array{Complex{Float64}, 3}, 1}(na)
    AL = Array{Array{Complex{Float64}, 3}, 1}(numv - 1)

    #initializing right environment

    for ll = 1:(numv - 1)
        FR[ll] = Array{Array{Complex{Float64}, 2}, 1}(na)
        FR[ll][na] = eye(1)
        for jj = na:-1:2
            FR[ll][jj - 1] = update_renv(Ain[jj], A[ll][jj], FR[ll][jj])
        end
        FL[ll] = coef[ll]*eye(1)
    end

    FRH[na] = ones(1, 1, 1)
    for jj = na:-1:2
        FRH[jj - 1] = update_re(Ain[jj], H[jj], A[numv][jj], FRH[jj])
    end

    FLH = coef[numv]*complex(ones(1, 1, 1))

    #sweep right

    for jj = 1:(na - 1)

        AL[1] = prod_LR(FL[1], A[1][jj])
        A0[jj] = prod_LR(AL[1], FR[1][jj])
        for ll = 2:(numv - 1)
            AL[ll] = prod_LR(FL[ll], A[ll][jj])
            A0[jj] += prod_LR(AL[ll], FR[ll][jj])
        end
        AOL = prod_AOL(A[numv][jj], H[jj], FLH)
        A0[jj] += prod_LR(AOL, FRH[jj])
        A0[jj] = leftorth!(A0[jj])
        for ll = 1:(numv - 1)
            FL[ll] = update_lenv(A0[jj], AL[ll])
        end
        FLH = update_le(A0[jj], AOL)

    end

    AL[1] = prod_LR(FL[1], A[1][na])
    A0[na] = prod_LR(AL[1], FR[1][na])
    for ll = 2:(numv - 1)
        AL[ll] = prod_LR(FL[ll], A[ll][na])
        A0[na] += prod_LR(AL[ll], FR[ll][na])
    end
    AOL = prod_AOL(A[numv][na], H[na], FLH)
    A0[na] += prod_LR(AOL, FRH[na])

    return A0

end

function compress_sum_var_apply_H_right!(A0, env, envop, Ain, coef, A, H)

    na = length(Ain)
    numv = length(coef)
    AL = similar(A0, numv - 1)

    #initializing right environment

    for ll = 1:(numv - 1)

        env[ll][na + 1][1] = 1.0
        for jj = na:-1:2
            update_renv!(env[ll][jj], Ain[jj], A[ll][jj], env[ll][jj + 1])
        end
        env[ll][1][1] = coef[ll]

    end

    envop[na + 1][1] = 1.0
    for jj = na:-1:2
        
        update_re!(envop[jj], Ain[jj], H[jj], A[numv][jj], envop[jj + 1])
    
    end
    envop[1][1] = coef[numv]

    #sweep right

    for jj = 1:(na - 1)

        AL[1] = prod_LR(env[1][jj], A[1][jj])
        prod_LR!(A0[jj], AL[1], env[1][jj + 1])
        for ll = 2:(numv - 1)
            AL[ll] = prod_LR(env[ll][jj], A[ll][jj])
            A0[jj] += prod_LR(AL[ll], env[ll][jj + 1])
        end
        AOL = prod_AOL(A[numv][jj], H[jj], envop[jj])
        A0[jj] += prod_LR(AOL, envop[jj + 1])
        A0[jj] = leftorth!(A0[jj])
        for ll = 1:(numv - 1)
            update_lenv!(env[ll][jj + 1], A0[jj], AL[ll])
        end
        update_le!(envop[jj + 1], A0[jj], AOL)

    end

    AL[1] = prod_LR(env[1][na], A[1][na])
    prod_LR!(A0[na], AL[1], env[1][na + 1])
    for ll = 2:(numv - 1)
        AL[ll] = prod_LR(env[ll][na], A[ll][na])
        A0[na] += prod_LR(AL[ll], env[ll][na + 1])
    end
    AOL = prod_AOL(A[numv][na], H[na], envop[na])
    A0[na] += prod_LR(AOL, envop[na + 1])
    nothing

end

"""
function to compress linear combination of mps stored in array A, which
have weights given by coef and where H is applied to the last MPS.
The compression uses the two site variational algorithm with A0 as initial guess.
A0 must be in left canonical form (normalisation not necessary)
"""

function compress_sum_var_apply_H_two_site(Ain,coef,A,H,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Array{Complex{Float64},2},1},1}(numv-1)
    FR = Array{Array{Array{Complex{Float64},2},1},1}(numv-1)
    FLH = Array{Array{Complex{Float64},3},1}(na)
    FRH = Array{Array{Complex{Float64},3},1}(na)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing left environment

    for ll = 1:numv-1
        #A[ll][na] = coef[ll]*Av[ll][na]
        FL[ll] = Array{Array{Complex{Float64},2},1}(na)
        FL[ll][1] = eye(1)
        for jj = 1:na-2
            FL[ll][jj+1] = update_lenv(Ain[jj],A[ll][jj],FL[ll][jj])
        end
        FR[ll] = Array{Array{Complex{Float64},2},1}(na)
        FR[ll][na] = coef[ll]*eye(1);
    end

    FLH[1] = ones(1,1,1)
    for jj = 1:na-1
        FLH[jj+1] = update_le(Ain[jj],H[jj],A[numv][jj],FLH[jj])
    end
    FRH[na] = coef[numv]*ones(1,1,1)

    for sweep = 1:maxsweeps

        U,S,Vd = eye(1),eye(1),eye(1)
        #sweep left

        for jj = na:-1:2;

            av = A_prod_LR_two_site(FL[1][jj-1],A[1][jj-1],A[1][jj],FR[1][jj])
            for ll = 2:numv-1
                av += A_prod_LR_two_site(FL[ll][jj-1],A[ll][jj-1],A[ll][jj],FR[ll][jj])
            end
            av += A_prod_LOR_two_site(FLH[jj-1],A[numv][jj-1],H[jj-1],A[numv][jj],H[jj],FRH[jj])
            U,S,Vd = svdecon(reshape(av,dims[jj-1]*size(A[1][jj-1],2),
                size(A[1][jj],2)*dims[jj+1]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(Vd[:,1:dims[jj]]',dims[jj],size(A[1][jj],2),dims[jj+1])
            for ll = 1:numv-1
                FR[ll][jj-1] = update_renv(A0[jj],A[ll][jj],FR[ll][jj])
            end
            FRH[jj-1] = update_re(A0[jj],H[jj],A[numv][jj],FRH[jj])

        end

        #A0[1] = reshape(U[:,1:Dims[2]]*diagm(S[1:Dims[2]]),Dims[1],size(A[1][1],2),Dims[2])

        #sweep right

        for jj = 1:na-1;

            av = A_prod_LR_two_site(FL[1][jj],A[1][jj],A[1][jj+1],FR[1][jj+1])
            for ll = 2:numv-1
                av += A_prod_LR_two_site(FL[ll][jj],A[ll][jj],A[ll][jj+1],FR[ll][jj+1])
            end
            av += A_prod_LOR_two_site(FLH[jj],A[numv][jj],H[jj],A[numv][jj+1],H[jj+1],FRH[jj+1])
            U,S,Vd = svdecon(reshape(av,dims[jj]*size(A[1][jj],2),
                size(A[1][jj+1],2)*dims[jj+2]))
            S = S[S.>0]
            stot = sum(S.^2)
            dims[jj+1] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
            A0[jj] = reshape(U[:,1:dims[jj+1]],dims[jj],size(A[1][jj],2),dims[jj+1])
            for ll = 1:numv-1
                FL[ll][jj+1] = update_lenv(A0[jj],A[ll][jj],FL[ll][jj])
            end
            FLH[jj+1] = update_le(A0[jj],H[jj],A[numv][jj],FLH[jj])

        end

        A0[na] = reshape(diagm(S[1:dims[na]])*(Vd[:,1:dims[na]]'),dims[na],size(A[1][na],2),dims[na+1])

    end

    return A0

end


function compress_sum_var_apply_H_two_site_left(Ain,coef,A,H,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Array{Complex{Float64},2},1},1}(numv-1)
    FR = Array{Array{Array{Complex{Float64},2},1},1}(numv-1)
    FLH = Array{Array{Complex{Float64},3},1}(na)
    FRH = Array{Array{Complex{Float64},3},1}(na)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing left environment

    for ll = 1:numv-1
        #A[ll][na] = coef[ll]*Av[ll][na]
        FL[ll] = Array{Array{Complex{Float64},2},1}(na)
        FL[ll][1] = eye(1)
        for jj = 1:na-2
            FL[ll][jj+1] = update_lenv(Ain[jj],A[ll][jj],FL[ll][jj])
        end
        FR[ll] = Array{Array{Complex{Float64},2},1}(na)
        FR[ll][na] = coef[ll]*eye(1);
    end

    FLH[1] = ones(1,1,1)
    for jj = 1:na-2
        FLH[jj+1] = update_le(Ain[jj],H[jj],A[numv][jj],FLH[jj])
    end
    FRH[na] = coef[numv]*ones(1,1,1)

    U,S,Vd = eye(1),eye(1),eye(1)
    #sweep left

    for jj = na:-1:2;

        av = A_prod_LR_two_site(FL[1][jj-1],A[1][jj-1],A[1][jj],FR[1][jj])
        for ll = 2:numv-1
            av += A_prod_LR_two_site(FL[ll][jj-1],A[ll][jj-1],A[ll][jj],FR[ll][jj])
        end
        av += A_prod_LOR_two_site(FLH[jj-1],A[numv][jj-1],H[jj-1],A[numv][jj],H[jj],FRH[jj])
        U,S,Vd = svdecon(reshape(av,dims[jj-1]*size(A[1][jj-1],2),
            size(A[1][jj],2)*dims[jj+1]))
        S = S[S.>0]
        stot = sum(S.^2)
        dims[jj] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
        A0[jj] = reshape(Vd[:,1:dims[jj]]',dims[jj],size(A[1][jj],2),dims[jj+1])
        for ll = 1:numv-1
            FR[ll][jj-1] = update_renv(A0[jj],A[ll][jj],FR[ll][jj])
        end
        FRH[jj-1] = update_re(A0[jj],H[jj],A[numv][jj],FRH[jj])

    end

    A0[1] = reshape(U[:,1:dims[2]]*diagm(S[1:dims[2]]),dims[1],size(A[1][1],2),dims[2])

    return A0

end


function compress_sum_var_apply_H_two_site_right(Ain,coef,A,H,maxsweeps,svd_tol,dims,dlim)

    na = length(Ain)
    numv = length(coef)
    FL = Array{Array{Array{Complex{Float64},2},1},1}(numv-1)
    FR = Array{Array{Array{Complex{Float64},2},1},1}(numv-1)
    FLH = Array{Array{Complex{Float64},3},1}(na)
    FRH = Array{Array{Complex{Float64},3},1}(na)
    A0 = Array{Array{Complex{Float64},3},1}(na)

    #initializing right environment

    for ll = 1:numv-1
        FR[ll] = Array{Array{Complex{Float64},2},1}(na)
        FR[ll][na] = eye(1)
        for jj = na:-1:2
            FR[ll][jj-1] = update_renv(Ain[jj],A[ll][jj],FR[ll][jj])
        end
        FL[ll] = Array{Array{Complex{Float64},2},1}(na)
        FL[ll][1] = coef[ll]*eye(1)
    end

    FLH[1] = coef[numv]*ones(1,1,1)
    FRH[na] = ones(1,1,1)
    for jj = na:-1:2;
        FRH[jj-1] = update_re(Ain[jj],H[jj],A[numv][jj],FRH[jj])
    end

    U,S,Vd = eye(1),eye(1),eye(1)
    #sweep right

    for jj = 1:na-1;

        av = A_prod_LR_two_site(FL[1][jj],A[1][jj],A[1][jj+1],FR[1][jj+1])
        for ll = 2:numv-1
            av += A_prod_LR_two_site(FL[ll][jj],A[ll][jj],A[ll][jj+1],FR[ll][jj+1])
        end
        av += A_prod_LOR_two_site(FLH[jj],A[numv][jj],H[jj],A[numv][jj+1],H[jj+1],FRH[jj+1])
        U,S,Vd = svdecon(reshape(av,dims[jj]*size(A[1][jj],2),
            size(A[1][jj+1],2)*dims[jj+2]))
        S = S[S.>0]
        stot = sum(S.^2)
        dims[jj+1] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,dlim)
        A0[jj] = reshape(U[:,1:dims[jj+1]],dims[jj],size(A[1][jj],2),dims[jj+1])
        for ll = 1:numv-1
            FL[ll][jj+1] = update_lenv(A0[jj],A[ll][jj],FL[ll][jj])
        end
        FLH[jj+1] = update_le(A0[jj],H[jj],A[numv][jj],FLH[jj])

    end

    A0[na] = reshape(diagm(S[1:dims[na]])*(Vd[:,1:dims[na]]'),dims[na],size(A[1][na],2),dims[na+1])

    return A0

end


"""
Compresses a left-canonical MPS to a maximum bond dimension D_max using a
relative tolerance tol in the trimming, giving a right-canonical MPS.
"""
function compressionSVD!(mps, dmax, tol = 1e-15)
    
    n = length(mps)
    lambda = Array{Array}(n - 1)
    err_trim = Array{Float64}(n)
    err_glob = Float64
    ent_ent = Array{Float64}(n)
    dims = Array{Int64}(n - 1)
    dnew = 1
    r = 1

    for i = n:-1:1
    
        dl, d, dr = size(mps[i])
        mps_site = reshape(mps[i], (dl*d, dr))*r
        #u, s, v = svdecon(reshape(mps_site, (dl, dnew*d)))
        u, s, v = svd(reshape(mps_site, (dl, dnew*d)))
        s = s[s .> 0]
        stot = sum(abs2.(s))
        b = dnew
        dnew = minimum([sum(cumsum(abs2.(s)/stot) .< (1 - tol)) + 1, dmax])
        err_trim[i] = sum(abs2.(s[(dnew + 1):end]))/stot
        ent_ent[i] = -2*sum(abs2.(s).*log.(s))
        mps[i] = reshape(v[:, 1:dnew]', (dnew, d, b))
        r = u[:, 1:dnew]*diagm(s[1:dnew])
    
    end

    err_glob = 1 - abs2(r[1])

    return err_glob, err_trim, ent_ent

end

