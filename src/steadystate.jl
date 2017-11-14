"""
function to find steady state for MPO L given input guess A in left
canonical form (normalisation not necessary). Matrix as linear map.
"""

function steady_state_search!(At,L,Lconj,maxsweeps,converge_rat,eigs_tol)

    na = length(At)
    FL = Array{Array{Complex{Float64},4},1}(na)
    FR = Array{Array{Complex{Float64},4},1}(na)
    ae = 0.0
    aestart = 0.0

    FL[1] = ones(1,1,1,1)
    for jj = 1:na-1
        FL[jj+1] = update_lenv_two_op(At[jj],L[jj],Lconj[jj],At[jj],FL[jj])
    end
    FR[na] = ones(1,1,1,1)

    for sweep = 1:maxsweeps

        println("sweeping left")
        #sweep left
        for jj = na:-1:2
            HM = LinearMap(x->A_prod_L_OO_R_eigs(FL[jj],x,L[jj],Lconj[jj],FR[jj]),
                size(At[jj][:],1),Complex128; ishermitian=true)
            ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[jj][:],
                maxiter = 1000, tol = eigs_tol)
            println("site ", jj, " evalue is ", ae[1])
            if jj == na
                aestart = ae[1]
            end
            #Q,R = qr(reshape(av,size(FL[jj],1),size(H[jj],2)*size(FR[jj],1))')
            #av = qrfact!(reshape(av,size(FL[jj],1),size(H[jj],2)*size(FR[jj],1))')
            #av = full(av[:Q])
            av,rv = qr(reshape(av,size(FL[jj],1),size(L[jj],2)*size(FR[jj],1))')
            At[jj] = reshape(av',size(av,2),size(At[jj],2),size(At[jj],3))
            FR[jj-1] = update_renv_two_op(At[jj],L[jj],Lconj[jj],At[jj],FR[jj])
            mo = At[jj-1]
            @tensor mo[-1,-2,-3] := mo[-1,-2,1]*rv[-3,1]
            At[jj-1] = mo
        end

        #HM = LinearMap(x->A_prod_LOR_eigs(FL[1],x,H[1],FR[1]),
        #       size(At[1][:],1),Complex128; ishermitian=true)
        #ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[1][:],
        #        maxiter = 1000, tol = eigs_tol)
        #println("site ", 1, " evalue is ", ae[1])
        #At[1] = reshape(av,size(FL[1],1),size(H[1],2),size(FR[1],1))

        println("sweeping right")
        #sweep right
        for jj = 1:na-1;
            HM = LinearMap(x->A_prod_L_OO_R_eigs(FL[jj],x,L[jj],Lconj[jj],FR[jj]),
                size(At[jj][:],1),Complex128; ishermitian=true)
            ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[jj][:],
                maxiter = 1000, tol = eigs_tol)
            println("site ", jj, " evalue is ", ae[1])
            #Q,R = qr(reshape(av,size(FL[jj],1)*size(H[jj],2),size(FR[jj],1)));
            #av = qrfact!(reshape(av,size(FL[jj],1)*size(H[jj],2),size(FR[jj],1)))
            #av = full(av[:Q])
            av, rv = qr(reshape(av,size(FL[jj],1)*size(L[jj],2),size(FR[jj],1)))
            At[jj] = reshape(av,size(At[jj],1),size(At[jj],2),size(av,2))
            FL[jj+1] = update_lenv_two_op(At[jj],L[jj],Lconj[jj],At[jj],FL[jj]);
            mo = At[jj+1]
            @tensor mo[-1,-2,-3] := rv[-1,1]*mo[1,-2,-3]
            At[jj+1] = mo
        end

        HM = LinearMap(x->A_prod_L_OO_R_eigs(FL[na],x,L[na],Lconj[na],FR[na]),
                size(At[na][:],1),Complex128; ishermitian=true)
        ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[na][:],
                maxiter = 1000, tol = eigs_tol)
        println("site ", na, " evalue is ", ae[1])
        At[na] = reshape(av,size(FL[na],1),size(L[na],2),size(FR[na],1))

        if abs(ae[1]) > converge_rat*abs(aestart) || abs(ae[1])<1e-10
            return
        end

    end

    no = sum(abs2(At[na][:]))
    return ae

end

function steady_state_search_two_site!(At,L,Lconj,maxsweeps,converge_rat,eigs_tol,D_lim,svd_tol,Dims)

    na = length(At)
    FL = Array{Array{Complex{Float64},4},1}(na)
    FR = Array{Array{Complex{Float64},4},1}(na)
    ae = 0.0
    aestart = 0.0

    FL[1] = ones(1,1,1,1)
    for jj = 1:na-2
        FL[jj+1] = update_lenv_two_op(At[jj],L[jj],Lconj[jj],At[jj],FL[jj])
    end
    FR[na] = ones(1,1,1,1)

    for sweep = 1:maxsweeps

        println("sweeping left")
        #sweep left
        for jj = na:-1:2
            vin = neighbour_contract(At[jj-1],At[jj])
            HM = LinearMap(x->A_prod_L_OO_R_two_site(FL[jj-1],x,L[jj-1],Lconj[jj-1],
                L[jj],Lconj[jj],FR[jj]),
                size(vin,1),Complex128; ishermitian=true)
            ae,av = eigs(HM; nev = 1, ncv = 20, which = :SR, v0 = vin,
                maxiter = 2000, tol = eigs_tol)
            if jj == na
                aestart = ae[1]
            end
            U,S,Vd = svdecon(reshape(av,Dims[jj-1]*size(At[jj-1],2),
                size(At[jj],2)*Dims[jj+1]))
            S = S[S.>0]
            stot = sum(S.^2)
            Dims[jj] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,D_lim)
            println("site ", jj, " evalue is ", ae[1], " bond dim ", Dims[jj])
            At[jj] = reshape(Vd[:,1:Dims[jj]]',Dims[jj],size(At[jj],2),Dims[jj+1])
            FR[jj-1] = update_renv_two_op(At[jj],L[jj],Lconj[jj],At[jj],FR[jj])
            At[jj-1] = reshape(U[:,1:Dims[jj]]*diagm(S[1:Dims[jj]]),Dims[jj-1],size(At[jj-1],2),Dims[jj])
        end

        #HM = LinearMap(x->A_prod_L_OO_R_eigs(FL[1],x,L[1],Lconj[1],FR[1]),
        #       size(At[1][:],1),Complex128; ishermitian=true)
        #ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[1][:],
        #        maxiter = 2000, tol = eigs_tol)
        #println("site ", 1, " evalue is ", ae[1])
        #At[1] = reshape(av,size(FL[1],1),size(L[1],2),size(FR[1],1))

        println("sweeping right")
        #sweep right
        for jj = 1:na-1;
            vin = neighbour_contract(At[jj],At[jj+1])
            HM = LinearMap(x->A_prod_L_OO_R_two_site(FL[jj],x,L[jj],Lconj[jj],
                L[jj+1],Lconj[jj+1],FR[jj+1]),
                size(vin,1),Complex128; ishermitian=true)
            ae,av = eigs(HM; nev = 1, ncv = 20, which = :SR, v0 = vin,
                maxiter = 2000, tol = eigs_tol)
            U,S,Vd = svdecon(reshape(av,Dims[jj]*size(At[jj],2),
                size(At[jj+1],2)*Dims[jj+2]))
            S = S[S.>0]
            stot = sum(S.^2)
            Dims[jj+1] = min(sum(cumsum(S.^2/stot) .< 1-svd_tol)+1,D_lim)
            println("site ", jj, " evalue is ", ae[1], " bond dim ", Dims[jj+1])
            At[jj] = reshape(U[:,1:Dims[jj+1]],Dims[jj],size(At[jj],2),Dims[jj+1])
            At[jj+1] = reshape(diagm(S[1:Dims[jj+1]])*(Vd[:,1:Dims[jj+1]]'),Dims[jj+1],size(At[jj+1],2),Dims[jj+2])
            FL[jj+1] = update_lenv_two_op(At[jj],L[jj],Lconj[jj],At[jj],FL[jj]);
        end

        #HM = LinearMap(x->A_prod_L_OO_R_eigs(FL[na],x,L[na],Lconj[na],FR[na]),
        #        size(At[na][:],1),Complex128; ishermitian=true)
        #ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[na][:],
        #        maxiter = 2000, tol = eigs_tol)
        #println("site ", na, " evalue is ", ae[1])
        #At[na] = reshape(av,size(FL[na],1),size(L[na],2),size(FR[na],1))

        if abs(ae[1]) > converge_rat*abs(aestart) || abs(ae[1])<1e-10
            return
        end

    end

    no = sum(abs2(At[na][:]))
    return ae, Dims

end
