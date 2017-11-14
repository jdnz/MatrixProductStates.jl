"""
function to find ground state MPS for MPO H given input guess A in left
canonical form (normalisation not necessary). Full matrix calc.
"""

function ground_state_search_full(A,H,maxsweeps)

    na = length(A)
    At = deepcopy(A)
    FL = Array{Array{Complex{Float64},3},1}(na)
    FR = Array{Array{Complex{Float64},3},1}(na)

    FL[1] = ones(1,1,1)
    for jj = 1:na-1
        FL[jj+1] = update_le(A[jj],H[jj],A[jj],FL[jj])
    end
    FR[na] = ones(1,1,1)

    for sweep = 1:maxsweeps

        #sweep left
        for jj = na:-1:2
            HM = reshape(O_prod_LR(FL[jj],H[jj],FR[jj]),
                    size(FL[jj],1)*size(H[jj],2)*size(FR[jj],1),
                    size(FL[jj],1)*size(H[jj],2)*size(FR[jj],1))
            ae,av = eigs(HM; nev = 1, which = :SM, v0 = At[jj][:])
            println(ae)
            Q,R = qr(reshape(av,size(FL[jj],1),size(H[jj],2)*size(FR[jj],1))')
            At[jj] = reshape(Q',size(Q,2),size(At[jj],2),size(At[jj],3))
            FR[jj-1] = update_re(At[jj],H[jj],At[jj],FR[jj])
        end
            HM = reshape(O_prod_LR(FL[1],H[1],FR[1]),
                    size(FL[1],1)*size(H[1],2)*size(FR[1],1),
                    size(FL[1],1)*size(H[1],2)*size(FR[1],1))
            ae,av = eigs(HM; nev = 1, which = :SM, v0 = At[1][:])
            At[1] = reshape(av,size(FL[1],1),size(H[1],2),size(FR[1],1))

        #sweep right
        for jj = 1:na-1;
            HM = reshape(O_prod_LR(FL[jj],H[jj],FR[jj]),
                    size(FL[jj],1)*size(H[jj],2)*size(FR[jj],1),
                    size(FL[jj],1)*size(H[jj],2)*size(FR[jj],1))
            ae,av = eigs(HM; nev = 1, which = :SM, v0 = At[jj][:])
            println(ae)
            Q,R = qr(reshape(av,size(FL[jj],1)*size(H[jj],2),size(FR[jj],1)));
            At[jj] = reshape(Q,size(At[jj],1),size(At[jj],2),size(Q,2))
            FL[jj+1] = update_le(At[jj],H[jj],At[jj],FL[jj]);
        end
            HM = reshape(O_prod_LR(FL[na],H[na],FR[na]),
                    size(FL[na],1)*size(H[na],2)*size(FR[na],1),
                    size(FL[na],1)*size(H[na],2)*size(FR[na],1))
            ae,av = eigs(HM; nev = 1, which = :SM, v0 = At[na][:])
            At[na] = reshape(av,size(FL[na],1),size(H[na],2),size(FR[na],1))

    end

    no = sum(abs2(At[na][:]))
    return At, no

end

"""
function to find ground state MPS for MPO H given input guess A in left
canonical form (normalisation not necessary). Matrix as linear map.
"""

function ground_state_search!(At,H,maxsweeps,converge_rat,eigs_tol)

    na = length(At)
    FL = Array{Array{Complex{Float64},3},1}(na)
    FR = Array{Array{Complex{Float64},3},1}(na)
    ae = 0.0
    aestart = 0.0

    FL[1] = ones(1,1,1)
    for jj = 1:na-1
        FL[jj+1] = update_le(At[jj],H[jj],At[jj],FL[jj])
    end
    FR[na] = ones(1,1,1)

    for sweep = 1:maxsweeps

        println("sweeping left")
        #sweep left
        for jj = na:-1:2
            HM = LinearMap(x->A_prod_LOR_eigs(FL[jj],x,H[jj],FR[jj]),
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
            av,rv = qr(reshape(av,size(FL[jj],1),size(H[jj],2)*size(FR[jj],1))')
            At[jj] = reshape(av',size(av,2),size(At[jj],2),size(At[jj],3))
            FR[jj-1] = update_re(At[jj],H[jj],At[jj],FR[jj])
            mo = At[jj-1]
            @tensor mo[-1,-2,-3] := mo[-1,-2,1]*rv[-3,1]
            At[jj-1] = mo
        end

        #HM = LinearMap(x->A_prod_LOR_eigs(FL[1],x,H[1],FR[1]),
        #    	size(At[1][:],1),Complex128; ishermitian=true)
        #ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[1][:],
        #        maxiter = 1000, tol = eigs_tol)
        #println("site ", 1, " evalue is ", ae[1])
        #At[1] = reshape(av,size(FL[1],1),size(H[1],2),size(FR[1],1))

        println("sweeping right")
        #sweep right
        for jj = 1:na-1;
            HM = LinearMap(x->A_prod_LOR_eigs(FL[jj],x,H[jj],FR[jj]),
            	size(At[jj][:],1),Complex128; ishermitian=true)
            ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[jj][:],
                maxiter = 1000, tol = eigs_tol)
            println("site ", jj, " evalue is ", ae[1])
            #Q,R = qr(reshape(av,size(FL[jj],1)*size(H[jj],2),size(FR[jj],1)));
            #av = qrfact!(reshape(av,size(FL[jj],1)*size(H[jj],2),size(FR[jj],1)))
            #av = full(av[:Q])
            av, rv = qr(reshape(av,size(FL[jj],1)*size(H[jj],2),size(FR[jj],1)))
            At[jj] = reshape(av,size(At[jj],1),size(At[jj],2),size(av,2))
            FL[jj+1] = update_le(At[jj],H[jj],At[jj],FL[jj]);
            mo = At[jj+1]
            @tensor mo[-1,-2,-3] := rv[-1,1]*mo[1,-2,-3]
            At[jj+1] = mo
        end

        HM = LinearMap(x->A_prod_LOR_eigs(FL[na],x,H[na],FR[na]),
            	size(At[na][:],1),Complex128; ishermitian=true)
        ae,av = eigs(HM; nev = 1, which = :SR, v0 = At[na][:],
                maxiter = 1000, tol = eigs_tol)
        println("site ", na, " evalue is ", ae[1])
        At[na] = reshape(av,size(FL[na],1),size(H[na],2),size(FR[na],1))

        if abs(ae[1]) > converge_rat*abs(aestart) || abs(ae[1])<1e-10
            return
        end

    end

    no = sum(abs2(At[na][:]))
    return ae

end
