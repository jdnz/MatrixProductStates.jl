using MAT
using MatrixProductStates
BLAS.set_num_threads(2)

# spin model parameters
const na = 12 # number of atoms
const gam_1d = ones(na) # coupling of atoms
const gam_eg = 0.0 # eg spontaneous decay rate
const del_p = 0.0 # detuning of pump beam
const k_wg = 0.2*pi # waveguide wavevector
const rj = collect(1.0:na) # atom positions
const k_in = na/(na + 1)*pi # pump beam wavevector
const f_amp = 0.0 # probe amplitude
f(t) = f_amp # time envelope of pump beam

# simulation parameters
const dt = 0.02 # time step
const t_fin = 200.0 # final time
const d = 2 # dimension of atom Hilbert space
const d_max = 100 # maximum bond dimension
const measure_int = 5 # number of time steps per observable measurement

const path_data = "/home/jdouglas/data/" # save directory
const base_filename = string(path_data,"SMPhaseSup_Dens_N", na, "_D", d_max,
	"_Tf", t_fin, "_dt", dt, "_kwg", round(k_wg/pi, 3))

# local spin operators
const hgg = [1 0; 0 0]
const hge = [0 1; 0 0]
const heg = hge'
const hee = [0 0; 0 1]
const id = eye(2)
const TN = Complex{Float64}
const TA = Array

function time_evolve()

    # construct left-canonical initial MPS product state where the 
    # site density matrix is represented as a vector 
    # [|g><g|, |g><e|, |e><g|, |e><e|]
    # rho, dims = mpsproductstate(TN, TA, na, d_max, d^2,
    #	[0.0, 0.0, 0.0, 1.0])
    rho, dims = mpsproductstate(TN, TA, na, d_max, d^2,
        jj -> [0.5, 0.5*exp(-im*k_in*rj[jj]),
               0.5*exp(im*k_in*rj[jj]), 0.5])

    # create measurement operators
    IDlmps = makemps(TN, TA, [jj->[1; 0; 0; 1]], na, d^2)
	ELmpo = makempo(TN, TA, [jj->id jj->im*sqrt(gam_1d[jj]/2)*hge*exp(im*k_wg*rj[jj]);
        				jj->0 jj->id], na, d)
	ILmpo = applyMPOtoMPO(ELmpo, conj_mpo(ELmpo))
	ILlmps = mpo_to_mps(TN, TA, ILmpo)
	IL2mpo = applyMPOtoMPO(applyMPOtoMPO(ELmpo, ILmpo), conj_mpo(ELmpo))
	IL2lmps = mpo_to_mps(TN, TA, IL2mpo)
	NEmpo = makempo(TN, TA, [jj->id jj->hee; jj->0 jj->id], na, d)
	NElmps = mpo_to_mps(TN, TA, NEmpo)
	NGmpo = makempo(TN, TA, [jj->id jj->hgg; jj->0 jj->id], na, d)
	NGlmps = mpo_to_mps(TN, TA, NGmpo)
	ERmpo = makempo(TN, TA, [jj->id jj->im*sqrt(gam_1d[jj]/2)*hge*exp(-im*k_wg*rj[jj]);
                		jj->0 jj->id], na, d)
    ERmpo[1][1, :, 2, :] = f(0.0)*id + 
    	im*sqrt(gam_1d[1]/2)*hge*exp(-im*k_wg*rj[1])
    EPmpo = makempo(TN, TA, [jj->id jj->im*sqrt(gam_1d[jj]/2)*hge*exp(-im*k_in*rj[jj]);
                        jj->0 jj->id], na, d)
    EPmpo[1][1, :, 2, :] = f(0.0)*id + 
        im*sqrt(gam_1d[1]/2)*hge*exp(-im*k_in*rj[1])
	IRmpo = applyMPOtoMPO(ERmpo,conj_mpo(ERmpo))
	IR2mpo = applyMPOtoMPO(applyMPOtoMPO(ERmpo, IRmpo),
		conj_mpo(ERmpo))
    ERlmps = mpo_to_mps(TN, TA, ERmpo)
    EPlmps = mpo_to_mps(TN, TA, EPmpo)
	IRlmps = mpo_to_mps(TN, TA, IRmpo)
	IR2lmps = mpo_to_mps(TN, TA, IR2mpo)

    # excitation manifold projectors
    max_ex = 10
    NEnlmps = Array{Array{TA{TN, 3}, 1}}(max_ex + 1)
    for ex = 0:max_ex
        NEnlmps[ex+1] = mpo_to_mps(TN, TA, nex_projector(TN, TA, na, ex))
    end
 
    # step times and measurement times
    t = 0.0:dt:t_fin
    t_m = t[1:measure_int:end]
    tstep = length(t) - 1

    # initialize output measures
    tr_rho = zeros(TN, length(t_m))
    e_pop = zeros(TN, length(t_m))
    e_pop_jl = zeros(TN, na, na, length(t_m))
    g_pop = zeros(TN, length(t_m))
    E_r = zeros(TN, length(t_m))
    E_p = zeros(TN, length(t_m))
    I_r = zeros(TN, length(t_m))
    I2_r = zeros(TN, length(t_m))
    I_l = zeros(TN, length(t_m))
    I2_l = zeros(TN, length(t_m))
    times = zeros(tstep, 4)
    nex = zeros(TN, max_ex + 1, length(t_m))

    #initial values of output measures
    e_pop[1] = scal_prod_no_conj(NElmps, rho)
    measure_excitations!(rho, (@view e_pop_jl[:,:,1]), IDlmps)
    g_pop[1] = scal_prod_no_conj(NGlmps, rho)
    tr_rho[1] = scal_prod_no_conj(IDlmps, rho)
    E_r[1] = scal_prod_no_conj(ERlmps, rho)
    E_p[1] = scal_prod_no_conj(EPlmps, rho)
    I_r[1] = scal_prod_no_conj(IRlmps, rho)
    I2_r[1] = scal_prod_no_conj(IR2lmps, rho)
    I_l[1] = scal_prod_no_conj(ILlmps, rho)
    I2_l[1] = scal_prod_no_conj(IL2lmps, rho)
    for n = 1:max_ex + 1
        nex[n, 1] = scal_prod_no_conj(NEnlmps[n], rho)
    end

    # setup temporary rho and environments for RK4 algorithm
    rho1, _ = mpsgroundstate(TN, TA, na, d_max, d^2)
    rho2, _ = mpsgroundstate(TN, TA, na, d_max, d^2)
    rho3, _ = mpsgroundstate(TN, TA, na, d_max, d^2)
    mpo_size = 6
    env1 = build_env(TN, TA, dims, dims)
    env2 = build_env(TN, TA, dims, dims)
    env3 = build_env(TN, TA, dims, dims)
    envop = build_env(TN, TA, dims, dims, ones(na + 1)*mpo_size)

    # time evolution operators for RK4 algorithm
    L1 = construct_L_spin_model(TN, TA, 1, dt/2, t[1])
    L2 = construct_L_spin_model(TN, TA, 0, 1, t[1] + dt/2)
    L3 = construct_L_spin_model(TN, TA, 1, dt/2, t[2])
    L = [L1, L2, L3]

    println("Initialisation complete")
    println(typeof(L))
    println(typeof(rho))
    println(typeof(env1))
    println(typeof(envop))

    #EVOLUTION
    for i = 1:tstep
        time_in = time();

        # update time-evolution operator (only for time-dependent L)
        #update_L_spin_model!(L1, 1, dt/2, t[i])
        #update_L_spin_model!(L2, 0, 1, t[i]+dt/2)
        #update_L_spin_model!(L3, 1, dt/2, t[i+1])
		#times[i,1] = time() - time_in

        # Runge-Kutta 4th order time step
        RK4stp_apply_H_half!(dt, L, rho, rho1, rho2, rho3, env1, env2, env3, envop)
        times[i,2] = time() - times[i,1] - time_in
                
        # measurements (every measure_int time steps)
        if rem(i, measure_int) == 0
            mind = div(i, measure_int) + 1
            # atomic state populations
            e_pop[mind] = scal_prod_no_conj(NElmps, rho)
            g_pop[mind] = scal_prod_no_conj(NGlmps, rho)
            measure_excitations!(rho, (@view e_pop_jl[:,:,mind]), IDlmps)
            # number in each excitation manifold
            for n = 1:max_ex + 1
                nex[n, mind] = scal_prod_no_conj(NEnlmps[n], rho)
            end
            # trace
            tr_rho[mind] = scal_prod_no_conj(IDlmps, rho)
            # output right field update
            ERmpo[1][1, :, 2, :] = f(t[i + 1])*id +
            	im*sqrt(gam_1d[1]/2)*hge*exp(-im*k_wg*rj[1])
            EPmpo[1][1, :, 2, :] = f(t[i + 1])*id +
                im*sqrt(gam_1d[1]/2)*hge*exp(-im*k_in*rj[1])
            IRupdate =  apply_site_MPOtoMPO(ERmpo[1],
             	conj_site_mpo(ERmpo[1]))
            IR2update =  apply_site_MPOtoMPO(
             	apply_site_MPOtoMPO(ERmpo[1], IRupdate),
             	conj_site_mpo(ERmpo[1]))
            ERlmps[1] = mpo_to_mps_site(ERmpo[1])
            EPlmps[1] = mpo_to_mps_site(EPmpo[1])
			IRlmps[1] = mpo_to_mps_site(IRupdate)
			IR2lmps[1] = mpo_to_mps_site(IR2update)
            # field observables
            E_r[mind] = scal_prod_no_conj(ERlmps, rho)
            E_p[mind] = scal_prod_no_conj(EPlmps, rho)
            I_r[mind] = scal_prod_no_conj(IRlmps, rho)
            I2_r[mind] = scal_prod_no_conj(IR2lmps, rho)
            I_l[mind] = scal_prod_no_conj(ILlmps, rho)
            I2_l[mind] = scal_prod_no_conj(IL2lmps, rho)
            times[i,3] = time() - times[i, 2] - times[i, 1] - time_in
        end

        # write temporary file
        if rem(i,measure_int*100) == 0
            write_data_file(string(base_filename, "_temp.mat"), t_m, e_pop,
                            e_pop_jl, g_pop, nex, tr_rho, E_r, E_p, I_r,
                            I2_r, I_l, I2_l, rho, times)
        end

    end

    # saving final data
    write_data_file(string(base_filename, ".mat"), t_m, e_pop,
                        e_pop_jl, g_pop, nex, tr_rho, E_r, E_p, I_r,
                        I2_r, I_l, I2_l, rho, times)

    return rho

end

function write_data_file(filename, t_m, e_pop, e_pop_jl, g_pop, nex,
                         tr_rho, E_r, E_p, I_r, I2_r, I_l, I2_l, rho, times)

    file = matopen(string(filename), "w")
    write(file, "TN",string(TN))
    write(file, "TA",string(TA))
    write(file, "na", na)
    write(file, "gam_1d", gam_1d)
    write(file, "gam_eg", gam_eg)
    write(file, "del_p", del_p)
    write(file, "k_wg", k_wg)
    write(file, "rj", rj)
    write(file, "k_in", k_in)
    write(file, "f_amp", f_amp)
    write(file, "f", f.(t_m))
    write(file, "dt", dt)
    write(file, "t_fin", t_fin)
    write(file, "d_max", d_max)
    write(file, "measure_int", measure_int)

    write(file, "t_m", collect(t_m))
    write(file, "e_pop", e_pop)
    write(file, "e_pop_jl", e_pop_jl)
    write(file, "g_pop", g_pop)
    write(file, "nex", nex)
    write(file, "tr_rho", tr_rho)
    write(file, "E_r", E_r)
    write(file, "E_p", E_p)
    write(file, "I_r", I_r)
    write(file, "I2_r", I2_r)
    write(file, "I_l", I_l)
    write(file, "I2_l", I2_l)
    write(file,"rho", collect.(rho))
    write(file, "times", sum(times,1))
    close(file)

end

# Construct MPO of L operator to evolve density matrix in time

function construct_L_spin_model(::Type{TN}, ::Type{TA}, con, delt, time) where {TN, TA}

    drj = diff(rj)
    LMPO = Array{TA{TN, 4}, 1}(na)

    ph = exp.(im*k_wg*drj)
    cp = sqrt.(delt*gam_1d/2)

    LMPO[1] = zeros(TN, 1, 4, 6, 4)
    LMPO[1][1, :, 1, :] = kron(id, id)
    LMPO[1][1, :, 2, :] = cp[1]*ph[1]*(kron(id, hge) - kron(hge', id))
    LMPO[1][1, :, 3, :] = cp[1]*ph[1]*(kron(hge, id))
    LMPO[1][1, :, 4, :] = cp[1]'*ph[1]'*(kron(id, hge))
    LMPO[1][1, :, 5, :] = cp[1]'*ph[1]'*(kron(hge, id) - kron(id, hge'))
    LMPO[1][1, :, 6, :] = local_l(1, con, delt, time)

    LMPO[na] = zeros(TN, 6, 4, 1, 4)
    LMPO[na][1, :, 1, :] = local_l(na, con, delt, time)
    LMPO[na][2, :, 1, :] = cp[na]*kron(hge, id)
    LMPO[na][3, :, 1, :] = -cp[na]*(kron(hge', id) - kron(id, hge))
    LMPO[na][4, :, 1, :] = -cp[na]'*(kron(id, hge') - kron(hge, id))
    LMPO[na][5, :, 1, :] = cp[na]'*kron(id, hge)
    LMPO[na][6, :, 1, :] = kron(id, id)

    for jj = 2:na-1
        LMPO[jj] = zeros(TN, 6, 4, 6, 4)
	    LMPO[jj][1, :, 1, :] = kron(id, id)
	    LMPO[jj][1, :, 2, :] = cp[jj]*ph[jj]*(kron(id, hge) - kron(hge', id))
	    LMPO[jj][1, :, 3, :] = cp[jj]*ph[jj]*(kron(hge, id))
	    LMPO[jj][1, :, 4, :] = cp[jj]'*ph[jj]'*(kron(id, hge))
	    LMPO[jj][1, :, 5, :] = cp[jj]'*ph[jj]'*(kron(hge, id) - kron(id, hge'))
        LMPO[jj][1, :, 6, :] = local_l(jj, con, delt, time)
        LMPO[jj][2, :, 6, :] = cp[jj]*kron(hge, id)
        LMPO[jj][3, :, 6, :] = -cp[jj]*(kron(hge', id) - kron(id, hge))
        LMPO[jj][4, :, 6, :] = -cp[jj]'*(kron(id, hge') - kron(hge, id))
        LMPO[jj][5, :, 6, :] = cp[jj]'*kron(id, hge)
        LMPO[jj][6, :, 6, :] = kron(id, id)
	    LMPO[jj][2, :, 2, :] = ph[jj]*kron(id, id)
        LMPO[jj][3, :, 3, :] = ph[jj]*kron(id, id)
        LMPO[jj][4, :, 4, :] = ph[jj]'*kron(id, id)
        LMPO[jj][5, :, 5, :] = ph[jj]'*kron(id, id)
    end

    return LMPO

end

# Update L for time-varying pump field

function update_L_spin_model!(LMPO, con, delt, time)

    LMPO[1][1, :, 6, :] = local_l(1, con, delt, time)
    LMPO[na][1, :, 1, :] = local_l(na, con, delt, time)
    for jj = 2:na-1
        LMPO[jj][1, :, 6, :] = local_l(jj, con, delt, time)
    end

end

# Local L terms

function local_l(jj, con, delt, time)

    con/na*kron(id, id) +
        delt*((im*del_p - (gam_eg + gam_1d[jj])/2)*kron(hee, id) -
        (im*del_p + (gam_eg + gam_1d[jj])/2)*kron(id, hee) +
        (gam_eg + gam_1d[jj])*kron(hge, hge) + 
        im*f(time)*sqrt(gam_1d[jj]/2)*exp(im*k_in*rj[jj])*
            (kron(hge', id) - kron(id, hge)) +
        im*f(time)*sqrt(gam_1d[jj]/2)*exp(-im*k_in*rj[jj])*
            (kron(hge, id) - kron(id, hge')))

end

# Measure <\sigma_ee^j \sigma_ee^l>

function measure_excitations!(rho, epopjl, IDlmps)

    na = length(rho)
    for jj = 1:na
        IDlmps[jj][1, 1, 1] = 0.0
        epopjl[jj, jj] = scal_prod_no_conj(IDlmps, rho)
        for ll = (jj + 1):na
            IDlmps[ll][1, 1, 1] = 0.0
            epopjl[jj, ll] = scal_prod_no_conj(IDlmps, rho)
            IDlmps[ll][1, 1, 1] = 1.0
        end
        IDlmps[jj][1, 1, 1] = 1.0
    end

end

function nex_projector(::Type{TN}, ::Type{TA}, na, nex) where {TN, TA}

    P = Array{TA{TN, 4}, 1}(na)
    dl = 1
    for n = 1:na
        dr = min(nex, n, na - n) + 1
        A = zeros(TN, dl, d, dr, d)
        if n <= na - nex
            for i = 1:min(dl, dr)
                A[i, :, i, :] = hgg
            end
            for i = 2:dr 
                A[i - 1, :, i, :] = hee
            end
        else
            for i = 1:min(dl, dr)
                A[i, :, i, :] = hee
            end
            for i = 2:dl
                A[i, :, i - 1, :] = hgg
            end
        end
        P[n] = A
        dl = dr
    end

    return P

end

time_evolve()
