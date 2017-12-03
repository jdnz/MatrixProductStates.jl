using MAT
using MatrixProductStates
BLAS.set_num_threads(1)

#Spin model parameters
const na = 70 
const od = 33
const ad = 2.5*60/na #atomic distance in micro m
const cloud_sig = 23.6344/ad
const gam_1d = exp.(-((1:na)-(na+1)/2).^2/2/cloud_sig^2)
const gam_1d = gam_1d/sum(gam_1d)/2*od
const gam_eg = 1.0
const del_p = 0.0
const k_wg = 0.5*pi 
const rj = collect(1.0:na) #sort!(rand(na)*na)#

#Pulse shape parameters
const k_in = k_wg    
const gam_exp = 2*pi*6.065e6
const f_amp_exp = sqrt.([0.0589,0.0721,0.0995,0.1637,0.3081,0.6395,1.2121,1.8898, 
    2.6039,4.2050,7.1291,10.4823,17.0816,27.1592,42.8378,71.7874]/gam_exp/1e-6)
const f_amp = f_amp_exp[1]

#Rydberg specific parameters
const om = 10.0/6.065/2.0
const k_c = 0.0
const del_s = 0.0
const gam_es = 0.0

# rmax = 20 na = 70 
const uu = [524.328 + 89.9671im, 524.328 - 89.9671im, 649.737 + 219.429im, 649.737 - 219.429im, 488.928]
const lam = [-0.228929 + 0.189648im, -0.228929 - 0.189648im, 0.146279 + 0.297638im, 0.146279 - 0.297638im, 0.480032] 

# rmax = 30 na = 60 
#uu = [1055.32, 1156.39 + 264.599im, 1156.39 - 264.599im, 781.661]
#lam = [-0.220397, 0.0354409 + 0.244048im, 0.0354409 - 0.244048im, 0.396262]
#uu = [15158.7, 11895.1 - 1929.93im, 11895.1 + 1929.93im, 2399.43, 16.4076]
#lam = [-0.103573, 0.015448 + 0.126927im, 0.015448 - 0.126927im, 0.30542, 0.606657]
#uu = [90866.8, 59173.0 - 18223.1im, 59173.0 + 18223.1im, 5257.17, 107.47, 1.26558]
#lam = [-0.0606795, 0.0112804 + 0.0784944im, 0.0112804 - 0.0784944im, 0.246432, 0.48336, 0.720283]

# rmax = 20 na = 60 
#const uu = [0.704528 + 135.691im, 0.704528 - 135.691im, 225.917]
#const lam = [0.226209 + 0.32374im, 0.226209 - 0.32374im, 0.475997]

#uu = [ 2255.07, 686.082 + 91.2384im, 686.082 - 91.2384im, 477.07]
#lam = [-0.0998312, 0.0646318 + 0.25377im, 0.0646318-0.25377im, 0.424938]
#uu = [-508.38 - 1018.61im, -508.38 + 1018.61im, -1546.1 - 1942.74im, -1546.1 + 1942.74im,
#        28.5203]
#lam = [ -0.0374226 + 0.193987im, -0.0374226 - 0.193987im, 0.250214 + 0.0887921im, 
#       0.250214 - 0.0887921im, 0.58104]
#uu = [-658.963 + 91.3732im, -658.963 - 91.3732im, 309.325 + 1085.49im, 309.325 - 1085.49im, 
#       709.204, 4.33617]
#lam = [-0.0921073 + 0.171392im, -0.0921073 - 0.171392im, 0.124872 + 0.194013im, 
#       0.124872 - 0.194013im, 0.382177, 0.670268]

# rmax = 20 na = 50
#uu = [1740.29, 2277.16 + 454.662im, 2277.16 - 454.662im, 945.959, 6.06471]
#lam = [-0.126487, 0.0183872 + 0.140738im, 0.0183872 - 0.140738im, 0.296793, 0.601841]
#uu = [ 57.1125, 170.91 + 211.667im, 170.91 - 211.667im, 336.281]
#lam = [ -0.298989, 0.0548962 + 0.260436im, 0.0548962 - 0.260436im, 0.382366]


# rmax = 20 na = 40
#uu = [19014.9 - 3038.05im, 19014.9 + 3038.05im, 2635.02, 47.4863, 0.531659]
#lam = [-0.0167917 + 0.0377784im, -0.0167917 - 0.0377784im, 0.155726, 0.386972, 0.653397]

# rmax = 20 na = 30
#uu = [-2652.81-1864.93im, -2652.81 + 1864.93im, 274.344, 5.91125, 0.0704647]
#lam = [0.0313486 + 0.0360902im, 0.0313486 - 0.0360902im, 0.179561, 0.407673, 0.666632]

# rmax = 20 na = 20
#uu = [1350.25, 707.234, 15.7581, 0.426333, 0.0053846]
#lam = [-0.0138335, 0.0500349, 0.19712, 0.418572, 0.672633]

# pure dephasing
const gam_gg = 0.0
const gam_ss = 40e3/6.065e6
const gam_ee = 0.0

# simulation parameters
const setrand = 1 # random number generator seed
const dt = 0.01
const t_fin = 10000.0
const d = 3
const d_max = 10
const measure_int = 5
const path_data = "/home/jdouglas/data/" # save directory
const base_filename = string(path_data, "Ryd_Jumps_N", na, "_D", d_max,
    "_Tf", t_fin, "_f", round(f_amp, 3), "_dt", dt, "_traj", setrand)

# envelope of input pulse
function f(t)

    trise = 0.8e-6*gam_exp
    tup = 2e-6*gam_exp

    if t < 0
        envelope = 0.0
    elseif t < trise
        envelope = (1.0 + cos(pi*(t - trise)/trise))/2
    elseif t <= tup + trise
        envelope = 1.0
    elseif t <= 2*trise + tup
        envelope = (1.0 + cos(pi*(t - tup - trise)/trise))/2
    else
        envelope = 0.0
    end

    f_amp*sqrt(envelope)

end

# local spin operators
const hgg = [1 0 0; 0 0 0; 0 0 0]
const hee = [0 0 0; 0 1 0; 0 0 0]
const hss = [0 0 0; 0 0 0; 0 0 1]
const hge = [0 1 0; 0 0 0; 0 0 0]
const heg = hge'
const hes = [0 0 0; 0 0 1; 0 0 0]
const hse = hes'
const id = eye(3)

# types used in mps representation
const TN = Complex{Float64}
const TA = Array

function time_evolve()

    # create jump and measurement operators
    jumpleft = makempo(TN, TA, [jj->id jj->im*sqrt(gam_1d[jj]/2)*exp(im*k_wg*rj[jj])*hge;
                                jj->0 jj->id], na, d)
    jumpright = makempo(TN, TA, [jj->id jj->im*sqrt(gam_1d[jj]/2)*exp(-im*k_wg*rj[jj])*hge;
                                 jj->0 jj->id], na, d)
    jumpright[1][1, :, 2, :] = f(0.0)*id + im*sqrt(gam_1d[1]/2)*exp(-im*k_wg*rj[1])*hge
    ir_mpo = applyMPOtoMPO(jumpright, conj_mpo(jumpright))
    ir2_mpo = applyMPOtoMPO(applyMPOtoMPO(jumpright, ir_mpo), conj_mpo(jumpright))

    # step times and measurement times
    t = 0.0:dt:t_fin
    t_m = t[1:measure_int:end]
    tstep = length(t) - 1

    # initialize output measures
    p_coh = zeros(tstep)
    p_sum = zeros(tstep)
    p_acc = zeros(tstep, 2na + 4)
    e_pop = zeros(na, length(t_m))
    s_pop = zeros(na, length(t_m))
    epop = zeros(na)
    spop = zeros(na)
    I_r = zeros(TN, length(t_m))
    I2_r = zeros(TN, length(t_m))
    times = zeros(tstep, 4)
    t_r = Float64[]
    t_l = Float64[]
    t_eg = Float64[]
    pos_eg = Float64[]
    t_ss = Float64[]
    pos_ss = Float64[]
    cont = zeros(Int64, 4)

    #Construct left-canonical MPS for the initial state and temp arrays for time evolution
    A, dims =  mpsgroundstate(TN, TA, na, d_max, d)
    A_coh = similar(A)
    A_r = similar(A)
    A_l = similar(A)
    A1 = similar(A)
    A2 = similar(A)
    A3 = similar(A)
    A_coh .= copy.(A)
    A_r .= copy.(A)
    A_l .= copy.(A)
    A1 .= copy.(A)
    A2 .= copy.(A)
    A3 .= copy.(A)
    mpo_size = 4 + length(lam)
    envop = build_env(TN, TA, dims, dims, ones(na + 1)*mpo_size)
    envop_jump = build_env(TN, TA, dims, dims, ones(na + 1)*2)
    env1 = build_env(TN, TA, dims, dims)
    env2 = build_env(TN, TA, dims, dims)
    env3 = build_env(TN, TA, dims, dims)    

    # initial measurements
    I_r[1] = scal_op_prod(A, ir_mpo, A)
    I2_r[1] = scal_op_prod(A, ir2_mpo, A)
    measure_excitations!((@view e_pop[:, 1]), (@view s_pop[:, 1]), A)

    # random jump variables
    srand(setrand)
    r1 = rand(tstep)
    r2 = rand(tstep)

    # create evolution operators for Runge-Kutta algorithm
    H1 = rydberg_hamiltonian(TN, TA, 1, dt/2, t[1])
    H2 = rydberg_hamiltonian(TN, TA, 0, 1, t[1] + dt/2)
    H3 = rydberg_hamiltonian(TN, TA, 1, dt/2, t[2])

    H = [H1, H2, H3]

    println("Initialisation complete")

    #EVOLUTION
    for i = 1:tstep
        time_in = time()

        # update time evolution operators
        update_rydberg_hamiltonian!(H1, 1, dt/2, t[i])
        update_rydberg_hamiltonian!(H2, 0, 1, t[i] + dt/2)
        update_rydberg_hamiltonian!(H3, 1, dt/2, t[i+1])

        # hoherent time evolution probability
        RK4stp_apply_H_half!(dt, H, A_coh, A, A1, A2, A3, env1, env2, env3, envop)
        p_coh[i] = normalize_lo!(A_coh)
        times[i, 1] = time() - time_in
        if r1[i] < p_coh[i]
            A .= copy.(A_coh)
            times[i,2] = 0
        else
            #Update waveguide jump operator
            jumpright[1][1,:,2,:] = f(t[i])*id + im*sqrt(gam_1d[1]/2)*exp(-im*k_wg*rj[1])*hge

            #Jump probabilities: decay into the waveguide
            compress_var_apply_H!(A_r, envop_jump, A, A, jumpright, 1)
            p_r = normalize_lo!(A_r)
            compress_var_apply_H!(A_l, envop_jump, A, A, jumpleft, 1)
            p_l = normalize_lo!(A_l)
        
            #Jump probabilities: decay outside the waveguide
            measure_excitations!(epop, spop, A)
            p_acc[i, 1] = 0.0
            for j = 1:na
                p_acc[i, j + 1] = p_acc[i, j] + dt*gam_eg*epop[j]
            end
            for j = 1:na
                p_acc[i, na + j + 1] = p_acc[i, na + j] + dt*gam_ss*spop[j]
            end
            p_acc[i, 2na + 2] = p_acc[i, 2na + 1] + dt*p_r
            p_acc[i, 2na + 3] = p_acc[i, 2na + 2] + dt*p_l
            p_sum[i] = p_acc[i, 2na + 3]
            println(p_sum[i] + p_coh[i])
            println(t[i])
            p_acc[i, :] = p_acc[i, :]/p_acc[i, 2na + 3]

            #Do jumps according to above probabilities
            if r2[i] < p_acc[i, na + 1]
                for j = 1:na
                    if r2[i] < p_acc[i, j + 1]
                        apply_site_operator!(A[j], hge)
                        leftorthonormalizationQR!(A)
                        push!(t_eg, t[i])
                        push!(pos_eg, j)
                        cont[1] += 1
                        break
                    end
                end
            elseif r2[i] < p_acc[i, 2na + 1]
                for j = 1:na
                    if r2[i] < p_acc[i, na + j + 1]
                        apply_site_operator!(A[j], hss)
                        leftorthonormalizationQR!(A)
                        push!(t_ss, t[i])
                        cont[2] += 1
                        break
                    end
                end
            elseif r2[i] < p_acc[i, 2na + 2]
                A .= copy.(A_r)
                push!(t_r, t[i])
                cont[3] += 1
            else
                A .= copy.(A_l)
                push!(t_l, t[i])
                cont[4] += 1
            end
            times[i, 2] = time() - times[i, 1] - time_in

        end

        # measurements (every measure_int time steps)
        if rem(i, measure_int) == 0

            mind = div(i, measure_int)+1
            # atomic state populations
            measure_excitations!((@view e_pop[:, mind]), (@view s_pop[:, mind]), A)
            times[i, 3] = time() - times[i, 2]  - times[i, 1] - time_in
                       
            # output right field update
            jumpright[1][1, :, 2, :] = f(t[i + 1])*id + 
                    im*sqrt(gam_1d[1]/2)*exp(-im*k_wg*rj[1])*hge
            
            ir_mpo[1] =  apply_site_MPOtoMPO(jumpright[1], conj_site_mpo(jumpright[1]))
            ir2_mpo[1] =  apply_site_MPOtoMPO(
                apply_site_MPOtoMPO(jumpright[1], ir_mpo[1]),
                conj_site_mpo(jumpright[1]))

            I_r[mind] = scal_op_prod(A, ir_mpo, A)
            I2_r[mind] = scal_op_prod(A, ir2_mpo, A)

            times[i,4] = time() - times[i, 3] - times[i, 2] - times[i, 1] - time_in

        end

        # saving temp file
        if rem(i, measure_int*2000) == 0

            write_data_file(string(base_filename,"_temp.mat"), t, t_m, e_pop, 
                            s_pop, p_coh, p_acc, p_sum, I_r, I2_r, cont, t_r,
                            t_l, t_eg, t_ss, pos_eg, A, times)

        end


    end
        
    # saving output file
    write_data_file(string(base_filename,".mat"), t, t_m, e_pop, 
                    s_pop, p_coh, p_acc, p_sum, I_r, I2_r, cont, t_r,
                    t_l, t_eg, t_ss, pos_eg, A, times)

end

function write_data_file(filename, t, t_m, e_pop, s_pop, p_coh, p_acc,
                         p_sum, I_r, I2_r, cont, t_r, t_l, t_eg, t_ss, 
                         pos_eg, A, times)

    file = matopen(string(filename),"w")
    write(file, "TN", string(TN))
    write(file, "TA", string(TA))
    write(file, "na", na)
    write(file, "gam_1d", gam_1d)
    write(file, "gam_eg", gam_eg)
    write(file, "del_p", del_p)
    write(file, "k_wg", k_wg)
    write(file, "rj", rj)
    write(file, "k_in", k_in)
    write(file, "f_amp", f_amp)
    write(file, "f", f.(t_m))
    write(file, "om", om)
    write(file, "k_c", k_c)
    write(file, "del_s", del_s)
    write(file, "gam_es", gam_es)
    write(file, "uu", uu)
    write(file, "lam", lam)
    write(file, "gam_gg", gam_gg)
    write(file, "gam_ss", gam_ss)
    write(file, "gam_ee", gam_ee)
    write(file, "dt", dt)
    write(file, "t_fin", t_fin)
    write(file, "d_max", d_max)
    write(file, "measure_int", measure_int)

    write(file, "t", collect(t))
    write(file, "t_m", collect(t_m))
    write(file, "e_pop", e_pop)
    write(file, "s_pop", s_pop)
    write(file, "p_coh", p_coh)
    write(file, "p_acc", p_acc)
    write(file, "p_sum", p_sum)
    write(file, "I_r", I_r)
    write(file, "I2_r", I2_r)
    write(file, "times", sum(times,1))
    write(file, "cont", cont)
    write(file, "t_r", t_r)
    write(file, "t_l", t_l)
    write(file, "t_eg", t_eg)
    write(file, "t_ss", t_ss)
    write(file, "pos_eg", pos_eg)
    write(file, "A", collect.(A))

    close(file)

end

function rydberg_hamiltonian(::Type{TN}, ::Type{TA}, con, delt, time) where {TN, TA}

    H = Array{TA{TN, 4}, 1}(na)

    no_exp = length(lam)
    max_ind = 4 + no_exp
    drj = diff(rj)
    ph = exp.(im*k_wg*drj)
    cp = sqrt.(delt*gam_1d/2)

    H[1] = zeros(TN, 1, 3, max_ind, 3)
    H[1][1, :, 1, :] = id
    H[1][1, :, 2, :] = -cp[1]*ph[1]*heg
    H[1][1, :, 3, :] = -cp[1]*ph[1]*hge
    for exp_ind = 1:no_exp
        H[1][1, :, 3 + exp_ind, :] = sqrt(-im*delt*uu[exp_ind]*lam[exp_ind])*hss
    end
    H[1][1, :, max_ind, :] = local_h(1, con, delt, time)

    for jj = 2:(na - 1)
        H[jj] = zeros(TN, max_ind, 3, max_ind, 3)
        H[jj][1, :, 1, :] = id
        H[jj][1, :, 2, :] = -cp[jj]*ph[jj]*heg
        H[jj][1, :, 3, :] = -cp[jj]*ph[jj]*hge
        H[jj][1, :, max_ind, :] = local_h(jj, con, delt, time)
        H[jj][2, :, max_ind, :] = cp[jj]*hge
        H[jj][3, :, max_ind, :] = cp[jj]*heg
        H[jj][2, :, 2, :] = ph[jj]*id
        H[jj][3, :, 3, :] = ph[jj]*id
        for exp_ind = 1:no_exp
            H[jj][1, :, 3 + exp_ind, :] = sqrt(-im*delt*uu[exp_ind]*lam[exp_ind])*hss
            H[jj][3 + exp_ind, :, max_ind, :] = sqrt(-im*delt*uu[exp_ind]*lam[exp_ind])*hss
            H[jj][3 + exp_ind, :, 3 + exp_ind, :] = lam[exp_ind]*id
        end
        H[jj][max_ind, :, max_ind, :] = id
    end

    H[na] = zeros(TN, max_ind, 3, 1, 3)
    H[na][1, :, 1, :] = local_h(na, con, delt, time)
    H[na][2, :, 1, :] = cp[na]*hge
    H[na][3, :, 1, :] = cp[na]*heg
    for exp_ind = 1:no_exp
        H[na][3 + exp_ind, :, 1, :] = sqrt(-im*delt*uu[exp_ind]*lam[exp_ind])*hss
    end
    H[na][max_ind, :, 1, :] = id

    return H

end

function local_h(jj, con, delt, time)

    con*id/na + delt*((im*del_p -
        (gam_eg + gam_ee + gam_1d[jj])/2)*hee +
        im*f(time)*sqrt(gam_1d[jj]/2)*exp(im*k_in*rj[jj])*heg -
        1/(2na)*abs2(f(time))*id + (im*del_s - gam_ss/2)*hss + 
        im*om*(exp(im*k_c*rj[jj])*hes + exp(-im*k_c*rj[jj])*hse))

end

function update_rydberg_hamiltonian!(H, con, delt, time)

    no_exp = length(lam)
    max_ind = 4 + no_exp

    H[1][1, :, max_ind, :] = local_h(1, con, delt, time)

    for jj = 2:(na - 1)
        H[jj][1, :, max_ind, :] = local_h(jj, con, delt, time)
    end

    H[na][1, :, 1, :] = local_h(na, con, delt, time)

end

# calculate local expectation values, assuming mps is already left normalized
function measure_excitations!(e_pop, s_pop, A)
    
    # initialize right environment
    env = ones(1, 1)
    
    for n = na:(-1):1
        Aenv = prod_LR(A[n], env)
        e_pop[n] = real(local_exp(Aenv, A[n], hee))
        s_pop[n] = real(local_exp(Aenv, A[n], hss))
        env = update_renv(A[n], Aenv)
    end
    
end

time_evolve()