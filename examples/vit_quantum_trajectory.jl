using MAT
using MatrixProductStates
using TensorOperations
BLAS.set_num_threads(2)

# spin model parameters
const na = 100 # number of atoms
const d_c = 10 # dimension of cavity Hilbert space (upper limit on photons in cavity)
const gam_1d = 2.0*ones(na) # coupling of probe to each atom
const gam_eg = 0.5 # eg spontaneous decay rate
const gam_es = 0.5 # es spontaneous decay rate
const kappa = 0.03 # cavity decay rate
const g = 2.0*ones(na) # cavity coupling with each atom
const del_p = 0.0 # detuning of pump beam from eg transition del_p = ωp - ωeg
const del_c = 0.0 # two-photon detuning of vit transition del_c = ωp - ωc - ωsg
const k_wg = 0.5*pi # waveguide wavevector
const rj = collect(1.0:na) # atom positions
const k_in = 0.5*pi # pump beam wavevector
const f_amp = 1.0 # probe amplitude
const sigma_t = 6.0 # time width of gaussian input pulse
const t_peak = 10.0 # time peak of gaussian input pulse
f(t) = f_amp*(pi*sigma_t^2/4)^(-1/4)*exp(-2(t - t_peak)^2/sigma_t^2) # time envelope of pump beam

# simulation parameters
const setrand = 21 # random number generator seed
const dt = 0.01 # time step
const t_fin = 50.0 # final time
const d = 3 # dimension of atom Hilbert space
const d_max = 50 # maximum bond dimension
const measure_int = 10 # number of time steps per observable measurement
const path_data = "/home/jdouglas/data/" # save directory
const base_filename = string(path_data,"VIT_N", na, "_D", d_max,
	"_Tf", t_fin, "_dt", dt, "_traj", setrand)

# local spin operators
const hgg = [1 0 0; 0 0 0; 0 0 0]
const hee = [0 0 0; 0 1 0; 0 0 0]
const hss = [0 0 0; 0 0 0; 0 0 1]
const hge = [0 1 0; 0 0 0; 0 0 0]
const heg = hge'
const hes = [0 0 0; 0 0 1; 0 0 0]
const hse = hes'
const id = eye(3)
# cavity operators
const anum = diagm(0:(d_c - 1))
const a = diagm(sqrt.(1:(d_c - 1)), 1)
const id_c = eye(d_c)

# types used in mps representation
const TN = Complex{Float64}
const TA = Array

function time_evolve()

    # construct left-canonical initial MPS ground state where the 
    A, dims = mpsgroundstate(TN, TA, na + 1, d_max, [d*ones(na)..., d_c])

    # create measurement operators
	jumpleft = Array{TA{TN, 4}, 1}(na + 1)
	jumpleft[1:na] = makempo(TN, TA, [jj->id jj->im*sqrt(gam_1d[jj]/2)*hge*exp(im*k_wg*rj[jj]);
        				jj->0 jj->id], na, d)
    jumpleft[na + 1] = zeros(TN, 1, d_c, 1, d_c)
    jumpleft[na + 1][1, :, 1, :] = eye(TN, d_c)
	jumpright = Array{TA{TN, 4}, 1}(na + 1)
	jumpright[1:na] = makempo(TN, TA, [jj->id jj->im*sqrt(gam_1d[jj]/2)*hge*exp(-im*k_wg*rj[jj]);
                		jj->0 jj->id], na, d)
    jumpright[na + 1] = zeros(TN, 1, d_c, 1, d_c)
    jumpright[na + 1][1, :, 1, :] = eye(TN, d_c)
    jumpright[1][1, :, 2, :] = f(0.0)*id + im*sqrt(gam_1d[1]/2)*hge*exp(-im*k_wg*rj[1])
	IRmpo = applyMPOtoMPO(jumpright, conj_mpo(jumpright))
	IR2mpo = applyMPOtoMPO(applyMPOtoMPO(jumpright, IRmpo), conj_mpo(jumpright))

    #step times and measurement times
    t = 0.0:dt:t_fin
    t_m = t[1:measure_int:end]
    tstep = length(t) - 1

    # initialize output measures
    p_coh = zeros(tstep)
    p_sum = zeros(tstep)
    p_acc = zeros(tstep, 2na + 4)
    e_pop = zeros(na, length(t_m))
    s_pop = zeros(na, length(t_m))
    cav_pop = zeros(length(t_m))
    epop = zeros(na)
    spop = zeros(na)
    I_r = zeros(TN, length(t_m))
    I2_r = zeros(TN, length(t_m))
    times = zeros(tstep, 4)
    t_r = Float64[]
    t_l = Float64[]
    t_cav = Float64[]
    t_eg = Float64[]
    pos_eg = Float64[]
    t_es = Float64[]
    pos_es = Float64[]
    cont = zeros(Int64,5)
 
    I_r[1] = abs2(f(t[1]))
    I2_r[1] = abs2(f(t[1]))^2
    cav_pop[1] = measure_excitations!((@view e_pop[:, 1]), (@view s_pop[:, 1]), A)
	
	# temporary arrays for time evolution
    envop = build_env(TN, TA, dims, dims, ones(na + 2)*6)
    envop_jump = build_env(TN, TA, dims, dims, [2*ones(na)..., 1, 1])
    A_coh = similar(A)
    A_r = similar(A)
    A_l = similar(A)
    A_coh .= copy.(A)
    A_r .= copy.(A)
    A_l .= copy.(A)

	# random jump variables
	srand(setrand)
	r1 = rand(tstep)
	r2 = rand(tstep)

    # create linear evolution MPO at initial time
    expH = vit_hamiltonian(TN, TA, 1, dt, t[1])

    for i = 1:tstep
        time_in = time()

        # update time-evolution operator
        update_h!(expH, 1, dt, t[i])

        # coherent time evolution probability
        compress_var_apply_H!(A_coh, envop, A, A, expH, 1)
        p_coh[i] = normalize_lo!(A_coh)
        times[i,1] = time() - time_in
        if r1[i] < p_coh[i]
            A .= copy.(A_coh)
        else
            #Update waveguide jump operator
            jumpright[1][1, :, 2, :] = f(t[i])*id + 
                    im*sqrt(gam_1d[1]/2)*hge*exp(-im*k_wg*rj[1])

            #Jump probabilities: decay into the waveguide
            compress_var_apply_H!(A_r, envop_jump, A, A, jumpright, 1)
            p_r = normalize_lo!(A_r)
            compress_var_apply_H!(A_l, envop_jump, A, A, jumpleft, 1)
            p_l = normalize_lo!(A_l)
            
            #Jump probabilities: decay outside the waveguide
            cpop = measure_excitations!(epop, spop, A)
            
            p_acc[i, 1] = 0.0
            for j = 1:na
                p_acc[i, j + 1] = p_acc[i, j] + dt*gam_eg*epop[j]
            end
            for j = 1:na
                p_acc[i, na + j + 1] = p_acc[i, na + j] + dt*gam_es*epop[j]
            end
            p_acc[i, 2na + 2] = p_acc[i, 2na + 1] + dt*p_r
            p_acc[i, 2na + 3] = p_acc[i, 2na + 2] + dt*p_l
            p_acc[i, 2na + 4] = p_acc[i, 2na + 3] + dt*kappa*cpop
            p_sum[i] = p_acc[i, end]
            println(p_sum[i] + p_coh[i])
            println(t[i])
            p_acc[i, :] = p_acc[i, :]/p_acc[i, end]
            
            # jumps according to above probabilities
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
                        apply_site_operator!(A[j], hse)
                        leftorthonormalizationQR!(A)
                        push!(t_es, t[i])
                        push!(pos_es, j)
                        cont[2] += 1
                        break
                    end
                end
            elseif r2[i] < p_acc[i, 2na + 2]
                A .= copy.(A_r)
                push!(t_r, t[i])
                cont[3] += 1
            elseif r2[i] < p_acc[i, 2na + 3]
                A .= copy.(A_l)
                push!(t_l, t[i])
                cont[4] += 1
            else
                apply_site_operator!(A[na + 1], a)
                leftorthonormalizationQR!(A)
                push!(t_cav, t[i])
                cont[5] += 1
            end
            times[i, 2] = time() - times[i, 1] - time_in

        end
        
        if rem(i, measure_int) == 0

            mind = div(i, measure_int) + 1
            # Atomic state populations and cavity occupation number
            cav_pop[mind] = measure_excitations!(
            	(@view e_pop[:, mind]), (@view s_pop[:, mind]), A)
            times[i, 3] = time() - times[i, 2]  - times[i, 1] - time_in
                       
            # Output right field update
            jumpright[1][1, :, 2, :] = f(t[i + 1])*id + 
                    im*sqrt(gam_1d[1]/2)*hge*exp(-im*k_wg*rj[1])
            
            IRmpo[1] =  apply_site_MPOtoMPO(jumpright[1], conj_site_mpo(jumpright[1]))
            IR2mpo[1] =  apply_site_MPOtoMPO(
                apply_site_MPOtoMPO(jumpright[1], IRmpo[1]),
                conj_site_mpo(jumpright[1]))

            # Output observables
            I_r[mind] = scal_op_prod(A, IRmpo, A)
            I2_r[mind] = scal_op_prod(A, IR2mpo, A)

            times[i,4] = time() - times[i, 3] - times[i, 2] - times[i, 1] - time_in

        end
        
        # write temporary file
        if rem(i,measure_int*10) == 0
            write_data_file(string(base_filename, "_temp.mat"),
            	            t_m, e_pop, s_pop, cav_pop, p_coh, p_acc,
                            p_sum, I_r, I2_r, cont, t_r, t_l, t_cav, t_eg, t_es, 
                            A, times)
        end

	end

    # saving final data
	write_data_file(string(base_filename, ".mat"),
		            t_m, e_pop, s_pop, cav_pop, p_coh, p_acc,
                    p_sum, I_r, I2_r, cont, t_r, t_l, t_cav, t_eg, t_es, 
                    A, times)


end

function measure_excitations_old!(e_pop, s_pop, A)
    
    #calculates local expectation values, assuming mps is already left normalized
    Acav = A[end]
    @tensor rhoR[-1, -2] := Acav[-1, 3, 1]*conj(Acav[-2, 3, 1]) 
   
    for n = na:(-1):1
        # Compute excit(n) and s_pop(n) from A[n], conj(A[n]), rhoR, O[n] and rhoR[n+1]
        An = A[n]
        @tensor ep = scalar(rhoR[3, 5]*An[1, 2, 3]*hee[4, 2]*conj(An[1, 4, 5]))
        e_pop[n] = real(ep)
        @tensor sp = scalar(rhoR[3, 5]*An[1, 2, 3]*hss[4, 2]*conj(An[1, 4, 5]))
        s_pop[n] = real(sp)
        # Update rhoR from A[n], conj(A[n]) and the previous rhoR 
        @tensor rhoR[-1, -2] := rhoR[1, 2]*An[-1, 3, 1]*conj(An[-2, 3, 2]) 
    end

    cav_pop = local_exp(Acav, Acav, anum)
    
end

# calculate local expectation values, assuming mps is already left normalized
function measure_excitations!(e_pop, s_pop, A)
    
    Acav = A[end]
    # initialize right environment
    env = ones(1, 1)
    env = update_renv(Acav, Acav, env)

    # sweep left taking expectation value
    for n = na:(-1):1
        # Compute excit(n) and s_pop(n) from A[n], conj(A[n]), rhoR, O[n] and rhoR[n+1]
        Aenv = prod_LR(A[n], env)
        e_pop[n] = real(local_exp(Aenv, A[n], hee))
        s_pop[n] = real(local_exp(Aenv, A[n], hss))
        # Update rhoR from A[n], conj(A[n]) and the previous rhoR 
        env = update_renv(A[n], Aenv)
    end

    cav_pop = real(local_exp(Acav, Acav, anum))
    
end

# calculate local expectation values, assuming mps is already left normalized
function measure_excitations_qr!(e_pop, s_pop, A)
    
    Acav = A[end]
    Acan = Acav
    # sweep left taking expectation value
    for n = na:(-1):1
        # Compute excit(n) and s_pop(n) from A[n], conj(A[n]), rhoR, O[n] and rhoR[n+1]
        _, r = rightorth(Acan)
        Acan = prod_LR(A[n], r)
        e_pop[n] = real(local_exp(Acan, Acan, hee))
        s_pop[n] = real(local_exp(Acan, Acan, hss))
    end

    cav_pop = real(local_exp(Acav, Acav, anum))
    
end


# creates VIT hamiltonian when con = 0 and delt = 1
# with con = 1 and delt = dt creates the linear time evolution operator 1 - im*dt*H
function vit_hamiltonian(::Type{TN}, ::Type{TA}, con, delt, time) where {TN, TA}

	H = Array{TA{TN, 4}, 1}(na + 1)
	
	drj = diff(rj)
	ph = exp.(im*k_wg*drj)
	cp = sqrt.(delt*gam_1d/2)

	H[1] = zeros(TN, 1, 3, 6, 3)
	H[1][1, :, 1, :] = id
	H[1][1, :, 2, :] = -cp[1]*ph[1]*heg
	H[1][1, :, 3, :] = -cp[1]*ph[1]*hge
	H[1][1, :, 4, :] = g[1]*hes
	H[1][1, :, 5, :] = conj(g[1])*hse
	H[1][1, :, 6, :] = local_h(1, con, delt, time)

    H[na] = zeros(TN, 6, 3, 6, 3)
    H[na][1, :, 1, :] = id
    H[na][1, :, 4, :] = g[na]*hes
    H[na][1, :, 5, :] = conj(g[na])*hse
    H[na][1, :, 6, :] = local_h(na, con, delt, time)
    H[na][2, :, 6, :] = cp[na]*hge
    H[na][3, :, 6, :] = cp[na]*heg
    H[na][4, :, 4, :] = id
    H[na][5, :, 5, :] = id
    H[na][6, :, 6, :] = id

	for jj = 2:(na - 1)
	    H[jj] = zeros(TN, 6, 3, 6, 3)
	    H[jj][1, :, 1, :] = id
	    H[jj][1, :, 2, :] = -cp[jj]*ph[jj]*heg
	    H[jj][1, :, 3, :] = -cp[jj]*ph[jj]*hge
	    H[jj][1, :, 4, :] = g[jj]*hes
	    H[jj][1, :, 5, :] = conj(g[jj])*hse
	    H[jj][1, :, 6, :] = local_h(jj, con, delt, time)
	    H[jj][2, :, 6, :] = cp[jj]*hge
	    H[jj][3, :, 6, :] = cp[jj]*heg
	    H[jj][2, :, 2, :] = ph[jj]*id
	    H[jj][3, :, 3, :] = ph[jj]*id
	    H[jj][4, :, 4, :] = id
	    H[jj][5, :, 5, :] = id
	    H[jj][6, :, 6, :] = id
	end

	H[na + 1] = complex(zeros(6, d_c, 1, d_c))
	H[na + 1][1, :, 1, :] = delt*(im*del_c - kappa/2)*anum
	H[na + 1][4, :, 1, :] = (-im*delt)*a
	H[na + 1][5, :, 1, :] = (-im*delt)*a'
	H[na + 1][6, :, 1, :] = id_c

	return H

end

# update of H created using vit_hamiltonian() for time dependent input pulse
function update_h!(H, con, delt, time)

    H[1][1, :, 6, :] = local_h(1, con, delt, time)

    H[na][1, :, 6, :] = local_h(na, con, delt, time)

    for jj = 2:(na - 1)
        H[jj][1, :, 6, :] = local_h(jj, con, delt, time)
    end

end

# local site Hamiltonian
function local_h(jj, con, delt, time)
	
	con*id/na + delt*((im*del_p - (gam_eg + gam_es + gam_1d[jj])/2)*hee +
		im*sqrt(gam_1d[jj]/2)*f(time)*exp(im*k_in*rj[jj])*heg - 
		(1/(2na))*abs2(f(time))*id)

end

function write_data_file(filename, t_m, e_pop, s_pop, cav_pop, p_coh, p_acc,
                         p_sum, I_r, I2_r, cont, t_r, t_l, t_cav, t_eg, t_es, 
                         A, times)

    file = matopen(string(filename), "w")
    write(file, "TN",string(TN))
    write(file, "TA",string(TA))
    write(file, "na", na)
    write(file, "d_c", d_c)
    write(file, "gam_1d", gam_1d)
    write(file, "gam_eg", gam_eg)
    write(file, "gam_es", gam_es)
    write(file, "kappa", kappa)
    write(file, "g", g)
    write(file, "del_p", del_p)
    write(file, "del_c", del_c)
    write(file, "k_wg", k_wg)
    write(file, "rj", rj)
    write(file, "k_in", k_in)
    write(file, "f_amp", f_amp)
    write(file, "t_peak", t_peak)
    write(file, "sigma_t",sigma_t)
    write(file, "f", f.(t_m))
    write(file, "dt", dt)
    write(file, "t_fin", t_fin)
    write(file, "d_max", d_max)
    write(file, "measure_int", measure_int)

    write(file, "t_m", collect(t_m))
    write(file, "e_pop", e_pop)
    write(file, "s_pop", s_pop)
    write(file, "cav_pop", cav_pop)
    write(file, "p_coh", p_coh)
    write(file, "p_acc", p_acc)
    write(file, "p_sum", p_sum)
    write(file, "I_r", I_r)
    write(file, "I2_r", I2_r)
    write(file, "cont", cont)
    write(file, "t_r", t_r)
    write(file, "t_l", t_l)
    write(file, "t_cav", t_cav)
    write(file, "t_eg", t_eg)
    write(file, "t_es", t_es)
    write(file, "A", collect.(A))
    write(file, "times", times)
    close(file)

end

time_evolve()