# Quantum trajectory simulation of spin model for two-level atoms as presented in 
# Marco T. Manzoni, Darrick E. Chang and James S. Douglas, 
# Nature Communications 8, 1743 (2017)

using MAT
using MatrixProductStates
BLAS.set_num_threads(2)

# spin model parameters
const na = 50 # number of atoms
const gam_1d = 1.0*ones(na) # coupling of probe to each atom
const gam_eg = 0.0 # eg spontaneous decay rate
const del_p = 0.0 # detuning of pump beam from eg transition del_p = ωp - ωeg
const k_wg = 0.2*pi # waveguide wavevector
const rj = collect(1.0:na) # atom positions
const k_in = 0.5*pi # pump beam wavevector
const f_amp = 0.0 # probe amplitude
const sigma_t = 6.0 # time width of gaussian input pulse
const t_peak = 10.0 # time peak of gaussian input pulse
f(t) = f_amp*(pi*sigma_t^2/4)^(-1/4)*exp(-2(t - t_peak)^2/sigma_t^2) # time envelope of pump beam

# simulation parameters
const setrand = 1 # random number generator seed
const dt = 0.01 # time step
const t_fin = 5.0 # final time
const d = 2 # dimension of atom Hilbert space
const d_max = 50 # maximum bond dimension
const measure_int = 10 # number of time steps per observable measurement
const path_data = "/home/jdouglas/data/" # save directory
const base_filename = string(path_data,"SM_N", na, "_D", d_max,
	"_Tf", t_fin, "_dt", dt, "_rk4_traj", setrand)

# local spin operators
const hgg = [1 0; 0 0]
const hee = [0 0; 0 1]
const hge = [0 1; 0 0]
const heg = hge'
const id = eye(2)

# types used in mps representation
const TN = Complex{Float64}
const TA = Array

function time_evolve()

    # construct left-canonical initial MPS ground state where the 
    #A, dims = mpsgroundstate(TN, TA, na, d_max, d)
    A, dims = mpsproductstate(TN, TA, na, d_max, d, [0, 1])

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
    p_acc = zeros(tstep, na + 3)
    e_pop = zeros(na, length(t_m))
    epop = zeros(na)
    I_r = zeros(TN, length(t_m))
    I2_r = zeros(TN, length(t_m))
    times = zeros(tstep, 4)
    t_r = Float64[]
    t_l = Float64[]
    t_eg = Float64[]
    pos_eg = Float64[]
    cont = zeros(Int64, 3)
 
    # initial measurements
    I_r[1] = scal_op_prod(A, ir_mpo, A)
    I2_r[1] = scal_op_prod(A, ir2_mpo, A)
    measure_excitations!((@view e_pop[:, 1]), A)
	
	# temporary arrays for time evolution
    envop = build_env(TN, TA, dims, dims, ones(na + 1)*4)
    envop_jump = build_env(TN, TA, dims, dims, ones(na + 1)*2)
    env1 = build_env(TN, TA, dims, dims)
    env2 = build_env(TN, TA, dims, dims)
    env3 = build_env(TN, TA, dims, dims)
    A_coh = similar(A)
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

	# random jump variables
	srand(setrand)
	r1 = rand(tstep)
	r2 = rand(tstep)

    # create linear evolution MPO at initial time
    H1 = sm_hamiltonian(TN, TA, 1, dt/2, t[1])
    H2 = sm_hamiltonian(TN, TA, 0, 1, t[1] + dt/2)
    H3 = sm_hamiltonian(TN, TA, 1, dt/2, t[2])
    H = [H1, H2, H3]

    for i = 1:tstep
        time_in = time()

        # update time-evolution operator
        update_h!(H1, 1, dt/2, t[i])
        update_h!(H2, 0, 1, t[i] + dt/2)
        update_h!(H3, 1, dt/2, t[i+1])

        # coherent time evolution probability
        RK4stp_apply_H_half!(dt, H, A_coh, A, A1, A2, A3, env1, env2, env3, envop)
        p_coh[i] = normalize_lo!(A_coh)
        times[i,1] = time() - time_in
        
        # coherent evolution or jumps
        if r1[i] < p_coh[i]
            A .= copy.(A_coh)
        else
            # update waveguide jump operator
            jumpright[1][1, :, 2, :] = f(t[i])*id + 
                    im*sqrt(gam_1d[1]/2)*exp(-im*k_wg*rj[1])*hge

            # jump probabilities for decay into the waveguide
            compress_var_apply_H!(A_r, envop_jump, A, A, jumpright, 1)
            p_r = normalize_lo!(A_r)
            compress_var_apply_H!(A_l, envop_jump, A, A, jumpleft, 1)
            p_l = normalize_lo!(A_l)
            
            # jump probabilities decay outside the waveguide
            measure_excitations!(epop, A)
            p_acc[i, 1] = 0.0
            for j = 1:na
                p_acc[i, j + 1] = p_acc[i, j] + dt*gam_eg*epop[j]
            end
            p_acc[i, na + 2] = p_acc[i, na + 1] + dt*p_r
            p_acc[i, na + 3] = p_acc[i, na + 2] + dt*p_l
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
            elseif r2[i] < p_acc[i, na + 2]
                A .= copy.(A_r)
                push!(t_r, t[i])
                cont[2] += 1
            else
                A .= copy.(A_l)
                push!(t_l, t[i])
                cont[3] += 1
            end
            times[i, 2] = time() - times[i, 1] - time_in

        end
        
        if rem(i, measure_int) == 0

            mind = div(i, measure_int) + 1
            # Atomic state populations and cavity occupation number
            measure_excitations!((@view e_pop[:, mind]), A)
            times[i, 3] = time() - times[i, 2]  - times[i, 1] - time_in
                       
            # Output right field update
            jumpright[1][1, :, 2, :] = f(t[i + 1])*id + 
                    im*sqrt(gam_1d[1]/2)*exp(-im*k_wg*rj[1])*hge
            ir_mpo[1] =  apply_site_MPOtoMPO(jumpright[1], conj_site_mpo(jumpright[1]))
            ir2_mpo[1] =  apply_site_MPOtoMPO(
                apply_site_MPOtoMPO(jumpright[1], ir_mpo[1]),
                conj_site_mpo(jumpright[1]))

            # right field output observables
            I_r[mind] = scal_op_prod(A, ir_mpo, A)
            I2_r[mind] = scal_op_prod(A, ir2_mpo, A)
            times[i,4] = time() - times[i, 3] - times[i, 2] - times[i, 1] - time_in

        end
        
        # write temporary file
        if rem(i,measure_int*10) == 0
            write_data_file(string(base_filename, "_temp.mat"),
            	            t_m, e_pop, p_coh, p_acc, p_sum, I_r,
                            I2_r, cont, t_r, t_l, t_eg, A, times)
        end

	end

    # saving final data
	write_data_file(string(base_filename, ".mat"),
                           t_m, e_pop, p_coh, p_acc, p_sum, I_r,
                           I2_r, cont, t_r, t_l, t_eg, A, times)


end

# calculate local expectation values, assuming mps is already left normalized
function measure_excitations!(e_pop, A)
    
    # initialize right environment
    env = ones(1, 1)
    # sweep left taking expectation value
    for n = na:(-1):1
        Aenv = prod_LR(A[n], env)
        e_pop[n] = real(local_exp(Aenv, A[n], hee))
        env = update_renv(A[n], Aenv)
    end

end

# creates the spin model hamiltonian when con = 0 and delt = 1
# with con = 1 and delt = dt creates the linear time evolution operator 1 - im*dt*H
function sm_hamiltonian(::Type{TN}, ::Type{TA}, con, delt, time) where {TN, TA}

	H = Array{TA{TN, 4}, 1}(na + 1)
	
	drj = diff(rj)
	ph = exp.(im*k_wg*drj)
	cp = sqrt.(delt*gam_1d/2)

	H[1] = zeros(TN, 1, 2, 4, 2)
	H[1][1, :, 1, :] = id
	H[1][1, :, 2, :] = -cp[1]*ph[1]*heg
	H[1][1, :, 3, :] = -cp[1]*ph[1]*hge
	H[1][1, :, 4, :] = local_h(1, con, delt, time)

    H[na] = zeros(TN, 4, 2, 1, 2)
    H[na][1, :, 1, :] = id
    H[na][1, :, 1, :] = local_h(na, con, delt, time)
    H[na][2, :, 1, :] = cp[na]*hge
    H[na][3, :, 1, :] = cp[na]*heg
    H[na][4, :, 1, :] = id

	for jj = 2:(na - 1)
	    H[jj] = zeros(TN, 4, 2, 4, 2)
	    H[jj][1, :, 1, :] = id
	    H[jj][1, :, 2, :] = -cp[jj]*ph[jj]*heg
	    H[jj][1, :, 3, :] = -cp[jj]*ph[jj]*hge
	    H[jj][1, :, 4, :] = local_h(jj, con, delt, time)
	    H[jj][2, :, 4, :] = cp[jj]*hge
	    H[jj][3, :, 4, :] = cp[jj]*heg
	    H[jj][2, :, 2, :] = ph[jj]*id
	    H[jj][3, :, 3, :] = ph[jj]*id
	    H[jj][4, :, 4, :] = id
	end

	return H

end

# update of H created using vit_hamiltonian() for time dependent input pulse
function update_h!(H, con, delt, time)

    H[1][1, :, 4, :] = local_h(1, con, delt, time)

    H[na][1, :, 1, :] = local_h(na, con, delt, time)

    for jj = 2:(na - 1)
        H[jj][1, :, 4, :] = local_h(jj, con, delt, time)
    end

end

# local site Hamiltonian
function local_h(jj, con, delt, time)
	
	con*id/na + delt*((im*del_p - (gam_eg + gam_1d[jj])/2)*hee +
		im*sqrt(gam_1d[jj]/2)*f(time)*exp(im*k_in*rj[jj])*heg - 
		(1/(2na))*abs2(f(time))*id)

end

function write_data_file(filename, t_m, e_pop, p_coh, p_acc,
                         p_sum, I_r, I2_r, cont, t_r, t_l, t_eg,
                         A, times)

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
    write(file, "t_peak", t_peak)
    write(file, "sigma_t",sigma_t)
    write(file, "f", f.(t_m))
    write(file, "dt", dt)
    write(file, "t_fin", t_fin)
    write(file, "d_max", d_max)
    write(file, "measure_int", measure_int)

    write(file, "t_m", collect(t_m))
    write(file, "e_pop", e_pop)
    write(file, "p_coh", p_coh)
    write(file, "p_acc", p_acc)
    write(file, "p_sum", p_sum)
    write(file, "I_r", I_r)
    write(file, "I2_r", I2_r)
    write(file, "cont", cont)
    write(file, "t_r", t_r)
    write(file, "t_l", t_l)
    write(file, "t_eg", t_eg)
    write(file, "A", collect.(A))
    write(file, "times", times)
    close(file)

end

time_evolve()