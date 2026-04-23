paras_bbo = {
"epoch": 20,
"pop_size": 50,
"p_m": 0.01,
"elites": 2,
}

paras_eoa = {
"epoch": 20,
"pop_size": 50,
"p_c": 0.9,
"p_m": 0.01,
"n_best": 2,
"alpha": 0.98,
"beta": 0.9,
"gamma": 0.9,
}

paras_iwo = {
"epoch": 20,
"pop_size": 50,
"seed_min": 3,
"seed_max": 9,
"exponent": 3,
"sigma_start": 0.6,
"sigma_end": 0.01,
}
paras_sbo = {
"epoch": 20,
"pop_size": 50,
"alpha": 0.9,
"p_m": 0.05,
"psw": 0.02,
}
paras_sma = {
"epoch": 20,
"pop_size": 50,
"p_t": 0.03,
}
paras_vcs = {
"epoch": 20,
"pop_size": 50,
"lamda": 0.5,
"sigma": 0.3,
}
paras_who = {
"epoch": 20,
"pop_size": 50,
"n_explore_step": 3,
"n_exploit_step": 3,
"eta": 0.15,
"p_hi": 0.9,
"local_alpha": 0.9,
"local_beta": 0.3,
"global_alpha": 0.2,
"global_beta": 0.8,
"delta_w": 2.0,
"delta_c": 2.0,
}
paras_cro = {
"epoch": 20,
"pop_size": 50,
"po": 0.4,
"Fb": 0.9,
"Fa": 0.1,
"Fd": 0.1,
"Pd": 0.5,
"GCR": 0.1,
"gamma_min": 0.02,
"gamma_max": 0.2,
"n_trials": 5,
}
paras_ocro = dict(paras_cro)
paras_ocro["restart_count"] = 5

paras_de = {
"epoch": 20,
"pop_size": 50,
"wf": 0.7,
"cr": 0.9,
"strategy": 0,
}
paras_jade = {
"epoch": 20,
"pop_size": 50,
"miu_f": 0.5,
"miu_cr": 0.5,
"pt": 0.1,
"ap": 0.1,
}
paras_sade = {
"epoch": 20,
"pop_size": 50,
}
paras_shade = paras_lshade = {
"epoch": 20,
"pop_size": 50,
"miu_f": 0.5,
"miu_cr": 0.5,
}
paras_sap_de = {
"epoch": 20,
"pop_size": 50,
"branch": "ABS"
}
paras_ep = paras_levy_ep = {
"epoch": 20,
"pop_size": 50,
"bout_size": 0.05
}
paras_es = paras_levy_es = {
"epoch": 20,
"pop_size": 50,
"lamda": 0.75
}
paras_fpa = {
"epoch": 20,
"pop_size": 50,
"p_s": 0.8,
"levy_multiplier": 0.2
}
paras_ga = {
"epoch": 20,
"pop_size": 50,
"pc": 0.9,
"pm": 0.05,
}
paras_single_ga = {
"epoch": 20,
"pop_size": 50,
"pc": 0.9,
"pm": 0.8,
"selection": "roulette",
"crossover": "uniform",
"mutation": "swap",
}
paras_multi_ga = {
"epoch": 20,
"pop_size": 50,
"pc": 0.9,
"pm": 0.05,
"selection": "roulette",
"crossover": "uniform",
"mutation": "swap",
}
paras_ma = {
"epoch": 20,
"pop_size": 50,
"pc": 0.85,
"pm": 0.15,
"p_local": 0.5,
"max_local_gens": 10,
"bits_per_param": 4,
}

paras_bro = {
"epoch": 20,
"pop_size": 50,
"threshold": 3,
}
paras_improved_bso = {
"epoch": 20,
"pop_size": 50,
"m_clusters": 5,
"p1": 0.2,
"p2": 0.8,
"p3": 0.4,
"p4": 0.5,
}
paras_bso = dict(paras_improved_bso)
paras_bso["slope"] = 20
paras_ca = {
"epoch": 20,
"pop_size": 50,
"accepted_rate": 0.15,
}
paras_chio = {
"epoch": 20,
"pop_size": 50,
"brr": 0.15,
"max_age": 3
}
paras_fbio = {
"epoch": 20,
"pop_size": 50,
}
paras_base_gska = {
"epoch": 20,
"pop_size": 50,
"pb": 0.1,
"kr": 0.9,
}
paras_gska = {
"epoch": 20,
"pop_size": 50,
"pb": 0.1,
"kf": 0.5,
"kr": 0.9,
"kg": 5,
}
paras_ica = {
"epoch": 20,
"pop_size": 50,
"empire_count": 5,
"assimilation_coeff": 1.5,
"revolution_prob": 0.05,
"revolution_rate": 0.1,
"revolution_step_size": 0.1,
"zeta": 0.1,
}
paras_lco = {
"epoch": 20,
"pop_size": 50,
"r1": 2.35,
}
paras_improved_lco = {
"epoch": 20,
"pop_size": 50,
}
paras_qsa = {
"epoch": 20,
"pop_size": 50,
}
paras_saro = {
"epoch": 20,
"pop_size": 50,
"se": 0.5,
"mu": 15
}
paras_ssdo = {
"epoch": 20,
"pop_size": 50,
}
paras_tlo = {
"epoch": 20,
"pop_size": 50,
}
paras_improved_tlo = {
"epoch": 20,
"pop_size": 50,
"n_teachers": 5,
}

paras_aoa = {
"epoch": 20,
"pop_size": 50,
"alpha": 5,
"miu": 0.5,
"moa_min": 0.2,
"moa_max": 0.9,
}
paras_cem = {
"epoch": 20,
"pop_size": 50,
"n_best": 20,
"alpha": 0.7,
}
paras_cgo = {
"epoch": 20,
"pop_size": 50,
}
paras_gbo = {
"epoch": 20,
"pop_size": 50,
"pr": 0.5,
"beta_min": 0.2,
"beta_max": 1.2,
}
paras_hc = {
"epoch": 20,
"pop_size": 50,
"neighbour_size": 50
}
paras_swarm_hc = {
"epoch": 20,
"pop_size": 50,
"neighbour_size": 10
}
paras_pss = {
"epoch": 20,
"pop_size": 50,
"acceptance_rate": 0.8,
"sampling_method": "LHS",
}
paras_sca = {
"epoch": 20,
"pop_size": 50,
}

paras_hs = {
"epoch": 20,
"pop_size": 50,
"c_r": 0.95,
"pa_r": 0.05
}

paras_aeo = {
"epoch": 20,
"pop_size": 50,
}
paras_gco = {
"epoch": 20,
"pop_size": 50,
"cr": 0.7,
"wf": 1.25,
}
paras_wca = {
"epoch": 20,
"pop_size": 50,
"nsr": 4,
"wc": 2.0,
"dmax": 1e-6
}

paras_archoa = {
"epoch": 20,
"pop_size": 50,
"c1": 2,
"c2": 5,
"c3": 2,
"c4": 0.5,
"acc_max": 0.9,
"acc_min": 0.1,
}
paras_aso = {
"epoch": 20,
"pop_size": 50,
"alpha": 50,
"beta": 0.2,
}
paras_efo = {
"epoch": 20,
"pop_size": 50,
"r_rate": 0.3,
"ps_rate": 0.85,
"p_field": 0.1,
"n_field": 0.45,
}
paras_eo = {
"epoch": 20,
"pop_size": 50,
}
paras_hgso = {
"epoch": 20,
"pop_size": 50,
"n_clusters": 3,
}
paras_mvo = {
"epoch": 20,
"pop_size": 50,
"wep_min": 0.2,
"wep_max": 1.0,
}
paras_nro = {
"epoch": 20,
"pop_size": 50,
}
paras_sa = {
"epoch": 20,
"pop_size": 50,
"max_sub_iter": 5,
"t0": 1000,
"t1": 1,
"move_count": 5,
"mutation_rate": 0.1,
"mutation_step_size": 0.1,
"mutation_step_size_damp": 0.99,
}
paras_two = {
"epoch": 20,
"pop_size": 50,
}
paras_wdo = {
"epoch": 20,
"pop_size": 50,
"RT": 3,
"g_c": 0.2,
"alp": 0.4,
"c_e": 0.4,
"max_v": 0.3,
}

paras_abc = {
"epoch": 20,
"pop_size": 50,
"n_elites": 16,
"n_others": 4,
"patch_size": 5.0,
"patch_reduction": 0.985,
"n_sites": 3,
"n_elite_sites": 1,
}
paras_acor = {
"epoch": 20,
"pop_size": 50,
"sample_count": 25,
"intent_factor": 0.5,
"zeta": 1.0,
}
paras_alo = {
"epoch": 20,
"pop_size": 50,
}
paras_ao = {
"epoch": 20,
"pop_size": 50,
}
paras_ba = {
"epoch": 20,
"pop_size": 50,
"loudness": 0.8,
"pulse_rate": 0.95,
"pf_min": 0.,
"pf_max": 10.,
}
paras_adaptive_ba = {
"epoch": 20,
"pop_size": 50,
"loudness_min": 1.0,
"loudness_max": 2.0,
"pr_min": 0.15,
"pr_max": 0.85,
"pf_min": 0.,
"pf_max": 10.,
}
paras_dev_ba = {
"epoch": 20,
"pop_size": 50,
"pulse_rate": 0.95,
"pf_min": 0.,
"pf_max": 10.,
}
paras_beesa = {
"epoch": 20,
"pop_size": 50,
"selected_site_ratio": 0.5,
"elite_site_ratio": 0.4,
"selected_site_bee_ratio": 0.1,
"elite_site_bee_ratio": 2.0,
"dance_radius": 0.1,
"dance_reduction": 0.99,
}
paras_prob_beesa = {
"epoch": 20,
"pop_size": 50,
"recruited_bee_ratio": 0.1,
"dance_radius": 0.1,
"dance_reduction": 0.99,
}
paras_bes = {
"epoch": 20,
"pop_size": 50,
"a_factor": 10,
"R_factor": 1.5,
"alpha": 2.0,
"c1": 2.0,
"c2": 2.0,
}
paras_bfo = {
"epoch": 20,
"pop_size": 50,
"Ci": 0.01,
"Ped": 0.25,
"Nc": 5,
"Ns": 4,
"d_attract": 0.1,
"w_attract": 0.2,
"h_repels": 0.1,
"w_repels": 10,
}
paras_abfo = {
"epoch": 20,
"pop_size": 50,
"C_s": 0.1,
"C_e": 0.001,
"Ped": 0.01,
"Ns": 4,
"N_adapt": 4,
"N_split": 40,
}
paras_bsa = {
"epoch": 20,
"pop_size": 50,
"ff": 10,
"pff": 0.8,
"c1": 1.5,
"c2": 1.5,
"a1": 1.0,
"a2": 1.0,
"fl": 0.5,
}
paras_coa = {
"epoch": 20,
"pop_size": 50,
"n_coyotes": 5,
}
paras_csa = {
"epoch": 20,
"pop_size": 50,
"p_a": 0.3,
}
paras_cso = {
"epoch": 20,
"pop_size": 50,
"mixture_ratio": 0.15,
"smp": 5,
"spc": False,
"cdc": 0.8,
"srd": 0.15,
"c1": 0.4,
"w_min": 0.4,
"w_max": 0.9,
"selected_strategy": 1,
}
paras_do = {
"epoch": 20,
"pop_size": 50,
}
paras_eho = {
"epoch": 20,
"pop_size": 50,
"alpha": 0.5,
"beta": 0.5,
"n_clans": 5,
}
paras_fa = {
"epoch": 20,
"pop_size": 50,
"max_sparks": 20,
"p_a": 0.04,
"p_b": 0.8,
"max_ea": 40,
"m_sparks": 5,
}
paras_ffa = {
"epoch": 20,
"pop_size": 50,
"gamma": 0.001,
"beta_base": 2,
"alpha": 0.2,
"alpha_damp": 0.99,
"delta": 0.05,
"exponent": 2,
}
paras_foa = {
"epoch": 20,
"pop_size": 50,
}
paras_goa = {
"epoch": 20,
"pop_size": 50,
"c_min": 0.00004,
"c_max": 1.0,
}
paras_gwo = {
"epoch": 20,
"pop_size": 50,
}
paras_hgs = {
"epoch": 20,
"pop_size": 50,
"PUP": 0.08,
"LH": 10000,
}
paras_hho = {
"epoch": 20,
"pop_size": 50,
}
paras_ja = {
"epoch": 20,
"pop_size": 50,
}
paras_mfo = {
"epoch": 20,
"pop_size": 50,
}
paras_mrfo = {
"epoch": 20,
"pop_size": 50,
"somersault_range": 2.0,
}
paras_msa = {
"epoch": 20,
"pop_size": 50,
"n_best": 5,
"partition": 0.5,
"max_step_size": 1.0,
}
paras_nmra = {
"epoch": 20,
"pop_size": 50,
"pb": 0.75,
}
paras_improved_nmra = {
"epoch": 20,
"pop_size": 50,
"pb": 0.75,
"pm": 0.01,
}
paras_pfa = {
"epoch": 20,
"pop_size": 50,
}
paras_pso = {
"epoch": 20,
"pop_size": 50,
"c1": 2.05,
"c2": 2.05,
"w_min": 0.4,
"w_max": 0.9,
}
paras_ppso = {
"epoch": 20,
"pop_size": 50,
}
paras_hpso_tvac = {
"epoch": 20,
"pop_size": 50,
"ci": 0.5,
"cf": 0.0,
}
paras_cpso = {
"epoch": 20,
"pop_size": 50,
"c1": 2.05,
"c2": 2.05,
"w_min": 0.4,
"w_max": 0.9,
}
paras_clpso = {
"epoch": 20,
"pop_size": 50,
"c_local": 1.2,
"w_min": 0.4,
"w_max": 0.9,
"max_flag": 7,
}
paras_sfo = {
"epoch": 20,
"pop_size": 50,
"pp": 0.1,
"AP": 4.0,
"epsilon": 0.0001,
}
paras_improved_sfo = {
"epoch": 20,
"pop_size": 50,
"pp": 0.1,
}
paras_sho = {
"epoch": 20,
"pop_size": 50,
"h_factor": 5.0,
"N_tried": 10,
}
paras_slo = paras_modified_slo = {
"epoch": 20,
"pop_size": 50,
}
paras_improved_slo = {
"epoch": 20,
"pop_size": 50,
"c1": 1.2,
"c2": 1.2
}
paras_srsr = {
"epoch": 20,
"pop_size": 50,
}
paras_ssa = {
"epoch": 20,
"pop_size": 50,
"ST": 0.8,
"PD": 0.2,
"SD": 0.1,
}
paras_sso = {
"epoch": 20,
"pop_size": 50,
}
paras_sspidera = {
"epoch": 20,
"pop_size": 50,
"r_a": 1.0,
"p_c": 0.7,
"p_m": 0.1
}
paras_sspidero = {
"epoch": 20,
"pop_size": 50,
"fp_min": 0.65,
"fp_max": 0.9
}
paras_woa = {
"epoch": 20,
"pop_size": 50,
}
paras_hi_woa = {
"epoch": 20,
"pop_size": 50,
"feedback_max": 10
}
