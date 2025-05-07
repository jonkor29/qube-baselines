import logging
import numpy as np
import tensorflow as tf
tf.disable_eager_execution() # Ensure TF1 graph behavior

logger = logging.getLogger(__name__)


class CMA(object):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implemented with TensorFlow v1.14.

    This implementation is essentially following "The CMA Evolution Strategy: A Tutorial" [1]

    [1] https://arxiv.org/abs/1604.00772
    """
    def __init__(
        self,
        initial_solution,
        initial_step_size,
        fitness_function,
        enforce_bounds=None,
        population_size=None,
        cc=None,
        cσ=None,
        c1=None,
        cμ=None,
        damps=None,
        termination_no_effect=1e-8,
        store_trace=False,
        callback_function=None,
        dtype=tf.float32,
    ):
        """
        Args:
          initial_solution
            Search starting point, a list or numpy array.

          initial_step_size
            Standard deviation of the covariance matrix at generation 0.

          fitness_function
            Function to be minimized. Function must have the following signature:
            ```
            Args:
              x: tf.Tensor of shape (M, N)

            Returns:
              Fitness evaluations: tf.Tensor of shape (M,)
            ```
            Where `M` is the number of solutions to evaluate and `N` is the dimension
            of a single solution. This function's operations will be part of the TF graph.

          enforce_bounds
            2D list, the min and max for each dimension, e.g. [[-1, 1], [-2, 2], [0, 1]].
            Ensures the fitness function is never called with out of bounds values.
            Out of bounds samples are clipped back to the minimum or maximum values and a penalty
            (scalar, sum of squared L2 norms of clipped differences) is added to the fitness evaluation of all samples.

          population_size
            Number of samples produced at each generation.
            Defaults to 8 + 3 * ln(dimension) (e.g. 10 for 2 dimensions, 14 for 10 dimensions)

          cc, cσ, c1, cμ, damps
            Core parameters of the algorithm. Set to appropriate values by default.

          termination_no_effect
            Set the threshold for NoEffectAxis and NoEffectCoord termination criteria.
            Decreasing this value can increase the number of significant decimals of the solution.
            Defaults to 1e-8.

          store_trace
            If True, core variables are stored in memory (attribute self.trace) at each generation.
            This is mostly a debugging mechanism and it should not be used in production.
            Defaults to False.

          callback_function
            User defined function called first after initialization, then at the end of each
            generation. Intended for logging purpose.
            Function must have the following signature:
            ```
            Args:
              cma: the parent CMA instance (i.e. self)
              logger: a python Logger instance
            ```
        """
        if not isinstance(initial_solution, (np.ndarray, list)):
            raise ValueError('Initial solution must be a list or numpy array')
        elif np.ndim(initial_solution) != 1:
            ndim = np.ndim(initial_solution)
            raise ValueError(f'Initial solution must be a 1D array but got an array of dim {ndim}')
        elif not np.isscalar(initial_step_size) or initial_step_size <= 0:
            raise ValueError(f'Initial step size must be a number greater than zero')
        elif not callable(fitness_function):
            raise ValueError(f'Fitness function must be callable')
        elif population_size is not None and population_size <= 4:
            raise ValueError(f'Population size must be at least 4')
        elif enforce_bounds is not None and not isinstance(enforce_bounds, (np.ndarray, list)):
            raise ValueError('Bounds must be a list or numpy array')
        elif enforce_bounds is not None and np.ndim(enforce_bounds) != 2:
            ndim = np.ndim(enforce_bounds)
            raise ValueError(f'Bounds must be a 2D array but got an array of dim {ndim}')
        elif callback_function is not None and not callable(callback_function):
            raise ValueError(f'Callback function must be callable')

        self.generation = 0
        self.initial_solution_np = np.array(initial_solution, dtype=np.float32) # Ensure consistent dtype
        self.initial_step_size_py = float(initial_step_size)
        self.fitness_fn_user = fitness_function # User-provided function
        self.population_size_py = population_size
        self.enforce_bounds_py = enforce_bounds
        self._cc_py = cc
        self._cσ_py = cσ
        self._c1_py = c1
        self._cμ_py = cμ
        self._damps_py = damps
        self.termination_no_effect_py = float(termination_no_effect)
        self.store_trace = store_trace
        self.callback_fn = callback_function
        self.dtype = dtype # tf.DType, e.g. tf.float32
        self.termination_criterion_met = False

        self._initialized = False
        self.graph = None
        self.sess = None

        # For _should_terminate_np logic (values from previous generation)
        self._prev_sigma_val_for_term = None
        self._prev_D_val_for_term = None

    def init(self):
        if self._initialized:
            raise ValueError('Already initialized - call reset method to start over')

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.generation = 0
            self.dimension = len(self.initial_solution_np) # Python int
            self._enforce_bounds_py_bool = self.enforce_bounds_py is not None
            self.trace = []

            # -------------------------
            # Non-trainable parameters (defined as TF ops/constants)
            # -------------------------
            self.N_tf = tf.constant(self.dimension, dtype=self.dtype, name="N_tf")

            if self.population_size_py is not None:
                self.λ_tf = tf.constant(self.population_size_py, dtype=self.dtype, name="lambda_tf")
            else:
                # For tf.floor, name argument is fine as it's a direct function call
                self.λ_tf = tf.floor(tf.log(tf.cast(self.N_tf, self.dtype)) * 3.0 + 8.0, name="lambda_tf")

            self.shape_op = tf.cast(tf.stack([self.λ_tf, self.N_tf]), tf.int32, name="population_shape")
            self.μ_tf = tf.floor(self.λ_tf / 2.0, name="mu_tf")
            μ_tf_int = tf.cast(self.μ_tf, tf.int32, name="mu_tf_int")

            log_mu_plus_half = tf.log(self.μ_tf + 0.5)
            log_range_terms = tf.log(tf.cast(tf.range(1, μ_tf_int + 1), dtype=self.dtype))
            weights_front = log_mu_plus_half - log_range_terms
            
            num_zeros_for_weights = tf.cast(self.λ_tf - self.μ_tf, tf.int32, name="num_zeros_for_weights")
            weights_back = tf.zeros([num_zeros_for_weights], dtype=self.dtype)
            
            _weights_unnormalized = tf.concat([weights_front, weights_back], axis=0)
            self.weights_tf = tf.divide(_weights_unnormalized, tf.reduce_sum(_weights_unnormalized), name="weights_tf")
            self.weights_tf = self.weights_tf[:, tf.newaxis] # Reshape after naming

            # Corrected Line 166 area (μeff_tf)
            _sum_weights_for_mueff = tf.reduce_sum(self.weights_tf)
            _sum_weights_sq_for_mueff = tf.square(_sum_weights_for_mueff)
            _weights_sq_for_mueff = tf.square(self.weights_tf)
            _sum_sq_weights_for_mueff = tf.reduce_sum(_weights_sq_for_mueff)
            self.μeff_tf = tf.divide(_sum_weights_sq_for_mueff, _sum_sq_weights_for_mueff, name="mueff_tf")

            if self._cc_py is not None:
                self.cc_tf = tf.constant(self._cc_py, dtype=self.dtype, name="cc_tf")
            else:
                # Corrected Line 173 area (cc_tf)
                _cc_numerator = 4.0 + self.μeff_tf / self.N_tf
                _cc_denominator = self.N_tf + 4.0 + 2.0 * self.μeff_tf / self.N_tf
                self.cc_tf = tf.divide(_cc_numerator, _cc_denominator, name="cc_tf")
            
            if self._cσ_py is not None:
                self.cσ_tf = tf.constant(self._cσ_py, dtype=self.dtype, name="c_sigma_tf")
            else:
                # Corrected Line 178 area (cσ_tf)
                _c_sigma_numerator = self.μeff_tf + 2.0
                _c_sigma_denominator = self.N_tf + self.μeff_tf + 5.0
                self.cσ_tf = tf.divide(_c_sigma_numerator, _c_sigma_denominator, name="c_sigma_tf")

            if self._c1_py is not None:
                self.c1_tf = tf.constant(self._c1_py, dtype=self.dtype, name="c1_tf")
            else:
                # Corrected Line 183 area (c1_tf)
                _c1_denominator_term1 = tf.square(self.N_tf + 1.3)
                _c1_denominator = _c1_denominator_term1 + self.μeff_tf
                self.c1_tf = tf.divide(2.0, _c1_denominator, name="c1_tf")

            if self._cμ_py is not None:
                self.cμ_tf = tf.constant(self._cμ_py, dtype=self.dtype, name="c_mu_tf")
            else:
                # Corrected Line 190 area (cμ_tf)
                _c_mu_factor1 = self.μeff_tf - 2.0 + 1.0 / self.μeff_tf
                _c_mu_numerator = 2.0 * _c_mu_factor1
                _c_mu_denominator_term1 = tf.square(self.N_tf + 2.0)
                _c_mu_denominator_term2 = self.μeff_tf # Simplified from 2.0 * self.μeff_tf / 2.0
                _c_mu_denominator = _c_mu_denominator_term1 + _c_mu_denominator_term2
                self.cμ_tf = tf.divide(_c_mu_numerator, _c_mu_denominator, name="c_mu_tf")

            if self._damps_py is not None:
                self.damps_tf = tf.constant(self._damps_py, dtype=self.dtype, name="damps_tf")
            else:
                # This line has tf.identity for naming, which is a good pattern.
                _damps_val = (
                    1.0 + 2.0 * tf.maximum(0.0, tf.sqrt((self.μeff_tf - 1.0) / (self.N_tf + 1.0)) - 1.0) + self.cσ_tf
                )
                self.damps_tf = tf.identity(_damps_val, name="damps_tf")

            # Corrected Line 200 area (chiN_tf)
            _chiN_term_in_paren = 1.0 - 1.0 / (4.0 * self.N_tf) + 1.0 / (21.0 * tf.square(self.N_tf))
            _chiN_sqrt_N = tf.sqrt(self.N_tf)
            self.chiN_tf = tf.multiply(_chiN_sqrt_N, _chiN_term_in_paren, name="chiN_tf")

            if self._enforce_bounds_py_bool:
                bounds_np = np.array(self.enforce_bounds_py, dtype=self.dtype.as_numpy_dtype())
                self.clip_value_min_tf = tf.constant(bounds_np[:, 0], dtype=self.dtype, name="clip_min")
                self.clip_value_max_tf = tf.constant(bounds_np[:, 1], dtype=self.dtype, name="clip_max")
            
            self._enforce_bounds_tf_bool = tf.constant(self._enforce_bounds_py_bool, dtype=tf.bool, name="enforce_bounds_bool")

            # ---------------------
            # Trainable parameters (TF Variables)
            # ---------------------
            self.m_var = tf.Variable(self.initial_solution_np, dtype=self.dtype, name="m")
            self.σ_var = tf.Variable(self.initial_step_size_py, dtype=self.dtype, name="sigma")
            self.C_var = tf.Variable(tf.eye(self.dimension, dtype=self.dtype), name="C")
            self.p_σ_var = tf.Variable(tf.zeros([self.dimension], dtype=self.dtype), name="p_sigma")
            self.p_C_var = tf.Variable(tf.zeros([self.dimension], dtype=self.dtype), name="p_C")
            self.B_var = tf.Variable(tf.eye(self.dimension, dtype=self.dtype), name="B")
            self.D_var = tf.Variable(tf.eye(self.dimension, dtype=self.dtype), name="D") # Diagonal matrix

            # --------------------------------------
            # Define graph operations for one generation
            # --------------------------------------
            # (1) Sample a new population
            z_op = tf.random_normal(self.shape_op, dtype=self.dtype, name="z")
            y_op = tf.matmul(z_op, tf.matmul(self.B_var, self.D_var), name="y")
            x_op = tf.add(self.m_var, self.σ_var * y_op, name="x_sampled")

            x_clipped_op = tf.clip_by_value(x_op, self.clip_value_min_tf, self.clip_value_max_tf, name="x_clipped")
            
            scalar_penalty_op = tf.cond(
                self._enforce_bounds_tf_bool,
                lambda: tf.norm(x_op - x_clipped_op)**2, 
                lambda: tf.constant(0.0, dtype=self.dtype)
            )
            scalar_penalty_op = tf.identity(scalar_penalty_op, name="scalar_penalty")
            
            self.x_final_op = tf.cond(
                self._enforce_bounds_tf_bool,
                lambda: x_clipped_op,
                lambda: x_op,
                name="x_final"
            )

            # (2) Selection and Recombination
            f_x_op_unnamed = self.fitness_fn_user(self.x_final_op) + scalar_penalty_op
            f_x_op = tf.identity(f_x_op_unnamed, name="f_x")

            sorted_indices_op = tf.argsort(f_x_op, name="sorted_indices")
            self.x_sorted_op = tf.gather(self.x_final_op, sorted_indices_op, name="x_sorted")

            x_diff_op = self.x_sorted_op - self.m_var 
            x_mean_op = tf.reduce_sum(x_diff_op * self.weights_tf, axis=0, name="x_mean_weighted_diff")
            m_new_op = tf.add(self.m_var, x_mean_op, name="m_new")

            # (3) Adapting Covariance Matrix
            y_mean_op_unnamed = x_mean_op / self.σ_var
            y_mean_op = tf.identity(y_mean_op_unnamed, name="y_mean_weighted")

            p_C_delta_op = tf.sqrt(self.cc_tf * (2.0 - self.cc_tf) * self.μeff_tf) * y_mean_op
            p_C_new_op_unnamed = (1.0 - self.cc_tf) * self.p_C_var + p_C_delta_op
            p_C_new_op = tf.identity(p_C_new_op_unnamed, name="p_C_new")
            p_C_matrix_op = p_C_new_op[:, tf.newaxis]

            y_k_list_op = (self.x_sorted_op - self.m_var) / self.σ_var 
            map_fn_elems = y_k_list_op[:, tf.newaxis, :] 
            
            C_mu_terms_op = tf.map_fn(
                fn=lambda e_row_tensor: e_row_tensor * tf.transpose(e_row_tensor),
                elems=map_fn_elems,
                dtype=self.dtype 
            ) 
            
            y_s_op = tf.reduce_sum(C_mu_terms_op * self.weights_tf[:, :, tf.newaxis], axis=0, name="y_s_rank_mu_update")

            term_old_C = (1.0 - self.c1_tf - self.cμ_tf) * self.C_var
            term_rank_one = self.c1_tf * tf.matmul(p_C_matrix_op, tf.transpose(p_C_matrix_op))
            term_rank_mu = self.cμ_tf * y_s_op
            C_new_op_unnamed = term_old_C + term_rank_one + term_rank_mu
            
            C_upper_op = tf.matrix_band_part(C_new_op_unnamed, 0, -1)
            C_diag_from_upper_op = tf.linalg.diag_part(C_upper_op) # tf.diag_part in TF1
            C_diag_matrix_op = tf.linalg.diag(C_diag_from_upper_op) # tf.diag in TF1
            C_upper_no_diag_op = C_upper_op - C_diag_matrix_op
            C_new_op_symmetric = C_diag_matrix_op + C_upper_no_diag_op + tf.transpose(C_upper_no_diag_op)
            C_new_op = tf.identity(C_new_op_symmetric, name="C_new")


            # (4) Step-size control (σ)
            D_inv_diag_elements = tf.math.reciprocal(tf.linalg.diag_part(self.D_var)) # tf.diag_part in TF1
            D_inv_op = tf.linalg.diag(D_inv_diag_elements, name="D_inv") # tf.diag in TF1
            C_inv_sqrt_op = tf.matmul(tf.matmul(self.B_var, D_inv_op), tf.transpose(self.B_var), name="C_inv_sqrt")
            C_inv_sqrt_y_op = tf.squeeze(tf.matmul(C_inv_sqrt_op, y_mean_op[:, tf.newaxis]), axis=[1], name="C_inv_sqrt_y")


            p_σ_delta_op = tf.sqrt(self.cσ_tf * (2.0 - self.cσ_tf) * self.μeff_tf) * C_inv_sqrt_y_op
            p_σ_new_op_unnamed = (1.0 - self.cσ_tf) * self.p_σ_var + p_σ_delta_op
            p_σ_new_op = tf.identity(p_σ_new_op_unnamed, name="p_sigma_new")

            exp_term = (self.cσ_tf / self.damps_tf) * ((tf.norm(p_σ_new_op) / self.chiN_tf) - 1.0)
            σ_new_op_unnamed = self.σ_var * tf.exp(exp_term)
            σ_new_op = tf.identity(σ_new_op_unnamed, name="sigma_new")

            # (5) Update B and D (eigen decomposition of C_new_op)
            s_op, u_op, _ = tf.linalg.svd(C_new_op, name="svd_C_new") 
            
            self.B_new_from_svd_op = tf.identity(u_op, name="B_new_from_svd")
            self.diag_D_elements_op = tf.sqrt(s_op, name="diag_D_elements_new") 
            self.D_new_from_svd_op = tf.linalg.diag(self.diag_D_elements_op, name="D_new_from_svd") # tf.diag in TF1

            # (6) Create assignment operations
            assign_m_op = tf.assign(self.m_var, m_new_op)
            assign_σ_op = tf.assign(self.σ_var, σ_new_op)
            assign_C_op = tf.assign(self.C_var, C_new_op)
            assign_p_σ_op = tf.assign(self.p_σ_var, p_σ_new_op)
            assign_p_C_op = tf.assign(self.p_C_var, p_C_new_op)
            assign_B_op = tf.assign(self.B_var, self.B_new_from_svd_op)
            assign_D_op = tf.assign(self.D_var, self.D_new_from_svd_op)

            self.update_op_group = tf.group(
                assign_m_op, assign_σ_op, assign_C_op,
                assign_p_σ_op, assign_p_C_op,
                assign_B_op, assign_D_op,
                name="update_op_group"
            )

            m_expanded_op = tf.expand_dims(self.m_var, 0)
            self.best_fitness_eval_op = self.fitness_fn_user(m_expanded_op)[0]

            self.init_vars_op = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_vars_op)

        self._initialized = True
        return self

    def search(self, max_generations=500):
        if not self._initialized:
            self.init()

        if self.callback_fn is not None:
            self.callback_fn(self, logger)

        for _generation_idx in range(max_generations):
            self.generation += 1

            # (0) Get current sigma_var and D_var for TolXUp termination criteria.
            # These are the values *before* this generation's updates.
            self._prev_sigma_val_for_term, self._prev_D_val_for_term = self.sess.run([self.σ_var, self.D_var])
            
            # (1-6) Run one generation: sampling, selection, updates.
            # Fetch values needed for trace and termination checks.
            fetches_for_run = {
                'update_group': self.update_op_group, # Run all assignment ops
                # For trace (values *after* update):
                'm_updated': self.m_var,
                'sigma_updated': self.σ_var,
                'C_updated': self.C_var,
                'p_sigma_updated': self.p_σ_var,
                'p_C_updated': self.p_C_var,
                'B_updated': self.B_var,
                'D_updated': self.D_var,
                'x_sorted_current': self.x_sorted_op, # Op result from current generation
                # For termination check:
                'diag_D_elements_current': self.diag_D_elements_op # Op result (sqrt of new eigenvalues)
            }
            
            run_results = self.sess.run(fetches_for_run)

            # Store trace if enabled
            if self.store_trace:
                trace_data = {
                    'm': run_results['m_updated'],
                    'σ': run_results['sigma_updated'],
                    'C': run_results['C_updated'],
                    'p_σ': run_results['p_sigma_updated'],
                    'p_C': run_results['p_C_updated'],
                    'B': run_results['B_updated'],
                    'D': run_results['D_updated'],
                    'population': run_results['x_sorted_current'],
                }
                self.trace.append(trace_data)

            # (7) Terminate early if necessary
            term_check_args = {
                'm_val': run_results['m_updated'],
                'sigma_val': run_results['sigma_updated'],
                'C_val': run_results['C_updated'],
                'B_val': run_results['B_updated'], # New B from SVD
                'diag_D_elements_val': run_results['diag_D_elements_current'], # Sqrt(new eigenvalues)
                'prev_sigma_val': self._prev_sigma_val_for_term,
                'prev_D_val': self._prev_D_val_for_term, # Full D matrix from before update
                'return_details': False 
            }
            self.termination_criterion_met = self._should_terminate_np(**term_check_args)

            if self.callback_fn is not None:
                self.callback_fn(self, logger)

            if self.termination_criterion_met:
                break
        
        return self.best_solution(), self.best_fitness()

    def best_solution(self):
        if not self._initialized or self.sess is None:
            # Fallback for the case where search might not have been called, but init was.
            # For example, if user wants to see initial solution through this method.
            if self.graph is None: # Not even init was called
                 raise RuntimeError("CMA not initialized. Call init() first.")
            # If graph exists but session is somehow None (should not happen with current logic)
            if self.sess is None:
                self.sess = tf.Session(graph=self.graph)
                # Best_solution might be called before search, so variables might not be initialized from search.
                # This relies on init_vars_op being run in init().
            return self.sess.run(self.m_var)
        return self.sess.run(self.m_var)
    
    def get_mean(self):
        """Returns the current mean vector (m) as a NumPy array."""
        if not self._initialized or self.sess is None:
            # This logic is similar to best_solution() for uninitialized state
            if self.graph is None:
                 raise RuntimeError("CMA not initialized. Call init() first.")
            if self.sess is None: # Should not happen if init was successful
                self.sess = tf.Session(graph=self.graph)
        return self.sess.run(self.m_var)

    def get_covariance_matrix(self):
        """Returns the current covariance matrix (C) as a NumPy array."""
        if not self._initialized or self.sess is None:
            if self.graph is None:
                 raise RuntimeError("CMA not initialized. Call init() first.")
            if self.sess is None:
                self.sess = tf.Session(graph=self.graph)
        return self.sess.run(self.C_var)


    def best_fitness(self):
        if not self._initialized or self.sess is None:
            if self.graph is None:
                 raise RuntimeError("CMA not initialized. Call init() first.")
            if self.sess is None:
                self.sess = tf.Session(graph=self.graph)
            return self.sess.run(self.best_fitness_eval_op)
        return self.sess.run(self.best_fitness_eval_op)

    def _should_terminate_np(self, m_val, sigma_val, C_val, B_val, diag_D_elements_val,
                             prev_sigma_val, prev_D_val, return_details=False):
        # All inputs are NumPy arrays from sess.run()

        # NoEffectAxis: stop if adding a 0.1-standard deviation vector in any principal axis
        # direction of C does not change m (i.e., the addition is numerically zero).
        # B_val columns are eigenvectors. diag_D_elements_val are sqrt of eigenvalues.
        idx_nea = (self.generation - 1) % self.dimension # Current axis index based on generation
        
        # Principal axis std dev vector: 0.1 * sigma_val * sqrt(eigenvalue_i) * eigenvector_i
        # diag_D_elements_val[idx_nea] is sqrt(eigenvalue) for the idx_nea axis.
        # B_val[:, idx_nea] is the eigenvector for the idx_nea axis.
        m_nea_add = 0.1 * sigma_val * diag_D_elements_val[idx_nea] * B_val[:, idx_nea]
        no_effect_axis = np.all(np.abs(m_nea_add) < self.termination_no_effect_py)

        # NoEffectCoord: stop if adding 0.2 stdev in any single coordinate does not change m
        # Original TF2 code: m_nec = self.m + 0.2 * self.σ * tf.linalg.diag_part(self.C)
        # This means 0.2 * sigma * C_kk (variance along coordinate k).
        # This is what's replicated here.
        m_nec_add = 0.2 * sigma_val * np.diag(C_val) # np.diag(C_val) gets diagonal elements (variances C_kk)
        no_effect_coord = np.any(np.abs(m_nec_add) < self.termination_no_effect_py)

        # ConditionCov: stop if condition number of C (max_eig / min_eig) is > 1e14
        # diag_D_elements_val are sqrt of eigenvalues of new C.
        max_eig_sqrt = np.max(diag_D_elements_val)
        min_eig_sqrt = np.min(diag_D_elements_val)
        
        condition_cov = False
        if min_eig_sqrt <= 0: # Eigenvalue is zero or negative (numerically unstable C)
            condition_cov = True # Effectively infinite or undefined condition number
        else:
            condition_number = (max_eig_sqrt / min_eig_sqrt)**2
            if condition_number > 1e14:
                 condition_cov = True
        
        # TolXUp: stop if σ × max(D_elements) increased by more than 10^4.
        # max_eig_sqrt is max(sqrt(eigenvalues)) from current D.
        # prev_D_val is the D matrix from previous generation. np.diag(prev_D_val) gets its diag elements.
        prev_max_D_diag_element = np.max(np.diag(prev_D_val))
        
        current_sigma_max_D_diag = sigma_val * max_eig_sqrt
        prev_sigma_max_D_diag = prev_sigma_val * prev_max_D_diag_element
        
        # Original TF2: tf.abs(self.σ * max_D - self._prev_sigma * prev_max_D) > 1e4. This was a symmetric check.
        # However, the text says "increased by more than 10^4".
        # Replicating original TF2 code's TolXUp: `tf.greater(tol_x_up_diff, 1e4)` where `tol_x_up_diff` has `tf.abs`.
        # This means a large increase OR decrease would trigger.
        # Let's stick to the `abs` version.
        tol_x_up_diff_abs = np.abs(current_sigma_max_D_diag - prev_sigma_max_D_diag)
        tol_x_up = tol_x_up_diff_abs > 1e4

        do_terminate = bool(no_effect_axis or no_effect_coord or condition_cov or tol_x_up)

        if not return_details:
            return do_terminate
        else:
            return (
                do_terminate,
                dict(
                    no_effect_axis=bool(no_effect_axis),
                    no_effect_coord=bool(no_effect_coord),
                    condition_cov=bool(condition_cov),
                    tol_x_up=bool(tol_x_up),
                )
            )

    def reset(self):
        """Resets the CMA optimizer to its initial state for a new search."""
        self.close() # Close existing session and clear graph reference
        self._initialized = False
        # Reset internal state for termination checks
        self._prev_sigma_val_for_term = None
        self._prev_D_val_for_term = None
        self.termination_criterion_met = False
        return self.init() # Re-initialize graph, session, and variables

    def close(self):
        """Closes the TensorFlow session and releases resources."""
        if self.sess is not None:
            self.sess.close()
            logger.info("CMA TensorFlow session closed.")
        self.sess = None
        self.graph = None # Allow graph to be garbage collected
        # Mark as not usable until re-init, but reset handles this.
        # self._initialized = False 

    # It's good practice to provide a way to clean up TF resources when the object is no longer needed.
    # A context manager (`__enter__`, `__exit__`) could also be an option if used like `with CMA(...) as cma:`.
    def __del__(self):
        self.close()