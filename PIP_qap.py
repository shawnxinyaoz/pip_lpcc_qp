import numpy as np
from pyomo.environ import *
from datetime import datetime
import pandas as pd
import os
import time


# The first functions all for QAP data processing from the QAPLIB instances files.

# Function to read .dat file and extract n, F, and D
def read_qap_dat_file_A(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]  # Remove empty lines and strip spaces
        n = int(lines[0])  # The first non-empty line contains the size n
        F = [[int(num) for num in line.split()] for line in lines[1:n+1]]
        D = [[int(num) for num in line.split()] for line in lines[n+1:2*n+1]]
    return n, F, D

# In case the QAPLIB instance file is not formatted to be easily read, which happens
def read_qap_dat_file_B(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]  # Remove empty lines and strip spaces
        n = int(lines[0])  # The first non-empty line contains the size n
        F = [[int(num) for num in line.split()] for line in lines[1:n + 1]]

        # Handling D where each row might be split into multiple lines
        raw_D_lines = lines[n + 1:]
        # Combine every two lines to form one complete row for D
        corrected_D_lines = [' '.join(raw_D_lines[i:i + 2]) for i in range(0, len(raw_D_lines), 2)]
        D = [[int(num) for num in line.split()] for line in corrected_D_lines]
    return n, F, D

# Function to create Cijkl
def create_cijkl(F, D):
    n = len(F)
    C = np.zeros((n, n, n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    C[i, j, k, l] = F[i][j] * D[k][l]
    return C

# Function to define matrix S
def create_Q(C):
    n = C.shape[0]
    S = np.zeros((n**2, n**2), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    S[(i)*n + k, (j)*n + l] = C[i, j, k, l]
    # Calculate the maximum row norm of S
    max_row_norm = np.max(np.sum(np.abs(S), axis=1))

    # Set alpha to be larger than the maximum row norm
    alpha = max_row_norm + 1  # or any value greater than max_row_norm based on your needs

    # Subtract alpha * I from S
    Q = S - alpha * np.eye(n ** 2)

    print("symmetric check:", np.array_equal(Q, Q.T))
    print("Q is", check_matrix_definiteness(Q))

    return Q

def generate_eq_constraint_matrix(n):
    constraint_matrix = np.zeros((2 * n, n**2), dtype=int)
    for i in range(n):
        constraint_matrix[i, i*n:(i+1)*n] = 1
    for j in range(n):
        constraint_matrix[n+j, j::n] = 1
    return constraint_matrix

def generate_nonnegative_constraints(n):
    A = np.eye(n**2, dtype=int)
    b = np.zeros(n**2, dtype=int)
    return A, b

# Function to calculate the original objective value
def calculate_original_objective(C, x):
    if x is None:
        return "NA"
    n = C.shape[0]
    objective_value = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    objective_value += C[i, j, k, l] * x[i*n+k] * x[j*n+l]
    return objective_value

# Wrapper function to process QAP instance
def process_qap_instance(file_path):
    # Attempt to read using method A
    n, F, D = read_qap_dat_file_A(file_path)

    # Check dimensions
    if len(F) == n and len(D) == n and all(len(row) == n for row in F) and all(len(row) == n for row in D):
        # Dimensions are correct
        pass
    else:
        # Attempt to read using method B
        n, F, D = read_qap_dat_file_B(file_path)
        # Check dimensions again
        if not (len(F) == n and len(D) == n and all(len(row) == n for row in F) and all(len(row) == n for row in D)):
            # If still incorrect, raise an error
            raise ValueError("Error in data file: dimensions of F or D are not correct")

    # If dimensions are correct, proceed to process QAP instance
    C = create_cijkl(F, D)
    Q = create_Q(C)
    Dx = generate_eq_constraint_matrix(n)
    A, b = generate_nonnegative_constraints(n)
    f = np.ones(2 * n, dtype=int)
    return C, Q, A, b, Dx, f

r'''
The above functions process the QAP data file from QAPLIB and reformulate into QP as stated in the manuscript,
C is pertained to recover the original objective value.
Q defines the objective min 0.5 x^T Q x.
A b Dx f define the constraints A x \ge b and Dx x = f.
Note that the notation for the equality cosntraint matrix here is Dx since D is used to represent the distance matrix from the orignal QAP data.
r'''

def save_df_qap(df, n_vars, dat_file):
    r'''
    This function saves the results of PIP runs on qap.
    r'''
    # Check if "result" directory exists, if not, create it
    if not os.path.exists('qap_results'):
        os.makedirs('qap_results')

    # Get the current date and time for timestamp (optional)
    now = datetime.now()

    filename = now.strftime(f'PIP_QAP_problem_{dat_file}.csv')

    # Construct the full path to save the file in the result subfolder
    full_path = os.path.join('qap_results', filename)

    # Save the DataFrame to the constructed path
    df.to_csv(full_path, index=False)
    print(f"Saved results to {full_path}")

def pip_solve_qap(n_vars, n_ineq_constraints, n_eq_constraints, C, Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, start_fixed_proportion, final_fixed_proportion, max_iters, max_repeat_times, timelimit, pip_step_size = 0.1):
    r'''
    This is the PIP main function modified for solving QAP.
    n_ineq_constraints and n_eq_constraints are the numbers of inequality and equality constraints in the converted formulation.
    C is pertained Flow-Distance matrix to recover orignal QAP objective.
    Q, c, A, b, D, f  define the QP reformulation of the QAP
    \min_x 0.5 x^T Q x + c  s.t. A x \ge b, D x = f.
    u_x, upperbound_for_lambda, upperbound_for_slacka are the upperbounds and big-M constant for the LPCC variable_result_saving_time.
    start_fixed_proportion and final_fixed_proportion are the fixed proportion of the first and last iterations. Note that p_max = 1-final_fixed_proportion.
    max_iters provides additional control of maximum numebr of iterations of PIP runs if needed.
    max_repeat_times controls number of iterations that the fixing proportion can remain unchanged.
    timelimit is the pip subproblem timelimit.
    id_to_save: notes to be saved in results if needed.
    pip_step_size: \alpha in paper, i.e. the amount of fixing proportion change at expansion steps.
    r'''
    # Initial model for first stationary solution
    print("Initializing PIP...")

    start_time = time.time() # Timer for entire PIP

    current_start_time = time.time() # Timer for current model

    model = ConcreteModel()

    # # Decision variables
    model.x = Var(range(n_vars), within=Reals, bounds = (-u_x-(1e-4), u_x+(1e-4)))


    # Objective
    model.obj = Objective(expr = 0.5 * sum(Q[i, j] * model.x[i] * model.x[j] for i in range(n_vars) for j in range(n_vars)) +
                          sum(c[i] * model.x[i] for i in range(n_vars)), sense=minimize)

    # INEQ Constraints
    def ineq_constraints_rule(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n_vars)) - b[k] >= 0
    model.ineq_constraints = Constraint(range(n_ineq_constraints), rule=ineq_constraints_rule)

    if D is not None:
        # EQ Constraints
        def eq_constraints_rule(model, k):
            return sum(D[k, j] * model.x[j] for j in range(n_vars)) - f[k] == 0
        model.eq_constraints = Constraint(range(n_eq_constraints), rule=eq_constraints_rule)

    # Declare the dual component to store the dual values
    model.dual = Suffix(direction=Suffix.IMPORT)

    # Solver
    solver = SolverFactory('ipopt')

    # solver.options['max_iter'] = 10
    results = solver.solve(model, tee=True)
    current_modeling_and_computation_time = time.time() - current_start_time

    # Start saving results
    variable_result_saving_start_time = time.time()
    ipopt_time = results.solver.time

    # Save the optimal values of x and the corresponding dual variables
    x_ = [model.x[i].value for i in range(n_vars)]
    lambda_ = [model.dual[model.ineq_constraints[k]] for k in range(n_ineq_constraints)]
    slack_ = [sum(A[k, i] * x_[i] for i in range(n_vars)) - b[k] for k in range(n_ineq_constraints)]

    number_of_degenerate_constraints = sum(1 for k in range(n_ineq_constraints) if lambda_[k] == 0 and slack_[k] == 0)

    # Output
    print("\nStatus:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)

    obj = value(model.obj)
    print("Initialization Done. \n Objective Value:", obj)  # Displaying the objective value

    variable_result_saving_time = time.time() - variable_result_saving_start_time
    #######################
    # Prepare for result recording
    recording_start_time = time.time()
    # Indices and parameters
    iteration_list = [0]
    fixed_proportion_list = ['NA']
    number_of_integer_variables_list = ["NA"]

    # Obj values
    objective_value_list = [obj]

    original_objective_value_list = []

    original_objective_value = calculate_original_objective(C, x_)
    original_objective_value_list.append(original_objective_value)

    improvement_list = ['NA']

    # Number of fixing variables
    number_of_lambda_fixed_zero_list = ['NA']
    number_of_slack_fixed_zero_list = ['NA']
    lambda_fixed_zero_indices_list = ["NA"]
    slack_fixed_zero_indices_list = ["NA"]

    number_of_degenerate_constraints_list = [number_of_degenerate_constraints]

    # Time records
    computation_time_list = [ipopt_time] # Solver reported computation time

    modeling_and_computation_time_list = [current_modeling_and_computation_time] # Modeling + solver time

    running_total = ipopt_time
    cumulative_computation_time_list = [running_total] # Solver Computation time sum up

    total_time_list = [time.time() - start_time] # Modeling + computation + result recording

    # Other data
    lambda_range_list = [(min(lambda_),max(lambda_))]
    slack_range_list = [(min(slack_), max(slack_))]

    notes_list = [f'u_x is {u_x}, u_l is {upperbound_for_lambda}, u_s is {upperbound_for_slack}']

    # strictness for positivity constraints
    epsilon_l_list = ['NA']
    epsilon_s_list = ['NA']

    variable_result_saving_time_list = [variable_result_saving_time]

    record_saving_time = time.time()-recording_start_time
    record_saving_time_list = [record_saving_time]
    #######################
    # Parameter initial settings
    epsilon_l = 1e-4
    epsilon_s = 1e-4

    epsilon = [epsilon_l, epsilon_s]

    i=0
    repeat = 0
    fixed_proportion = start_fixed_proportion

    ##################
    # Building base model
    current_start_time = time.time()
    model = ConcreteModel()

    # Decision variables
    model.x = Var(range(n_vars), within=Reals, bounds = (-u_x, u_x))
    model.l = Var(range(n_ineq_constraints), within=Reals)
    model.z = Var(range(n_ineq_constraints), within=Binary)
    if n_eq_constraints != 0:
        model.u = Var(range(n_eq_constraints), within=Reals)

        # Objective
        model.obj = Objective(expr=0.5 * sum(c[i] * model.x[i] for i in range(n_vars)) +
                              0.5 * sum(b[k] * model.l[k] for k in range(n_ineq_constraints)) +
                              0.5 * sum(f[k] * model.u[k] for k in range(n_eq_constraints)),
                              sense=minimize)
        def kkt_1_constraint_rule(model, i):
            return c[i] + sum(Q[i, j] * model.x[j] for j in range(n_vars)) - sum(A[k, i] * model.l[k] for k in range(n_ineq_constraints)) - sum(D[k, i] * model.u[k] for k in range(n_eq_constraints)) == 0

    else:
        model.obj = Objective(expr=0.5 * sum(c[i] * model.x[i] for i in range(n_vars)) +
                      0.5 * sum(b[k] * model.l[k] for k in range(n_ineq_constraints)),
                      sense=minimize)
        def kkt_1_constraint_rule(model, i):
            return c[i] + sum(Q[i, j] * model.x[j] for j in range(n_vars)) - sum(A[k, i] * model.l[k] for k in range(n_ineq_constraints)) == 0

    model.kkt_1_constraint_constraints = Constraint(range(n_vars), rule=kkt_1_constraint_rule)

    def l_bound_rule_lower(model, k):
        return model.l[k] >= 0

    def l_bound_rule_upper(model, k):
        return model.l[k] <= upperbound_for_lambda * model.z[k]

    model.l_bound_constraints_lower = Constraint(range(n_ineq_constraints), rule=l_bound_rule_lower)
    model.l_bound_constraints_upper = Constraint(range(n_ineq_constraints), rule=l_bound_rule_upper)

    def ax_bound_rule_lower(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n_vars)) - b[k] >= 0

    def ax_bound_rule_upper(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n_vars)) - b[k] <= upperbound_for_slack * (1 - model.z[k])

    model.ax_bound_constraints_lower = Constraint(range(n_ineq_constraints), rule=ax_bound_rule_lower)
    model.ax_bound_constraints_upper = Constraint(range(n_ineq_constraints), rule=ax_bound_rule_upper)

    # # This line computes the dot product of the k-th row of D with x and subtracts the k-th element of f
    if n_eq_constraints != 0:
        def dx_f_constraint_rule(model, k):
            return sum(D[k, j] * model.x[j] for j in range(n_vars)) - f[k] == 0

        model.dx_f_constraints = Constraint(range(n_eq_constraints), rule=dx_f_constraint_rule)

    #######################
    # Iterative Leverage
    while i < max_iters:
        i+=1
        print("Iteration", i,": Fixing proportion,", fixed_proportion, "\n")

        #######################
        # Fixing determination

        # Calculate number of positive lambda and slacks
        pl = sum(l > 0 for l in lambda_)
        ps = sum(s > 0 for s in slack_)

        # Determine the number of indices to select based on pl and ps
        num_indices_to_select_lambda = int(fixed_proportion * pl)
        num_indices_to_select_slacks = int(fixed_proportion * ps)

        # Get the indices for the largest lambdas and constraint_values based on the new criteria
        largest_lambda_indices = sorted(range(len(lambda_)), key=lambda k: lambda_[k], reverse=True)[:num_indices_to_select_lambda]
        largest_slack_indices = sorted(range(len(slack_)), key=lambda k: slack_[k], reverse=True)[:num_indices_to_select_slacks]

        slack_fixed_zero_indices = []
        lambda_fixed_zero_indices = []

        # Modify the constraints based on the obtained indices
        for k in range(n_ineq_constraints):
            # For the largest lambda values
            if k in largest_lambda_indices and lambda_[k] > epsilon[0]:
                slack_fixed_zero_indices.append(k)
            # For the largest constraint_values
            elif k in largest_slack_indices and slack_[k] > epsilon[1]:
                lambda_fixed_zero_indices.append(k)

        #######################
        # FIXING
        for k in slack_fixed_zero_indices:
            model.z[k].fix(1)
        for k in lambda_fixed_zero_indices:
            model.z[k].fix(0)

        ########################
        # Warm start
        for k in range(len(x_)):
            model.x[k]=x_[k]
        for j in range(len(lambda_)):
            model.l[j]=lambda_[j]

        ########################
        # Solve
        solver = SolverFactory('gurobi')

        solver.options['timelimit']= timelimit

        results = solver.solve(model, tee=True, warmstart = True)

        current_modeling_and_computation_time = time.time() - current_start_time

        # start saving results
        variable_result_saving_start_time = time.time()
        # Save the new optimal values of x and the corresponding dual variables and the slack variables
        x_ = [model.x[i].value for i in range(n_vars)]
        lambda_ = [model.l[k].value for k in range(n_ineq_constraints)]
        slack_ = [sum(A[k, i] * x_[i] for i in range(n_vars)) - b[k] for k in range(n_ineq_constraints)]

        number_of_degenerate_constraints = sum(1 for k in range(n_ineq_constraints) if lambda_[k] == 0  and slack_[k] == 0)


        ########################

        current_obj = value(model.obj)
        comp_time = results.solver.time  # This gets the solver time

        variable_result_saving_time = time.time() - variable_result_saving_start_time

        #######################
        # UNFIX MODEL AFTER SOLVE
        for k in slack_fixed_zero_indices:
            model.z[k].unfix()
        for k in lambda_fixed_zero_indices:
            model.z[k].unfix()
        #######################


        # Result recording
        recording_start_time = time.time()
        # Indices and parameters
        iteration_list.append(i)
        fixed_proportion_list.append(fixed_proportion)

        number_of_slack_fixed_zero =  len(slack_fixed_zero_indices)
        number_of_lambda_fixed_zero = len(lambda_fixed_zero_indices)

        number_of_integer_variables_list.append(n_ineq_constraints - number_of_slack_fixed_zero - number_of_lambda_fixed_zero)

        number_of_degenerate_constraints_list.append(number_of_degenerate_constraints)
        # Obj values
        objective_value_list.append(current_obj)

        original_objective_value = calculate_original_objective(C, x_)
        original_objective_value_list.append(original_objective_value)
        print("Original QAP Objective value: ", original_objective_value_list)

        improvement = obj - current_obj
        obj = current_obj
        improvement_list.append(improvement)

        # Number of fixing variables
        number_of_lambda_fixed_zero_list.append(number_of_lambda_fixed_zero)
        number_of_slack_fixed_zero_list.append(number_of_slack_fixed_zero)
        lambda_fixed_zero_indices_list.append(lambda_fixed_zero_indices)
        slack_fixed_zero_indices_list.append(slack_fixed_zero_indices)

        # Other data
        lambda_range_list.append((min(lambda_),max(lambda_)))
        slack_range_list.append((min(slack_), max(slack_)))

        notes_list.append('NA')

        # strictness for positivity constraints
        epsilon_l_list.append(epsilon_l)
        epsilon_s_list.append(epsilon_s)

        # Time records
        computation_time_list.append(comp_time) # Solver reported computation time

        modeling_and_computation_time_list.append(current_modeling_and_computation_time) # Modeling + solver time

        running_total += comp_time
        cumulative_computation_time_list.append(running_total) # Solver Computation time sum up

        variable_result_saving_time_list.append(variable_result_saving_time)

        record_saving_time = time.time()-recording_start_time
        record_saving_time_list.append(record_saving_time)

        total_time_list.append(time.time() - start_time) # Modeling + computation + result recording

        current_start_time = time.time() # restart
        #######################
        # Logical check for iterations
        repeat += 1

        if abs(improvement) <= 1e-4 or repeat >= max_repeat_times:
            repeat = 0
            fixed_proportion -= pip_step_size

        if fixed_proportion < final_fixed_proportion:
            break

    #####################
    return {
        'Iteration': iteration_list,
        'fixed_proportion (proportion of fixed int variables)': fixed_proportion_list,
        # 'Objective Value': objective_value_list,
        'Original Objective Value': original_objective_value_list,
        'total_time': total_time_list
        # , 'cumulative_solver_time': cumulative_computation_time_list,
        # 'Obj Improvement = last obj - current obj': improvement_list,
        # 'Number of lambda fixed zero (z_j = 0)': number_of_lambda_fixed_zero_list,
        # 'Number of slack fixed zero (z_j = 1)': number_of_slack_fixed_zero_list,
        # 'Number of unfixed integer variables': number_of_integer_variables_list,
        # 'lambda_fixed_zero_indices': lambda_fixed_zero_indices_list,
        # 'slack_fixed_zero_indices': slack_fixed_zero_indices_list,
        # 'number_of_degenerate_constraints': number_of_degenerate_constraints_list,
        # 'lambda range': lambda_range_list,
        # 'slack range': slack_range_list,
        # 'notes': notes_list,
        # 'epsilon_l_list': epsilon_l_list,
        # 'epsilon_s_list': epsilon_s_list,
        # 'computation_time': computation_time_list,
        # 'modeling+computation': modeling_and_computation_time_list,
        # 'variable result saving time post solver': variable_result_saving_time_list,
        # 'record_saving_time': record_saving_time_list
    }

def gurobi_full_qap(C, Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, timelimit, id_to_save, problem_type):
    r'''
    This function solves the QAP in FMIP approach.
    C is pertained Flow-Distance matrix to recover orignal QAP objective.
    Q, c, A, b, D, f define the QP reformulation of the QAP
    \min_x 0.5 x^T Q x + c  s.t. A x \ge b, D x = f.
    u_x, upperbound_for_lambda, upperbound_for_slacka are the upperbounds and big-M constant for the LPCC variable_result_saving_time.
    timelimit is the time limit provided to the solver.
    id_to_save and problem_type provide additional marking options to differentiate results.
    r'''
    results_list = []

    n_vars = Q.shape[0]
    n_ineq_constraints = A.shape[0]

    if D is not None:
        n_eq_constraints = D.shape[0]
    else:
        n_eq_constraints = 0

    start_time = time.time()

    model = ConcreteModel()

    # Decision variables
    model.x = Var(range(n_vars), within=Reals, bounds = (-u_x, u_x))
    model.l = Var(range(n_ineq_constraints), within=Reals)
    model.z = Var(range(n_ineq_constraints), within=Binary)
    if n_eq_constraints != 0:
        model.u = Var(range(n_eq_constraints), within=Reals)

        # Objective
        model.obj = Objective(expr=0.5 * sum(c[i] * model.x[i] for i in range(n_vars)) +
                              0.5 * sum(b[k] * model.l[k] for k in range(n_ineq_constraints)) +
                              0.5 * sum(f[k] * model.u[k] for k in range(n_eq_constraints)),
                              sense=minimize)
        def kkt_1_constraint_rule(model, i):
            return c[i] + sum(Q[i, j] * model.x[j] for j in range(n_vars)) - sum(A[k, i] * model.l[k] for k in range(n_ineq_constraints)) - sum(D[k, i] * model.u[k] for k in range(n_eq_constraints)) == 0

    else:
        model.obj = Objective(expr=0.5 * sum(c[i] * model.x[i] for i in range(n_vars)) +
                      0.5 * sum(b[k] * model.l[k] for k in range(n_ineq_constraints)),
                      sense=minimize)
        def kkt_1_constraint_rule(model, i):
            return c[i] + sum(Q[i, j] * model.x[j] for j in range(n_vars)) - sum(A[k, i] * model.l[k] for k in range(n_ineq_constraints)) == 0

    model.kkt_1_constraint_constraints = Constraint(range(n_vars), rule=kkt_1_constraint_rule)

    def l_bound_rule_lower(model, k):
        return model.l[k] >= 0

    def l_bound_rule_upper(model, k):
        return model.l[k] <= upperbound_for_lambda * model.z[k]

    model.l_bound_constraints_lower = Constraint(range(n_ineq_constraints), rule=l_bound_rule_lower)
    model.l_bound_constraints_upper = Constraint(range(n_ineq_constraints), rule=l_bound_rule_upper)

    def ax_bound_rule_lower(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n_vars)) - b[k] >= 0

    def ax_bound_rule_upper(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n_vars)) - b[k] <= upperbound_for_slack * (1 - model.z[k])

    model.ax_bound_constraints_lower = Constraint(range(n_ineq_constraints), rule=ax_bound_rule_lower)
    model.ax_bound_constraints_upper = Constraint(range(n_ineq_constraints), rule=ax_bound_rule_upper)

    # # This line computes the dot product of the k-th row of D with x and subtracts the k-th element of f
    if n_eq_constraints != 0:
        def dx_f_constraint_rule(model, k):
            return sum(D[k, j] * model.x[j] for j in range(n_vars)) - f[k] == 0

        model.dx_f_constraints = Constraint(range(n_eq_constraints), rule=dx_f_constraint_rule)

    # Solver
    solver = SolverFactory('gurobi')
    ########
    solver.options['TimeLimit'] = timelimit


    try:
        results = solver.solve(model, tee=True)
        objective_value = value(model.obj)
        x_ = [model.x[i].value for i in range(n_vars)]
    except Exception:  # Catching generic exception, can be specified
        objective_value = "NA"
        x_ = None

    # print("\nStatus:", results.solver.status)
    # print("Termination Condition:", results.solver.termination_condition)
    print("\n Objective value is:", objective_value)

    results_list.append({
        'dim_x': n_vars,
        'dim_z': n_ineq_constraints,
        'num_of_ineq_cosntraints': n_ineq_constraints,
        'num_of_eq_constraints': n_eq_constraints,
        'instance (note)': id_to_save,
        # 'objective_value': objective_value,
        'original_objective_value': calculate_original_objective(C, x_),
        'total_time': time.time() - start_time
        # ,'gurobi_solve_time': results.solver.time if objective_value != "NA" else timelimit,
        # 'upperbound_for_lambda': upperbound_for_lambda,
        # 'upperbound_for_slack': upperbound_for_slack,
    })

    # Convert list of dictionaries to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    if not os.path.exists(f'{problem_type}_results'):
        os.makedirs(f'{problem_type}_results')

    filename = f'FMIP_QAP_problem_{id_to_save}.csv'
    full_path = os.path.join(f'{problem_type}_results', filename)

    df.to_csv(full_path, index=False)

    print("Full MILP done. \n results saved in {}".format(full_path))

def check_matrix_definiteness(Q):
    r'''
    Check the definiteness of a given matrix Q.

    Args:
    - Q (numpy.ndarray): The matrix to be checked.

    Returns:
    - str: The definiteness type of the matrix (PD, ND, PSD, NSD, or Indefinite).
    r'''
    eigenvalues = np.linalg.eigvals(Q)

    if all(eigenvalue > 0 for eigenvalue in eigenvalues):
        return "Positive Definite (PD)"
    elif all(eigenvalue < 0 for eigenvalue in eigenvalues):
        return "Negative Definite (ND)"
    elif all(eigenvalue >= 0 for eigenvalue in eigenvalues):
        return "Positive Semi-Definite (PSD)"
    elif all(eigenvalue <= 0 for eigenvalue in eigenvalues):
        return "Negative Semi-Definite (NSD)"
    else:
        return "Indefinite"

def algorithm(folder_path = None, max_iters = 18, max_repeat_times = 3 , start_fixed_proportion = 0.8, final_fixed_proportion = 0.4, pip_subproblem_timelimit = 600, pip_step_size = 0.1, pip_run = 1, fmip_run = 1, fmip_timelimit = 600, problem_type = 'QAP', additional_input = "NA"):
    r'''
    This function combines instance loading, pip runs, and fmip runs.
    folder_path is the path of the folder containing the targeting instances.
    max_iters, max_repeat_times, start_fixed_proportion, final_fixed_proportion are parameters of the pip main function above.
    pip_run, fmip_run = 1 or 0 controls if the user wants to run each method on the targeting instances.
    fmip_timelimit is the time limit in seconds provided to the solver in fmip runs.
    r'''
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.dat')]

    # Sort the files to ensure they're processed in the correct order
    txt_files.sort()

    for file in txt_files:
        file_path = os.path.join(folder_path, file)

        C, Q, A, b, D, f = process_qap_instance(file_path)

        n_vars = Q.shape[0]
        c = np.zeros(n_vars)

        n_ineq_constraints = A.shape[0]
        n_eq_constraints = D.shape[0]


        # beta = np.max(Q) - np.min(Q)
        u_x = 1
        upperbound_for_lambda = 2 * Q.shape[0] * (np.max(np.abs(Q)))
        upperbound_for_slack = 1

        print(f"{problem_type}...data loaded from {file_path}")

        #######################
        # GUROBI FULL MILP
        if fmip_run == 1:
            print(f"RUNNING GUROBI FULL MILP...")
            id_to_save = f"{file}" # can add aditional notes for marking
            gurobi_full_qap(C, Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, fmip_timelimit, id_to_save, problem_type)
        if pip_run == 0:
            print("Gurobi only. Skipping PIP.")
            continue

        ########################

        result_dict = pip_solve_qap(n_vars, n_ineq_constraints, n_eq_constraints, C, Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, start_fixed_proportion, final_fixed_proportion, max_iters, max_repeat_times, pip_subproblem_timelimit, pip_step_size = pip_step_size)

        print('Outputing to dataframe....')
        results_df = pd.DataFrame(result_dict)

        print('Outputing to csv....')
        save_df_qap(results_df, n_vars, file)



r'''
The section below executes the experiments for the instances in the targeting folder path.
The user may control p_max, max_repeat (r_max in the manuscript), pip subproblem time limit, pip step size (\alpha), and other parameters here.
For FMIP runs, the fmip_timelimit can be controled at the bottom of the code.
r'''

print("Solving QAP with PIP...")

folder_path = 'instances/qap_instances/samples'

p_max = 0.6 # version of PIP (_)

max_repeat = 3 # max_repeat (r_max in the manuscript),
pip_subproblem_timelimit = 60 # time limit for each pip subproblem solving is 1 min for qap below 30 and 10 min everywhere else
pip_step_size = 0.1

start_fixed_proportion = 0.8
final_fixed_proportion = 1 - p_max
max_iters = np.rint((start_fixed_proportion - final_fixed_proportion + pip_step_size)/pip_step_size * max_repeat)

algorithm(folder_path = folder_path,
max_iters = max_iters,
max_repeat_times = max_repeat,
start_fixed_proportion = start_fixed_proportion,
final_fixed_proportion = final_fixed_proportion,
pip_subproblem_timelimit = pip_subproblem_timelimit,
pip_step_size = pip_step_size,
pip_run = 1,
fmip_run = 1,
fmip_timelimit = 600)
