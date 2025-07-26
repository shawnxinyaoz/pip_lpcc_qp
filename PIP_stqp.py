import scipy.sparse.linalg as sla
import numpy as np
from scipy.sparse import csr_matrix
from pyomo.environ import *
from datetime import datetime
import pandas as pd
import os
from scipy.sparse import random
from scipy.stats import norm
import time

def gurobi_full(Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, timelimit, id_to_save, problem_type, gurobi_warm = 1):
    r'''
    This is the function for FMIP and FMIP-W (warm started) calculation.
    Q, c, A, b, D, f are the matrices and vectors defining the corresponding QP:
    \min_x 0.5 x^T Q x + c  s.t. A x \ge b, D x = f.
    u_x, upperbound_for_lambda, upperbound_for_lambda are the upperbounds and big-M constant for the LPCC variable_result_saving_time.
    timelimit is the time limit provided to the solver for early termination.
    id_to_save, problem_type provide an option to mark some additional notes in results file saved if needed.
    Use gurobi_warm = 1 for FMIP-W calculation and 0 for FMIP calculation.
    r'''
    results_list = []

    n_vars = Q.shape[0]
    n_ineq_constraints = A.shape[0]

    if D is not None:
        n_eq_constraints = D.shape[0]
    else:
        n_eq_constraints = 0

    ######################## 0501
    # Initial model for first stationary solution
    print("Initializing FMIP GUROBI...")

    current_start_time = time.time() # Timer for current model

    model = ConcreteModel()

    # # Decision variables
    model.x = Var(range(n_vars), within=Reals, bounds = (-u_x, u_x))

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

    # Output
    print("\nStatus:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)

    initial_obj = value(model.obj)

    fmip_initial_time = time.time() - current_start_time
    print("FMIP Initialization Done. \n Objective Value:", initial_obj, '\n in', fmip_initial_time)  # Displaying the objective value

    #######################

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
    solver.options['TimeLimit'] = timelimit

    if gurobi_warm == 1:
        ######################################## Warm start ##################################
        for k in range(len(x_)):
            model.x[k] = x_[k]
        for j in range(len(lambda_)):
            model.l[j] = lambda_[j]
        ######################################################################################
        try:
            results = solver.solve(model, tee=True, warmstart=True)
            objective_value = value(model.obj)
        except Exception:  # Catching generic exception, can be specified
            objective_value = "NA"
    else:
        try:
            results = solver.solve(model, tee=True, warmstart=False)
            objective_value = value(model.obj)
        except Exception:  # Catching generic exception, can be specified
            objective_value = "NA"

    total_time = time.time()-current_start_time
    print("\n Objective value is:", objective_value, "\n Total time:", total_time)

    results_list.append({
        'dim_x': n_vars,
        'dim_z': n_ineq_constraints,
        # 'num_of_ineq_cosntraints': n_ineq_constraints,
        # 'num_of_eq_constraints': n_eq_constraints,
        'instance notes': id_to_save,
        'objective_value': objective_value,
        'total_time': time.time()-current_start_time,
        # 'total_gurobi_time': total_time,
        # 'gurobi_solve_time': results.solver.time if objective_value != "NA" else timelimit,
        # 'fmip_initial_time': fmip_initial_time,
        # 'initial_obj': initial_obj,
        # 'upperbound_for_lambda': upperbound_for_lambda,
        # 'upperbound_for_slack': upperbound_for_slack,
        'FMIP-W': gurobi_warm
    })

    # Convert list of dictionaries to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    if not os.path.exists("stqp_results"):
        os.makedirs("stqp_results")

    if gurobi_warm == 1:
        filename = f'FMIP_W_StQP_nvars_{n_vars}_problem_{id_to_save}.csv'
        full_path = os.path.join("stqp_results", filename)
        df.to_csv(full_path, index=False)
        print("Full MILP WARM done. \n results saved in {}".format(full_path))
    else:
        filename = f'FMIP_StQP_nvars_{n_vars}_problem_{id_to_save}.csv'
        full_path = os.path.join("stqp_results", filename)
        df.to_csv(full_path, index=False)
        print("Full MILP done. \n results saved in {}".format(full_path))

def save_df_fwd_sqp(df, n_vars, n_cons, notes):
    r'''
    This function is created to record the results of PIP runs
    r'''
    # Check if "result" directory exists, if not, create it
    if not os.path.exists('stqp_results'):
        os.makedirs('stqp_results')

    # Get the current date and time
    now = datetime.now()

    filename = now.strftime(f'PIP_StQP_nvars_{n_vars}_problem_{notes}.csv')

    # Construct the full path to save the file in the "result" subfolder
    full_path = os.path.join('stqp_results', filename)

    # Save the DataFrame to the constructed path
    df.to_csv(full_path, index=False)
    print(f"Saved results to {full_path}")

def pip_solve(n_vars, n_ineq_constraints, n_eq_constraints, Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, start_fixed_proportion, final_fixed_proportion, max_iters, max_repeat_times, timelimit, id_to_save, pip_step_size = 0.1):
    r'''
    This is the PIP main function solving stqp.
    n_vars: dimension of x.
    n_ineq_constraints: number of inequality constraints in the qp, also the dimension of the complementarity pairs.
    n_eq_constraints is the numebr of equality constraints.
    Q, c, A, b, D, f are the matrices and vectors defining the corresponding QP:
    \min_x 0.5 x^T Q x + c  s.t. A x \ge b, D x = f.
    u_x, upperbound_for_lambda, upperbound_for_slack are the upperbounds and big-M constant for the LPCC variable_result_saving_time.
    start_fixed_proportion and final_fixed_proportion are the fixed proportion of the first and last iterations.
    Note that p_max = 1-final_fixed_proportion.
    max_iters provides additional control of maximum numebr of iterations of PIP runs if needed.
    max_repeat_times controls number of iterations that the fixing proportion can remain unchanged.
    timelimit is the pip subproblem timelimit.
    id_to_save: additional notes to be saved in results if needed.
    pip_step_size: \alpha in paper, i.e. the amount of fixing proportion change at expansion steps.
    r'''
    ########################
    # Initial model for first stationary solution
    print("Initializing PIP...")

    start_time = time.time() # Timer for entire PIP

    current_start_time = time.time() # Timer for current model

    model = ConcreteModel()

    # # Decision variables
    model.x = Var(range(n_vars), within=Reals, bounds = (-u_x, u_x))

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
        print("Iteration", i,":\n")

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

        notes_list.append(id_to_save)

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
    # The dictionary saves a wide range of insights for the user to analyze.
    return {
        'Iteration': iteration_list,
        'fixed_proportion (proportion of fixed int variables)': fixed_proportion_list,
        'Objective Value': objective_value_list,
        'total_time': total_time_list
        # ,
        # 'cumulative_solver_time': cumulative_computation_time_list,
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

def algorithm(folder_path = None,
              max_iters = 15, max_repeat_times = 3 , start_fixed_proportion = 0.8, final_fixed_proportion = 0.4,
              pip_subproblem_timelimit = 600, pip_step_size = 0.1,
              pip_run = 1,
              fmip_run = 1,
              fmip_timelimit = 600,
              fmip_w_run = 1,
              fmip_w_timelimit = 600,
              problem_type = 'stqp'):
    r'''
    This function combines instance loading, pip runs, and fmip (-w) runs.
    folder_path is the path of the folder containing the targeting instances.
    max_iters, max_repeat_times, start_fixed_proportion, final_fixed_proportion are parameters of the pip main function above.
    pip_run, fmip_run, fmip_w_run = 1 or 0 controls if the user wants to run each method on the targeting instances.
    fmip_timelimit, fmip_w_timelimit are the time limit in seconds provided to the solver in fmip and fmip_w runs.
    r'''

    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Sort the files to ensure they're processed in the correct order
    txt_files.sort()

    for file in txt_files:
        file_path = os.path.join(folder_path, file)

        Q = np.loadtxt(file_path)

        n_vars = Q.shape[0]
        n_ineq_constraints = n_vars
        n_eq_constraints = 1

        c = np.zeros(n_vars)

        # beta = np.max(Q) - np.min(Q)
        u_x = 1
        upperbound_for_lambda = 2 * Q.shape[0] * (np.max(np.abs(Q)))
        upperbound_for_slack = 1

        # x >= 0
        A = np.eye(n_vars)
        b = np.zeros(n_vars)

        # Define D as a 1 x n_vars matrix where every entry is 1
        D = np.ones((1, n_vars))
        f = np.array([1])

        print(f"{problem_type}...data loaded from {file_path}")

        #######################
        # GUROBI FULL MILP
        print(f"RUNNING GUROBI FULL MILP...")

        id_to_save = f"{file}"

        if fmip_run == 1:
            gurobi_full(Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, fmip_timelimit, id_to_save, problem_type, gurobi_warm = 0)
        if fmip_w_run == 1:
            gurobi_full(Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, fmip_w_timelimit, id_to_save, problem_type, gurobi_warm = 1)
        if pip_run == 0:
            print("Gurobi only. Skipping PIP.")
            continue

        result_dict = pip_solve(n_vars, n_ineq_constraints, n_eq_constraints, Q, c, A, b, D, f, u_x, upperbound_for_lambda, upperbound_for_slack, start_fixed_proportion, final_fixed_proportion, max_iters, max_repeat_times, pip_subproblem_timelimit, file, pip_step_size = pip_step_size)

        print('Outputing to dataframe....')
        results_df = pd.DataFrame(result_dict)

        print('Outputing to csv....')
        save_df_fwd_sqp(results_df, n_vars, n_ineq_constraints, file)

r'''
The section below executes the experiments for the instances in the targeting folder path.
The user may control p_max, max_repeat (r_max in the manuscript), pip subproblem time limit, pip step size (\alpha), and other parameters here.
For FMIP and FMIP-W runs, the fmip_timelimit and fmip_w_timelimit can be controled at the bottom of the code.
r'''

print("Solving StQP with PIP...")

folder_path = 'instances/stqp_instances/samples'

p_max = 0.8 # version of PIP (_)

max_repeat = 3 # max_repeat (r_max in the manuscript),
pip_subproblem_timelimit = 600 # time limit for each pip subproblem solving
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
fmip_timelimit = 60,
fmip_w_run = 1,
fmip_w_timelimit = 60)
