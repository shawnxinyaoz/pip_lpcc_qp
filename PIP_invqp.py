import numpy as np
from pyomo.environ import *
from datetime import datetime
import pandas as pd
import os
from scipy.sparse import random as sparse_random
from scipy.stats import norm
from sklearn.datasets import make_sparse_spd_matrix
import time


def pip_solve_for_inv_cqp(Q, A, xbar, bbar, cbar, ux, ub, uc, ul, us, start_fixed_proportion, final_fixed_proportion, max_repeat_times, max_iters, timelimit, initial_solver = 'ipopt', pip_step_size = 0.1):
    r'''
    This is the PIP main function solving the LPCC of inverse qp. Please check the manuscript for detailed formulation.
    Note that there is slight different in notation in the paper: A in the code is D in the paper, and b in the code is d in the paper. And l stands for lambda.
    r'''
    m = A.shape[0]
    n_ineq_constraints = m
    n = Q.shape[0]
    ########################
    # initializing PIP
    print("Data generated. PIP initializing....")
    start_time = time.time()

    x_, b_, c_, lambda_, slack_, obj, comp_time = initial_solve(Q, A, xbar, bbar, cbar,
                                                                 ux, ub, uc, ul, us, initial_solver, timelimit)



    print("PIP initialization done.")

    iterations = [0]

    number_of_integer_values_list = ["NA"]

    objective_value_list = [obj]

    improvement_list = ['NA']

    computation_times = [comp_time]

    number_of_lambda_fixed_zero_list = ['NA']
    number_of_slack_fixed_zero_list = ['NA']
    lambda_fixed_zero_indices_list = ["NA"]
    slack_fixed_zero_indices_list = ["NA"]

    number_of_degenerate_constraints_list = ["NA"]


    i = 0
    fixed_proportion = start_fixed_proportion
    repeat = 1

    epsilon_l = 1e-4
    epsilon_s = 1e-4
    epsilon = [epsilon_l, epsilon_s]

    fixed_proportion_list = ['NA']

    lambda_range_list = [(min(lambda_),max(lambda_))]
    slack_range_list = [(min(slack_), max(slack_))]

    running_total = comp_time
    cumulative_computation_times = [comp_time]

    total_time = [time.time()-start_time]

    ########################
    print("Building base model of PIP...")

    # New model
    model = ConcreteModel()

    # Decision variables
    model.x = Var(range(n), within=Reals)
    model.b = Var(range(m), within=Reals)
    model.c = Var(range(n), within=Reals)

    model.l = Var(range(m), within=NonNegativeReals)
    model.z = Var(range(m), within=Binary)

    model.sx = Var(range(n), within=Reals)
    model.sb = Var(range(m), within=Reals)
    model.sc = Var(range(n), within=Reals)

    # Objective
    model.obj = Objective(expr=sum(model.sx[i] for i in range(n))+
                               sum(model.sb[k] for k in range(m))+
                               sum(model.sc[k] for k in range(n)),
                          sense=minimize)

    # Constraints
    def eq_constraint_rule(model, i):
        return model.c[i] + sum(Q[i, j] * model.x[j] for j in range(n)) - sum(
            A[k, i] * model.l[k] for k in range(m)) == 0

    model.eq_constraints = Constraint(range(n), rule=eq_constraint_rule)

    def l_bound_rule(model, k):
        return model.l[k] <= ul * model.z[k]

    model.l_bound_constraints = Constraint(range(m), rule=l_bound_rule)

    def ax_bound_rule_lower(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k] >= 0

    def ax_bound_rule_upper(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k] <= us * (1 - model.z[k])

    model.ax_bound_constraints_lower = Constraint(range(m), rule=ax_bound_rule_lower)
    model.ax_bound_constraints_upper = Constraint(range(m), rule=ax_bound_rule_upper)

    # Deviation constraints for x, b, c
    def x_deviation_constraints_lower(model, i):
        return -model.sx[i] <= model.x[i] - xbar[i]

    def x_deviation_constraints_upper(model, i):
        return model.x[i] - xbar[i] <= model.sx[i]

    def b_deviation_constraints_lower(model, j):
        return -model.sb[j] <= model.b[j] - bbar[j]

    def b_deviation_constraints_upper(model, j):
        return model.b[j] - bbar[j] <= model.sb[j]

    def c_deviation_constraints_lower(model, i):
        return -model.sc[i] <= model.c[i] - cbar[i]

    def c_deviation_constraints_upper(model, i):
        return model.c[i] - cbar[i] <= model.sc[i]

    # Bound constraints for x, b, c
    def x_bound_constraints_lower(model, i):
        return -ux <= model.x[i]

    def x_bound_constraints_upper(model, i):
        return model.x[i] <= ux

    def b_bound_constraints_lower(model, j):
        return -ub <= model.b[j]

    def b_bound_constraints_upper(model, j):
        return model.b[j] <= ub

    def c_bound_constraints_lower(model, i):
        return -uc <= model.c[i]

    def c_bound_constraints_upper(model, i):
        return model.c[i] <= uc

    model.x_deviation_constraints_lower = Constraint(range(n), rule=x_deviation_constraints_lower)
    model.x_deviation_constraints_upper = Constraint(range(n), rule=x_deviation_constraints_upper)
    model.b_deviation_constraints_lower = Constraint(range(m), rule=b_deviation_constraints_lower)
    model.b_deviation_constraints_upper = Constraint(range(m), rule=b_deviation_constraints_upper)
    model.c_deviation_constraints_lower = Constraint(range(n), rule=c_deviation_constraints_lower)
    model.c_deviation_constraints_upper = Constraint(range(n), rule=c_deviation_constraints_upper)

    model.x_bound_constraints_lower = Constraint(range(n), rule=x_bound_constraints_lower)
    model.x_bound_constraints_upper = Constraint(range(n), rule=x_bound_constraints_upper)
    model.b_bound_constraints_lower = Constraint(range(m), rule=b_bound_constraints_lower)
    model.b_bound_constraints_upper = Constraint(range(m), rule=b_bound_constraints_upper)
    model.c_bound_constraints_lower = Constraint(range(n), rule=c_bound_constraints_lower)
    model.c_bound_constraints_upper = Constraint(range(n), rule=c_bound_constraints_upper)


    while i < max_iters:

        i+=1
        print("\n Iteration ", i, " out of ", max_iters, "\n")

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

        # Save the new optimal values of x and the corresponding dual variables
        x_ = [model.x[i].value for i in range(n)]
        b_ = [model.b[k].value for k in range(m)]
        c_ = [model.c[i].value for i in range(n)]
        lambda_ = [model.l[k].value for k in range(m)]
        z_ = [model.z[k].value for k in range(m)]
        slack_ = [sum(A[k, i] * x_[i] for i in range(n)) - b_[k] for k in range(m)]

        # UNFIXING
        for k in slack_fixed_zero_indices:
            model.z[k].unfix()
        for k in lambda_fixed_zero_indices:
            model.z[k].unfix()


        number_of_degenerate_constraints = sum(1 for k in range(m) if lambda_[k] == 0  and slack_[k] == 0)
        number_of_slack_fixed_zero =  len(slack_fixed_zero_indices)
        number_of_lambda_fixed_zero = len(lambda_fixed_zero_indices)

        current_obj = value(model.obj)
        comp_time = results.solver.time  # This gets the solver time

        ########################

        improvement = obj - current_obj
        print("At iteration", i, "Objective value decreased by", improvement)

        obj = current_obj

        iterations.append(i)
        number_of_integer_values_list.append(m - number_of_slack_fixed_zero - number_of_lambda_fixed_zero)
        objective_value_list.append(current_obj)

        number_of_degenerate_constraints_list.append(number_of_degenerate_constraints)
        # Number of fixing variables
        number_of_lambda_fixed_zero_list.append(number_of_lambda_fixed_zero)
        number_of_slack_fixed_zero_list.append(number_of_slack_fixed_zero)
        lambda_fixed_zero_indices_list.append(lambda_fixed_zero_indices)
        slack_fixed_zero_indices_list.append(slack_fixed_zero_indices)

        fixed_proportion_list.append(fixed_proportion)
        computation_times.append(comp_time)

        improvement_list.append(improvement)

        running_total += comp_time
        cumulative_computation_times.append(running_total)

        # Other data
        lambda_range_list.append((min(lambda_),max(lambda_)))
        slack_range_list.append((min(slack_), max(slack_)))

        total_time.append(time.time()-start_time)

        ### LOOP LOGIC
        repeat += 1

        if repeat > max_repeat_times or abs(improvement) < 1e-4:
            fixed_proportion -= pip_step_size
            repeat = 1

        if fixed_proportion < final_fixed_proportion:
            break

    return {
        'Iteration': iterations,
        'fixed_proportion': fixed_proportion_list,
        'Objective Value': objective_value_list,
        'total time': total_time
        # ,'Number of unfixed integer variables': number_of_integer_values_list,
        # 'Objective improvement (obj - current obj)': improvement_list,
        # 'Number of lambda fixed zero (z_j = 0)': number_of_lambda_fixed_zero_list,
        # 'Number of slack fixed zero (z_j = 1)': number_of_slack_fixed_zero_list,
        # 'lambda_fixed_zero_indices': lambda_fixed_zero_indices_list,
        # 'slack_fixed_zero_indices': slack_fixed_zero_indices_list,
        # 'number_of_degenerate_constraints': number_of_degenerate_constraints_list,
        # 'Computation Time': computation_times,
        # 'Cumulative Time': cumulative_computation_times,
        # 'lambda range': lambda_range_list,
        # 'slack range': slack_range_list
    }

def save_df(df, m, n, seed, notes):
    r'''
    This function is created to record the results of inverse qp runs
    r'''
    # Check if "result" directory exists, if not, create it
    if not os.path.exists('invqp_results'):
        os.makedirs('invqp_results')

    # Get the current date and time
    now = datetime.now()

    # Construct filename with format YYYYMMDD_HHMMSS_seed_{seed}.csv
    filename = now.strftime(f'PIP_{notes}_InvQP_m_{m}_seed_{seed}.csv')

    # Construct the full path to save the file in the "result" subfolder
    full_path = os.path.join('invqp_results', filename)

    # Save the DataFrame to the constructed path
    df.to_csv(full_path, index=False)
    print(f"Saved results to {full_path}")

def data_generation(m, n, seed, sparsity = 0.5):
    """
    Generates data as per the specified steps in the manuscript.
    The scheme is proposed in Section 6.3 in F. Jara-Moroni, J.S. Pang, and A. W¨achter. A study of the difference-of-convex approach for solving linear programs with complementarity constraints. Mathematical Programming 169: 221–254 (2018).

    Args:
    m (int): The size of vectors lambda and w_tilde and one dimension of matrix A.
    n (int): The size of vector x and one dimension of matrices Q and A.
    sparsity controls the sparsity of both Q and A matrices. (A in the code is D in the paper, and b in the code is d in the paper)

    Returns:
    Q, A, x_bar, b_bar, c_bar, u_x, u_b, u_c, u_lambda, u_slack to define the corresponding LPCC.
    """

    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate a sparse random symmetric positive definite matrix Q and a sparse matrix A
    m = int(m)
    n = int(n)
    print("m:", m,"n", n)


    Q = make_sparse_spd_matrix(dim=n, alpha=sparsity, random_state=seed)

    A = sparse_random(m, n, density=sparsity, format='csr')

    # Step 2: Generate a random vector x following a normal distribution
    x = norm.rvs(size=n)

    # Step 3: Generate vectors lambda and w_tilde uniformly at random between 0 and 10
    lambda_vec = np.random.uniform(0, 10, size=m)
    w_tilde = np.random.uniform(0, 10, size=m)

    # # Step 4: Generate a random binary vector v
    # v = np.random.choice([0, 1], size=m)

    # Step 5: Define b and c
    b = A.dot(x) - w_tilde
    c = A.T.dot(lambda_vec) - Q.dot(x)

    # Step 6: Perturb b, c and with normally distributed noise
    x_bar = x + norm.rvs(size=n)
    b_bar = b + norm.rvs(size=m)
    c_bar = c + norm.rvs(size=n)

    # Step 7: Set upper bounds
    u_x = 10 * np.max(np.abs(x_bar))
    u_b = 10 * np.max(np.abs(b_bar))
    u_c = 10 * np.max(np.abs(c_bar))
    u_lambda = 10 * np.max(np.abs(lambda_vec))
    u_slack =  10 * np.max(np.abs(w_tilde))

    return Q, A, x_bar, b_bar, c_bar, u_x, u_b, u_c, u_lambda, u_slack

def full_model(Q, A, xbar, bbar, cbar, ux, ub, uc, ul, us, solver_name, timelimit):
    r'''
    This is the function for FMIP and FMIP-W (warm started) calculation.
    timelimit is the time limit provided to the solver for early termination.
    solver_name is always gurobi in our experiments for FMIP
    r'''
    m = A.shape[0]
    n = A.shape[1]
    print("m, n", m, n)
    print("len xbar bbar cbar", len(xbar), len(bbar), len(cbar))
    M = ul
    uxv = np.full(A.shape[1], ux)  # Vector of length equal to the number of columns in A
    ubv = np.full(A.shape[0], ub)  # Vector of length equal to the number of rows in A
    ucv = np.full(A.shape[1], uc)
    N = us

    # New model
    model = ConcreteModel()

    # Decision variables
    model.x = Var(range(n), within=Reals)
    model.b = Var(range(m), within=Reals)
    model.c = Var(range(n), within=Reals)

    model.sx = Var(range(n), within=Reals)
    model.sb = Var(range(m), within=Reals)
    model.sc = Var(range(n), within=Reals)

    model.l = Var(range(m), within=NonNegativeReals)
    model.z = Var(range(m), within=Binary)

    # Objective
    model.obj = Objective(expr=sum(model.sx[i] for i in range(n)) +
                               sum(model.sb[k] for k in range(m)) +
                               sum(model.sc[k] for k in range(n)),
                          sense=minimize)

    # Constraints
    def eq_constraint_rule(model, i):
        return model.c[i] + sum(Q[i, j] * model.x[j] for j in range(n)) - sum(
            A[k, i] * model.l[k] for k in range(m)) == 0

    model.eq_constraints = Constraint(range(n), rule=eq_constraint_rule)

    def l_bound_rule(model, k):
        return model.l[k] <= M * model.z[k]

    model.l_bound_constraints = Constraint(range(m), rule=l_bound_rule)

    def ax_bound_rule_lower(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k] >= 0

    def ax_bound_rule_upper(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k] <= N * (1 - model.z[k])

    model.ax_bound_constraints_lower = Constraint(range(m), rule=ax_bound_rule_lower)
    model.ax_bound_constraints_upper = Constraint(range(m), rule=ax_bound_rule_upper)

    ########
    # Deviation constraints for x, b, c
    def x_deviation_constraints_lower(model, i):
        return -model.sx[i] <= model.x[i] - xbar[i]

    def x_deviation_constraints_upper(model, i):
        return model.x[i] - xbar[i] <= model.sx[i]

    def b_deviation_constraints_lower(model, j):
        return -model.sb[j] <= model.b[j] - bbar[j]

    def b_deviation_constraints_upper(model, j):
        return model.b[j] - bbar[j] <= model.sb[j]

    def c_deviation_constraints_lower(model, i):
        return -model.sc[i] <= model.c[i] - cbar[i]

    def c_deviation_constraints_upper(model, i):
        return model.c[i] - cbar[i] <= model.sc[i]

    # Bound constraints for x, b, c
    def x_bound_constraints_lower(model, i):
        return -uxv[i] <= model.x[i]

    def x_bound_constraints_upper(model, i):
        return model.x[i] <= uxv[i]

    def b_bound_constraints_lower(model, j):
        return -ubv[j] <= model.b[j]

    def b_bound_constraints_upper(model, j):
        return model.b[j] <= ubv[j]

    def c_bound_constraints_lower(model, i):
        return -ucv[i] <= model.c[i]

    def c_bound_constraints_upper(model, i):
        return model.c[i] <= ucv[i]

    model.x_deviation_constraints_lower = Constraint(range(n), rule=x_deviation_constraints_lower)
    model.x_deviation_constraints_upper = Constraint(range(n), rule=x_deviation_constraints_upper)
    model.b_deviation_constraints_lower = Constraint(range(m), rule=b_deviation_constraints_lower)
    model.b_deviation_constraints_upper = Constraint(range(m), rule=b_deviation_constraints_upper)
    model.c_deviation_constraints_lower = Constraint(range(n), rule=c_deviation_constraints_lower)
    model.c_deviation_constraints_upper = Constraint(range(n), rule=c_deviation_constraints_upper)

    model.x_bound_constraints_lower = Constraint(range(n), rule=x_bound_constraints_lower)
    model.x_bound_constraints_upper = Constraint(range(n), rule=x_bound_constraints_upper)
    model.b_bound_constraints_lower = Constraint(range(m), rule=b_bound_constraints_lower)
    model.b_bound_constraints_upper = Constraint(range(m), rule=b_bound_constraints_upper)
    model.c_bound_constraints_lower = Constraint(range(n), rule=c_bound_constraints_lower)
    model.c_bound_constraints_upper = Constraint(range(n), rule=c_bound_constraints_upper)

    # Solver
    solver = SolverFactory(solver_name)
    if solver_name == 'gurobi':
        solver.options['TimeLimit'] = timelimit
        # solver.options['MIPfocus'] = 1

    results = solver.solve(model, tee=True)
    solve_time = results.solver.time  # This gets the solver time

    # Save the new optimal values of x and the corresponding dual variables
    x_new = [model.x[i].value for i in range(n)]
    b_new = [model.b[k].value for k in range(m)]
    c_new = [model.c[i].value for i in range(n)]
    lambda_new = [model.l[k].value for k in range(m)]
    z_new = [model.z[k].value for k in range(m)]

    objective_value = value(model.obj)

    # Output
    print("\nStatus:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)
    print("\n Objective Value:", objective_value)  # Displaying the objective value

    return x_new, b_new, c_new, lambda_new, z_new, objective_value, solve_time

def initial_solve(Q, A, xbar, bbar, cbar, ux, ub, uc, ul, us, solver_name, timelimit):
    r'''
    This function solves the inverse qp to an initial solution for PIP and FMIP-W.
    solver_name controls the approach to obtain initial solutions. We use solver_name = 'ipopt' for m = 200 and 'gurobi' for m = 1000 in our experiments.
    timelimit is used to when solver_name is 'gurobi', since then the initialization process is a time limited FMIP run, which is used in our experiments for m = 1000. Please check manuscript for details.
    r'''
    print("\n Starting to solve initial using", solver_name, '\n')

    m = A.shape[0]
    n = Q.shape[0]
    if solver_name == 'ipopt':
        # New model
        model = ConcreteModel()

        # Decision variables
        model.x = Var(range(n), within=Reals)
        model.b = Var(range(m), within=Reals)
        model.c = Var(range(n), within=Reals)

        model.l = Var(range(m), within=NonNegativeReals)

        model.sx = Var(range(n), within=Reals)
        model.sb = Var(range(m), within=Reals)
        model.sc = Var(range(n), within=Reals)

        # Objective
        model.obj = Objective(expr=sum(model.sx[i] for i in range(n))+
                                   sum(model.sb[k] for k in range(m))+
                                   sum(model.sc[k] for k in range(n)),
                              sense=minimize)

        # Constraints
        def eq_constraint_rule(model, i):
            return model.c[i] + sum(Q[i, j] * model.x[j] for j in range(n)) - sum(
                A[k, i] * model.l[k] for k in range(m)) == 0

        model.eq_constraints = Constraint(range(n), rule=eq_constraint_rule)

        def complementarity_rule(model, k):
            return (sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k]) * model.l[k] <= 0 # change to 0 or from 0 for tolerance

        model.complementarity_constraints = Constraint(range(m), rule=complementarity_rule)

        def ax_bound_rule_lower(model, k):
            return sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k] >= 0

        model.ax_bound_constraints_lower = Constraint(range(m), rule=ax_bound_rule_lower)

        # Deviation constraints for x, b, c
        def x_deviation_constraints_lower(model, i):
            return -model.sx[i] <= model.x[i] - xbar[i]

        def x_deviation_constraints_upper(model, i):
            return model.x[i] - xbar[i] <= model.sx[i]

        def b_deviation_constraints_lower(model, j):
            return -model.sb[j] <= model.b[j] - bbar[j]

        def b_deviation_constraints_upper(model, j):
            return model.b[j] - bbar[j] <= model.sb[j]

        def c_deviation_constraints_lower(model, i):
            return -model.sc[i] <= model.c[i] - cbar[i]

        def c_deviation_constraints_upper(model, i):
            return model.c[i] - cbar[i] <= model.sc[i]

        # Bound constraints for x, b, c
        def x_bound_constraints_lower(model, i):
            return -ux <= model.x[i]

        def x_bound_constraints_upper(model, i):
            return model.x[i] <= ux

        def b_bound_constraints_lower(model, j):
            return -ub <= model.b[j]

        def b_bound_constraints_upper(model, j):
            return model.b[j] <= ub

        def c_bound_constraints_lower(model, i):
            return -uc <= model.c[i]

        def c_bound_constraints_upper(model, i):
            return model.c[i] <= uc

        model.x_deviation_constraints_lower = Constraint(range(n), rule=x_deviation_constraints_lower)
        model.x_deviation_constraints_upper = Constraint(range(n), rule=x_deviation_constraints_upper)
        model.b_deviation_constraints_lower = Constraint(range(m), rule=b_deviation_constraints_lower)
        model.b_deviation_constraints_upper = Constraint(range(m), rule=b_deviation_constraints_upper)
        model.c_deviation_constraints_lower = Constraint(range(n), rule=c_deviation_constraints_lower)
        model.c_deviation_constraints_upper = Constraint(range(n), rule=c_deviation_constraints_upper)

        model.x_bound_constraints_lower = Constraint(range(n), rule=x_bound_constraints_lower)
        model.x_bound_constraints_upper = Constraint(range(n), rule=x_bound_constraints_upper)
        model.b_bound_constraints_lower = Constraint(range(m), rule=b_bound_constraints_lower)
        model.b_bound_constraints_upper = Constraint(range(m), rule=b_bound_constraints_upper)
        model.c_bound_constraints_lower = Constraint(range(n), rule=c_bound_constraints_lower)
        model.c_bound_constraints_upper = Constraint(range(n), rule=c_bound_constraints_upper)

        # Solve
        solver = SolverFactory(solver_name)

        if solver_name == 'gurobi':
            solver.options['timelimit'] = timelimit
            results = solver.solve(model, tee=True)
        elif solver_name == 'ipopt':
            results = solver.solve(model, tee=True)

        # Save the new optimal values of x and the corresponding dual variables
        x_ = [model.x[i].value for i in range(n)]
        b_ = [model.b[k].value for k in range(m)]
        c_ = [model.c[i].value for i in range(n)]
        lambda_ = [model.l[k].value for k in range(m)]
        slack_ = [sum(A[k, i] * x_[i] for i in range(n)) - b_[k] for k in range(m)]
        obj = value(model.obj)
        comp_time = results.solver.time

    elif solver_name == 'gurobi':
        x_, b_, c_, lambda_, z_, obj, comp_time = full_model(Q, A, xbar, bbar, cbar, ux, ub, uc, ul, us, "gurobi", timelimit)
        slack_ = [sum(A[k, i] * x_[i] for i in range(n)) - b_[k] for k in range(m)]

    print("Initial solving done")

    return x_, b_, c_, lambda_, slack_, obj, comp_time

def full_model_w(Q, A, xbar, bbar, cbar, ux, ub, uc, ul, us, solver_name, timelimit, initial_solver=None, warmstart='yes'):
    r'''
    This function controls the FMIP-W runs, which will call the initial_solve function to warm start.
    Essencially this is basically a modified function from the full_model function. We leave it here to additional modification if needed.
    r'''
    m = A.shape[0]
    n = A.shape[1]
    print("m, n", m, n)
    print("len xbar bbar cbar", len(xbar), len(bbar), len(cbar))
    M = ul
    uxv = np.full(A.shape[1], ux)  # Vector of length equal to the number of columns in A
    ubv = np.full(A.shape[0], ub)  # Vector of length equal to the number of rows in A
    ucv = np.full(A.shape[1], uc)
    N = us

    # New model
    model = ConcreteModel()

    # Decision variables
    model.x = Var(range(n), within=Reals)
    model.b = Var(range(m), within=Reals)
    model.c = Var(range(n), within=Reals)

    model.sx = Var(range(n), within=Reals)
    model.sb = Var(range(m), within=Reals)
    model.sc = Var(range(n), within=Reals)

    model.l = Var(range(m), within=NonNegativeReals)
    model.z = Var(range(m), within=Binary)

    # Objective
    model.obj = Objective(expr=sum(model.sx[i] for i in range(n)) +
                               sum(model.sb[k] for k in range(m)) +
                               sum(model.sc[k] for k in range(n)),
                          sense=minimize)

    # Constraints
    def eq_constraint_rule(model, i):
        return model.c[i] + sum(Q[i, j] * model.x[j] for j in range(n)) - sum(
            A[k, i] * model.l[k] for k in range(m)) == 0

    model.eq_constraints = Constraint(range(n), rule=eq_constraint_rule)

    def l_bound_rule(model, k):
        return model.l[k] <= M * model.z[k]

    model.l_bound_constraints = Constraint(range(m), rule=l_bound_rule)

    def ax_bound_rule_lower(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k] >= 0

    def ax_bound_rule_upper(model, k):
        return sum(A[k, j] * model.x[j] for j in range(n)) - model.b[k] <= N * (1 - model.z[k])

    model.ax_bound_constraints_lower = Constraint(range(m), rule=ax_bound_rule_lower)
    model.ax_bound_constraints_upper = Constraint(range(m), rule=ax_bound_rule_upper)

    ########
    # Deviation constraints for x, b, c
    def x_deviation_constraints_lower(model, i):
        return -model.sx[i] <= model.x[i] - xbar[i]

    def x_deviation_constraints_upper(model, i):
        return model.x[i] - xbar[i] <= model.sx[i]

    def b_deviation_constraints_lower(model, j):
        return -model.sb[j] <= model.b[j] - bbar[j]

    def b_deviation_constraints_upper(model, j):
        return model.b[j] - bbar[j] <= model.sb[j]

    def c_deviation_constraints_lower(model, i):
        return -model.sc[i] <= model.c[i] - cbar[i]

    def c_deviation_constraints_upper(model, i):
        return model.c[i] - cbar[i] <= model.sc[i]

    # Bound constraints for x, b, c
    def x_bound_constraints_lower(model, i):
        return -uxv[i] <= model.x[i]

    def x_bound_constraints_upper(model, i):
        return model.x[i] <= uxv[i]

    def b_bound_constraints_lower(model, j):
        return -ubv[j] <= model.b[j]

    def b_bound_constraints_upper(model, j):
        return model.b[j] <= ubv[j]

    def c_bound_constraints_lower(model, i):
        return -ucv[i] <= model.c[i]

    def c_bound_constraints_upper(model, i):
        return model.c[i] <= ucv[i]

    model.x_deviation_constraints_lower = Constraint(range(n), rule=x_deviation_constraints_lower)
    model.x_deviation_constraints_upper = Constraint(range(n), rule=x_deviation_constraints_upper)
    model.b_deviation_constraints_lower = Constraint(range(m), rule=b_deviation_constraints_lower)
    model.b_deviation_constraints_upper = Constraint(range(m), rule=b_deviation_constraints_upper)
    model.c_deviation_constraints_lower = Constraint(range(n), rule=c_deviation_constraints_lower)
    model.c_deviation_constraints_upper = Constraint(range(n), rule=c_deviation_constraints_upper)

    model.x_bound_constraints_lower = Constraint(range(n), rule=x_bound_constraints_lower)
    model.x_bound_constraints_upper = Constraint(range(n), rule=x_bound_constraints_upper)
    model.b_bound_constraints_lower = Constraint(range(m), rule=b_bound_constraints_lower)
    model.b_bound_constraints_upper = Constraint(range(m), rule=b_bound_constraints_upper)
    model.c_bound_constraints_lower = Constraint(range(n), rule=c_bound_constraints_lower)
    model.c_bound_constraints_upper = Constraint(range(n), rule=c_bound_constraints_upper)

    # Solver
    solver = SolverFactory(solver_name)
    if solver_name == 'gurobi':
        solver.options['TimeLimit'] = timelimit

    if warmstart is not None:
        print('Initializing for Warmstarting FULL MILP....')
        x_, b_, c_, lambda_, slack_, obj, comp_time = initial_solve(Q, A, xbar, bbar, cbar,
                                                                    ux, ub, uc, ul, us, initial_solver, timelimit)
        print("Warmstarting using initial solution obtained in:", comp_time, "seconds")
        for k in range(len(x_)):
            model.x[k] = x_[k]
        for j in range(len(lambda_)):
            model.l[j] = lambda_[j]

        results = solver.solve(model, tee=True, warmstart=True)
    else:
        results = solver.solve(model, tee=True)

    solve_time = results.solver.time  # This gets the solver time

    # Save the new optimal values of x and the corresponding dual variables
    x_new = [model.x[i].value for i in range(n)]
    b_new = [model.b[k].value for k in range(m)]
    c_new = [model.c[i].value for i in range(n)]
    lambda_new = [model.l[k].value for k in range(m)]
    z_new = [model.z[k].value for k in range(m)]

    objective_value = value(model.obj)

    # Output
    print("\nStatus:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)
    print("\n Objective Value:", objective_value)  # Displaying the objective value

    return x_new, b_new, c_new, lambda_new, z_new, objective_value, solve_time

def algorithm(sizes, seeds, max_iters, max_repeat_times, start_fixed_proportion, final_fixed_proportion, pip_subproblem_timelimit, pip_step_size = 0.1, initial_solver="ipopt", pip_run = 1, fmip_run = 0, fmip_timelimit = 600, fmip_w_run = 0, fmip_w_timelimit = 600):
    r'''
    This function controls the main execution of experiments.
    sizes is the list of m to be used in the problems. seeds controls the random generation process.
    max_repeat_times, start_fixed_proportion, final_fixed_proportion, pip_subproblem_timelimit, pip_step_size are the parameter of the PIP function.
    initial_solver="ipopt" controls the initialization process. Use 'ipopt' for m = 200 and 'gurobi' for m = 1000.
    pip_run, fmip_run, fmip_w_run = 1 or 0 controls if each method is used in the executed run.
    fmip_timelimit and fmip_w_timelimit provides time limit to the solver in fmip and fmip-w runs.
    r'''
    notes = f"initialized_by_{initial_solver}"
    for size in sizes:
        for seed in seeds:
            Q, A, xbar, bbar, cbar, ux, ub, uc, ul, us = data_generation(size, int(np.rint(
                0.75 * size)), seed)
            fmip_results = []
            fmip_w_results = []
            if fmip_run == 1:
                print(f"RUNNING GUROBI FULL MILP \n Size: {size} \n Seed: {seed}")

                start_time = time.time()
                print("Data generated. Now running gurobi full milp...")
                x_, b_, c_, lambda_, z_, obj, comp_time = full_model(Q, A, xbar, bbar, cbar,
                                                                     ux, ub, uc, ul, us, "gurobi",
                                                                     fmip_timelimit)
                # Append the results as a dictionary to the results list
                fmip_results.append({
                    'size': size,
                    'seed': seed,
                    # 'x_': x_,
                    # 'b_': b_,
                    # 'c_': c_,
                    # 'lambda_': lambda_,
                    # 'z_': z_,
                    'obj': obj,
                    # 'comp_time': comp_time,
                    'total_time': time.time() - start_time
                })

                df_results = pd.DataFrame(fmip_results)

                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if not os.path.exists('invqp_results'):
                    os.makedirs('invqp_results')
                csv_file_path = f"FMIP_InvQP_m_{size}_seed_{seed}.csv"
                full_path = os.path.join('invqp_results', csv_file_path)

                df_results.to_csv(full_path, index=False)
                print(f"Saved results to {full_path}")
            if fmip_w_run == 1:
                print(f"RUNNING GUROBI FULL MILP with warm start \n Size: {size} \n Seed: {seed}")

                start_time = time.time()
                print("Data generated. Now running gurobi full milp...")
                x_, b_, c_, lambda_, z_, obj, comp_time = full_model_w(Q, A, xbar, bbar, cbar,
                                                                       ux, ub, uc, ul, us, "gurobi",
                                                                       fmip_w_timelimit,
                                                                       initial_solver=initial_solver,
                                                                       warmstart='yes')
                # Append the results as a dictionary to the results list
                fmip_w_results.append({
                    'size': size,
                    'seed': seed,
                    # 'x_': x_,
                    # 'b_': b_,
                    # 'c_': c_,
                    # 'lambda_': lambda_,
                    # 'z_': z_,
                    'obj': obj,
                    # 'comp_time': comp_time,
                    'total_time': time.time() - start_time
                })

                df_results = pd.DataFrame(fmip_w_results)

                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if not os.path.exists('invqp_results'):
                    os.makedirs('invqp_results')
                csv_file_path = f"FMIP_W_initialized_by_{initial_solver}_InvQP_m_{size}_seed_{seed}.csv"
                full_path = os.path.join('invqp_results', csv_file_path)

                df_results.to_csv(full_path, index=False)
                print(f"Saved results to {full_path}")
            if pip_run == 1:
                results_dict = pip_solve_for_inv_cqp(Q, A, xbar, bbar, cbar, ux, ub, uc, ul, us, start_fixed_proportion,
                                                     final_fixed_proportion, max_repeat_times, max_iters, pip_subproblem_timelimit,
                                                     initial_solver=initial_solver, pip_step_size = pip_step_size)
                results_df = pd.DataFrame(results_dict)
                m = size
                n = int(np.rint(0.75 * size))
                save_df(results_df, m, n, seed, notes)


r'''
The section below executes the experiments for the instances generated in the targeting sizes and seeds.
The user may control pip and fmip(-w) parameters here.
In our experiments:
The sizes are [200, 1000]. The seeds are [1,2,3,4,5].
For m = 200, fmip_timelimit = 6000 and fmip_w_timelimit = 2400, i.e. 100 min and 40 min.
For m = 1000, fmip_timelimit = 9000, i.e. 2.5 hours, and fmip_w is not included due to numerical issues (Please see manuscript for details).
r'''

print("Solving InvQP with PIP...")

sizes = [1000]
seeds = [1 ,2]

p_max = 0.4 # version of PIP (_)

max_repeat_times = 3 # max_repeat (r_max in the manuscript),
pip_subproblem_timelimit = 600 # time limit for each pip subproblem solving
pip_step_size = 0.1

start_fixed_proportion = 0.8
final_fixed_proportion = 1 - p_max
max_iters = np.rint((start_fixed_proportion - final_fixed_proportion + pip_step_size)/pip_step_size * max_repeat_times)

algorithm(sizes,
seeds,
max_iters,
max_repeat_times,
start_fixed_proportion,
final_fixed_proportion,
pip_subproblem_timelimit,
pip_step_size = pip_step_size,
initial_solver = 'gurobi', # we use 'ipopt' when m = 200 and 'gurobi' when m = 1000
pip_run = 1,
fmip_run = 1,
fmip_timelimit = 9000,
fmip_w_run = 0,
fmip_w_timelimit = 2400)
