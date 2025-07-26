import numpy as np
from pyomo.environ import *
from datetime import datetime
import pandas as pd
import os
import time

# os.environ['NEOS_EMAIL'] = 'USER_NEOS_EMAL@ABC.COM'


def initial_solve(c, e, f, A, B, C, b, lx, ux, uy, uz, solver_name, timelimit, x_feasible = None,y_feasible= None, z_feasible = None):


    """
    Solves a linear program with complementarity constraints (LPCC).

    Args:
        c (array-like): Coefficients for the x variables in the objective function.
        e (array-like): Coefficients for the y variables in the objective function.
        f (array-like): Coefficients for the z variables in the objective function.
        A (2D array-like): Coefficients matrix for x variables in the equality constraints.
        B (2D array-like): Coefficients matrix for y variables in the equality constraints.
        C (2D array-like): Coefficients matrix for z variables in the equality constraints.
        b (array-like): Right-hand side vector for the equality constraints.
        solver_name (str): The solver to use for optimization (e.g., 'gurobi', 'ipopt').
        timelimit (float): The time limit for the solver in seconds.

    Returns:
        tuple: A tuple containing:
            - x_ (list): Optimal values of the x variables.
            - y_ (list): Optimal values of the y variables.
            - z_ (list): Optimal values of the z variables.
            - obj (float): Optimal value of the objective function.
            - comp_time (float): Computation time taken by the solver.
    """


    print("\nStarting to solve initial using", solver_name, '\n')
    n = len(c)
    m = len(e)
    print("m:", m, " n:", n)

    if x_feasible is not None:
        # Check feasibility
        x_within_bounds = np.all((lx <= x_feasible) & (x_feasible <= ux))
        y_within_bounds = np.all(y_feasible <= uy)
        z_within_bounds = np.all(z_feasible <= uz)

        print("x_feasible within bounds?", x_within_bounds)
        print("y_feasible within bounds?", y_within_bounds)
        print("z_feasible within bounds?", z_within_bounds)

        # Check feasibility of constraints
        lhs = A @ x_feasible + B @ y_feasible + C @ z_feasible
        constraints_feasible = np.allclose(lhs, b, atol=1e-6)

        print("Constraints feasible?", constraints_feasible)

        print("Complementarity feasible?", y_feasible @ z_feasible == 0)

    # New model
    model = ConcreteModel()

    # Decision variables
    model.x = Var(range(n), within=Reals, bounds=(lx, ux))
    model.y = Var(range(m), within=NonNegativeReals)
    model.z = Var(range(m), within=NonNegativeReals)

    # Objective
    model.obj = Objective(expr=sum(c[i] * model.x[i] for i in range(n)) +
                               sum(e[j] * model.y[j] for j in range(m)) +
                               sum(f[j] * model.z[j] for j in range(m)),
                          sense=minimize)

    # Constraints
    def eq_constraint_rule(model, i):
        return (sum(A[i, j] * model.x[j] for j in range(n))
                + sum(B[i, j] * model.y[j] for j in range(m))
                + sum(C[i, j] * model.z[j] for j in range(m)) == b[i])

    model.eq_constraints = Constraint(range(len(b)), rule=eq_constraint_rule)

    def complementarity_rule(model, j):
        return model.y[j] * model.z[j] <= 1e-7

    model.complementarity_constraints = Constraint(range(m), rule=complementarity_rule)

    # Solve
    solver = SolverFactory(solver_name)

    if solver_name == 'gurobi':
        solver.options['timelimit'] = timelimit
    elif solver_name == 'ipopt':
        if x_feasible is not None:
            solver.options['warm_start_init_point'] = 'yes'
            for i in range(n):
                model.x[i].value = x_feasible[i]
            for i in range(m):
                model.y[i].value = y_feasible[i]
                model.z[i].value = z_feasible[i]
        results = solver.solve(model, tee=True)
    elif solver_name == 'knitro' or solver_name == 'filter':
        solver = SolverManagerFactory('neos')
        results = solver.solve(model, opt=solver_name)

    # Save the new optimal values of x, y, and z
    x_ = [model.x[i].value for i in range(n)]
    y_ = [model.y[j].value for j in range(m)]
    z_ = [model.z[j].value for j in range(m)]
    obj = value(model.obj)
    comp_time = results.solver.time

    feasibility_check(A, B, C, b, lx, ux, uy, uz, x_, y_, z_)

    return x_, y_, z_, obj, comp_time

def pip_solve_for_lpcc(c, e, f, A, B, C, b, lx  , ux, uy, uz,
                       start_control_number, final_control_number,
                       max_repeat_times, max_iters, timelimit, initial_solver='ipopt', x_feasible = None,y_feasible= None, z_feasible = None):
    """
    Solves a Linear Program with Complementarity Constraints (LPCC) using a Progressive Iterative Process (PIP).

    Args:
        c (array-like): Coefficients for the x variables in the objective function.
        e (array-like): Coefficients for the y variables in the objective function.
        f (array-like): Coefficients for the z variables in the objective function.
        A (2D array-like): Coefficients matrix for x variables in the equality constraints.
        B (2D array-like): Coefficients matrix for y variables in the equality constraints.
        C (2D array-like): Coefficients matrix for z variables in the equality constraints.
        b (array-like): Right-hand side vector for the equality constraints.
        ux (float): Upper bound for x variables.
        uy (float): Upper bound for y variables.
        uz (float): Upper bound for z variables.
        start_control_number (float): Initial control number for fixing variables.
        final_control_number (float): Final control number for fixing variables.
        max_repeat_times (int): Maximum number of times to repeat fixing variables.
        max_iters (int): Maximum number of iterations.
        timelimit (float): Time limit for the solver in seconds.
        initial_solver (str): The solver to use for the initial solve (default: 'ipopt').

    Returns:
        dict: A dictionary containing the iteration details, objective values, and other relevant information.
    """
    m = len(e)
    n = len(c)

    # initializing PIP
    print("PIP initializing....")
    start_time = time.time()

    x_, y_, z_, obj, comp_time = initial_solve(c, e, f, A, B, C, b, lx, ux, uy, uz, initial_solver, timelimit,
                                                   x_feasible = x_feasible , y_feasible = y_feasible, z_feasible = z_feasible)

    print("PIP initialization done. Initial obj:", obj)

    # setting up lists for results saving...
    iterations = [0]
    number_of_integer_values_list = ["NA"]
    objective_value_list = [obj]
    improvement_list = ['NA']
    computation_times = [comp_time]

    number_of_lambda_fixed_zero_list = ['NA']
    number_of_slack_fixed_zero_list = ['NA']
    y_fixed_zero_indices_list = ["NA"]
    z_fixed_zero_indices_list = ["NA"]

    number_of_degenerate_constraints_list = ["NA"]

    # initial parameters
    i = 0
    control_number = start_control_number
    repeat = 1

    epsilon_l = 1e-4
    epsilon_s = 1e-4
    epsilon = [epsilon_l, epsilon_s]

    control_number_list = ['NA']
    lambda_range_list = [(min(y_), max(y_))]
    slack_range_list = [(min(z_), max(z_))]

    running_total = comp_time
    cumulative_computation_times = [comp_time]
    total_time = [time.time() - start_time]

    ########################
    print("Building base model of PIP...")

    # New model
    model = ConcreteModel()

    # Decision variables
    if ux is not None:
        model.x = Var(range(n), within=Reals, bounds=(lx, ux))
    else:
        model.x = Var(range(n), within=Reals)
    model.y = Var(range(m), within=Reals)
    model.z = Var(range(m), within=Reals)

    model.w = Var(range(m), within=Binary)

    # Objective
    model.obj = Objective(expr=sum(c[i] * model.x[i] for i in range(n)) +
                               sum(e[j] * model.y[j] for j in range(m)) +
                               sum(f[j] * model.z[j] for j in range(m)),
                          sense=minimize)

    # Constraints
    def eq_constraint_rule(model, i):
        return (sum(A[i, j] * model.x[j] for j in range(n)) +
                sum(B[i, j] * model.y[j] for j in range(m)) +
                sum(C[i, j] * model.z[j] for j in range(m)) <= b[i] + 1e-5)

    def eq_constraint_rule_2(model, i):
        return (sum(A[i, j] * model.x[j] for j in range(n)) +
                sum(B[i, j] * model.y[j] for j in range(m)) +
                sum(C[i, j] * model.z[j] for j in range(m)) >= b[i] - 1e-5)

    model.eq_constraints = Constraint(range(len(b)), rule=eq_constraint_rule)

    model.eq_constraints2 = Constraint(range(len(b)), rule=eq_constraint_rule_2)

    def y_bound_rule_upper(model, k):
        return model.y[k] <= uy * model.w[k]

    def y_bound_rule_lower(model, k):
        return model.y[k] >= 0

    model.y_bound_constraints_lower = Constraint(range(m), rule = y_bound_rule_lower)
    model.y_bound_constraints_upper = Constraint(range(m), rule = y_bound_rule_upper)

    def z_bound_rule_upper(model, k):
        return model.z[k] <= uz * (1 - model.w[k])

    def z_bound_rule_lower(model, k):
        return model.z[k] >= 0

    model.z_bound_constraints_upper = Constraint(range(m), rule = z_bound_rule_upper)
    model.z_bound_constraints_lower = Constraint(range(m), rule = z_bound_rule_lower)

    print("PIP Base model built... ")

    while i < max_iters:
        i += 1
        print("\n Iteration ", i, " out of ", max_iters, "\n")

        #######################
        # Fixing determination

        # Calculate number of positive y and z
        py = sum(l > 0 for l in y_)
        pz = sum(s > 0 for s in z_)

        # Determine the number of indices to select based on py and pz
        num_indices_to_select_y = int(control_number * py)
        num_indices_to_select_z = int(control_number * pz)

        # Get the indices for the largest y and z based on the new criteria
        largest_y_indices = sorted(range(len(y_)), key=lambda k: y_[k], reverse=True)[:num_indices_to_select_y]
        largest_z_indices = sorted(range(len(z_)), key=lambda k: z_[k], reverse=True)[:num_indices_to_select_z]

        y_fixed_zero_indices = []
        z_fixed_zero_indices = []

        # Modify the constraints based on the obtained indices
        for k in range(m):
            if k in largest_y_indices and y_[k] > epsilon[0]:
                z_fixed_zero_indices.append(k)
            elif k in largest_z_indices and z_[k] > epsilon[1]:
                y_fixed_zero_indices.append(k)

        #######################
        # FIXING
        for k in y_fixed_zero_indices:
            model.w[k].fix(0)
        for k in z_fixed_zero_indices:
            model.w[k].fix(1)

        ########################
        # Warm start
        for k in range(n):
            model.x[k] = x_[k]
        for j in range(m):
            model.y[j] = y_[j]
            model.z[j] = z_[j]

        ########################
        # Solve
        solver = SolverFactory('gurobi')

        solver.options['timelimit'] = timelimit
        results = solver.solve(model, tee=True, warmstart=True)

        # Save the new optimal values of x, y, and z
        x_ = [model.x[i].value for i in range(n)]
        y_ = [model.y[j].value for j in range(m)]
        z_ = [model.z[j].value for j in range(m)]

        w_ = [model.w[j].value for j in range(m)]

        # UNFIXING
        for k in y_fixed_zero_indices:
            model.w[k].unfix()
        for k in z_fixed_zero_indices:
            model.w[k].unfix()

        number_of_degenerate_constraints = sum(1 for k in range(m) if y_[k] == 0 and z_[k] == 0)
        number_of_y_fixed_zero = len(y_fixed_zero_indices)
        number_of_z_fixed_zero = len(z_fixed_zero_indices)

        current_obj = value(model.obj)
        comp_time = results.solver.time  # This gets the solver time

        ########################

        improvement = obj - current_obj
        print("At iteration", i, "Objective value decreased by", improvement)

        obj = current_obj

        iterations.append(i)
        number_of_integer_values_list.append(m - number_of_y_fixed_zero - number_of_z_fixed_zero)
        objective_value_list.append(current_obj)

        number_of_degenerate_constraints_list.append(number_of_degenerate_constraints)
        number_of_lambda_fixed_zero_list.append(number_of_y_fixed_zero)
        number_of_slack_fixed_zero_list.append(number_of_z_fixed_zero)
        y_fixed_zero_indices_list.append(y_fixed_zero_indices)
        z_fixed_zero_indices_list.append(z_fixed_zero_indices)

        control_number_list.append(control_number)
        computation_times.append(comp_time)
        improvement_list.append(improvement)

        running_total += comp_time
        cumulative_computation_times.append(running_total)

        lambda_range_list.append((min(y_), max(y_)))
        slack_range_list.append((min(z_), max(z_)))

        total_time.append(time.time() - start_time)

        # Loop logic
        repeat += 1

        feasibility_check(A, B, C, b, lx, ux, uy, uz, x_, y_, z_)

        if repeat > max_repeat_times or abs(improvement) < 1e-4:
            control_number -= 0.1
            repeat = 1

        if control_number < final_control_number:
            break


    return {
        'Iteration': iterations,
        'Control_number (max number of predetermined 0 or 1 int variables)': control_number_list,
        'Objective Value': objective_value_list,
        'total time': total_time,
        'Number of unfixed integer variables': number_of_integer_values_list,
        'Objective improvement (obj - current obj)': improvement_list,
        'Number of y fixed zero (y_j = 0)': number_of_lambda_fixed_zero_list,
        'Number of z fixed zero (z_j = 0)': number_of_slack_fixed_zero_list,
        'y_fixed_zero_indices': y_fixed_zero_indices_list,
        'z_fixed_zero_indices': z_fixed_zero_indices_list,
        'number_of_degenerate_constraints': number_of_degenerate_constraints_list,
        'Computation Time': computation_times,
        'Cumulative Time': cumulative_computation_times,
        'lambda range': lambda_range_list,
        'slack range': slack_range_list,
    }


def full_model(c, e, f, A, B, C, b, lx, ux, uy, uz, solver_name, timelimit,
               initial_solver=None, warmstart=None, heuristic_focus=0, x_feasible = None,y_feasible= None, z_feasible = None):
    """
    Solves a Linear Program with Complementarity Constraints (LPCC) using the full MILP model approach.

    Args:
        c (array-like): Coefficients for the x variables in the objective function.
        e (array-like): Coefficients for the y variables in the objective function.
        f (array-like): Coefficients for the z variables in the objective function.
        A (2D array-like): Coefficients matrix for x variables in the equality constraints.
        B (2D array-like): Coefficients matrix for y variables in the equality constraints.
        C (2D array-like): Coefficients matrix for z variables in the equality constraints.
        b (array-like): Right-hand side vector for the equality constraints.
        ux (float): Upper bound for x variables.
        uy (float): Upper bound for y variables.
        uz (float): Upper bound for z variables.
        solver_name (str): The solver to use for optimization (e.g., 'gurobi', 'ipopt').
        timelimit (float): The time limit for the solver in seconds.
        initial_solver (str, optional): The solver to use for the initial solve (default: None).
        warmstart (bool, optional): Whether to use a warm start (default: None).
        heuristic_focus (int, optional): Heuristic focus for the solver (default: 0).

    Returns:
        tuple: A tuple containing:
            - x_new (list): Optimal values of the x variables.
            - y_new (list): Optimal values of the y variables.
            - z_new (list): Optimal values of the z variables.
            - w_new (list): Values of the binary variables.
            - objective_value (float): Optimal value of the objective function.
            - solve_time (float): Computation time taken by the solver.
    """
    m = len(e)
    n = len(c)
    print("m:", m, "n:" , n)

    # New model
    model = ConcreteModel()

    # Decision variables
    if ux is not None:
        model.x = Var(range(n), within=Reals, bounds=(lx, ux))
    else:
        model.x = Var(range(n), within=Reals)
    model.y = Var(range(m), within=Reals)
    model.z = Var(range(m), within=Reals)
    model.w = Var(range(m), within=Binary)

    # Objective
    model.obj = Objective(expr=sum(c[i] * model.x[i] for i in range(n)) +
                               sum(e[j] * model.y[j] for j in range(m)) +
                               sum(f[j] * model.z[j] for j in range(m)),
                          sense=minimize)

    # Constraints
    def eq_constraint_rule(model, i):
        return (sum(A[i, j] * model.x[j] for j in range(n)) +
                sum(B[i, j] * model.y[j] for j in range(m)) +
                sum(C[i, j] * model.z[j] for j in range(m)) <= b[i] + 1e-5)

    def eq_constraint_rule_2(model, i):
        return (sum(A[i, j] * model.x[j] for j in range(n)) +
                sum(B[i, j] * model.y[j] for j in range(m)) +
                sum(C[i, j] * model.z[j] for j in range(m)) >= b[i] - 1e-5)

    model.eq_constraints = Constraint(range(len(b)), rule=eq_constraint_rule)

    model.eq_constraints2 = Constraint(range(len(b)), rule=eq_constraint_rule_2)

    # Upper bound constraints for y
    def y_bound_rule_upper(model, k):
        return model.y[k] <= uy * model.w[k]

    # Lower bound constraints for y
    def y_bound_rule_lower(model, k):
        return model.y[k] >= 0

    model.y_bound_constraints_lower = Constraint(range(m), rule=y_bound_rule_lower)
    model.y_bound_constraints_upper = Constraint(range(m), rule=y_bound_rule_upper)

    # Upper bound constraints for z
    def z_bound_rule_upper(model, k):
        return model.z[k] <= uz * (1 - model.w[k])

    # Lower bound constraints for z
    def z_bound_rule_lower(model, k):
        return model.z[k] >= 0

    model.z_bound_constraints_upper = Constraint(range(m), rule=z_bound_rule_upper)
    model.z_bound_constraints_lower = Constraint(range(m), rule=z_bound_rule_lower)


    # Solver
    solver = SolverFactory(solver_name)
    if solver_name == 'gurobi':
        solver.options['TimeLimit'] = timelimit
        # solver.options['IntFeasTol'] = 1e-1
    if heuristic_focus == 1:
        solver.options['MIPFocus'] = 1
        solver.options['Heuristics'] = 1

    if warmstart == 1:
        print('Initializing for Warmstarting FULL MILP....')
        x_, y_, z_, obj, comp_time = initial_solve(c, e, f, A, B, C, b, lx, ux, uy, uz, initial_solver, timelimit,
                                                   x_feasible = x_feasible , y_feasible = y_feasible, z_feasible = z_feasible)
        print("Warmstarting using initial solution obtained in:", comp_time, "seconds")
        for k in range(len(x_)):
            model.x[k] = x_[k]
        for j in range(len(y_)):
            model.y[j] = y_[j]
        for j in range(len(z_)):
            model.z[j] = z_[j]

        try:
            results = solver.solve(model, tee=True, warmstart=True)
            solve_time = results.solver.time  # This gets the solver time
        except Exception as e:
            print("An error occurred during solving with warmstart:", str(e))
            results = None
            solve_time = 'NA'
    else:
        try:
            results = solver.solve(model, tee=True)
            solve_time = results.solver.time  # This gets the solver time
        except Exception as e:
            print("An error occurred during solving:", str(e))
            results = None
            solve_time = 'NA'

    if results is not None:
        # Save the new optimal values of x, y, z, and w
        x_new = [model.x[i].value for i in range(n)]
        y_new = [model.y[k].value for k in range(m)]
        z_new = [model.z[k].value for k in range(m)]
        w_new = [model.w[k].value for k in range(m)]

        feasibility_check(A, B, C, b, lx, ux, uy, uz, x_new, y_new, z_new)

        objective_value = value(model.obj)

        # Output
        print("\nStatus:", results.solver.status)
        print("Termination Condition:", results.solver.termination_condition)
        print("\n Objective Value:", objective_value)  # Displaying the objective value

    else:
        # Save NA for the new optimal values if the solver fails
        x_new = ['NA'] * n
        y_new = ['NA'] * m
        z_new = ['NA'] * m
        w_new = ['NA'] * m
        objective_value = 'NA'

        # Output the failure status
        print("\nStatus: NA")
        print("Termination Condition: NA")
        print("\n Objective Value: NA")

    return x_new, y_new, z_new, w_new, objective_value, solve_time


def algorithm(sizes, seeds, q_dens,
              max_iters, max_repeat_times, start_control_number, final_control_number, timelimit,
              gurobi_yes, gurobi_only, gurobi_time_limit, gurobi_warmstart = 0,
              initial_solver="ipopt", heuristic_focus=0, folder_path=None):

    if folder_path is not None:
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        # Sort the files to ensure they're processed in the correct order
        txt_files.sort()

        file_path_all = []

        for file in txt_files:
            file_path_all.append(os.path.join(folder_path, file))
    else:
        file_path_all = [None]

    for file_path in file_path_all:
        if file_path is not None:
            file = os.path.splitext(os.path.basename(file_path))[0]
        else:
            file = "random"
        notes = f"{file}_initializedby_{initial_solver}_heuristicfocus_{heuristic_focus}_qdens_{q_dens}"
        for size in sizes:
            for seed in seeds:
                print(f"Generating data for size {size} and seed {seed}...")
                c, e, f, A, B, C, b, lx, ux, uy, uz,  x_feasible, y_feasible, z_feasible= data_generation(size, int(np.rint(0.25 * size)), seed, density= q_dens, file_path=file_path)
                print("lx:", lx, "ux:", ux, "uy:", uy, "uz:", uz)
                if gurobi_yes == 1:
                    results = []
                    print(f"RUNNING GUROBI FULL MILP \n Size: {size} \n Seed: {seed}")
                    start_time = time.time()
                    print("Data generated. Now running gurobi full milp...")
                    x_, y_, z_, w_, obj, comp_time = full_model(c, e, f, A, B, C, b, lx, ux, uy, uz,
                                                                "gurobi", gurobi_time_limit,
                                                                initial_solver=initial_solver, warmstart= gurobi_warmstart,
                                                                heuristic_focus=heuristic_focus,
                                                                x_feasible=x_feasible, y_feasible= y_feasible, z_feasible = z_feasible)
                    # Append the results as a dictionary to the results list
                    results.append({
                        'size': size,
                        'seed': seed,
                        'x_': x_,
                        'y_': y_,
                        'z_': z_,
                        'w_': w_,
                        'obj': obj,
                        'comp_time': comp_time,
                        'total_time': time.time() - start_time,
                        'warm start': gurobi_warmstart
                    })

                    # Convert the list of dictionaries to a pandas DataFrame
                    df_results = pd.DataFrame(results)

                    # Generate the formatted timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Save the DataFrame to a CSV file
                    if not os.path.exists('lpcc_results'):
                        os.makedirs('lpcc_results')

                    csv_file_path = f"GUROBI_FULL_MILP_results_{notes}_initialized_{gurobi_warmstart}_by_{initial_solver}_size_{size}_seed_{seed}_{timestamp}.csv"

                    # Construct the full path to save the file in the "result" subfolder
                    full_path = os.path.join('lpcc_results', csv_file_path)

                    # Save the DataFrame to the constructed path
                    df_results.to_csv(full_path, index=False)

                    print(f"Saved results to {full_path}")

                if gurobi_only != 1:
                    print(f"RUNNING PIP FOR SIZE {size} AND SEED {seed}...")
                    results_dict = pip_solve_for_lpcc(c, e, f, A, B, C, b, lx, ux, uy, uz,
                                                      start_control_number, final_control_number,
                                                      max_repeat_times, max_iters, timelimit, initial_solver=initial_solver,
                                                      x_feasible=x_feasible, y_feasible= y_feasible, z_feasible = z_feasible)

                    results_df = pd.DataFrame(results_dict)

                    save_df(results_df, size, int(np.rint(0.25 * size)), seed, notes)


def save_df(df, m, n, seed, notes):
    # Check if "result" directory exists, if not, create it
    if not os.path.exists('lpcc_results'):
        os.makedirs('lpcc_results')

    # Get the current date and time
    now = datetime.now()

    # Construct filename with format YYYYMMDD_HHMMSS_seed_{seed}.csv
    filename = now.strftime(f'{notes}_%Y%m%d_%H%M%S_LPCC_m_{m}_n_{n}_seed_{seed}.csv')

    # Construct the full path to save the file in the "result" subfolder
    full_path = os.path.join('lpcc_results', filename)

    # Save the DataFrame to the constructed path
    df.to_csv(full_path, index=False)
    print(f"Saved results to {full_path}")


def feasibility_check(A, B, C, b, lx, ux, uy, uz, x_feasible, y_feasible, z_feasible):
    """
    Check the feasibility of the provided feasible points.

    Parameters:
    A (np.ndarray): Coefficient matrix for x variables in the equality constraints.
    B (np.ndarray): Coefficient matrix for y variables in the equality constraints.
    C (np.ndarray): Coefficient matrix for z variables in the equality constraints.
    b (np.ndarray): Right-hand side vector for the equality constraints.
    lx (float): Lower bound for x variables.
    ux (float): Upper bound for x variables.
    uy (float): Upper bound for y variables.
    uz (float): Upper bound for z variables.
    x_feasible (np.ndarray): Feasible values for x variables.
    y_feasible (np.ndarray): Feasible values for y variables.
    z_feasible (np.ndarray): Feasible values for z variables.

    Returns:
    dict: A dictionary with the results of the feasibility checks.
    """
    # Ensure lx and ux are arrays with the same shape as x_feasible
    lx = np.full_like(x_feasible, lx)
    ux = np.full_like(x_feasible, ux)

    # Check feasibility of bounds
    x_within_bounds = np.all((lx <= x_feasible) & (x_feasible <= ux))
    y_within_bounds = np.all(y_feasible <= uy)
    z_within_bounds = np.all(z_feasible <= uz)

    print("x_feasible within bounds?", x_within_bounds)
    print("y_feasible within bounds?", y_within_bounds)
    print("z_feasible within bounds?", z_within_bounds)

    # Check feasibility of constraints
    lhs = A @ x_feasible + B @ y_feasible + C @ z_feasible
    constraints_feasible = np.allclose(lhs, b, atol=1e-5)
    # print(lhs-b)
    print("Constraints feasible?", constraints_feasible)

    # Check complementarity
    y_feasible = np.array(y_feasible)
    z_feasible = np.array(z_feasible)
    # complementarity_feasible = np.all(y_feasible * z_feasible <= 1e-6)
    complementarity_feasible = np.all(
        (np.minimum(y_feasible, z_feasible) >= -1e-5) & (np.minimum(y_feasible, z_feasible) <= 1e-5))
    print("Complementarity feasible?", complementarity_feasible, np.min(np.minimum(y_feasible, z_feasible)), np.max(np.minimum(y_feasible, z_feasible)))

    return {
        "x_within_bounds": x_within_bounds,
        "y_within_bounds": y_within_bounds,
        "z_within_bounds": z_within_bounds,
        "constraints_feasible": constraints_feasible,
        "complementarity_feasible": complementarity_feasible
    }


def data_generation(m, n, seed, density=0.5 , file_path = None):
    np.random.seed(seed)

    if file_path:
        # Load Q from the specified file
        Q = np.loadtxt(file_path)
        m = Q.shape[0]  # Dimension of Q
        n = int(np.rint(0.25 * m))
        print("Loaded Q from file: ", file_path, "m:", m, "n:", n)

    else:
        # Generate a random matrix
        Q = np.random.randn(m, m)

    # Make the matrix symmetric
    # Q = (Q + Q.T) / 2

    print(check_matrix_definiteness(Q))

    # Generate l_x and u_x
    lx = 0.1  # l_x in R^n
    ux = 1 # u_x in R^n

    a = np.random.rand(n) + 0.1

    # Generate x_feasible randomly within the bounds
    x_feasible = np.random.uniform(low=lx, high=ux, size=n)

    # Generate random feasible points for y and z
    y_feasible = np.zeros(m)
    z_feasible = np.random.uniform(1, 10, m)

    # Ensure about half of z_feasible[i] = 0 and y_feasible[i] > 0
    indices = np.random.choice(m, m // 2, replace=False)
    z_feasible[indices] = 0
    # y_feasible[indices] = np.random.uniform(1, 10, len(indices))
    y_feasible[indices] = (1 + a.T @ x_feasible ) / len(indices)

    # print(- a.T @ x_feasible + np.sum(y_feasible) - 1)

    # Generate random vectors c, e, and f
    c = np.random.randn(n)
    e = np.eye(m) @ z_feasible - Q @ y_feasible
    # Objective function components
    f = np.zeros(m)  # No direct z variables in the objective

    # Upper bounds for y, z
    uy = 1 + n * np.max(a) * ux  # u_y in R^m
    uz = m * np.max(Q) * uy + np.max(e) # u_z in R^m

    # Forming the matrices for the given LPCC structure
    A = np.vstack([np.zeros((m, n)), - a.reshape(1, -1)])  # A should be (m+1) x n
    B = np.vstack([Q, np.ones((1, m))])  # B should be (m+1) x m
    C = np.vstack([-np.eye(m), np.zeros((1, m))])  # C should be (m+1) x m

    b = np.hstack([-e, [1]])  # b should be (m+1)

    e = np.random.randn(m)

    # Check dimensions
    assert A.shape == (m+1, n), f"A has incorrect dimensions: {A.shape}"
    assert B.shape == (m+1, m), f"B has incorrect dimensions: {B.shape}"
    assert C.shape == (m+1, m), f"C has incorrect dimensions: {C.shape}"
    assert b.shape == (m+1,), f"b has incorrect dimensions: {b.shape}"

    # Check feasibility
    feasibility_check(A, B, C, b, lx, ux, uy, uz, x_feasible, y_feasible, z_feasible)

    # uz = 10* np.max(z_feasible)

    return c, e, f, A, B, C, b, lx, ux, uy, uz, x_feasible, y_feasible, z_feasible


def check_matrix_definiteness(Q):
    """
    Check the definiteness of a given matrix Q.

    Args:
    - Q (numpy.ndarray): The matrix to be checked.

    Returns:
    - str: The definiteness type of the matrix (PD, ND, PSD, NSD, or Indefinite).
    """
    eigenvalues = np.linalg.eigvals(Q)

    if all(eigenvalue > 0 for eigenvalue in eigenvalues):
        return "Q is Positive Definite (PD)"
    elif all(eigenvalue < 0 for eigenvalue in eigenvalues):
        return "Q is Negative Definite (ND)"
    elif all(eigenvalue >= 0 for eigenvalue in eigenvalues):
        return "Q is Positive Semi-Definite (PSD)"
    elif all(eigenvalue <= 0 for eigenvalue in eigenvalues):
        return "Q is Negative Semi-Definite (NSD)"
    else:
        return "Q is Indefinite"


# sizes = [100, 150, 200, 300, 350, 400]
# seeds = [1,2,3,4,5]

sizes = [20]
seeds = [1]

# infeasible seeds: 5
max_repeat_times = 3
start_p = 0.8
final_p = 0.2
max_iters = np.rint((start_p - final_p + 0.1)/0.1 * max_repeat_times)

algorithm(sizes, seeds, 1,
          max_iters=max_iters, max_repeat_times=max_repeat_times,
          start_control_number=start_p, final_control_number=final_p, timelimit=600,
          gurobi_yes=1, gurobi_only = 0,
          gurobi_time_limit=3600, gurobi_warmstart = 1,
          initial_solver="ipopt",
          folder_path = None)

