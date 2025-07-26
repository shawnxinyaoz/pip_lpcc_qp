# PIP

This repository contains implementations of the PIP methods and experiments in "Improving the Solution of Indefinite Quadratic Programs and Linear Programs with Complementarity Constraints by a Progressive MIP Method," by Xinyao Zhang, Shaoning Han, and Jong-Shi Pang. The manuscript has been accepted by Mathematical Programming Computation.

## Instances

- **instances**: This folder contains subfolders of StQP and QAP instances studied in the experiments.

  -- stqp_instances: Contains samples of StQP instances along with the generator Julia scripts proposed in "A new semidefinite programming bound for indefinite quadratic forms over a simplex" by I. Nowak.
  -- qap_instances: Contains samples of QAP instances from QAPLIB (R.E. Burkard and S. E. Karisch, and F. Rendl. QAPLIB – A quadratic assignment problem library. Journal of Global Optimization 10: 391–403 (1997). https://coral.ise.lehigh.edu/data-sets/qaplib/)
  --  InvQP instances, following the scheme proposed by F. Jara Moroni, J.S. Pang, and A. Wachter in "A study of the difference-of-convex approach for solving linear programs with complementarity constraints" (Mathematical Programming 169: 221–254 (2018)).

## Usage

- **PIP_stqp.py**: Implements the methods on solving Standard Quadratic Programming problems (StQP).
- **PIP_qap.py**: Contains functions to process and solve Quadratic Assignment Problems (QAP).
- **PIP_invqp.py**: Contains functions to generate and solve inverse QP (InvQP) instances.

At the bottom section of each .py file, the parameters of each method can be adjusted.

  - 'folder_path': Please modify this to be the directory containing the targeting instances

  - pip_run, fmip_run, fmip_w_run: [binary - 1 if the method is included in the run and 0 otherwise]

For PIP:

  In StQP and QAP runs:

  - p_max: [a number between 1 and 0 - set to 0.4 / 0.6 /0.8 / 0.9 in our experiments]
    Controls the maximum proportion of integer variables to solve in PIP. Please see manuscript for details. It is the number in parenthesis after "PIP" shown in the tables in the manuscript.

  - max_repeat: [positive integer - set to 3 in our experiments]
    Controls the maximum iterations of PIP allowed to fixing same proportion of integer variables. It is r_max in the manuscript.

  - pip_subproblem_timelimit: [positive number - set to 60 for instances in 'instances/qap_instances/below_30'; set to 600 everywhere else in our experiments]
    Controls the time limit given to solver at each subproblem of every iteration of PIP.

  - pip_step_size:  [between 1 and 0 - set to 0.1 in our experiments]
    Controls the change of proportion of fixed integer variables at eligible iterations. It is \alpha in the manuscript.

  - start_fixed_proportion: [between 1 and 0 - set to 0.8 in our experiments]
    Controls the proportion of fixed integer variables in first subproblems.

  - Others: final_fixed_proportion is always equal to 1 - p_max; max_iters is automatically set to number of iterations when PIP repeats to max times at each fixing proportion.

  In InvQP runs:

    In addition to the parameters above,
  - "initial solver" controls the approach to obtain the initial solution of PIP and FMIP-W.
    Specifically, we use 'ipopt' when m = 200, and 'gurobi' when m = 1000. Please see manuscript for details.

For FMIP and FMIP-W:

  - fmip_timelimit & fmip_w_timelimit: [positive number; in seconds] control the time limit given to the solver in FMIP (-w) runs.

    In StQP, we set both to
      60 for StQP with n = 200, (folder_path = 'instances/stqp_instances/200_05' or 'instances/stqp_instances/200_075')
      600 for StQP with n = 500 and 1000, (folder_path = 'instances/stqp_instances/500_05'; '.../500_075'; '.../1k_05'; '.../1k_075')
      3600 for StQP with n = 2000, (folder_path = 'instances/stqp_instances/2k_05' or 'instances/stqp_instances/2k_075')

    In QAP, we set fmip_timelimit to
      600 for QAP with n < 30, (folder_path = 'instances/qap_instances/below_30')
      14400 for QAP with n = 30 and 35, (folder_path = 'instances/qap_instances/30_35')
      28800 for QAP with n = 40, (folder_path = 'instances/qap_instances/40')

    In InvQP, we set
      fmip_timelimit = 6000 & fmip_w_timelimit = 2400 for InvQP with m = 200,
      fmip_timelimit = 9000 & fmip_w_run = 0 for InvQP with m = 1000.


    Please also see manuscript for specific settings on different groups of instances. They are stated in the parenthesis following method names in the results tables.

    The goal is to align FMIP (-w) times with PIP times on same group of instances to compare their solution quality in comparable times.

## Results

After adjusting the parameters, execute the scripts. The results for each experiment will be saved in their respective folders (stqp_results/, qap_results/, invqp_results/) as CSV files. Each file contains instance information, objective values, and the total solution time (all in seconds). Additional details can be saved with slight edit in the "results" dictionaries in the scripts.

## Setup

Python 3.x packages used:
- numpy
- scipy
- pandas
- pyomo
- scikit-learn

Solvers used:
- gurobi
- ipopt

## Acknowledgments

We acknowledge the authors of the developers of Pyomo, Gurobi, and IPOPT for their excellent optimization tools.

Special thanks to I. Nowak for his design of the StQP instances,
the contributors to QAPLIB and the QAP instances: R.E. Burkard and S. E. Karisch, and F. Rendl, as well as C.E. Nugent, T.E. Vollmann, and J. Ruml; B. Eschermann and H.J. Wunderlich; E.D. Taillard,
and F. Jara-Moroni, J.S. Pang, and A. Wachter for their scheme of the inverse QP instances used in our experiments.

Last but not least, we appreciate the guidance from the editors and reviewers from Mathematical Programming Computation.

Contact: Xinyao Zhang - xinyaoz@usc.edu
