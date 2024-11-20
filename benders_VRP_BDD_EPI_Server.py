# Import Gurobi Library
# import csv

import gurobipy as gb
import numpy as np
from scipy.io import loadmat
import time
import sys
from csv import writer
import pandas as pd
# from . import benders_VRP_EPI_BDD_opt_MP
# from . import benders_VRP_EPI_BDD_opt_MP


# Class which can have attributes set.
class expando(object):
    pass


def EPI(model, where):
    # model = self.model
    GRB = gb.GRB
    if where == GRB.Callback.MIPNODE:  #
        nodecnt = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
        nodecnt = nodecnt / model._numcheckedNode
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)  #
        # print(nodecnt)
        if status == GRB.OPTIMAL:
            if (nodecnt.is_integer()) & (model._numEPI <= model._numEPImax):  # model._numEPImax
                # if nodecnt <= 5e3:
                # A relaxed sol. obtained at the  explored node
                sol = model.cbGetNodeRel(model._vars)
                SubFun = model._Gfun
                N = model._coeff[0]
                K = model._coeff[1]
                cut = np.zeros((K, N ** 2))
                Z = np.zeros(K)
                label_cut = np.zeros((K, N ** 2))
                cut_reindex = np.zeros((K, N ** 2))
                pi = np.zeros((K, N ** 2))

                for i in range(K):
                    # for j in range(N ** 2):
                    cut[i, :] = sol[i * (N ** 2): (i + 1) * (N ** 2)]

                for i in range(K):
                    Z[i] = sol[i + K * (N ** 2)]

                #  Indeices of relaxed sol in descending order (max-min).
                for i in range(K):
                    label_cut[i, :] = np.argsort(-cut[i, :])
                # label_cut.astype(int)

                for i in range(K):
                    for j in range(N ** 2):
                        pi[i, j] = SubFun[int(label_cut[i, j])]
                        cut_reindex[i, j] = cut[i, int(label_cut[i, j])]
                # EPIs construction
                # for i in range(K):
                #     if np.dot(pi[i, :],  cut[i, :]) > Z[i]:
                #             model.cbCut( sum((pi[i, j] * model._vars[j + i * (N ** 2)]) for j in range(N ** 2)) <= model._vars[ i + K * (N ** 2)])
                # model._numEPI += 1
                if sum(np.dot(pi[i, :], cut_reindex[i, :]) for i in range(K)) > sum((Z[i]) for i in range(K)):
                    model.cbCut(
                        sum(sum((pi[i, j] * model._vars[int(label_cut[i, j]) + i * (N ** 2)]) for i in range(K))
                            for j in range(N ** 2)) <= sum((model._vars[k + K * (N ** 2)]) for k in range(K)))
                    model._numEPI += 1


# Master problem
class Benders_Master:
    def __init__(self, benders_gap, run, node, vehicle, delta_val, gamma, chi, EPI_index, IteNum_MP_LP):
        self.max_iters = 5e3
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(benders_gap, run, node, vehicle,
                        delta_val, gamma, chi, EPI_index, IteNum_MP_LP)
        self.data.run = run
        self.data.node = node
        self.data.vehicle = vehicle
        self.data.delta_val = delta_val
        self.data.z_fea_sub = []
        self._build_model()

    def optimize(self, simple_results=False):
        data = self.data
        N = data.N
        K = data.K
        T = data.T
        c = data.c
        maxEPI = 1e4
        # tlimit = 1e4

        # Initial solution        # Submodular set function (quadratic form)
        GFun = []
        for i in range(N):
            for j in range(N):
                GFun.append(T[i, j] + c[i])
        self.model._Gfun = GFun
        self.model._coeff = [N, K]
        self.model._numEPI = 0
        self.model._numEPImax = maxEPI
        self.model._numcheckedNode = 1e4
        self.model._num_inte_sol = 0
        self.model._vars = self.model.getVars()
        # self.model.setParam('Heuristics', 0.00)
        # self.model.setParam('Threads', 1)
        # self.model.setParam('PreCrush', 1)
        print('\n' + '#' * 50)
        print('Master problem optimization_Initial(MILP)')
        self.model.optimize()

        print('\n' + '#' * 50)
        print('Subproblem optimization_Initial(LP)')
        self.submodel = Benders_Subproblem(self, data.run, data.node, data.vehicle,
                                           data.delta_val, data.gamma, data.chi)     # Build subproblem from solution
        self.submodel.update_fixed_vars(self)
        self.submodel.optimize()

        self.submodel_MIP = Benders_Subproblem_MIP(self, self.submodel, data.run,
                                                   data.node,
                                                   data.vehicle, data.delta_val,
                                                   data.gamma, data.chi)  # Build subproblem from solution
        self.submodel_MIP.update_fixed_vars(self)
        self.submodel_MIP.optimize()
        self._add_cut()
        self._update_bounds()
        self._save_vars()
        print('\n' + '#' * 50)
        print('The # of iteration = {}'.format(len(self.data.cutlist)))
        print('LB, UB = {},{}'.format(self.data.lb, self.data.ub))

        while np.abs((self.data.ub - self.data.lb)/self.data.lb) > self.data.benders_gap and len(self.data.cutlist) < self.max_iters and (self.data.lb < self.data.ub):
            print('\n' + '#' * 50)
            # if len(self.data.cutlist) not in self.data.IteNum_MP_LP:  # num in which the program switch from MP LP to MIP, （MP:MIP     SP:LP）
            #     print('Master problem optimization(MILP)_{}'.format(len(self.data.cutlist)))
            #     self.data.MP_relaxed = 0
            #     if self.data.EPI_index == 1:
            #         self.model.optimize(EPI)
            #     else:
            #         self.model.optimize()
            #
            #     print('\n' + '#' * 50)
            #     print('Subproblem optimization(LP)_{}'.format(len(self.data.cutlist)))
            #     self.submodel.update_fixed_vars(self)  # Update the decision var. of MP
            #     self.submodel.optimize()  # Optimize subproblem
            #     if self.submodel.model.Status == 2:  # 2: optimal; 3:infeasible
            #         print('\n' + '#' * 50)
            #         print('Subproblem optimization(MILP)_{}'.format(len(self.data.cutlist)))
            #         self.submodel_MIP.update_fixed_vars(self)
            #         self.submodel_MIP.optimize()
            #         self._add_cut()  # Add Generalized Benders cut
            #         self._update_bounds()
            #         self._save_vars()
            #     elif self.submodel.model.Status == 3:
            #         self._add_cut()
            #         self._update_bounds()
            #         self._save_vars()
            #     print('\n' + '#' * 50)
            #     print('The # of iteration = {}'.format(len(self.data.cutlist)))
            #     print('LB, UB = {},{}'.format(self.data.lb, self.data.ub))
            #
            # else:
            # print('Master problem optimization(LP)_{}'.format(len(self.data.cutlist)))
            # self.data.MP_relaxed = 1
            # self.m_continuous = Benders_Master_Relaxed(self.data.benders_gap, self.data.run, self.data.node,
            #                                            self.data.vehicle, self.data.deltabar, self.data.gamma,
            #                                            self.data.chi,
            #                                            self.data.EPI_index)
            # self.m_continuous.optimize()
            # self.m_continuous.data.xs
            # print('\n' + '#' * 50)
            # print('\n' + '#' * 50)
            # self.submodel.update_fixed_vars(self)
            # self.submodel.optimize()
            #
            # if self.submodel.model.Status == 2:  # 2: optimal; 3:infeasible
            #     self.submodel_MIP = Benders_Subproblem_MIP(self.m_continuous, self.submodel, data.run,
            #                                                data.node,
            #                                                data.vehicle, data.delta_val,
            #                                                data.gamma, data.chi)  # Build subproblem from solution
            #     print('\n' + '#' * 50)
            #     print('Subproblem optimization(MILP)_{}'.format(len(self.data.cutlist)))
            #     self.submodel_MIP.update_fixed_vars(self)
            #     self.submodel_MIP.optimize()
            #     self._add_cut()  # Add Generalized Benders cut
            #     self._update_bounds()
            #     self._save_vars()
            # elif self.submodel.model.Status == 3:
            #     self._add_cut()
            #     self._update_bounds()
            #     self._save_vars()
            # print('\n' + '#' * 50)
            # print('The # of iteration = {}'.format(len(self.data.cutlist)))
            # print('LB, UB = {},{}'.format(self.data.lb, self.data.ub))

            ites = len(self.data.cutlist)/2
            if ites.is_integer():
                print('Master problem optimization(MILP)_{}'.format(
                    len(self.data.cutlist)))
                self.data.MP_relaxed = 0
                if self.data.EPI_index == 1:
                    self.model.optimize(EPI)
                else:
                    self.model.optimize()

                print('\n' + '#' * 50)
                print('Subproblem optimization(LP)_{}'.format(
                    len(self.data.cutlist)))
                # Update the decision var. of MP
                self.submodel.update_fixed_vars(self)
                self.submodel.optimize()  # Optimize subproblem
                if self.submodel.model.Status == 2:  # 2: optimal; 3:infeasible
                    print('\n' + '#' * 50)
                    print('Subproblem optimization(MILP)_{}'.format(
                        len(self.data.cutlist)))
                    self.submodel_MIP.update_fixed_vars(self)
                    self.submodel_MIP.optimize()
                    self._add_cut()  # Add Generalized Benders cut
                    self._update_bounds()
                    self._save_vars()
                elif self.submodel.model.Status == 3:
                    self._add_cut()
                    self._update_bounds()
                    self._save_vars()
                print('\n' + '#' * 50)
                print('The # of iteration = {}'.format(len(self.data.cutlist)))
                print('LB, UB = {},{}'.format(self.data.lb, self.data.ub))
            else:
                print('Master problem optimization(LP)_{}'.format(
                    len(self.data.cutlist)))
                self.data.MP_relaxed = 1
                self.m_continuous = Benders_Master_Relaxed(self.data.benders_gap, self.data.run, self.data.node,
                                                           self.data.vehicle, self.data.deltabar, self.data.gamma,
                                                           self.data.chi,
                                                           self.data.EPI_index)
                self.m_continuous.optimize()
                self.m_continuous.data.xs
                print('\n' + '#' * 50)
                print('\n' + '#' * 50)
                self.submodel.update_fixed_vars(self)
                self.submodel.optimize()

                if self.submodel.model.Status == 2:  # 2: optimal; 3:infeasible
                    self.submodel_MIP = Benders_Subproblem_MIP(self.m_continuous, self.submodel, data.run,
                                                               data.node,
                                                               data.vehicle, data.delta_val,
                                                               data.gamma, data.chi)  # Build subproblem from solution
                    print('\n' + '#' * 50)
                    print('Subproblem optimization(MILP)_{}'.format(
                        len(self.data.cutlist)))
                    self.submodel_MIP.update_fixed_vars(self)
                    self.submodel_MIP.optimize()
                    self._add_cut()  # Add Generalized Benders cut
                    self._update_bounds()
                    self._save_vars()
                elif self.submodel.model.Status == 3:
                    self._add_cut()
                    self._update_bounds()
                    self._save_vars()
                print('\n' + '#' * 50)
                print('The # of iteration = {}'.format(len(self.data.cutlist)))
                print('LB, UB = {},{}'.format(self.data.lb, self.data.ub))

        print('\n' + '#' * 50)
        print('The # of iteration = {}'.format(len(self.data.cutlist)))
        print('LB, UB = {},{}'.format(self.data.lb, self.data.ub))
        print('# of added cut={}'.format(len(self.data.cutlist)))

    ###
    #   Loading functions
    ###

    def _load_data(self, benders_gap, run, node, vehicle, delta_val, gamma, chi, EPI_index, IteNum_MP_LP):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.benders_gap = benders_gap
        self.data.gamma = gamma
        self.data.chi = chi
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.xs = []
        self.data.ys = []
        self.data.thetas = []
        self.data.EPI_index = EPI_index
        self.data.IteNum_MP_LP = IteNum_MP_LP

        self.data.x_val = np.zeros((vehicle, node, node))
        self.data.omega1_val = np.zeros(node)
        self.data.omega2_val = np.zeros(node)
        self.data.omega3_val = np.zeros(node)
        self.data.omega4_val = np.zeros(node)
        self.data.MP_relaxed = 0
        self.data.num_BDD = 2

        data = loadmat(
            '/Users/vulcanyao/OneDrive - 南方科技大学/PDF Expert/Research Yao/code_VRPPD/[J3]Bilevel EVRP with time flexibility/J3-V1/DataMap/Vehicles_{}/RealMap_{}_test.mat'.format(run, node))
        self.data.Emax = data['Emax'][0][0]
        self.data.rmax = self.data.Emax
        self.data.e = data['e']
        self.data.T = 1e0 * data['T']
        self.data.M = int(data['M'])
        self.data.p = data['p'][0]
        self.data.g = data['g'][0]
        self.data.t = 1e0 * data['t'][0]
        self.data.c = data['c'][0]
        self.data.K = int(vehicle)
        self.data.N = int(data['N'])
        self.data.alpha = 0.05
        self.data.Q = 200
        self.data.deltabar = np.array(delta_val)

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        K = self.data.K
        N = self.data.N

        # MIP-type MP
        self.variables.x = m.addVars(K, N, N, vtype=gb.GRB.BINARY, name="x")
        self.variables.Z = m.addVars(K, vtype=gb.GRB.CONTINUOUS, name="Z")
        self.variables.theta = m.addVar(lb=0, ub=gb.GRB.INFINITY, name='theta')
        self.variables.omega1 = m.addVars(
            N, vtype=gb.GRB.BINARY, name="omega1")
        self.variables.omega2 = m.addVars(
            N, vtype=gb.GRB.BINARY, name="omega2")
        self.variables.omega3 = m.addVars(
            N, vtype=gb.GRB.BINARY, name="omega3")
        self.variables.omega4 = m.addVars(
            N, vtype=gb.GRB.BINARY, name="omega4")

        if len(self.data.cutlist) == 0:
            m = self.model
            N = self.data.N
            self.variables.eta = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="eta")
            self.variables.epsilon = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="epsilon")
            self.variables.q = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="q")
            self.variables.u = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="u")
            self.variables.v = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="v")
            self.variables.zeta1 = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="zeta1")
            self.variables.zeta2 = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="zeta2")
            self.variables.delta = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="delta")
            self.variables.tau = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="tau")
        m.update()

    def _build_objective(self):
        K = self.data.K
        zz = 0
        for k in range(K):
            zz += self.variables.Z[k]
        for k in range(K):
            self.model.setObjective(self.variables.theta + zz, gb.GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model
        K = self.data.K
        N = self.data.N
        T = self.data.T
        c = self.data.c

        x = self.variables.x
        Z = self.variables.Z
        for k in range(K):
            zz = 0
            for i in range(N):
                for j in range(N):
                    zz += T[i, j] * x[k, i, j] + c[i] * x[k, i, j]
            self.constraints.c8final = m.addConstr(Z[k] >= zz)

        self.constraints.c1 = m.addConstrs(self.variables.x.sum(
            '*', i, '*') <= 1 for i in range(N) if i != 0)
        self.constraints.c2 = m.addConstrs(
            self.variables.x[k, i, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c3 = m.addConstrs(
            self.variables.x[k, N - 1, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c4 = m.addConstrs(
            self.variables.x[k, i, 0] == 0 for i in range(N) for k in range(K))
        # Flow conservation
        self.constraints.c5 = m.addConstrs(
            self.variables.x.sum(k, 0, '*') - self.variables.x.sum(k, '*', 0) == 1 for k in range(K))
        self.constraints.c6 = m.addConstrs(
            self.variables.x.sum(k, N - 1, '*') - self.variables.x.sum(k, '*', N - 1) == -1 for k in range(K))
        self.constraints.c7 = m.addConstrs(
            self.variables.x.sum(k, i, '*') - self.variables.x.sum(k, '*', i) == 0 for i in range(N - 1) if i != 0 for k
            in range(K))

        if len(self.data.cutlist) == 0:
            M = self.data.M
            N = self.data.N
            t = self.data.t
            T = self.data.T
            deltabar = self.data.deltabar
            gamma = self.data.gamma
            chi = self.data.chi

            x = self.variables.x
            omega1 = self.variables.omega1
            omega2 = self.variables.omega2
            omega3 = self.variables.omega3
            omega4 = self.variables.omega4
            tau = self.variables.tau
            delta = self.variables.delta
            u = self.variables.u
            v = self.variables.v
            eta = self.variables.eta
            epsilon = self.variables.epsilon
            zeta1 = self.variables.zeta1
            zeta2 = self.variables.zeta2
            q = self.variables.q

            self.constraints.c8 = m.addConstrs(
                t[j] - tau[j] <= 0 for j in range(N))
            self.constraints.c81 = m.addConstrs(
                tau[j] - t[j] - delta[j] <= 0 for j in range(N))

            self.constraints.c9 = m.addConstrs(
                tau[j] >= tau[i] + T[i, j] - M * (1 - x[k, i, j]) for j in range(N) for i in range(N) for k in range(K))
            self.constraints.c10 = m.addConstrs(
                eta[j] >= epsilon[j] + u[j] * deltabar - zeta1[j]*chi - zeta2[j]*chi - M * (1 - x.sum('*', '*', j)) for j in range(N))
            # KKT condition of lower level problem
            self.constraints.c11 = m.addConstrs(
                -gamma * zeta1[j] + gamma * zeta2[j] - q[j] + u[j] - v[j] == 0 for j in range(N) if 0 < j < N - 1)
            self.constraints.c12 = m.addConstrs(
                1 - zeta1[j] - zeta2[j] == 0 for j in range(N) if 0 < j < N - 1)
            self.constraints.c13a = m.addConstrs(0 <= epsilon[j] + gamma * delta[j] - chi
                                                 for j in range(N) if 0 < j < N - 1)
            self.constraints.c13b = m.addConstrs(epsilon[j] + gamma * delta[j] - chi <= M * omega1[j]
                                                 for j in range(N) if 0 < j < N - 1)
            self.constraints.c14 = m.addConstrs(
                zeta1[j] <= M * (1 - omega1[j]) for j in range(N) if 0 < j < N - 1)
            self.constraints.c15 = m.addConstrs(
                0 <= epsilon[j] - gamma * delta[j] + chi for j in range(N) if 0 < j < N - 1)

            self.constraints.c15 = m.addConstrs(
                epsilon[j] - gamma * delta[j] + chi <= M * omega2[j] for j in range(N) if 0 < j < N - 1)
            self.constraints.c16 = m.addConstrs(
                zeta2[j] <= M * (1 - omega2[j]) for j in range(N) if 0 < j < N - 1)
            self.constraints.c17 = m.addConstrs(
                deltabar - delta[j] <= M * omega3[j] for j in range(N) if 0 < j < N - 1)
            self.constraints.c18 = m.addConstrs(
                u[j] <= M * (1 - omega3[j]) for j in range(N) if 0 < j < N - 1)
            self.constraints.c19 = m.addConstrs(
                delta[j] <= M * omega4[j] for j in range(N) if 0 < j < N - 1)
            self.constraints.c20 = m.addConstrs(
                v[j] <= M * (1 - omega4[j]) for j in range(N) if 0 < j < N - 1)

            self.constraints.c8a = m.addConstrs(
                delta[j] >= 0 for j in range(N))
            self.constraints.c8b = m.addConstrs(
                delta[j] <= deltabar for j in range(N))

        self.constraints.cuts = {}
        pass

    ###
    # Cut adding
    ###

    def _add_cut(self):
        if len(self.data.cutlist) in self.data.IteNum_MP_LP:   # Initial ite. just like  GBD
            if self.submodel_MIP.model.Status == 2:
                K = self.data.K
                N = self.data.N
                x = self.variables.x
                omega1 = self.variables.omega1
                omega2 = self.variables.omega2
                omega3 = self.variables.omega3
                omega4 = self.variables.omega4
                cut = len(self.data.cutlist)
                self.data.cutlist.append(cut)
                # Get sensitivity from subproblem
                sens_x = 0
                sens_omega1 = 0
                sens_omega2 = 0
                sens_omega3 = 0
                sens_omega4 = 0
                for k in range(K):
                    for i in range(N):
                        for j in range(N):
                            sens_x += (x[k, i, j] - self.m_continuous.variables.x[k,
                                       i, j].x) * self.submodel.constraints.fix_x[k, i, j].pi
                            sens_omega1 += (omega1[j] - self.m_continuous.variables.omega1[j].x) * \
                                self.submodel.constraints.fix_omega1[j].pi
                            sens_omega2 += (omega2[j] - self.m_continuous.variables.omega2[j].x) * \
                                self.submodel.constraints.fix_omega2[j].pi
                            sens_omega3 += (omega3[j] - self.m_continuous.variables.omega3[j].x) * \
                                self.submodel.constraints.fix_omega3[j].pi
                            sens_omega4 += (omega4[j] - self.m_continuous.variables.omega4[j].x) * \
                                self.submodel.constraints.fix_omega4[j].pi

                # z_sub = self.submodel_MIP.model.ObjVal
                z_sub = 0
                for i in range(self.data.N - 1):
                    z_sub += self.submodel_MIP.variables.eta[i].x

                self.constraints.cuts[cut] = self.model.addConstr(self.variables.theta >= z_sub + sens_x + sens_omega1
                                                                  + sens_omega2 + sens_omega3 + sens_omega4)
            elif self.submodel_MIP.model.Status == 3:
                data = self.data
                print('\n' + '#' * 50)
                print('Feasibility  Subproblem optimization')
                self.feasibility_submodel_MIP = Benders_Feasibility_Subproblem_MIP(self, data.run, data.node, data.vehicle,
                                                                                   data.delta_val,
                                                                                   data.gamma, data.chi)  # Build feasibility subproblem from solution
                self.feasibility_submodel.optimize()
                K = self.data.K
                N = self.data.N
                x = self.variables.x
                omega1 = self.variables.omega1
                omega2 = self.variables.omega2
                omega3 = self.variables.omega3
                omega4 = self.variables.omega4
                cut = len(self.data.cutlist)
                self.data.cutlist.append(cut)
                # Get sensitivity from subproblem
                sens_x = 0
                sens_omega1 = 0
                sens_omega2 = 0
                sens_omega3 = 0
                sens_omega4 = 0
                for k in range(K):
                    for i in range(N):
                        for j in range(N):
                            sens_x += (x[k, i, j] - self.m_continuous.variables.x[k, i, j].x) * self.feasibility_submodel_MIP.constraints.fix_x[
                                k, i, j].pi
                            sens_omega1 += (omega1[j] - self.m_continuous.variables.omega1[j].x) * self.feasibility_submodel_MIP.constraints.fix_omega1[
                                j].pi
                            sens_omega2 += (omega2[j] - self.m_continuous.variables.omega2[j].x) * self.feasibility_submodel_MIP.constraints.fix_omega2[
                                j].pi
                            sens_omega3 += (omega3[j] - self.m_continuous.variables.omega3[j].x) * self.feasibility_submodel_MIP.constraints.fix_omega3[
                                j].pi
                            sens_omega4 += (omega4[j] - self.m_continuous.variables.omega4[j].x) * self.feasibility_submodel_MIP.constraints.fix_omega4[
                                j].pi

                z_sub = self.feasibility_submodel_MIP.model.ObjVal

                self.data.z_fea_sub.append(z_sub)
                # Generate feasibility cut
                self.constraints.cuts[cut] = self.model.addConstr(0 >= z_sub + sens_x + sens_omega1 + sens_omega2
                                                                  + sens_omega3 + sens_omega4)

        else:  # Later ite., BDD is used to solve this MIP.
            if self.submodel.model.Status == 2:
                K = self.data.K
                N = self.data.N
                x = self.variables.x
                omega1 = self.variables.omega1
                omega2 = self.variables.omega2
                omega3 = self.variables.omega3
                omega4 = self.variables.omega4
                cut = len(self.data.cutlist)
                self.data.cutlist.append(cut)
                # Get sensitivity from subproblem
                sens_x = 0
                sens_omega1 = 0
                sens_omega2 = 0
                sens_omega3 = 0
                sens_omega4 = 0
                for k in range(K):
                    for i in range(N):
                        for j in range(N):
                            sens_x += (x[k, i, j] - x[k, i, j].x) * \
                                self.submodel.constraints.fix_x[k, i, j].pi
                            sens_omega1 += (omega1[j] - omega1[j].x) * \
                                self.submodel.constraints.fix_omega1[j].pi
                            sens_omega2 += (omega2[j] - omega2[j].x) * \
                                self.submodel.constraints.fix_omega2[j].pi
                            sens_omega3 += (omega3[j] - omega3[j].x) * \
                                self.submodel.constraints.fix_omega3[j].pi
                            sens_omega4 += (omega4[j] - omega4[j].x) * \
                                self.submodel.constraints.fix_omega4[j].pi

                z_sub = self.submodel.model.ObjVal

                # Generate cut
                self.constraints.cuts[cut] = self.model.addConstr(self.variables.theta >= z_sub + sens_x
                                                                  + sens_omega1 + sens_omega2 + sens_omega3 + sens_omega4)
            elif self.submodel.model.Status == 3:
                data = self.data
                print('\n' + '#' * 50)
                print('Feasibility  Subproblem optimization')
                self.feasibility_submodel = Benders_Feasibility_Subproblem(self, data.run, data.node, data.vehicle,
                                                                           data.delta_val,
                                                                           data.gamma, data.chi)  # Build feasibility subproblem from solution
                self.feasibility_submodel.optimize()
                K = self.data.K
                N = self.data.N
                x = self.variables.x
                omega1 = self.variables.omega1
                omega2 = self.variables.omega2
                omega3 = self.variables.omega3
                omega4 = self.variables.omega4
                cut = len(self.data.cutlist)
                self.data.cutlist.append(cut)
                # Get sensitivity from subproblem
                sens_x = 0
                sens_omega1 = 0
                sens_omega2 = 0
                sens_omega3 = 0
                sens_omega4 = 0
                for k in range(K):
                    for i in range(N):
                        for j in range(N):
                            sens_x += (x[k, i, j] - x[k, i, j].x) * self.feasibility_submodel.constraints.fix_x[
                                k, i, j].pi
                            sens_omega1 += (omega1[j] - omega1[j].x) * self.feasibility_submodel.constraints.fix_omega1[
                                j].pi
                            sens_omega2 += (omega2[j] - omega2[j].x) * self.feasibility_submodel.constraints.fix_omega2[
                                j].pi
                            sens_omega3 += (omega3[j] - omega3[j].x) * self.feasibility_submodel.constraints.fix_omega3[
                                j].pi
                            sens_omega4 += (omega4[j] - omega4[j].x) * self.feasibility_submodel.constraints.fix_omega4[
                                j].pi

                z_sub = self.feasibility_submodel.model.ObjVal
                # z_sub = self.feasibility_submodel_MIP.model.ObjVal

                self.data.z_fea_sub.append(z_sub)
                # Generate feasibility cut
                self.constraints.cuts[cut] = self.model.addConstr(0 >= z_sub + sens_x + sens_omega1 + sens_omega2
                                                                  + sens_omega3 + sens_omega4)

    ###
    # Update upper and lower bounds
    ###

    def _update_bounds(self):
        if self.submodel_MIP.model.Status == 2:
            if self.data.MP_relaxed == 0:
                z_sub = 0
                for i in range(self.data.N - 1):
                    z_sub += self.submodel.variables.eta[i].x
                z_master = self.model.ObjVal
                self.data.ub = z_master - self.variables.theta.x + z_sub
                self.data.lb = self.model.ObjBound
                self.data.upper_bounds.append(self.data.ub)
                self.data.lower_bounds.append(self.data.lb)
            elif self.data.MP_relaxed == 1:
                self.data.lb = self.data.lower_bounds[-1]
                self.data.ub = self.data.upper_bounds[-1]
        elif self.submodel_MIP.model.Status == 3:
            self.data.lb = self.data.lower_bounds[-1]
            self.data.ub = self.data.upper_bounds[-1]

        # if self.data.MP_relaxed == 0:
        #     if self.submodel.model.Status == 2:
        #         z_sub = 0
        #         for i in range(self.data.N - 1):
        #             z_sub += self.submodel.variables.eta[i].x
        #         z_master = self.model.ObjVal
        #         self.data.ub = z_master - self.variables.theta.x + z_sub
        #         self.data.lb = self.model.ObjBound
        #         self.data.upper_bounds.append(self.data.ub)
        #         self.data.lower_bounds.append(self.data.lb)
        # elif self.data.MP_relaxed == 1:
        #     if self.submodel.model.Status == 2:
        #         self.data.lb = self.data.lower_bounds[-1]
        #         self.data.ub =  self.data.upper_bounds[-1]

    def _save_vars(self):
        K = self.data.K
        N = self.data.N
        self.data.thetas.append(self.variables.theta.x)
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    self.data.xs.append(self.variables.x[k, i, j].x)
                    # self.data.ys.append(self.submodel.variables.y.x)


#  Relaxed Master problem
class Benders_Master_Relaxed:
    def __init__(self, benders_gap, run, node, vehicle, delta_val, gamma, chi, EPI_index):
        self.max_iters = 1e4
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(benders_gap, run, node, vehicle,
                        delta_val, gamma, chi, EPI_index)
        self.data.run = run
        self.data.node = node
        self.data.vehicle = vehicle
        self.data.delta_val = delta_val
        self.data.z_fea_sub = []
        self._build_model()

    def optimize(self, simple_results=False):
        data = self.data
        N = data.N
        K = data.K
        T = data.T
        c = data.c
        maxEPI = 1e4
        # tlimit = 1e4

        # Initial solution
        # print('\n' + '#' * 50)
        # print('Master problem optimization(LP)_{}'.format(len(self.data.cutlist)))
        # Submodular set function (quadratic form)
        GFun = []
        for i in range(N):
            for j in range(N):
                GFun.append(T[i, j] + c[i])
        self.model._Gfun = GFun
        self.model._coeff = [N, K]
        self.model._numEPI = 0
        self.model._numEPImax = maxEPI
        self.model._num_inte_sol = 0
        self.model._vars = self.model.getVars()
        # self.model.setParam('Heuristics', 0.00)
        # self.model.setParam('Threads', 1)
        # self.model.setParam('PreCrush', 1)
        self.model.optimize()  # Optimize MP with EPIs
        self._save_vars()

    ###
    #   Loading functions
    ###

    def _load_data(self, benders_gap, run, node, vehicle, delta_val, gamma, chi, EPI_index):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.benders_gap = benders_gap
        self.data.gamma = gamma
        self.data.chi = chi
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.xs = []
        self.data.ys = []
        self.data.thetas = []
        self.data.EPI_index = EPI_index

        self.data.x_val = np.zeros((vehicle, node, node))
        self.data.omega1_val = np.zeros(node)
        self.data.omega2_val = np.zeros(node)
        self.data.omega3_val = np.zeros(node)
        self.data.omega4_val = np.zeros(node)

        data = loadmat(
            '/Users/vulcanyao/OneDrive - 南方科技大学/PDF Expert/Research Yao/code_VRPPD/[J3]Bilevel EVRP with time flexibility/J3-V1/DataMap/Vehicles_{}/RealMap_{}_test.mat'.format(run, node))
        self.data.Emax = data['Emax'][0][0]
        self.data.rmax = self.data.Emax
        self.data.e = data['e']
        self.data.T = 1e0 * data['T']
        self.data.M = int(data['M'])
        self.data.p = data['p'][0]
        self.data.g = data['g'][0]
        self.data.t = 1e0 * data['t'][0]
        self.data.c = data['c'][0]
        self.data.K = int(vehicle)
        self.data.N = int(data['N'])
        self.data.alpha = 0.05
        self.data.Q = 200
        self.data.deltabar = np.array(delta_val)

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        K = self.data.K
        N = self.data.N

        # Relaxed-type of MP
        self.variables.x = m.addVars(
            K, N, N, vtype=gb.GRB.CONTINUOUS, name="x")
        self.variables.Z = m.addVars(K, vtype=gb.GRB.CONTINUOUS, name="Z")
        self.variables.theta = m.addVar(lb=0, ub=gb.GRB.INFINITY, name='theta')
        self.variables.omega1 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="omega1")
        self.variables.omega2 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="omega2")
        self.variables.omega3 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="omega3")
        self.variables.omega4 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="omega4")

        if len(self.data.cutlist) == 0:
            m = self.model
            N = self.data.N
            self.variables.eta = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="eta")
            self.variables.epsilon = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="epsilon")
            self.variables.q = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="q")
            self.variables.u = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="u")
            self.variables.v = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="v")
            self.variables.zeta1 = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="zeta1")
            self.variables.zeta2 = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="zeta2")
            self.variables.delta = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="delta")
            self.variables.tau = m.addVars(
                N, vtype=gb.GRB.CONTINUOUS, name="tau")
        m.update()

    def _build_objective(self):
        K = self.data.K
        zz = 0
        for k in range(K):
            zz += self.variables.Z[k]
        for k in range(K):
            self.model.setObjective(self.variables.theta + zz, gb.GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model
        K = self.data.K
        N = self.data.N
        T = self.data.T
        c = self.data.c

        x = self.variables.x
        Z = self.variables.Z
        for k in range(K):
            zz = 0
            for i in range(N):
                for j in range(N):
                    zz += T[i, j] * x[k, i, j] + c[i] * x[k, i, j]
            self.constraints.c8final = m.addConstr(Z[k] >= zz)

        self.constraints.c1 = m.addConstrs(self.variables.x.sum(
            '*', i, '*') <= 1 for i in range(N) if i != 0)
        self.constraints.c2 = m.addConstrs(
            self.variables.x[k, i, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c3 = m.addConstrs(
            self.variables.x[k, N - 1, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c4 = m.addConstrs(
            self.variables.x[k, i, 0] == 0 for i in range(N) for k in range(K))
        # Flow conservation
        self.constraints.c5 = m.addConstrs(
            self.variables.x.sum(k, 0, '*') - self.variables.x.sum(k, '*', 0) == 1 for k in range(K))
        self.constraints.c6 = m.addConstrs(
            self.variables.x.sum(k, N - 1, '*') - self.variables.x.sum(k, '*', N - 1) == -1 for k in range(K))
        self.constraints.c7 = m.addConstrs(
            self.variables.x.sum(k, i, '*') - self.variables.x.sum(k, '*', i) == 0 for i in range(N - 1) if i != 0 for k
            in range(K))

        if len(self.data.cutlist) == 0:
            M = self.data.M
            N = self.data.N
            t = self.data.t
            T = self.data.T
            deltabar = self.data.deltabar
            gamma = self.data.gamma
            chi = self.data.chi

            x = self.variables.x
            omega1 = self.variables.omega1
            omega2 = self.variables.omega2
            omega3 = self.variables.omega3
            omega4 = self.variables.omega4
            tau = self.variables.tau
            delta = self.variables.delta
            u = self.variables.u
            v = self.variables.v
            eta = self.variables.eta
            epsilon = self.variables.epsilon
            zeta1 = self.variables.zeta1
            zeta2 = self.variables.zeta2
            q = self.variables.q

            self.constraints.c8 = m.addConstrs(
                t[j] - tau[j] <= 0 for j in range(N))
            self.constraints.c81 = m.addConstrs(
                tau[j] - t[j] - delta[j] <= 0 for j in range(N))

            self.constraints.c9 = m.addConstrs(
                tau[j] >= tau[i] + T[i, j] - M * (1 - x[k, i, j]) for j in range(N) for i in range(N) for k in range(K))
            self.constraints.c10 = m.addConstrs(
                eta[j] >= epsilon[j] + u[j] * deltabar - zeta1[j]*chi - zeta2[j]*chi - M * (1 - x.sum('*', '*', j)) for j in range(N))
            # KKT condition of lower level problem
            self.constraints.c11 = m.addConstrs(
                -gamma * zeta1[j] + gamma * zeta2[j] - q[j] + u[j] - v[j] == 0 for j in range(N) if 0 < j < N - 1)
            self.constraints.c12 = m.addConstrs(
                1 - zeta1[j] - zeta2[j] == 0 for j in range(N) if 0 < j < N - 1)
            self.constraints.c13a = m.addConstrs(0 <= epsilon[j] + gamma * delta[j] - chi
                                                 for j in range(N) if 0 < j < N - 1)
            self.constraints.c13b = m.addConstrs(epsilon[j] + gamma * delta[j] - chi <= M * omega1[j]
                                                 for j in range(N) if 0 < j < N - 1)
            self.constraints.c14 = m.addConstrs(
                zeta1[j] <= M * (1 - omega1[j]) for j in range(N) if 0 < j < N - 1)
            self.constraints.c15 = m.addConstrs(
                0 <= epsilon[j] - gamma * delta[j] + chi for j in range(N) if 0 < j < N - 1)

            self.constraints.c15 = m.addConstrs(
                epsilon[j] - gamma * delta[j] + chi <= M * omega2[j] for j in range(N) if 0 < j < N - 1)
            self.constraints.c16 = m.addConstrs(
                zeta2[j] <= M * (1 - omega2[j]) for j in range(N) if 0 < j < N - 1)
            self.constraints.c17 = m.addConstrs(
                deltabar - delta[j] <= M * omega3[j] for j in range(N) if 0 < j < N - 1)
            self.constraints.c18 = m.addConstrs(
                u[j] <= M * (1 - omega3[j]) for j in range(N) if 0 < j < N - 1)
            self.constraints.c19 = m.addConstrs(
                delta[j] <= M * omega4[j] for j in range(N) if 0 < j < N - 1)
            self.constraints.c20 = m.addConstrs(
                v[j] <= M * (1 - omega4[j]) for j in range(N) if 0 < j < N - 1)

            self.constraints.c8a = m.addConstrs(
                delta[j] >= 0 for j in range(N))
            self.constraints.c8b = m.addConstrs(
                delta[j] <= deltabar for j in range(N))

        self.constraints.cuts = {}
        pass

    def _save_vars(self):
        K = self.data.K
        N = self.data.N
        # self.data.thetas.append(self.variables.theta.x)
        self.data.xs = np.zeros((K, N, N))
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    self.data.xs[k, i, j] = self.variables.x[k, i, j].x

        # print('\n' + '#' * 50)
        # print('save to folder = solution_Benders')
        # print('\n' + '#' * 50)
        # # save data as csv
        # for k in range(K):
        #     stacked = pd.DataFrame(self.data.xs[k,:,:])
        #     stacked.to_csv('solution_Benders/stacked{}.csv'.format(k))
        #
        #
        # # read csv
        # x_sol_relaxed = np.zeros((K,N,N))
        # for k in range(K):
        #      aa = pd.read_csv('solution_Benders/stacked{}.csv'.format(k))
        #      x_sol_relaxed[k,:,:] = aa.iloc[:, 1:]


# Subproblem
class Benders_Subproblem:
    def __init__(self, MP, run, node, vehicle, delta_val, gamma, chi):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(run, node, vehicle, delta_val, gamma, chi)
        self._build_model()
        self.data.MP = MP
        self.update_fixed_vars()

    def optimize(self):
        self.model.Params.InfUnbdInfo = 1
        self.model.optimize()
        qq = 0

    def _load_data(self, run, node, vehicle, delta_val, gamma, chi):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.gamma = gamma
        self.data.chi = chi
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.xs = []
        self.data.ys = []
        self.data.thetas = []

        data = loadmat(
            '/Users/vulcanyao/OneDrive - 南方科技大学/PDF Expert/Research Yao/code_VRPPD/[J3]Bilevel EVRP with time flexibility/J3-V1/DataMap/Vehicles_{}/RealMap_{}_test.mat'.format(run, node))
        self.data.Emax = data['Emax'][0][0]
        self.data.rmax = self.data.Emax
        self.data.e = data['e']
        self.data.T = 1e0 * data['T']
        self.data.M = int(data['M'])
        self.data.p = data['p'][0]
        self.data.g = data['g'][0]
        self.data.t = 1e0 * data['t'][0]
        self.data.c = data['c'][0]
        self.data.K = int(vehicle)
        self.data.N = int(data['N'])
        # self.data.alpha = 0.05
        # self.data.Q = 200
        self.data.deltabar = delta_val

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        K = self.data.K
        N = self.data.N
        deltabar = self.data.deltabar

        self.variables.theta = m.addVar(
            lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name='theta')
        self.variables.x = m.addVars(
            K, N, N, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="x")
        self.variables.omega1 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega1")
        self.variables.omega2 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega2")
        self.variables.omega3 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega3")
        self.variables.omega4 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega4")

        self.variables.eta = m.addVars(
            N, lb=0, vtype=gb.GRB.CONTINUOUS, name="eta")
        self.variables.epsilon = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="epsilon")
        self.variables.q = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="q")
        self.variables.u = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="u")
        self.variables.v = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="v")
        self.variables.zeta1 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta1")
        self.variables.zeta2 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta2")
        self.variables.delta = m.addVars(
            N, lb=0, ub=deltabar, vtype=gb.GRB.CONTINUOUS, name="delta")
        self.variables.tau = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="tau")

        m.update()

    def _build_objective(self):
        m = self.model
        N = self.data.N
        qq = 0
        for i in range(N):
            if i < N-1:
                if i > 0:
                    qq += self.variables.eta[i]
        m.setObjective(qq, gb.GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model
        K = self.data.K
        M = self.data.M
        N = self.data.N
        t = self.data.t
        T = self.data.T
        deltabar = self.data.deltabar
        gamma = self.data.gamma
        chi = self.data.chi

        x = self.variables.x
        omega1 = self.variables.omega1
        omega2 = self.variables.omega2
        omega3 = self.variables.omega3
        omega4 = self.variables.omega4
        tau = self.variables.tau
        delta = self.variables.delta
        u = self.variables.u
        v = self.variables.v
        eta = self.variables.eta
        epsilon = self.variables.epsilon
        zeta1 = self.variables.zeta1
        zeta2 = self.variables.zeta2
        q = self.variables.q

        self.constraints.c1 = m.addConstrs(
            x.sum('*', i, '*') <= 1 for i in range(N) if i != 0)
        self.constraints.c2 = m.addConstrs(
            x[k, i, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c3 = m.addConstrs(
            x[k, N - 1, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c4 = m.addConstrs(
            x[k, i, 0] == 0 for i in range(N) for k in range(K))
        # Flow conservation
        self.constraints.c5 = m.addConstrs(
            x.sum(k, 0, '*') - x.sum(k, '*', 0) == 1 for k in range(K))
        self.constraints.c6 = m.addConstrs(
            x.sum(k, N - 1, '*') - x.sum(k, '*', N - 1) == -1 for k in range(K))
        self.constraints.c7 = m.addConstrs(x.sum(k, i, '*') - x.sum(k, '*', i) == 0 for i in range(N - 1) if i != 0 for k
                                           in range(K))
        self.constraints.c8a = m.addConstrs(
            t[j] - tau[j] <= 0 for j in range(N))
        self.constraints.c8b = m.addConstrs(
            tau[j] - t[j]-delta[j] <= 0 for j in range(N))

        self.constraints.c9 = m.addConstrs(tau[j] >= tau[i] + T[i, j] - M*(
            1-x[k, i, j]) for j in range(N) for i in range(N) for k in range(K))
        self.constraints.c10 = m.addConstrs(eta[j] >= epsilon[j] + u[j]*deltabar - zeta1[j]
                                            * chi - zeta2[j]*chi - M*(1 - x.sum('*', '*', j)) for j in range(N) if 0 < j < N - 1)

        # KKT condition of lower level problem

        self.constraints.c11 = m.addConstrs(
            - gamma * zeta1[j] + gamma * zeta2[j] - q[j] + u[j] - v[j] == 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c12 = m.addConstrs(
            1 - zeta1[j] - zeta2[j] == 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c13a = m.addConstrs(
            0 <= epsilon[j] + gamma * delta[j] - chi for j in range(N) if 0 < j < N - 1)
        self.constraints.c13b = m.addConstrs(
            epsilon[j] + gamma * delta[j] - chi <= M * omega1[j] for j in range(N) if 0 < j < N - 1)

        self.constraints.c14 = m.addConstrs(
            zeta1[j] <= M * (1 - omega1[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c15a = m.addConstrs(
            0 <= epsilon[j] - gamma * delta[j] + chi for j in range(N) if 0 < j < N - 1)
        self.constraints.c15b = m.addConstrs(
            epsilon[j] - gamma * delta[j] + chi <= M * omega2[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c16 = m.addConstrs(
            zeta2[j] <= M * (1 - omega2[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c17 = m.addConstrs(
            deltabar - delta[j] <= M * omega3[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c18 = m.addConstrs(
            u[j] <= M * (1 - omega3[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c19 = m.addConstrs(
            delta[j] <= M * omega4[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c20 = m.addConstrs(
            v[j] <= M * (1 - omega4[j]) for j in range(N) if 0 < j < N - 1)

        self.constraints.fix_x = m.addConstrs(
            x[k, i, j] == 0 for k in range(K) for i in range(N) for j in range(N))
        self.constraints.fix_omega1 = m.addConstrs(
            omega1[i] == 0 for i in range(N))
        self.constraints.fix_omega2 = m.addConstrs(
            omega2[i] == 0 for i in range(N))
        self.constraints.fix_omega3 = m.addConstrs(
            omega3[i] == 0 for i in range(N))
        self.constraints.fix_omega4 = m.addConstrs(
            omega4[i] == 0 for i in range(N))

    def update_fixed_vars(self, MP=None):
        K = self.data.K
        N = self.data.N
        if MP is None:
            MP = self.data.MP
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    self.constraints.fix_omega1[j].rhs = MP.variables.omega1[j].x
                    self.constraints.fix_omega2[j].rhs = MP.variables.omega2[j].x
                    self.constraints.fix_omega3[j].rhs = MP.variables.omega3[j].x
                    self.constraints.fix_omega4[j].rhs = MP.variables.omega4[j].x
                    self.constraints.fix_x[k, i,
                                           j].rhs = MP.variables.x[k, i, j].x


# Subproblem MIP
class Benders_Subproblem_MIP:
    def __init__(self, MP, SP, run, node, vehicle, delta_val, gamma, chi):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(run, node, vehicle, delta_val, gamma, chi)
        self._build_model(MP, SP)
        self.data.MP = MP
        self.update_fixed_vars()

    def optimize(self):
        self.model.setParam('MIPGap', 0.001)
        self.model.optimize()
        qq = 0

    def _load_data(self, run, node, vehicle, delta_val, gamma, chi):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.gamma = gamma
        self.data.chi = chi
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.xs = []
        self.data.ys = []
        self.data.thetas = []

        data = loadmat(
            '/Users/vulcanyao/OneDrive - 南方科技大学/PDF Expert/Research Yao/code_VRPPD/[J3]Bilevel EVRP with time flexibility/J3-V1/DataMap/Vehicles_{}/RealMap_{}_test.mat'.format(run, node))
        self.data.Emax = data['Emax'][0][0]
        self.data.rmax = self.data.Emax
        self.data.e = data['e']
        self.data.T = 1e0 * data['T']
        self.data.M = int(data['M'])
        self.data.p = data['p'][0]
        self.data.g = data['g'][0]
        self.data.t = 1e0 * data['t'][0]
        self.data.c = data['c'][0]
        self.data.K = int(vehicle)
        self.data.N = int(data['N'])
        # self.data.alpha = 0.05
        # self.data.Q = 200
        self.data.deltabar = delta_val

    ###
    #   Model Building
    ###
    def _build_model(self, MP, SP):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective(MP, SP)
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        K = self.data.K
        N = self.data.N
        deltabar = self.data.deltabar

        self.variables.theta = m.addVar(
            lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name='theta')
        self.variables.x = m.addVars(K, N, N, vtype=gb.GRB.BINARY, name="x")
        self.variables.omega1 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega1")
        self.variables.omega2 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega2")
        self.variables.omega3 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega3")
        self.variables.omega4 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega4")

        self.variables.eta = m.addVars(
            N, lb=0, vtype=gb.GRB.CONTINUOUS, name="eta")
        self.variables.epsilon = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="epsilon")
        self.variables.q = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="q")
        # self.variables.r = m.addVars(N, K, vtype=gb.GRB.CONTINUOUS, name="r")
        # self.variables.E = m.addVars(N, K, vtype=gb.GRB.CONTINUOUS, name="E")
        self.variables.u = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="u")
        self.variables.v = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="v")
        self.variables.zeta1 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta1")
        self.variables.zeta2 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta2")
        self.variables.delta = m.addVars(
            N, lb=0, ub=deltabar, vtype=gb.GRB.CONTINUOUS, name="delta")
        self.variables.tau = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="tau")

        m.update()

    def _build_objective(self, MP, SP):
        m = self.model
        N = self.data.N
        K = self.data.K
        x = self.variables.x
        omega1 = self.variables.omega1
        omega2 = self.variables.omega2
        omega3 = self.variables.omega3
        omega4 = self.variables.omega4

        sens_x = 0
        sens_omega1 = 0
        sens_omega2 = 0
        sens_omega3 = 0
        sens_omega4 = 0
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    sens_x += (x[k, i, j] - MP.variables.x[k, i, j].x) * \
                        SP.constraints.fix_x[k, i, j].pi
                    sens_omega1 += (omega1[j] - MP.variables.omega1[j].x) * \
                        SP.constraints.fix_omega1[j].pi
                    sens_omega2 += (omega2[j] - MP.variables.omega2[j].x) * \
                        SP.constraints.fix_omega2[j].pi
                    sens_omega3 += (omega3[j] - MP.variables.omega3[j].x) * \
                        SP.constraints.fix_omega3[j].pi
                    sens_omega4 += (omega4[j] - MP.variables.omega4[j].x) * \
                        SP.constraints.fix_omega4[j].pi
        qq = 0
        for i in range(N):
            if i < N-1:
                if i > 0:
                    qq += self.variables.eta[i]
        m.setObjective(qq - sens_x - sens_omega1 - sens_omega2 -
                       sens_omega3 - sens_omega4, gb.GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model
        K = self.data.K
        M = self.data.M
        N = self.data.N
        t = self.data.t
        T = self.data.T
        deltabar = self.data.deltabar
        gamma = self.data.gamma
        chi = self.data.chi

        x = self.variables.x
        omega1 = self.variables.omega1
        omega2 = self.variables.omega2
        omega3 = self.variables.omega3
        omega4 = self.variables.omega4
        tau = self.variables.tau
        delta = self.variables.delta
        u = self.variables.u
        v = self.variables.v
        eta = self.variables.eta
        epsilon = self.variables.epsilon
        zeta1 = self.variables.zeta1
        zeta2 = self.variables.zeta2
        q = self.variables.q

        self.constraints.c1 = m.addConstrs(
            x.sum('*', i, '*') <= 1 for i in range(N) if i != 0)
        self.constraints.c2 = m.addConstrs(
            x[k, i, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c3 = m.addConstrs(
            x[k, N - 1, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c4 = m.addConstrs(
            x[k, i, 0] == 0 for i in range(N) for k in range(K))
        # Flow conservation
        self.constraints.c5 = m.addConstrs(
            x.sum(k, 0, '*') - x.sum(k, '*', 0) == 1 for k in range(K))
        self.constraints.c6 = m.addConstrs(
            x.sum(k, N - 1, '*') - x.sum(k, '*', N - 1) == -1 for k in range(K))
        self.constraints.c7 = m.addConstrs(x.sum(k, i, '*') - x.sum(k, '*', i) == 0 for i in range(N - 1) if i != 0 for k
                                           in range(K))
        self.constraints.c8a = m.addConstrs(
            t[j] - tau[j] <= 0 for j in range(N))
        self.constraints.c8b = m.addConstrs(
            tau[j] - t[j]-delta[j] <= 0 for j in range(N))

        self.constraints.c9 = m.addConstrs(tau[j] >= tau[i] + T[i, j] - M*(
            1-x[k, i, j]) for j in range(N) for i in range(N) for k in range(K))
        self.constraints.c10 = m.addConstrs(eta[j] >= epsilon[j] + u[j]*deltabar - zeta1[j]
                                            * chi - zeta2[j]*chi - M*(1 - x.sum('*', '*', j)) for j in range(N) if 0 < j < N - 1)

        # KKT condition of lower level problem
        self.constraints.c11 = m.addConstrs(
            - gamma * zeta1[j] + gamma * zeta2[j] - q[j] + u[j] - v[j] == 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c12 = m.addConstrs(
            1 - zeta1[j] - zeta2[j] == 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c13a = m.addConstrs(
            0 <= epsilon[j] + gamma * delta[j] - chi for j in range(N) if 0 < j < N - 1)
        self.constraints.c13b = m.addConstrs(
            epsilon[j] + gamma * delta[j] - chi <= M * omega1[j] for j in range(N) if 0 < j < N - 1)

        self.constraints.c14 = m.addConstrs(
            zeta1[j] <= M * (1 - omega1[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c15a = m.addConstrs(
            0 <= epsilon[j] - gamma * delta[j] + chi for j in range(N) if 0 < j < N - 1)
        self.constraints.c15b = m.addConstrs(
            epsilon[j] - gamma * delta[j] + chi <= M * omega2[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c16 = m.addConstrs(
            zeta2[j] <= M * (1 - omega2[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c17 = m.addConstrs(
            deltabar - delta[j] <= M * omega3[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c18 = m.addConstrs(
            u[j] <= M * (1 - omega3[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c19 = m.addConstrs(
            delta[j] <= M * omega4[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c20 = m.addConstrs(
            v[j] <= M * (1 - omega4[j]) for j in range(N) if 0 < j < N - 1)

        # self.constraints.fix_x = m.addConstrs(x[k, i, j] == 0 for k in range(K) for i in range(N) for j in range(N))
        # self.constraints.fix_omega1 = m.addConstrs(omega1[i] == 0  for i in range(N))
        # self.constraints.fix_omega2 = m.addConstrs(omega2[i] == 0  for i in range(N))
        # self.constraints.fix_omega3 = m.addConstrs(omega3[i] == 0  for i in range(N))
        # self.constraints.fix_omega4 = m.addConstrs(omega4[i] == 0  for i in range(N))

    def update_fixed_vars(self, MP=None):
        K = self.data.K
        N = self.data.N
        if MP is None:
            MP = self.data.MP
        # for k in range(K):
        #     for i in range(N):
        #         for j in range(N):
        #             self.constraints.fix_omega1[j].rhs = MP.variables.omega1[ j].x
        #             self.constraints.fix_omega2[j].rhs = MP.variables.omega2[ j].x
        #             self.constraints.fix_omega3[j].rhs = MP.variables.omega3[ j].x
        #             self.constraints.fix_omega4[j].rhs = MP.variables.omega4[ j].x
        #             self.constraints.fix_x[k, i, j].rhs = MP.variables.x[k, i, j].x


# Feasibility Subproblem
class Benders_Feasibility_Subproblem:
    def __init__(self, MP, run, node, vehicle, delta_val, gamma, chi):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(run, node, vehicle, delta_val, gamma, chi)
        self._build_model()
        self.data.MP = MP
        self.update_fixed_vars()

    def optimize(self):
        self.model.Params.InfUnbdInfo = 1
        self.model.optimize()
        qq = 0

    def _load_data(self, run, node, vehicle, delta_val, gamma, chi):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.gamma = gamma
        self.data.chi = chi
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.xs = []
        self.data.ys = []
        self.data.thetas = []

        data = loadmat(
            '/Users/vulcanyao/OneDrive - 南方科技大学/PDF Expert/Research Yao/code_VRPPD/[J3]Bilevel EVRP with time flexibility/J3-V1/DataMap/Vehicles_{}/RealMap_{}_test.mat'.format(run, node))
        self.data.Emax = data['Emax'][0][0]
        self.data.rmax = self.data.Emax
        self.data.e = data['e']
        self.data.T = 1e0 * data['T']
        self.data.M = int(data['M'])
        self.data.p = data['p'][0]
        self.data.g = data['g'][0]
        self.data.t = 1e0 * data['t'][0]
        self.data.c = data['c'][0]
        self.data.K = int(vehicle)
        self.data.N = int(data['N'])
        # self.data.alpha = 0.05
        # self.data.Q = 200
        self.data.deltabar = delta_val

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        K = self.data.K
        N = self.data.N
        deltabar = self.data.deltabar

        self.variables.theta = m.addVar(
            lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name='theta')
        self.variables.x = m.addVars(
            K, N, N, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="x")
        self.variables.omega1 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega1")
        self.variables.omega2 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega2")
        self.variables.omega3 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega3")
        self.variables.omega4 = m.addVars(
            N, lb=0, ub=1,  vtype=gb.GRB.CONTINUOUS, name="omega4")
        self.variables.eta = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="eta")
        self.variables.epsilon = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="epsilon")
        self.variables.q = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="q")
        # self.variables.r = m.addVars(N, K, vtype=gb.GRB.CONTINUOUS, name="r")
        # self.variables.E = m.addVars(N, K, vtype=gb.GRB.CONTINUOUS, name="E")
        self.variables.u = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="u")
        self.variables.v = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="v")
        self.variables.zeta1 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta1")
        self.variables.zeta2 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta2")
        self.variables.delta = m.addVars(
            N, lb=0, ub=deltabar, vtype=gb.GRB.CONTINUOUS, name="delta")
        self.variables.tau = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="tau")

        self.variables.Sthree = m.addVars(
            K, N, N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sthree")  # Slack var.for feasibility

        self.variables.Sone_1 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_1")
        self.variables.Sone_2 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_2")
        self.variables.Sone_3 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_3")
        self.variables.Sone_4 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_4")
        self.variables.Sone_5 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_5")
        self.variables.Sone_6 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_6")
        self.variables.Sone_7 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_7")
        self.variables.Sone_8 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_8")
        self.variables.Sone_9 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_9")
        self.variables.Sone_10 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_10")
        self.variables.Sone_11 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_11")
        self.variables.Sone_12 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_12")
        self.variables.Sone_13 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_13")
        self.variables.Sone_14 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_14")
        self.variables.Sone_15 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_15")

        self.variables.Sone_m1 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m1")
        self.variables.Sone_m2 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m2")
        self.variables.Sone_m3 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m3")
        self.variables.Sone_m4 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m4")
        self.variables.Sone_m5 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m5")
        self.variables.Sone_m6 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m6")
        self.variables.Sone_m7 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m7")
        self.variables.Sone_m8 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m8")

        m.update()

    def _build_objective(self):
        m = self.model
        N = self.data.N
        vars = self.variables
        m.setObjective(vars.Sthree.sum('*', '*', '*') + vars.Sone_1.sum('*') + vars.Sone_2.sum('*') + vars.Sone_3.sum('*')
                       + vars.Sone_4.sum('*') + vars.Sone_5.sum('*') +
                       vars.Sone_6.sum('*') + vars.Sone_7.sum('*')
                       + vars.Sone_8.sum('*') + vars.Sone_9.sum('*') +
                       vars.Sone_10.sum('*') + vars.Sone_11.sum('*')
                       + vars.Sone_12.sum('*') + vars.Sone_13.sum('*') +
                       vars.Sone_14.sum('*') + vars.Sone_15.sum('*')
                       + vars.Sone_m1.sum('*') + vars.Sone_m2.sum('*') +
                       vars.Sone_m3.sum('*') + vars.Sone_m4.sum('*')
                       + vars.Sone_m5.sum('*') + vars.Sone_m6.sum('*') +
                       vars.Sone_m7.sum('*') + vars.Sone_m8.sum('*'),
                       gb.GRB.MINIMIZE)  # - vars.Sone_3[0] -vars.Sone_3[N-1]

    def _build_constraints(self):
        m = self.model
        K = self.data.K
        M = self.data.M
        N = self.data.N
        t = self.data.t
        T = self.data.T
        deltabar = self.data.deltabar
        gamma = self.data.gamma
        chi = self.data.chi

        x = self.variables.x
        omega1 = self.variables.omega1
        omega2 = self.variables.omega2
        omega3 = self.variables.omega3
        omega4 = self.variables.omega4
        tau = self.variables.tau
        delta = self.variables.delta
        u = self.variables.u
        v = self.variables.v
        eta = self.variables.eta
        epsilon = self.variables.epsilon
        zeta1 = self.variables.zeta1
        zeta2 = self.variables.zeta2
        q = self.variables.q
        Sthree = self.variables.Sthree
        Sone_1 = self.variables.Sone_1
        Sone_2 = self.variables.Sone_2
        Sone_3 = self.variables.Sone_3
        Sone_4 = self.variables.Sone_4
        Sone_5 = self.variables.Sone_5
        Sone_6 = self.variables.Sone_6
        Sone_7 = self.variables.Sone_7
        Sone_8 = self.variables.Sone_8
        Sone_9 = self.variables.Sone_9
        Sone_10 = self.variables.Sone_10
        Sone_11 = self.variables.Sone_11
        Sone_12 = self.variables.Sone_12
        Sone_13 = self.variables.Sone_13
        Sone_14 = self.variables.Sone_14
        Sone_15 = self.variables.Sone_15

        Sone_m1 = self.variables.Sone_m1
        Sone_m2 = self.variables.Sone_m2
        Sone_m3 = self.variables.Sone_m3
        Sone_m4 = self.variables.Sone_m4
        Sone_m5 = self.variables.Sone_m5
        Sone_m6 = self.variables.Sone_m6
        Sone_m7 = self.variables.Sone_m7
        Sone_m8 = self.variables.Sone_m8

        self.constraints.c1 = m.addConstrs(
            x.sum('*', i, '*') <= 1 for i in range(N) if i != 0)
        self.constraints.c2 = m.addConstrs(
            x[k, i, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c3 = m.addConstrs(
            x[k, N - 1, i] == 0 for i in range(N) for k in range(K))
        self.constraints.c4 = m.addConstrs(
            x[k, i, 0] == 0 for i in range(N) for k in range(K))

        # Flow conservation
        self.constraints.c5 = m.addConstrs(
            x.sum(k, 0, '*') - x.sum(k, '*', 0) == 1 for k in range(K))
        self.constraints.c6 = m.addConstrs(
            x.sum(k, N - 1, '*') - x.sum(k, '*', N - 1) == -1 for k in range(K))
        self.constraints.c7 = m.addConstrs(x.sum(k, i, '*') - x.sum(k, '*', i) == 0 for i in range(N - 1) if i != 0 for k
                                           in range(K))
        self.constraints.c8a = m.addConstrs(
            t[j] - tau[j] - Sone_1[j] <= 0 for j in range(N))
        self.constraints.c8b = m.addConstrs(
            tau[j] - t[j]-delta[j] - Sone_2[j] <= 0 for j in range(N))

        self.constraints.c9 = m.addConstrs(tau[j] + Sthree[k, i, j] >= tau[i] + T[i, j] - M*(
            1-x[k, i, j]) for j in range(N) for i in range(N) for k in range(K))
        self.constraints.c10 = m.addConstrs(eta[j] + Sone_3[j] >= epsilon[j] + u[j]*deltabar - zeta1[j]
                                            * chi - zeta2[j]*chi - M*(1 - x.sum('*', '*', j)) for j in range(N) if 0 < j < N - 1)

        # KKT condition of lower level problem
        self.constraints.c11a = m.addConstrs(- gamma * zeta1[j] + gamma * zeta2[j] -
                                             q[j] + u[j] - v[j] - Sone_4[j] <= 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c11b = m.addConstrs(- gamma * zeta1[j] + gamma * zeta2[j] -
                                             q[j] + u[j] - v[j] + Sone_5[j] >= 0 for j in range(N) if 0 < j < N - 1)

        self.constraints.c12a = m.addConstrs(
            1 - zeta1[j] - zeta2[j] - Sone_6[j] <= 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c12b = m.addConstrs(
            1 - zeta1[j] - zeta2[j] + Sone_7[j] >= 0 for j in range(N) if 0 < j < N - 1)

        # m=1, Slack var. for piecewise inconv. func.
        self.constraints.c13a = m.addConstrs(
            0 <= epsilon[j] + gamma * delta[j] - chi + Sone_m1[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c13b = m.addConstrs(
            epsilon[j] + gamma * delta[j] - chi <= Sone_m2[j] + M * omega1[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c14a = m.addConstrs(
            0 <= zeta1[j] + Sone_m3[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c14b = m.addConstrs(
            zeta1[j] <= Sone_m4[j] + M * (1 - omega1[j]) for j in range(N) if 0 < j < N - 1)
        # m=2, Slack var. for piecewise inconv. func.
        self.constraints.c15a = m.addConstrs(
            0 <= epsilon[j] - gamma * delta[j] + chi + Sone_m5[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c15b = m.addConstrs(
            epsilon[j] - gamma * delta[j] + chi <= Sone_m6[j] + M * omega2[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c16a = m.addConstrs(
            0 <= zeta2[j] + Sone_m7[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c16b = m.addConstrs(
            zeta2[j] <= Sone_m8[j] + M * (1 - omega2[j]) for j in range(N) if 0 < j < N - 1)

        # Linearizatio of comple. cons.
        self.constraints.c17a = m.addConstrs(
            0 <= deltabar - delta[j] + Sone_8[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c17b = m.addConstrs(
            deltabar - delta[j] <= Sone_9[j] + M * omega3[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c18a = m.addConstrs(
            0 <= u[j] + Sone_10[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c18b = m.addConstrs(
            u[j] <= Sone_11[j] + M * (1 - omega3[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c19a = m.addConstrs(
            0 <= delta[j] + Sone_12[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c19b = m.addConstrs(
            delta[j] <= Sone_13[j] + M * omega4[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c20a = m.addConstrs(
            0 <= v[j] + Sone_14[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c20b = m.addConstrs(
            v[j] <= Sone_15[j] + M * (1 - omega4[j]) for j in range(N) if 0 < j < N - 1)

        self.constraints.fix_x = m.addConstrs(
            x[k, i, j] == 0 for k in range(K) for i in range(N) for j in range(N))
        self.constraints.fix_omega1 = m.addConstrs(
            omega1[i] == 0 for i in range(N))
        self.constraints.fix_omega2 = m.addConstrs(
            omega2[i] == 0 for i in range(N))
        self.constraints.fix_omega3 = m.addConstrs(
            omega3[i] == 0 for i in range(N))
        self.constraints.fix_omega4 = m.addConstrs(
            omega4[i] == 0 for i in range(N))

    def update_fixed_vars(self, MP=None):
        K = self.data.K
        N = self.data.N
        if MP is None:
            MP = self.data.MP
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    self.constraints.fix_omega1[j].rhs = MP.variables.omega1[j].x
                    self.constraints.fix_omega2[j].rhs = MP.variables.omega2[j].x
                    self.constraints.fix_omega3[j].rhs = MP.variables.omega3[j].x
                    self.constraints.fix_omega4[j].rhs = MP.variables.omega4[j].x
                    self.constraints.fix_x[k, i,
                                           j].rhs = MP.variables.x[k, i, j].x


# Feasibility Subproblem MIP
class Benders_Feasibility_Subproblem_MIP:
    def __init__(self, MP, run, node, vehicle, delta_val, gamma, chi):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(run, node, vehicle, delta_val, gamma, chi)
        self._build_model()
        self.data.MP = MP
        self.update_fixed_vars()

    def optimize(self):
        self.model.Params.InfUnbdInfo = 1
        self.model.setParam('MIPGap', 0.001)
        self.model.optimize()
        qq = 0

    def _load_data(self, run, node, vehicle, delta_val, gamma, chi):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.gamma = gamma
        self.data.chi = chi
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.xs = []
        self.data.ys = []
        self.data.thetas = []

        data = loadmat(
            '/Users/vulcanyao/OneDrive - 南方科技大学/PDF Expert/Research Yao/code_VRPPD/[J3]Bilevel EVRP with time flexibility/J3-V1/DataMap/Vehicles_{}/RealMap_{}_test.mat'.format(run, node))
        self.data.Emax = data['Emax'][0][0]
        self.data.rmax = self.data.Emax
        self.data.e = data['e']
        self.data.T = 1e0 * data['T']
        self.data.M = int(data['M'])
        self.data.p = data['p'][0]
        self.data.g = data['g'][0]
        self.data.t = 1e0 * data['t'][0]
        self.data.c = data['c'][0]
        self.data.K = int(vehicle)
        self.data.N = int(data['N'])
        # self.data.alpha = 0.05
        # self.data.Q = 200
        self.data.deltabar = delta_val

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        K = self.data.K
        N = self.data.N
        deltabar = self.data.deltabar

        self.variables.x = m.addVars(K, N, N, vtype=gb.GRB.BINARY, name="x")
        self.variables.omega1 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega1")
        self.variables.omega2 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega2")
        self.variables.omega3 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega3")
        self.variables.omega4 = m.addVars(
            N,  vtype=gb.GRB.BINARY, name="omega4")

        self.variables.theta = m.addVar(
            lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name='theta')

        self.variables.eta = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="eta")
        self.variables.epsilon = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="epsilon")
        self.variables.q = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="q")
        # self.variables.r = m.addVars(N, K, vtype=gb.GRB.CONTINUOUS, name="r")
        # self.variables.E = m.addVars(N, K, vtype=gb.GRB.CONTINUOUS, name="E")
        self.variables.u = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="u")
        self.variables.v = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="v")
        self.variables.zeta1 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta1")
        self.variables.zeta2 = m.addVars(
            N, vtype=gb.GRB.CONTINUOUS, name="zeta2")
        self.variables.delta = m.addVars(
            N, lb=0, ub=deltabar, vtype=gb.GRB.CONTINUOUS, name="delta")
        self.variables.tau = m.addVars(N, vtype=gb.GRB.CONTINUOUS, name="tau")

        self.variables.Sthree = m.addVars(
            K, N, N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sthree")  # Slack var.for feasibility

        self.variables.Sone_1 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_1")
        self.variables.Sone_2 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_2")
        self.variables.Sone_3 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_3")
        self.variables.Sone_4 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_4")
        self.variables.Sone_5 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_5")
        self.variables.Sone_6 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_6")
        self.variables.Sone_7 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_7")
        self.variables.Sone_8 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_8")
        self.variables.Sone_9 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_9")
        self.variables.Sone_10 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_10")
        self.variables.Sone_11 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_11")
        self.variables.Sone_12 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_12")
        self.variables.Sone_13 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_13")
        self.variables.Sone_14 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_14")
        self.variables.Sone_15 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_15")

        self.variables.Sone_m1 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m1")
        self.variables.Sone_m2 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m2")
        self.variables.Sone_m3 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m3")
        self.variables.Sone_m4 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m4")
        self.variables.Sone_m5 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m5")
        self.variables.Sone_m6 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m6")
        self.variables.Sone_m7 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m7")
        self.variables.Sone_m8 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_m8")

        self.variables.Stwo_1 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_1")
        self.variables.Stwo_2 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_2")
        self.variables.Stwo_3 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_3")
        self.variables.Stwo_4 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_4")
        self.variables.Stwo_5 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_5")
        self.variables.Stwo_6 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_6")
        self.variables.Stwo_7 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_7")
        self.variables.Stwo_8 = m.addVars(
            N, K, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Stwo_8")

        self.variables.Sone_16 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_16")

        self.variables.Sone_k1 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_k1")
        self.variables.Sone_k2 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_k2")
        self.variables.Sone_k3 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_k3")
        self.variables.Sone_k4 = m.addVars(
            N, lb=0, ub=gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="Sone_k4")

        m.update()

    def _build_objective(self):
        m = self.model
        N = self.data.N
        K = self.data.K
        vars = self.variables
        x = vars.x
        omega1 = vars.omega1
        omega2 = vars.omega2
        omega3 = vars.omega3
        omega4 = vars.omega4

        sens_x = 0
        sens_omega1 = 0
        sens_omega2 = 0
        sens_omega3 = 0
        sens_omega4 = 0
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    sens_x += (x[k, i, j] - x[k, i, j].x) * \
                        self.feasibility_submodel.constraints.fix_x[k, i, j].pi
                    sens_omega1 += (omega1[j] - omega1[j].x) * \
                        self.feasibility_submodel.constraints.fix_omega1[j].pi
                    sens_omega2 += (omega2[j] - omega2[j].x) * \
                        self.feasibility_submodel.constraints.fix_omega2[j].pi
                    sens_omega3 += (omega3[j] - omega3[j].x) * \
                        self.feasibility_submodel.constraints.fix_omega3[j].pi
                    sens_omega4 += (omega4[j] - omega4[j].x) * \
                        self.feasibility_submodel.constraints.fix_omega4[j].pi

        m.setObjective(vars.Sthree.sum('*', '*', '*') + vars.Sone_1.sum('*') + vars.Sone_2.sum('*') + vars.Sone_3.sum('*')
                       + vars.Sone_4.sum('*') + vars.Sone_5.sum('*') +
                       vars.Sone_6.sum('*') + vars.Sone_7.sum('*')
                       + vars.Sone_8.sum('*') + vars.Sone_9.sum('*') +
                       vars.Sone_10.sum('*') + vars.Sone_11.sum('*')
                       + vars.Sone_12.sum('*') + vars.Sone_13.sum('*') +
                       vars.Sone_14.sum('*') + vars.Sone_15.sum('*')
                       + vars.Sone_m1.sum('*') + vars.Sone_m2.sum('*') +
                       vars.Sone_m3.sum('*') + vars.Sone_m4.sum('*')
                       + vars.Sone_m5.sum('*') + vars.Sone_m6.sum('*') +
                       vars.Sone_m7.sum('*') + vars.Sone_m8.sum('*')
                       + vars.Stwo_1.sum('*', '*') + vars.Stwo_2.sum('*', '*') +
                       vars.Stwo_3.sum('*', '*') + vars.Stwo_4.sum('*', '*')
                       + vars.Stwo_5.sum('*', '*') + vars.Stwo_6.sum('*', '*') +
                       vars.Stwo_7.sum('*', '*') + vars.Stwo_8.sum('*', '*')
                       + vars.Sone_16.sum('*') + vars.Sone_k1.sum('*') + vars.Sone_k2.sum(
                           '*') + vars.Sone_k3.sum('*') + vars.Sone_k4.sum('*')
                       - sens_x - sens_omega1 - sens_omega2 - sens_omega3 - sens_omega4, gb.GRB.MINIMIZE)
        #- vars.Sone_3[0] -vars.Sone_3[N-1]

    def _build_constraints(self):
        m = self.model
        K = self.data.K
        M = self.data.M
        N = self.data.N
        t = self.data.t
        T = self.data.T
        deltabar = self.data.deltabar
        gamma = self.data.gamma
        chi = self.data.chi

        x = self.variables.x
        omega1 = self.variables.omega1
        omega2 = self.variables.omega2
        omega3 = self.variables.omega3
        omega4 = self.variables.omega4
        tau = self.variables.tau
        delta = self.variables.delta
        u = self.variables.u
        v = self.variables.v
        eta = self.variables.eta
        epsilon = self.variables.epsilon
        zeta1 = self.variables.zeta1
        zeta2 = self.variables.zeta2
        q = self.variables.q
        Sthree = self.variables.Sthree
        Sone_1 = self.variables.Sone_1
        Sone_2 = self.variables.Sone_2
        Sone_3 = self.variables.Sone_3
        Sone_4 = self.variables.Sone_4
        Sone_5 = self.variables.Sone_5
        Sone_6 = self.variables.Sone_6
        Sone_7 = self.variables.Sone_7
        Sone_8 = self.variables.Sone_8
        Sone_9 = self.variables.Sone_9
        Sone_10 = self.variables.Sone_10
        Sone_11 = self.variables.Sone_11
        Sone_12 = self.variables.Sone_12
        Sone_13 = self.variables.Sone_13
        Sone_14 = self.variables.Sone_14
        Sone_15 = self.variables.Sone_15

        Sone_m1 = self.variables.Sone_m1
        Sone_m2 = self.variables.Sone_m2
        Sone_m3 = self.variables.Sone_m3
        Sone_m4 = self.variables.Sone_m4
        Sone_m5 = self.variables.Sone_m5
        Sone_m6 = self.variables.Sone_m6
        Sone_m7 = self.variables.Sone_m7
        Sone_m8 = self.variables.Sone_m8

        Stwo_1 = self.variables.Stwo_1
        Stwo_2 = self.variables.Stwo_2
        Stwo_3 = self.variables.Stwo_3
        Stwo_4 = self.variables.Stwo_4
        Stwo_5 = self.variables.Stwo_5
        Stwo_6 = self.variables.Stwo_6
        Stwo_7 = self.variables.Stwo_7
        Stwo_8 = self.variables.Stwo_8

        Sone_16 = self.variables.Sone_16

        Sone_k1 = self.variables.Sone_k1
        Sone_k2 = self.variables.Sone_k2
        Sone_k3 = self.variables.Sone_k3
        Sone_k4 = self.variables.Sone_k4

        # Flow cons
        self.constraints.c1 = m.addConstrs(
            x.sum('*', i, '*') <= 1 + Sone_16[i] for i in range(N) if i != 0)
        self.constraints.c2a = m.addConstrs(
            x[k, i, i] <= 0 + Stwo_1[i, k] for i in range(N) for k in range(K))
        self.constraints.c2b = m.addConstrs(
            0 + Stwo_2[i, k] <= x[k, i, i] for i in range(N) for k in range(K))

        self.constraints.c3a = m.addConstrs(
            x[k, N - 1, i] <= 0 + Stwo_3[i, k] for i in range(N) for k in range(K))
        self.constraints.c3b = m.addConstrs(
            0 + Stwo_4[i, k] <= x[k, N - 1, i] for i in range(N) for k in range(K))

        self.constraints.c4a = m.addConstrs(
            x[k, i, 0] <= 0 + Stwo_5[i, k] for i in range(N) for k in range(K))
        self.constraints.c4b = m.addConstrs(
            0 + Stwo_6[i, k] <= x[k, i, 0] for i in range(N) for k in range(K))

        self.constraints.c5a = m.addConstrs(
            x.sum(k, 0, '*') - x.sum(k, '*', 0) <= 1 + Sone_k1[k] for k in range(K))
        self.constraints.c5b = m.addConstrs(
            1 + Sone_k2[k] <= x.sum(k, 0, '*') - x.sum(k, '*', 0) for k in range(K))

        self.constraints.c6a = m.addConstrs(
            x.sum(k, N - 1, '*') - x.sum(k, '*', N - 1) <= -1 + Sone_k3[k] for k in range(K))
        self.constraints.c6b = m.addConstrs(-1 + Sone_k4[k] <= x.sum(
            k, N - 1, '*') - x.sum(k, '*', N - 1) for k in range(K))

        self.constraints.c7a = m.addConstrs(x.sum(k, i, '*') - x.sum(k, '*', i) <= 0 + Stwo_7[i, k] for i in range(N - 1) if i != 0 for k
                                            in range(K))
        self.constraints.c7b = m.addConstrs(0 + Stwo_8[i, k] <= x.sum(k, i, '*') - x.sum(k, '*', i) for i in range(N - 1) if i != 0 for k
                                            in range(K))

        # Time cons.
        self.constraints.c8a = m.addConstrs(
            t[j] - tau[j] - Sone_1[j] <= 0 for j in range(N))
        self.constraints.c8b = m.addConstrs(
            tau[j] - t[j]-delta[j] - Sone_2[j] <= 0 for j in range(N))

        self.constraints.c9 = m.addConstrs(tau[j] + Sthree[k, i, j] >= tau[i] + T[i, j] - M*(
            1-x[k, i, j]) for j in range(N) for i in range(N) for k in range(K))
        self.constraints.c10 = m.addConstrs(eta[j] + Sone_3[j] >= epsilon[j] + u[j]*deltabar - zeta1[j]
                                            * chi - zeta2[j]*chi - M*(1 - x.sum('*', '*', j)) for j in range(N) if 0 < j < N - 1)

        # KKT condition of lower level problem
        self.constraints.c11a = m.addConstrs(- gamma * zeta1[j] + gamma * zeta2[j] -
                                             q[j] + u[j] - v[j] - Sone_4[j] <= 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c11b = m.addConstrs(- gamma * zeta1[j] + gamma * zeta2[j] -
                                             q[j] + u[j] - v[j] + Sone_5[j] >= 0 for j in range(N) if 0 < j < N - 1)

        self.constraints.c12a = m.addConstrs(
            1 - zeta1[j] - zeta2[j] - Sone_6[j] <= 0 for j in range(N) if 0 < j < N - 1)
        self.constraints.c12b = m.addConstrs(
            1 - zeta1[j] - zeta2[j] + Sone_7[j] >= 0 for j in range(N) if 0 < j < N - 1)

        # m=1, Slack var. for piecewise inconv. func.
        self.constraints.c13a = m.addConstrs(
            0 <= epsilon[j] + gamma * delta[j] - chi + Sone_m1[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c13b = m.addConstrs(
            epsilon[j] + gamma * delta[j] - chi <= Sone_m2[j] + M * omega1[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c14a = m.addConstrs(
            0 <= zeta1[j] + Sone_m3[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c14b = m.addConstrs(
            zeta1[j] <= Sone_m4[j] + M * (1 - omega1[j]) for j in range(N) if 0 < j < N - 1)
        # m=2, Slack var. for piecewise inconv. func.
        self.constraints.c15a = m.addConstrs(
            0 <= epsilon[j] - gamma * delta[j] + chi + Sone_m5[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c15b = m.addConstrs(
            epsilon[j] - gamma * delta[j] + chi <= Sone_m6[j] + M * omega2[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c16a = m.addConstrs(
            0 <= zeta2[j] + Sone_m7[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c16b = m.addConstrs(
            zeta2[j] <= Sone_m8[j] + M * (1 - omega2[j]) for j in range(N) if 0 < j < N - 1)

        # Linearizatio of comple. cons.
        self.constraints.c17a = m.addConstrs(
            0 <= deltabar - delta[j] + Sone_8[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c17b = m.addConstrs(
            deltabar - delta[j] <= Sone_9[j] + M * omega3[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c18a = m.addConstrs(
            0 <= u[j] + Sone_10[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c18b = m.addConstrs(
            u[j] <= Sone_11[j] + M * (1 - omega3[j]) for j in range(N) if 0 < j < N - 1)
        self.constraints.c19a = m.addConstrs(
            0 <= delta[j] + Sone_12[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c19b = m.addConstrs(
            delta[j] <= Sone_13[j] + M * omega4[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c20a = m.addConstrs(
            0 <= v[j] + Sone_14[j] for j in range(N) if 0 < j < N - 1)
        self.constraints.c20b = m.addConstrs(
            v[j] <= Sone_15[j] + M * (1 - omega4[j]) for j in range(N) if 0 < j < N - 1)

        # self.constraints.fix_x = m.addConstrs(x[k, i, j] == 0 for k in range(K) for i in range(N) for j in range(N))
        # self.constraints.fix_omega1 = m.addConstrs(omega1[i] == 0  for i in range(N))
        # self.constraints.fix_omega2 = m.addConstrs(omega2[i] == 0  for i in range(N))
        # self.constraints.fix_omega3 = m.addConstrs(omega3[i] == 0  for i in range(N))
        # self.constraints.fix_omega4 = m.addConstrs(omega4[i] == 0  for i in range(N))

    def update_fixed_vars(self, MP=None):
        K = self.data.K
        N = self.data.N
        if MP is None:
            MP = self.data.MP
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    self.constraints.fix_omega1[j].rhs = MP.variables.omega1[j].x
                    self.constraints.fix_omega2[j].rhs = MP.variables.omega2[j].x
                    self.constraints.fix_omega3[j].rhs = MP.variables.omega3[j].x
                    self.constraints.fix_omega4[j].rhs = MP.variables.omega4[j].x
                    self.constraints.fix_x[k, i,
                                           j].rhs = MP.variables.x[k, i, j].x


if __name__ == '__main__':
    gamma0 = np.array([0.05, 0.01])
    chi0 = np.array([0.01, 0.01])
    deltabar0 = np.array([1, 1])
    # tlimit = 3600 * 2
    map = 100
    vehicle = 5
    index = 1
    runmax = 1
    benders_gap = 1e-4
    EPI_index = 0
    # ************************
    IteNum_MP_LP = np.arange(start=1, stop=2, step=1)
    # IteNum_MP_LP = np.array([1,2,3,4,5,6,7])

    for ci in range(1):
        chi = chi0[ci]
        for gam in range(1):
            gamma = gamma0[gam]
            for del1 in range(1):
                deltabar = deltabar0[del1]
                for i in range(index):
                    for run in range(runmax):
                        node = 21 + 2 * i
                        run += 2
                        ini_time = time.time()
                        m = Benders_Master(
                            benders_gap, run, node, vehicle, deltabar, gamma, chi, EPI_index, IteNum_MP_LP)
                        m.optimize()
                        end_time = time.time()
                        Runtime = end_time - ini_time

                        biterm = 0
                        for j in range(node):
                            if m.submodel.variables.eta[j].x != 0:
                                biterm += 1

                        # if EPI_index == 1:
                        #     with open(r'solution_Benders/BVRP_deltabar_{}_gamma_{}_chi_{}_gap_{}'
                        #               r'/Server_Sol_BDD_EPI_K_{}.csv'.format((deltabar), (gamma),chi, benders_gap,vehicle),
                        #               mode='a') as f:
                        #         Sol = writer(f, delimiter='\t', lineterminator='\n')
                        #         Sol.writerow(
                        #             ['sol_map{}_N{}_K{}_run{}'.format(map, node, vehicle, run), '%.2f' % (Runtime),
                        #              '%.8f' % (m.submodel.model.ObjVal), '%.4f ' % (m.data.lb), '%.4f ' % (m.data.ub),
                        #              '%i ' % (len(m.data.cutlist))])
                        #         f.close()
                        # elif EPI_index == 0:
                        #     with open(r'solution_Benders/BVRP_deltabar_{}_gamma_{}_chi_{}_gap_{}'
                        #               r'/Server_Sol_BDD_no_EPI_K_{}_Gamma.csv'.format( (deltabar), (gamma),chi, benders_gap,vehicle),
                        #               mode='a') as f:
                        #         Sol = writer(f, delimiter='\t', lineterminator='\n')
                        #         Sol.writerow(
                        #             ['sol_map{}_N{}_K{}_run{}'.format(map, node, vehicle, run), '%.2f' % (Runtime),
                        #              '%.8f' % (m.submodel.model.ObjVal), '%.4f ' % (m.data.lb),
                        #              '%.4f ' % (m.data.ub),'%.4f ' % (m.data.ub-m.submodel.model.ObjVal),
                        #              '%i ' % (len(m.data.cutlist)),'%i'%(biterm)  ])
                        #         f.close()
