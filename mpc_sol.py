import numpy as np
import casadi as ca
import dynamics as dyn
from dataclasses import dataclass

"""
x : quat0(0:4), r0(4:7), qm(7:13), u0(13:19), um(19:25) 
y : quat_ee(0:4), r_ee(4:7), t_ee(7:13)
"""

@dataclass
class MPCParams:
    dt: float = 1
    N: int = 10
    Q_elem: np.array = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
    Q: ca.DM = ca.DM(np.diag(Q_elem))
    R: ca.DM = ca.DM(np.eye(6))
    Qf: ca.DM = ca.DM(np.diag(Q_elem))

    u_min: ca.DM = ca.DM(-np.ones(6)*50)
    u_max: ca.DM = ca.DM(np.ones(6)*50)
    
    angle_min: ca.DM = ca.DM(-np.pi)
    angle_max: ca.DM = ca.DM(np.pi)
    
    

class MPCSolver:
    def __init__(self, dt=1, N=10):
        self.dt = dt
        self.cfg = MPCParams()
        # defnine dynamics model
        # quat0 = [q1, q2, q3, q0]
        self.x = ca.MX.sym('x', 25) 
        self.u = ca.MX.sym('u', 6)
        x_dot = dyn.dynamics(self.x, self.u)
        self.f = ca.Function('f', [self.x, self.u], [x_dot], {
            'jit': True,
            'jit_options': {'compiler': 'gcc', 'verbose': True}
        })
        y = dyn.output(self.x)
        self.h = ca.Function('h', [self.x], [y], {
            'jit': False,
            'jit_options': {'compiler': 'gcc', 'verbose': True}
        })
        
        # define state and control variables
        self.X = ca.MX.sym('X', 25, N+1) # state 
        self.U = ca.MX.sym('U', 6, N)
        # define initial and reference states 
        self.X0 = ca.MX.sym('X0', 25)
        self.Y_ref = ca.MX.sym('X_ref', 13) 
        
        # define cost function
        self.obj = 0
        self._set_objective()
        
        # set constraints
        self.g = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self._set_constraints()
        
        X_flat = ca.reshape(self.X, (25*(self.cfg.N + 1), 1))
        U_flat = ca.reshape(self.U, ( 6* self.cfg.N     , 1))
        opt_vars = ca.vertcat(X_flat, U_flat)
        
        nlp_prob = {'f': self.obj, 'x': opt_vars, 'g': ca.vertcat(*self.g), 'p': ca.vertcat(self.X0, self.Y_ref)}
        opts = {'ipopt.print_level': 0, 'ipopt.max_iter': 1000, 'ipopt.tol': 1e-8, 
                'ipopt.acceptable_obj_change_tol': 1e-6, 'jit': False,
                'jit_options': {'compiler': 'gcc', 'verbose': True},}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
    def solve(self, x0, y_ref):
        # set initial guess
        x0_guess = ca.vertcat(ca.repmat(x0, self.cfg.N+1, 1), ca.repmat(ca.DM.zeros(6), self.cfg.N,1))
        args = {
            'lbx': self.lbx,
            'ubx': self.ubx,
            'lbg': ca.vertcat(*self.lbg),
            'ubg': ca.vertcat(*self.ubg),
            'x0': x0_guess,  
            'p': ca.vertcat(x0, y_ref)
        }
        sol = self.solver(**args)
        # extract the solution
        u_sol = ca.reshape(sol['x'][25*(self.cfg.N+1):], (6, self.cfg.N))
        
        return u_sol
        
    def _set_objective(self):
        # cost function
        q_unit = ca.DM([0, 0, 0, 1])
        q_ref = self.Y_ref[:4]
        Y_res_ref = self.Y_ref[4:]
        for k in range(self.cfg.N):
            Yk = self.h(self.X[:,k])
            q = Yk[:4]
            q_error = self._q_error(q, q_ref)
            Y_error = ca.vertcat(q_error - q_unit, Yk[4:] - Y_res_ref)
            self.obj += ca.mtimes(Y_error.T, ca.mtimes(self.cfg.Q, Y_error))
            self.obj += ca.mtimes(self.U[:,k].T, ca.mtimes(self.cfg.R, self.U[:,k]))
        # terminal cost
        Yf = self.h(self.X[:,-1])
        q = Yf[:4]
        qf_error = self._q_error(q, q_ref)
        Yf_error = ca.vertcat(qf_error - q_unit, Yf[4:] - Y_res_ref)
        self.obj += ca.mtimes(Yf_error.T, ca.mtimes(self.cfg.Qf, Yf_error))
        
    def _set_constraints(self):
        #initial state constraints
        self.g.append(self.X[:,0] - self.X0)
        self.lbg.append(ca.DM.zeros(25))
        self.ubg.append(ca.DM.zeros(25))
        
        # dynamics constraints
        for k in range(self.cfg.N):
            x_next_pred = self.X[:, k] + self.dt * self.f(self.X[:, k], self.U[:, k])
            q_next = x_next_pred[0:4] / ca.norm_2(x_next_pred[0:4])   
            self.g.append(self.X[:, k+1] - ca.vertcat(q_next, x_next_pred[4:]))
            self.lbg.append(ca.DM.zeros(25))
            self.ubg.append(ca.DM.zeros(25))
            
            self.g.append(self.X[7:13,k])
            self.lbg.append(ca.repmat(self.cfg.angle_min, 6, 1))
            self.ubg.append(ca.repmat(self.cfg.angle_max, 6, 1))
            
        self.lbx = ca.vertcat(ca.repmat(-ca.inf, 25 * (self.cfg.N+1), 1),
                              ca.repmat(self.cfg.u_min, self.cfg.N, 1))
        self.ubx = ca.vertcat(ca.repmat(ca.inf, 25 * (self.cfg.N+1), 1),
                              ca.repmat(self.cfg.u_max, self.cfg.N, 1))
        
    def _q_error(self, q, q_ref):
        # quaternion error
        q_error = self._q_mult(self._q_inv(q_ref), q)
        q_error /= ca.norm_2(q_error)
        return q_error
    
    def _q_inv(self, q):
        # quaternion inverse
        q_inv = ca.vertcat(-q[0:3], q[3])
        return q_inv
    
    def _q_mult(self, q1, q2):
        # quaternion multiplication
        q_mult = ca.vertcat(q1[3]*q2[0:3] + q2[3]*q1[0:3] + ca.cross(q1[0:3], q2[0:3]), q1[3]*q2[3] - ca.mtimes(q1[0:3].T, q2[0:3]))
        return q_mult