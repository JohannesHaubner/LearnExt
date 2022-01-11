class Solver(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def solve(self):
        """solve PDE"""
        raise NotImplementedError

class Context(object):
    def __init__(self, params):
        """
        :param params: contains deltat, t and T, boundary_cond
        """
        self.t = params["t"]
        self.T = params["T"]
        self.dt = params["deltat"]
        self.bc = params["boundary_cond"]

    def check_termination(self):
        return (not self.t < self.T)

    def advance_time(self):
        """
        update of time variables and boundary conditions
        """
        self.t += self.dt
        self.bc.t = self.t

class FSI(Context):
    def __init__(self, mesh, param, FSI_params):
        super().__init__(FSI_params)
        self.init = FSI_params["initial_cond"]
        self.mesh = mesh
        self.param = param
        self.FSI_params = FSI_params

class FSIsolver(Solver):
    def __init__(self, mesh, param, FSI_params, extension_operator):
        """
        solves the FSI system on mesh
        :param mesh: computational mesh (with fluid and solid part)
        :param param: contains id's for subdomains and boundary parts
        :param FSI_params: contains FSI parameters
        lambdas, mys, rhos, rhof, nyf
        also contains the information for the time-stepping
        deltat, t (start time), and T (end time)
        and initial and boundary conditions
        initial_cond, boundary_cond
        :param extension_operator: object of the ExtensionOperator-class
        """
        super().__init__(mesh)
        self.param = param
        self.FSI_params = FSI_params
        self.extension_operator = extension_operator

    def solve(self):
        print('here')



