import dolfin as df
import ufl


class ClementInterpolant(object):
    '''
    This class implements efficient construction of Clement interpolant of an
    UFL-built expression. Here, the Clement interpolant is a CG_1 function over 
    mesh constructed in two steps (See Braess' Finite element book):
        1) For each mesh vertex xj let wj the union of cells that share the vertex 
           (i.e wj is the support of vj - the basis function of CG_1 function
           space such that vj(xj) = 1). Then Qj(expr) is an L2 projection of
           expr into constant field on wj.
        2) Set Ih(expr) = sum_j Qj(expr)vj.
    '''
    def __init__(self, expr, use_averaging=True):
        '''For efficient interpolation things are precomuputed here'''
        # Analyze expr and raise if invalid
        terminals = _analyze_expr(expr)
        # Analyze shape and raise if expr cannot be represented
        _analyze_shape(expr.ufl_shape)
        shape = expr.ufl_shape
        # Extract mesh from expr operands and raise if it is not unique or missing
        mesh = _extract_mesh(terminals)
        # Compute things for constructing Q
        Q = df.FunctionSpace(mesh, 'DG', 0)
        q = df.TestFunction(Q)
        # Forms for L2 means [rhs]
        # Scalar, Vectors, Tensors are built from components
        # Translate expression into forms for individual components
        if len(shape) == 0:
            forms = [df.inner(expr, q)*df.dx]
        elif len(shape) == 1:
            forms = [df.inner(expr[i], q)*df.dx for i in range(shape[0])]
        else:
            forms = [df.inner(expr[i, j], q)*df.dx for i in range(shape[0]) for j in range(shape[1])]

        # Build averaging or summation operator for computing the interpolant
        # from L2 averaged components.
        V = df.FunctionSpace(mesh, 'CG', 1)
        volumes = df.assemble(df.inner(df.Constant(1), q)*df.dx)
        # Ideally we compute the averaging operator, then the interpolant is
        # simply A*component. I have not implemented this for backends other 
        # than PETSc. 

        A = _construct_averaging_operator(V, volumes)
        # We can precompute maps for assigning the components
        if len(shape) == 0:
            W = df.FunctionSpace(mesh, 'CG', 1)
            assigner = None
        else:
            if len(shape) == 1:
                W = df.VectorFunctionSpace(mesh, 'CG', 1, dim=shape[0])
            else:
                W = df.TensorFunctionSpace(mesh, 'CG', 1, shape=shape)
            assigner = df.FunctionAssigner(W, [V]*len(forms))

        # Collect stuff
        self.shape, self.V, self.A, self.forms, self.W, self.assigner = shape, V, A, forms, W, assigner

    def __call__(self):
        '''Return the interpolant.'''
        shape, V, A, forms = self.shape, self.V, self.A, self.forms
        
        # L2 means of comps to indiv. cells
        means = map(df.assemble, forms)

        # The interpolant (scalar, vector, tensor) is build from components
        components = []
        for mean in means:
            component = df.Function(V)
            # In case we have the averaging operator A*component computed the
            # final value. Otherwise it is rhs for L2 patch projection
            A.mult(mean, component.vector()) 
            # And to complete the interpolant we need to apply the precomputed
            # mass matrix inverse
            components.append(component)
        
        # Finalize the interpolant
        # Scalar has same space as component
        if len(shape) == 0: 
            uh = components.pop()
        # Other ranks
        else:
            uh = df.Function(self.W)
            self.assigner.assign(uh, components)
            # NOTE: assign might not use apply correctly. see 
            # https://bitbucket.org/fenics-project/dolfin/issues/587/functionassigner-does-not-always-call
            # So just to be sure
            uh.vector().apply('insert')

        return uh

# Workers--

def _analyze_expr(expr):
    '''
    A valid expr for Clement interpolation is defined only in terms of pointwise
    operations on finite element functions.
    '''
    # Elliminate forms
    if isinstance(expr, ufl.Form):
        raise ValueError('Expression is a form')
    # Elliminate expressions build from Trial/Test functions, FacetNormals 
    return tuple(ufl.corealg.traversal.traverse_unique_terminals(expr))


def _analyze_shape(shape):
    '''
    The shape of expr that UFL can build is arbitrary but we only support
    scalar, rank-1 and rank-2(square) tensors.
    '''
    is_valid = len(shape) < 3 and (shape[0] == shape[1] if len(shape) == 2 else True)
    if not is_valid:
        raise ValueError('Interpolating Expr does not result rank-0, 1, 2 function')


def _extract_mesh(terminals):
    '''Get the common mesh of operands that make the expression.'''
    pairs = []
    for t in terminals:
        try: 
            mesh = t.function_space().mesh()
            pairs.append((mesh.id(), mesh))
        except AttributeError: 
            pass

    # Unique mesh        
    id, = set(id_ for id_, _ in pairs)

    return pairs.pop()[1]


def _construct_summation_operator(V):
    '''
    Summation matrix has the following properties: It is a map from DG0 to CG1.
    It has the same sparsity pattern as the mass matrix and in each row the nonzero
    entries are 1. Finally let v \in DG0 then (A*v)_i is the sum of entries of v
    that live on the support of i-th basis function of CG1.
    '''
    mesh = V.mesh()
    Q = df.FunctionSpace(mesh, 'DG', 0)
    q = df.TrialFunction(Q)
    v = df.TestFunction(V)
    tdim = mesh.topology().dim()
    K = df.CellVolume(mesh)
    dX = df.dx(metadata={'form_compiler_parameters': {'quadrature_degree': 1,
                                                      'quadrature_scheme': 'vertex'}})
    # This is a nice trick which uses properties of the vertex quadrature to get
    # only ones as nonzero entries.
    # NOTE: Its is designed spec for CG1. In particular does not work CG2 etc so
    # for such spaces a difference construction is required, e.g. rewrite nnz
    # entries of mass matric V, Q to 1. That said CG2 is the highest order where
    # clement interpolation makes sense. With higher ordered the dofs that are
    # interior to cell (or if there are multiple dofs par facet interior) are
    # assigned the same value.
    A = df.assemble((1./K)*df.Constant(tdim+1)*df.inner(v, q)*dX)

    return A


def _construct_averaging_operator(V, c):
    '''
    If b is the vectors of L^2 means of some u on the mesh, v is the vector
    of cell volumes and A is the summation oparotr then x=(Ab)/(Ac) are the
    coefficient of Clement interpolant of u in V. Here we construct an operator
    B such that x = Bb.
    '''
    A = _construct_summation_operator(V)

    Ac = df.Function(V).vector()
    A.mult(c, Ac)
    # 1/Ac
    Ac = df.as_backend_type(Ac).vec()
    Ac.reciprocal()     
    # Scale rows
    mat = df.as_backend_type(A).mat()
    mat.diagonalScale(L=Ac)

    return A


def clement_interpolate(expr, with_CI=False):
    '''
    A free function for construting Clement interpolant of an expr. This is
    done by creating instance of ClementInterpolant and applying it. The
    instance is not cached. The function is intended for one time interpolation.
    However, should you decide to do the repeated interpolation use with_CI=True
    to return the interpolated function along with the ClementInterpolant
    instance. 
    '''
    ci = ClementInterpolant(expr)
    return (ci(), ci) if with_CI else ci()

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np

    msg = '|e|_0 = {:.4E}[{:.2f}] |e|_1 = {:.4E}[{:.2f}]'
    ncells = (4, 8, 16, 32, 64, 128)

    print("Function value:")
    f = df.Expression(('sin(pi*(x[0]-x[1]))', 'cos(pi*(x[0]+x[1]))'), degree=5)
    # Fake displacement
    errors0, h0 = None, None
    for n in ncells:
        mesh = df.UnitSquareMesh(n, n)
        Q = df.VectorFunctionSpace(mesh, 'DG', 0)
        qh = df.interpolate(f, Q)
        # At this point qh is only in L2 and difficult for standard
        # interpolation. Want to reconstruct this in P1 and look at
        # convergence
        vh = clement_interpolate(qh)

        eL2 = df.errornorm(f, vh, 'L2')
        eH10 = df.errornorm(f, vh, 'H10')
        h = mesh.hmin()
        
        errors = np.array([eL2, eH10])
        if errors0 is None:
            rates = np.nan*np.ones_like(errors)
        else:
            rates = np.log(errors/errors0)/np.log(h/h0)
        errors0, h0 = errors, h
        print(msg.format(*sum(zip(errors, rates), ())))

    print()

    print("Gradient:")
    grad_f = df.Expression((('pi*cos(pi*(x[0]-x[1]))', '-pi*cos(pi*(x[0]-x[1]))'),
                            ('-pi*sin(pi*(x[0]+x[1]))', '-pi*sin(pi*(x[0]+x[1]))')), degree=5)
    # Reconstruct gradient
    errors0, h0 = None, None
    for n in ncells:
        mesh = df.UnitSquareMesh(n, n)
        Q = df.VectorFunctionSpace(mesh, 'DG', 1)
        qh = df.interpolate(f, Q)
        # At this point qh is only in L2 and difficult for standard
        # interpolation. Want to reconstruct this in P1 and look at
        # convergence

        gh = clement_interpolate(df.grad(qh))

        eL2 = df.errornorm(grad_f, gh, 'L2')
        eH10 = df.errornorm(grad_f, gh, 'H10')
        h = mesh.hmin()
        
        errors = np.array([eL2, eH10])
        if errors0 is None:
            rates = np.nan*np.ones_like(errors)
        else:
            rates = np.log(errors/errors0)/np.log(h/h0)
        errors0, h0 = errors, h
        print(msg.format(*sum(zip(errors, rates), ())))

    x, y = mesh.coordinates().T
    fh_ = vh.compute_vertex_values().reshape((2, -1)).T   # This are ordered as vertices
    gh_ = gh.compute_vertex_values().reshape((4, -1)).T
    # Data for NN input
    data = np.c_[x, y, fh_, gh_]

    for row in data:
        x0, y0, vh_, gh_ = row[0], row[1], row[2:4], row[4:]
        assert np.linalg.norm(vh_ - vh(x0, y0)) < 1E-14
        assert np.linalg.norm(gh_ - gh(x0, y0)) < 1E-14

    print()
    
    print("Hessian:")
    hess_f_1 = df.Expression((('-pi*pi*sin(pi*(x[0]-x[1]))', 'pi*pi*sin(pi*(x[0]-x[1]))'),
                             ('pi*pi*sin(pi*(x[0]-x[1]))', '-pi*pi*sin(pi*(x[0]-x[1]))')), degree=5)
    
    hess_f_2 = df.Expression((('-pi*pi*cos(pi*(x[0]+x[1]))', '-pi*pi*cos(pi*(x[0]+x[1]))'),
                             ('-pi*pi*cos(pi*(x[0]+x[1]))', '-pi*pi*cos(pi*(x[0]+x[1]))')), degree=5)
    
    # Reconstruct Hessian?
    errors0, h0 = None, None
    for n in ncells:
        mesh = df.UnitSquareMesh(n, n)
        Q = df.VectorFunctionSpace(mesh, 'DG', 2)
        qh = df.interpolate(f, Q)
        qh_1, qh_2 = qh.split()
        # At this point qh is only in L2 and difficult for standard
        # interpolation. Want to reconstruct this in P1 and look at
        # convergence
        vh_1 = clement_interpolate(df.grad(df.grad(qh_1)))
        vh_2 = clement_interpolate(df.grad(df.grad(qh_2)))

        eL2_1 = df.errornorm(hess_f_1, vh_1, 'L2')
        eL2_2 = df.errornorm(hess_f_2, vh_2, 'L2')
        eL2 = np.sqrt(eL2_1**2 + eL2_2**2)
        eH10_1 = df.errornorm(hess_f_1, vh_1, 'H10')
        eH10_2 = df.errornorm(hess_f_2, vh_2, 'H10')
        eH10 = np.sqrt(eH10_1**2 + eH10_2**2)
        h = mesh.hmin()
        
        errors = np.array([eL2, eH10])
        if errors0 is None:
            rates = np.nan*np.ones_like(errors)
        else:
            rates = np.log(errors/errors0)/np.log(h/h0)
        errors0, h0 = errors, h
        print(msg.format(*sum(zip(errors, rates), ())))

    print()

    print("Laplacian:")
    # For Hessians? (This is laplacian, not hessian)
    f = df.Expression('sin(pi*(x[0]-x[1]))', degree=5)
    #target = (cos(pi*(x[0]-x[1]))*pi, -cos(pi*(x[0]-x[1]))*pi)
    target = df.Expression('-sin(pi*(x[0]-x[1]))*pi*pi -sin(pi*(x[0]-x[1]))*pi*pi', degree=5)
    # Fake displacement
    errors0, h0 = None, None
    for n in ncells:
        mesh = df.UnitSquareMesh(n, n)
        Q = df.FunctionSpace(mesh, 'CG', 2)
        qh = df.interpolate(f, Q)

        vh = clement_interpolate(df.div(df.grad(qh)))

        eL2 = df.errornorm(target, vh, 'L2')
        eH10 = df.errornorm(target, vh, 'H10')
        h = mesh.hmin()
        
        errors = np.array([eL2, eH10])
        if errors0 is None:
            rates = np.nan*np.ones_like(errors)
        else:
            rates = np.log(errors/errors0)/np.log(h/h0)
        errors0, h0 = errors, h
        print(msg.format(*sum(zip(errors, rates), ())))

    print()
