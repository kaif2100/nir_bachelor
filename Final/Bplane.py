import numpy as np
import constants
import warnings
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


class Bplane:
    """
    This class contains 3 vectors of a B-plane basis
    And different (1 so far) methods of it's calculation
    """

    def __init__(self, renc, Vga, Vinf):
        self.set_basis(None, None, None)
        if renc is not None and Vga is not None and Vinf is not None:
            self._calc_basis1(renc, Vga, Vinf)

    def set_basis(self, e1, e2, e3):
        """
        This method sets basis but doesn't check it for orthogonality and normalization
        """
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def _calc_basis1(self, renc, Vga, Vinf):
        """
        This method calculates and sets basis as it's done in the Natan Strange's dissertation (T, R, S) 
        """
        p3 = np.cross(renc, Vga)
        p3 /= np.linalg.norm(p3)
        
        self.e3 = Vinf / np.linalg.norm(Vinf)
        self.e1 = np.cross(self.e3, p3)
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.e3, self.e1)

        # save Vinf, renc, Vga just in case
        self._Vinf = np.array(Vinf)
        self._renc = np.array(renc)
        self._Vga = np.array(Vga)

    def get_basis(self):
        return self.e1, self.e2, self.e3

    def get_matrix(self):
        """
        Returns transition matrix in the meaning x_old = M @ x_new
        """
        return np.vstack((self.e1, self.e2, self.e3)).T
    
    def _get_piercing_point_old(self, R, V):
        """
        Calculates at which point (x, y, 0) will spacecraft pierce the B-plane
        If it starts at R with velocity V\
        
        Input:
        R - initial radius-vector, np.ndarray, shape is (3,), type is float
        V - initial velocity, np.ndarray, shape is (3,), type is float
        
        Output:
        [x, y, 0] - np.ndarray where x, y are the coordinates of the piercing-point in the B-plane
        None if the spacecraft doesn't pierce the B-plane
        """
        M = self.get_matrix().T
        
        R = M @ R
        V = M @ V
        time = -R[2] / V[2]
        
        if time < 0:
            warnings.warn("No piercing-point was found", RuntimeWarning)
            return None
        
        res = R + V * time
        return res
    
    def get_piercing_point(self, R, V):
        """
        R, V - in inertial basis (rotation to the Bplane basis is made here)
        """
        M = self.get_matrix().T
        
        R = M @ R
        V = M @ V
        time = -R[2] / V[2]
        
        if time < 0:
            warnings.warn("No piercing-point was found", RuntimeWarning)
            return None, None
        
        return R[0], R[1]
    
    def get_rp_not_corrected(self, x: float, y: float, mu: float, Vinf: float):
        """
        This function calculates periapsis distance (not height!)
        without correction for finite size of influence sphere
        
        Input:
        x, y - piercing-point coordinates, m
        mu - gravitational parameter of the gravitating body, m^3/s^2
        Vinf - relative speed at the moment of entering sphere of influence, m/s
        
        Returns:
        rp - periapsis distance, m
        """
        
        buf = mu / Vinf**2
        rp = np.sqrt(buf ** 2 + x**2 + y**2) - buf
        return rp
    
    def get_rp(self, x: float, y: float, mu: float, Vinf: float, ri: float):
        """
        This function calculates periapsis distance (not height!) 
        with correction for finite size of influence sphere
        
        Input:
        x, y - piercing-point coordinates, m
        mu - gravitational parameter of the gravitating body, m^3/s^2
        Vinf - relative speed at the moment of entering sphere of influence, m/s
        ri - radius of the sphere of influence, m
        
        Returns:
        rp - periapsis distance, m
        """
        B = np.sqrt(x**2 + y**2)
        C = B * Vinf
        h = Vinf**2 - 2 * mu / ri
        
        e = np.sqrt(1 + h * C**2 / mu**2)
        p = C ** 2 / mu
        rp = p / (e + 1)
        return rp
        
    def get_B(self, rp: float, mu: float, Vinf: float, ri: float):
        buf = 2 * mu / Vinf**2
        B = rp * np.sqrt(1 + buf * (1/rp - 1/ri))
        return B
    
    
    
class BplanePlotter:
    _units_factors = {"km": 1e3, "m": 1.}
    
    def __init__(self, dpi = 100, units_in = "m", units_out = "m"):
        self.fig, self.ax = plt.subplots(dpi = dpi, tight_layout = True)
        self._dpi = dpi
        
        self._size = None
        
        self._default_cm = matplotlib.cm.get_cmap('viridis')
        
        self._points = {}
        
        self._ellipses = {}
        
        self._lines = {}
        
        self._hps_circles = {}
#         self._hps_colormap = matplotlib.cm.get_cmap('viridis')
#         self._hps_labels = []
        
        if units_in in BplanePlotter._units_factors and units_out in BplanePlotter._units_factors:
            self._u = BplanePlotter._units_factors[units_out] / BplanePlotter._units_factors[units_in]
            self._units = units_out
        else:
            warnings.warn("The units provided wasn't found. Units miltiplier will be set 1.", RuntimeWarning)
            self._u = 1.
            self._units = "Unknown"
    
    
    def add_point(self, x: float, y: float, name: str, color = None):
        """
        Adds point (x, y) with a name (must be unique), adds the name into the names list
        If the name already exists it will be replaced
        
        Input:
        x - x-coordinate in meters
        y - y-coordinate in meters
        name
        """
        if not isinstance(name, str):
            warnings.warn("The name provided is not a string! The point wasn't added", RuntimeWarning)
            return -1
        
        if name in self._points:
            warnings.warn(f"The point {name} had already existed, it was replaced", RuntimeWarning)
        
        if color is None:
            color = self._default_cm(np.random.rand())
    
        self._points[name] = (x, y, color)
    
    def del_point(self, name: str):
        """
        Deletes the point (x, y) with a name given and also deletes the name from the names list
        """
        if not name in self._points:
            warnings.warn("The name provided hadn't been added before. Nothing was deleted.", RuntimeWarning)
            return -1
        
        self._points.pop(name)
    
    def add_line(self, x1, y1, x2, y2, name, color = None):
        """
        If color is None (deafult) it will be chosen arbitrary
        """
        if name in self._lines:
            warnings.warn(f"The line {name} had already existed, it was replaced", RuntimeWarning)
        
        if color is None:
            color = self._default_cm(np.random.rand())
        
        self._lines[name] = matplotlib.lines.Line2D((x1 * self._u, x2 * self._u), (y1 * self._u, y2 * self._u), color = color)
    
    
    def _add_hp_not_corrected(self, hp: float, Rbody: float, mu_body: float, Vinf: float):
        """
        This function adds a periapsis height isoline onto the B-plane
        without correction for finite radius of influence sphere
        
        Input:
        hp - periapsis height, m
        Rbody - radius of the gravitating body, m
        mu_body - gravitational parameter of the gravitating body, m^3/s^2
        Vinf - relative speed at the moment of entering sphere of influence, m/s
        """
        
        if hp in self._hps_circles:
            warnings.warn("The hp provided had already existed. It was recalculated.", RuntimeWarning)
        
        rp = Rbody + hp
        B = rp * np.sqrt(1 + 2 * mu_body / (rp * Vinf**2))
        circle = plt.Circle((0, 0), B * self._u, fill = False)
        self._hps_circles[hp] = circle
    
    def add_hp(self, hp: float, Rbody: float, mu_body: float, Vinf: float, ri: float, color = None):
        """
        This function adds a periapsis height isoline onto the B-plane
        with correction for finite radius of influence sphere
        
        Input:
        hp - periapsis height, m
        Rbody - radius of the gravitating body, m
        mu_body - gravitational parameter of the gravitating body, m^3/s^2
        Vinf - relative speed at the moment of entering sphere of influence, m/s
        """
        
        name = f"{hp * self._u:0.0f}"
        if name in self._hps_circles:
            warnings.warn("The hp provided had already existed. It was recalculated.", RuntimeWarning)
        
        if color is None:
            color = self._default_cm(np.random.rand())
        
        rp = Rbody + hp
        buf = 2 * mu_body / Vinf**2
        B = rp * np.sqrt(1 + buf * (1/rp - 1/ri))

        circle = plt.Circle((0, 0), B * self._u, fill = False, color = color)
        
        self._hps_circles[name] = circle
        
    
    def add_ellipse(self, x_center, y_center, a, b, v, name, color = None, multiplier = None):
        """
        Adds an ellipse on the plot
        
        Input:
        x_center, y_center - coordinates of the ellipse center
        a - major semi-axis
        b - minor semi-axis
        v - angle from Ox axis to the major semi-axis anti-clockwise, in DEGREES
        name - name (str)
        """
        if name in self._ellipses:
            warnings.warn(f"The ellipse {name} had already existed, it was replaced", RuntimeWarning)
            
        if color is None:
            color = self._default_cm(np.random.rand())
        
        if multiplier is None:
            m = self._u
        else:
            m = multiplier
        
        self._ellipses[name] = matplotlib.patches.Ellipse((x_center * m, y_center * m), 2 * a * m, 2 * b * m,
                                                          v, fill = False, color = color)
    
    
    def set_size(self, x_center, y_center, x_delta, y_delta):
        """
        Sets boundaries of the picture: [x_center - x_delta, x_center + x_delta] X [y_center - y_delta, y_center + y_delta]
        """
        self._size = [x_center * self._u, y_center * self._u, x_delta * self._u, y_delta * self._u]
        
    def plot(self, show = True):
        self.ax.clear()
        
        # plotting points
        for n in self._points:
            self.ax.plot(self._points[n][0] * self._u, self._points[n][1] * self._u, 'o', color = self._points[n][2])
        
        # plotting lines
        for n, l in self._lines.items():
            self.ax.add_artist(l)
        buf = plt.legend(self._lines.values(), self._lines.keys(), loc = 'upper right', title = "Lines")
        self.ax.add_artist(buf)
        
        #plotting ellipses
        for n, el in self._ellipses.items():
            self.ax.add_artist(el)
        buf = plt.legend(self._ellipses.values(), self._ellipses.keys(), loc = 'lower right', title = "Ellipses")
        self.ax.add_artist(buf)
        
        # plotting hp isolines
        N = len(self._hps_circles)
        if N:
            for hp, circ in self._hps_circles.items():
                self.ax.add_artist(circ)
            
            buf = plt.legend(self._hps_circles.values(), self._hps_circles.keys(), loc = 'upper left', title = f"hp, {self._units}")
            self.ax.add_artist(buf)
         
        # Hidden code (hp isolines with cmap)
        if False:   
            N = len(self._hps_circles)
            if N:
                n = N - 1 if N > 1 else 1 # for colormap

                ticks = []
                labels = np.zeros(n + 1)

                i = 0
                for hp, circ in self._hps_circles.items():
                    circ.set_color(self._hps_colormap(i / n))
                    self.ax.add_artist(circ)
                    ticks.append(i)
                    labels[i] = f'{hp * self._u: 0.1f}'
                    i += 1
                buf = plt.legend(self._hps_circles.values(), self._hps_circles.keys(), loc = 'upper left', title = f"hp, {self._units}")
                self.ax.add_artist(buf)
                norm = matplotlib.colors.Normalize(vmin = 0, vmax = n)
                cb = self.fig.colorbar(matplotlib.cm.ScalarMappable(norm = norm, cmap = self._hps_colormap), ticks = ticks,
                                       format = matplotlib.ticker.FuncFormatter(lambda i, y: labels[i]),
                                       ax = self.ax, orientation = 'vertical')

                fs = 10
                cb.ax.set_ylabel(f'$h_p, {self._units}$', fontsize=fs)               
                cb.ax.tick_params(labelsize=fs) # make customizable! 
         
        
        
        if self._size:
            self.ax.set_xlim(self._size[0] - self._size[2], self._size[0] + self._size[2])
            self.ax.set_ylim(self._size[1] - self._size[3], self._size[1] + self._size[3])
        self.ax.grid(True)
        self.ax.set_xlabel(f"$\\xi, {self._units}$")
        self.ax.set_ylabel(f"$\eta, {self._units}$")
        if show:
            plt.show()


            
def Jac2Body(x, mu):
    """
    df/dx, where f is right-side function for 2-body problem
    """
    res = np.zeros((6, 6))
    r = np.linalg.norm(x[:3])
    r2 = r**2
    mur3 = mu / r**3
    
    res[[0, 1, 2], [3, 4, 5]] = 1 
    
    res[3, 0] = mur3 * (3 * x[0]**2 / r2 - 1)
    res[3, 1] = mur3 * (3 * x[0] * x[1] / r2)
    res[3, 2] = mur3 * (3 * x[0] * x[2] / r2)

    res[4, 0] = mur3 * (3 * x[0] * x[1] / r2)
    res[4, 1] = mur3 * (3 * x[1]**2 / r2 - 1)
    res[4, 2] = mur3 * (3 * x[1] * x[2] / r2)

    res[5, 0] = mur3 * (3 * x[0] * x[2] / r2)
    res[5, 1] = mur3 * (3 * x[1] * x[2] / r2)
    res[5, 2] = mur3 * (3 * x[2]**2 / r2 - 1)
    
    return res

def TransitionMatrix(x_ref, t1, t2, r_units, V_units, mu, Jac = Jac2Body, tolerance = 1e-10, step = 10):
    """
    This function integrates transition matrix f (Ф = W @ f, Ф - dimensionalized, f - non-dimensionalized)
    from time t1 where Ф is assumed to be E (identity matrix) to the final time t2
    df/dt = W.-1 @ Jac(x_ref(t), t) @ W @ f
    
    Input:
    x_ref - reference trjectory, must be a function of t and return phase vector in m, m/s
    t1, t2 - start and finish times
    r_units - units for non-dimensiolizing of radius-vector expressed in meters (if you want to non-dimensionalize using kilometers you should pass 1000)
    V_units - as r_units, but for speed
    mu - gravitational parameter
    Jac - optional, Jacobian dF/dx, where dx/dt = F, default - for 2 body problems
    tolerance - tolerance
    step - step for integrator to give out results
    
    Returns:
    TM(t) - transition matrix as a function of time (cubic-spline interpolation of integration results)
    shape 6x6, TM[i, j] = dx_i / dx_j
    """
    # Matrix of units conversion (Ф = W f, where Ф is the target)
    W = np.zeros((6, 6))
    Winv = np.zeros((6, 6))
    W[[0, 1, 2], [0, 1, 2]] = r_units
    W[[3, 4, 5], [3, 4, 5]] = V_units
    Winv[[0, 1, 2], [0, 1, 2]] = 1 / r_units
    Winv[[3, 4, 5], [3, 4, 5]] = 1 / V_units
            
    def G(t, y):
        return Winv @ Jac(x_ref(t), mu) @ W @ y
    
    initial = np.eye(6)
    initial = Winv @ initial
    dxd0 = []
    
    N = int(np.ceil(np.abs(t2 - t1) // step))
    t_eval = np.linspace(t1, t2, N)
    
    # since dx/dx_i (i = 1, 2, ..., 6) are independent, we can integrate them separately
    for i in range(initial.shape[1]):
        y0 = initial[:, i]
        dxd0.append(solve_ivp(G, (t1, t2), y0, t_eval = t_eval, rtol=tolerance, atol=tolerance).y)
    dxd0 = np.asarray(dxd0).transpose((1, 0, 2)) # now time is the last index
    
    return interp1d(t_eval, dxd0, kind = 'cubic')
  
def transfer_errors(K1, TM, t1, t2, r_units, V_units):
    """
    This function calculates covariation matrix K2 at the time t2
    from given covariation matrix K1 at the moment t1
    (propagation of errors over time)
    
    Input:
    K1 - initial covariation matrix DIMENSIONALIZED (in meters, meters/second etc.)!
    TM - transition matrix as function of time (see function TransitionMatrix, make sure TM is defined on [t0, t2])
    t1 - time at which cov. matrix is K1
    t2 - time at which cov. matrix is desired
    r_units - units for non-dimensiolizing of radius-vector expressed in meters (if you want to non-dimensionalize using kilometers you should pass 1000)
    V_units - as r_units, but for speed
    (units are necessary for W calculation)
    
    
    Returns:
    K2 - covariation matrix at the moment t2
    K2inv - inverse covariation matrix at the moment t2
    
    P.S. if TM is non-dimensionalized, you have to multiply K1 and K1inv by corrspondong factors to receive it in meters
    Kdim = W @ Knondim @ W
    """
    
    Winv = np.zeros((6, 6))
    Winv[[0, 1, 2], [0, 1, 2]] = 1 / r_units
    Winv[[3, 4, 5], [3, 4, 5]] = 1 / V_units
    
    f1inv = np.linalg.inv(TM(t1))
    F1inv = f1inv @ Winv
    
    f2 = TM(t2)
    f = f2 @ F1inv
    
    finv = np.linalg.inv(f)
    K2 = f @ K1 @ f.T
    
    K1inv = np.linalg.inv(K1)
    print("Inv error K1:", np.max(np.abs(np.eye(6) - K1inv @ K1)))
    K2inv = finv.T @ K1inv @ finv
    print("Inv error K2:", np.max(np.abs(np.eye(6) - K2inv @ K2)))

    return K2, K2inv


def ScatteringEllipse(K1, TM, M, t1, t2, r_units, V_units, c = 12.766):
    """
    Input:
    K1 - initial covariation matrix DIMENSIONALIZED (in meters, meters/second etc.)!
    TM - transition matrix as function of time (see function TransitionMatrix, make sure TM is defined on [t0, t2])
    M - matrix of frame rotation (only for x, y, z) in the meaning x_old = M @ x_new
    t1 - time at which cov. matrix is 1
    t2 - time at which ellipse is desired
    r_units - units for non-dimensiolizing of radius-vector expressed in meters (if you want to non-dimensionalize using kilometers you should pass 1000)
    V_units - as r_units, but for speed
    
    Returns:
    a - first semi-axis
    b - second semi-axis
    v - angle form Ox to the first semi-axis in RADIANS
    """
    K2, Q_err = transfer_errors(K1, TM, t1, t2, r_units, V_units)
    Q_err = ellipse_6D_to_2D(Q_err, M1 = M, c = c)
    a_err, b_err, v_err = canonical2D(Q_err, c = 1)
    return a_err, b_err, v_err

def InfluenceEllipse(TM, M, dV, t1, t2):   
    """
    Calculates ellipse of influence for velocity correction (|delta velocity| <= dV)
    at the moment t1 and fly-in an the moment t2
    
    Input:
    TM - transition matrix as function of time (see function TransitionMatrix, make sure TM is defined on [t0, t2])
    M - matrix of frame rotation (only for x, y, z) in the meaning x_old = M @ x_new
    dV - magnitude of the correction IN UNITS (dV in m/s divided by V_units)
    t1 - time of the correction
    t2 - time of fly-in
    
    Returns:
    a, b, v - ellipse parameters
    (if TM is non-dimensionalized don't forget to multiply a and b with r_units factor)
    """
    
    f1inv = np.linalg.inv(TM(t1))
    f2 = TM(t2)
    
    A = (f2 @ f1inv)[:3, 3:]
    Ainv = np.linalg.inv(A)
    print("Inv error A:", np.max(np.abs(np.eye(3) - Ainv @ A)))
    
    Q = M.T @ Ainv.T @ Ainv @ M / dV**2
    a, b, v = canonical2D(Q)
    
    return a, b, v

def ellipse_6D_to_2D(Q, c = 12.766, M1 = np.eye(3), M2 = np.eye(3)):
    """
    This function calculates 2D projection of a 6D-ellipsoid x.T @ Q @ x <= c onto the x-y plane

    Input:
    Q - 6x6 ellipse matrix (must be symmetrical)
    c - constant in the ellipse equation, 12.766 by default (3-sigma errors)
    M1 - 3x3 basis change matrix for the first 3 axes in the meaning x_old = M1 @ x_new
    M2 - 3x3 basis change matrix for the last 3 axes in the meaning x_old = M2 @ x_new

    Returns:
    QQ - 2x2 symmetric matrix of the projected ellipsoid (y.T @ QQ @ y <= 1, where y - 2D coordinates (pay attention to <= 1!!!!))
    """
    # 1. Rotate the basis (position, velocity's basis doesn't change)
    buf1 = np.hstack((M1, np.zeros((3, 3), dtype = np.float64)))
    buf2 = np.hstack((np.zeros((3, 3), dtype = np.float64), M2))
    S = np.vstack((buf1, buf2))
    Q = S.T @ Q @ S
    
    # 2. Calculate the view x = L.-T @ u, |u| <= 1
    Q /= c
    L = np.linalg.cholesky(Q)

    # 3. Operator of projection on the plane
    T = np.zeros((6, 2))
    T[0, 0] = 1
    T[1, 1] = 1
    P = T @ T.T

    # 4. SVD decomposition of T.T L.-T
    Linv = np.linalg.inv(L)
    buf = T.T @ Linv.T # MxN
    U, E, V = np.linalg.svd(buf, full_matrices=False)  # MxM, MxN, NxN
    E = np.diag(E)

    # 5. 2D-Ellipsoid matrix
    buf = np.linalg.inv(U @ E).T
    QQ = buf @ buf.T

    return QQ

def canonical2D(Q, c = 1):
    """
    Function for calculating the canonical form (and projection) of an ellipse given by x.T @ Q @ x <= c
    
    Input:
    Q - initial ellipsoid matrix. The size of Q is either 3 by 3, and then the projection on xy-plane is taken, or 2 by 2
    c - constant in the ellipsoid equation

    Returns:
    a - length of the first semi-axes
    b - length of the second semi-axes
    v - angle of rotation (counterclockwise) from Ox to the first semi-axis (a) in radians
    """

    if Q.shape[0] == 3 and Q.shape[1] == 3:
        A = Q[0, 0] * Q[2, 2] - Q[0, 2]**2
        B = Q[0, 1] * Q[2, 2] - Q[0, 2] * Q[1, 2]
        C = Q[1, 1] * Q[2, 2] - Q[1, 2]**2
        D = c * Q[2, 2]
        A /= D
        B /= D
        C /= D
        D = 1
    elif Q.shape[0] == 2 and Q.shape[1] == 2:
        A = Q[0, 0]
        B = Q[0, 1]
        C = Q[1, 1]
        D = c
        A /= D
        B /= D
        C /= D
        D = 1
    else:
        raise ValueError("Shape of Q is improper")

    # If B = 0 it's already in cannonical form
    if np.allclose(B, 0):
        B = 0
        v = 0
    else:
        # Algorithm is taken form linear algebra book written by Umnov A.E.
        # If A = C
        if np.allclose(A, C):
            A = C
            v = np.pi / 4
        else:
            v = 0.5 * np.arctan(2 * B / (A - C))

    A2 = A * np.cos(v)**2 + B * np.sin(2 * v) + C * np.sin(v)**2
    C2 = A * np.sin(v)**2 - B * np.sin(2 * v) + C * np.cos(v)**2

    a = np.sqrt(1 / A2)
    b = np.sqrt(1 / C2)
    return a, b, v


def get_correction(TM, M, t_corr, t_in, dxi, deta, r_units, V_units):
    """
    Calculates recuired two-parameter correction in linear approximation

    Input:
    TM - fucntion of t which returns transition matrix (non-dim) 6x6
    M - rotation matrix 3x3 in the meaning x_old = M @ x_new
    t_corr - time of correction
    t_in - time of fly-in
    dxi - delta xi desired(in units of r)
    deta - delta eta desired(in units of r)
    r_units - the divider of r (radius-vector), necessary because TM contains d(r/runits)/dV0
    V_units - the divider of V (velocity), necessary because TM contains d(V/runits)/dV0

    Output:
    V - velocity for correction required (units are determined by the TM), np.array, shape (3,)
    """
    dxi /= r_units
    deta /= r_units

    dRdV = (TM(t_in) @ np.linalg.inv(TM(t_corr)))[:3, 3:] / V_units

    N = M.T @ dRdV

    A = N[0, :]
    B = N[1, :]
    An = np.linalg.norm(A)
    Bn = np.linalg.norm(B)
    a = A / An
    b = B / Bn
    cos = np.dot(a, b)
    sin2 = 1 - cos**2

    if cos == 1.:
        warnings.warn("Transition matrix provided is degenerated", RuntimeWarning)

    V1 = (dxi / An - deta / Bn * cos) / sin2
    V2 = (deta / Bn - dxi / An * cos) / sin2

    return V1 * a + V2 * b

def get_Vinf_correction(Vinf_des, Vinf_cur):
    """
    Calculates recuired one-parameter correction in linear approximation

    Input:
    Vinf_des - desired Vinf (in inertial)
    Vinf_cur - current Vinf (in inertial)

    Output:
    dV - velocity for correction required
    """
    
    return Vinf_des - Vinf_cur
