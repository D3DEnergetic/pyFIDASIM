import numpy as np
import numpy.polynomial.hermite as hermite
import scipy.integrate as integrate
import scipy.optimize as optimize
import os
import sys
import pickle
from pathlib import Path
try:
    import numba
    numba_is_available = True
except:
    numba_is_available = False
    print('Numba is not available; consider installing Numba')

def conditional_numba(skip_numba=False):
    def decorator(func):
        if numba_is_available and not skip_numba:
            return numba.jit(func, 
                             cache=True, 
                             nopython=True, 
                             nogil=True,
                             debug = False
                             )
        else:
            return func
    return decorator

@conditional_numba(skip_numba=False)
def reflection_coeff_fitting_formula(E0,a1,a2,a3,a4,eL):
    epsilon=E0/eL
    R= (a1*epsilon**a2)/ (1.+(a3*epsilon**a4))
    return(R)

@conditional_numba(skip_numba=False)
def BoxGauss(x, A, sigma, equi_cut):
        mu=0
        if abs(x) > equi_cut:
            unGauss = 0.
        else:
            unGauss = normal(x, mu, sigma)*A
        return unGauss

@conditional_numba(skip_numba=False)
def cross_line_plane(p0, v_los, p_plane, v_norm):
    ''' 
    This code calculates the point at which a given line crosses a plane as 
    shown in https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    
    The line is defined by a point p0 and a vector v_los
    The plane is defined by a point p_plane and the vector normal to the plane v_norm
    '''   
    d = -np.dot(p0 - p_plane, v_norm) / np.dot(v_norm, v_los)
    return p0 + v_los * d

@conditional_numba(skip_numba=False)
def normal(x, mu, sigma):
    if sigma ==0:
        return(x*0.)
    return np.exp(-((x - mu)/sigma)**2/2)/sigma/(2*np.pi)**0.5

@conditional_numba(skip_numba=True)
def rotate_uvw(rot_mat, uvw):
    """Routine which rotates vector uvw (u) v = M x u.
    From R^3 -> R^3. (Euler angles)

    Parameters
    ----------
    rot_mat : np.array([3,3]), rotation matrix
    uvw: np.array(3), vector in the NBI source's reference system

    Returns
    ----------
    xyz : np.array(3), vector in the machine's reference system
    """

    return (rot_mat@uvw.T).T

def rotate_plane(x, y, Arot, Brot, p):
    uu, vv = np.meshgrid(x, y)
    ww = np.zeros_like(vv)
    uu = uu.flatten()
    vv = vv.flatten()
    ww = ww.flatten()
    uvw = np.zeros([3, len(uu)])
    xyz = np.zeros_like(uvw)
    uvw[0, :] = ww
    uvw[1, :] = uu
    uvw[2, :] = vv
    for i in range(len(uu)):
        xyz[:, i] = rotate_uvw(Arot, Brot, uvw[:, i])
    xx = xyz[0, :] + p[0]
    xx = np.reshape(xx, [len(x), -1])
    yy = xyz[1, :] + p[1]
    yy = np.reshape(yy, [len(y), -1])
    zz = xyz[2, :] + p[2]
    zz = np.reshape(zz, [len(x), -1])
    return xx, yy, zz

def load_dict(filename):
    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.resolve()
    if not filename.exists():
        print(filename.as_posix())
        raise FileNotFoundError(f'{filename.as_posix()} does not exist')
    print(f'Loading {filename.as_posix()}')
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    return(data_dict)
    
    
def save_dict(data_dict, filename):
    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.resolve()
    print(f'Saving {filename.as_posix()}')
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, protocol=5)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def clean(A, t, dt):
    # function for making A binary for t+-dt
    # t is the target value I want in the matrix A with tolerance dt
    new_A = np.copy(A)
    new_A[np.logical_and(new_A > t - dt, new_A < t + dt)] = -1
    new_A[new_A != -1] = 0
    new_A[new_A == -1] = 1
    return (new_A)

def printMatrix(a):
    rows = a.shape[0]
    cols = a.shape[1]
    for i in range(0,rows):
        string=' '.join(["%5.2e" % a[i,j] for j in range(cols)])
        print(string)
     

def get_surface(new_A, grid3d):
    x_vals = []
    y_vals = []
    z_vals = []

    # Retrieve (x,y,z) coordinates of surface
    for i in range(new_A.shape[0]):
        for j in range(new_A.shape[1]):
            for k in range(new_A.shape[2]):
                if new_A[i, j, k] == 1.0:
                    r = i * grid3d['dR'] + grid3d['Rmin']
                    phi = k * grid3d['dphi'] + grid3d['phimin']
                    x_vals.append(r * np.cos(phi))
                    y_vals.append(r * np.sin(phi))
                    z_vals.append(j * grid3d['dZ'] + grid3d['Zmin'])
    return (np.array(x_vals), np.array(y_vals), np.array(z_vals))

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print('this window is not defined')
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    y = y[int(0.5 * window_len):int(len(y) - 0.5 * window_len) + 1]
    return y

def radial_profile( ra, f_0=1., mu=1.,offset=0.,sol_decay=0.1,exp=False):
    F=np.zeros(len(ra))
    index=ra<=1
    
    
    if exp:
        ff= np.exp(mu*(1.-ra[index])**2)
        ff/=np.max(ff)
    else:
        ff=(1.-ra[index]**2)**mu
       
    F[index]=(f_0-offset) *ff  + offset
        
        
    index=ra > 1
    if sum(index)>0:
        sep_val=F[np.argmin(np.abs(ra-1.))]
        F[index]=sep_val*np.exp((1-ra[index])/sol_decay)

    return (F)
    
    
def rotation_matrix(v1, v2):
    """
    Calculates the rotation matrix that changes v1 into v2.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_angle = np.dot(v1, v2)
    d = np.cross(v1, v2)
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        M = np.identity(3) if cos_angle > 0. else -np.identity(3)
    else:
        d /= sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array([[0, d[2], -d[1]],
                         [-d[2], 0, d[0]],
                         [d[1], -d[0], 0]], dtype=np.float64)

        M = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return M

@conditional_numba(skip_numba=False)
def ER_Matrix(axis, angle):
    ##rotation matrix from orientation to Vin to orientation to Vref
    '''
    A = np.array([[0,-axis[2],axis[1]],\
         [axis[2],0,-axis[0]],\
         [axis[1],axis[0],0]])
           
    ER_Mat = np.identity(3)+np.dot(np.sin(np.pi - angle),A)+np.dot((1-np.cos(np.pi - angle)), np.matmul(A,A))   
    '''
    angle = -angle
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    r = np.cos(angle / 2)
    i, j, k = axis * np.sin(angle / 2)
    '''
    ER_Mat = np.transpose( np.array([
        [a**2 + b**2 - c**2 - d**2, 2 * (b*c - a*d), 2 * (b*d + a*c)],
        [2 * (b*c + a*d), a**2 + c**2 - b**2 - d**2, 2 * (c*d - a*b)],
        [2 * (b*d - a*c), 2 * (c*d + a*b), a**2 + d**2 - b**2 - c**2]
        ]) )
    '''
    ER_Mat = np.array([
        [1-2*(j**2+k**2) , 2*(i*j-k*r) , 2*(i*k+j*r)],
        [2*(i*j+k*r) , 1-2*(i**2+k**2) , 2*(j*k-i*r)],
        [2*(i*k-j*r) , 2*(j*k+i*r) , 1-2*(i**2+j**2)],
        ])
    
    return ER_Mat

@conditional_numba(skip_numba=False)
def rotate_vectors_ER(B,v,vec):
    # =========================================================================
    #rotates magnetic field, neutral beam injection, and line-of-sight vectors
    #in accordance with the information described in netCDF_stark files:
    #"B-field shows in z-direction, E-field shows in x-direction"
    # ========================================================================
    
    
    #-----------------------------
    #---normalize input vectors---
    #-----------------------------
    
    B_abs = np.linalg.norm(B)
    
    v_abs = np.linalg.norm(v)
    
    B_unit=B/B_abs
    
    v_unit=v/v_abs
    
    los_vec_unit=vec/np.linalg.norm(vec)
    
    #--------------------
    #---Rotate vectors---
    #--------------------
    #Determine B-field rotation to z axis
    #Cross product and cross product norm are invariant of z component of B_unit
    #This results in an incorrect rotation matrix for opposite sign z
    
    if round(B_unit[2],3) == 1:
        #checks if B_unit is already aligned in z direction
        B_in_z = B_unit
        v_B = v_unit
        los_B = los_vec_unit
        #print("+ve Z")
    elif round(B_unit[2],3) == -1:
        #checks if B_unit is aligned in -z direction
        RotAxis = np.array([1.,0.,0.])
        RotAngle = np.pi
        RotMat = ER_Matrix(RotAxis,RotAngle)
        B_in_z = B_unit @ RotMat
        v_B = v_unit @ RotMat
        los_B = los_vec_unit @ RotMat
        
    else:
        RotAxis = np.cross(B_unit,np.array([0.,0.,1.]))/np.linalg.norm(np.cross(B_unit,np.array([0.,0.,1.])))
        RotAngle = np.arccos(np.dot(B_unit,np.array([0.,0.,1.])))
        RotMat = ER_Matrix(RotAxis,RotAngle)
         
        B_in_z = B_unit @ RotMat

        if round(B_in_z[2],3) == 1:
            #applies rotation if z component is positive
            v_B = v_unit @ RotMat
            los_B = los_vec_unit @ RotMat
        else:
            #corrects negative z component by rotating by pi and subtracting original angle
            RotMat = ER_Matrix(RotAxis, np.pi - RotAngle)
            B_in_z = B_unit @ RotMat
            if round(B_in_z[2],3) == 1:
                v_B = v_unit @ RotMat
                los_B = los_vec_unit @ RotMat
            else:
                raise ValueError("Failed to Align B-field in z Direction")
    
    #Determine v_unit rotation to -yz plane s.t. Bxv is in x direction
    #z component of v_unit should NOT change, so we rotate its normalized xy projection to [0,-1]
    #similar to the B-field rotation, we correct for negative x components

    if (round(v_B[0],3) == 0 and v_B[1] < 0) or round(v_B[2],3) == round(B_in_z[2],3):
        #checks if v_B is already aligned in -yz plane
        #or if v_unit and B_unit are in the same direction
        ##v_Rot = v_B
        ##B_Rot = B_in_z
        los_Rot = los_B
        # print(v_Rot)
        # print(B_Rot)
        # print(los_Rot)
    else:         
        RotAxis = np.array([0.,0.,1.])
        if v_B[0]*v_B[1]>0:
            RotAngle = np.arctan(abs(v_B[0])/abs(v_B[1]))
        else:
            RotAngle = np.pi - np.arctan(abs(v_B[0])/abs(v_B[1]))
        if v_B[0] >0:
            RotAngle += np.pi
        RotMat = ER_Matrix(RotAxis,RotAngle)
        v_Rot = v_B @ RotMat

        if round(v_Rot[0],3) == 0 and v_Rot[1] <= 0:
            #applies rotation if x is negative
            #B_Rot = B_in_z @ RotMat
            los_Rot = los_B @ RotMat
        else:
            raise ValueError("Failed to Align Bxv field in x Direction")
    
    ##produces resulting Bxv field
    #E_Rot = cross_(B_Rot,v_Rot)
    #if norm(E_Rot) != 0:
    #    E_Rot /= norm(E_Rot)
        
    ##un-normalize inputs
    #B_Rot *= B_abs
    #v_Rot *= v_abs
    #return B_Rot,v_Rot,los_Rot,E_Rot
    return los_Rot

def FWHM(x,mat):
    matdiff = mat-(np.max(mat)/2.)
    zero_crossings = np.where(np.diff(np.sign(matdiff)))[0]
    x0_0 = x[zero_crossings[0]]
    x1_0 = x[zero_crossings[0]+1]
    y0_0 = matdiff[zero_crossings[0]]
    y1_0 = matdiff[zero_crossings[0]+1]
    xval_0 = x0_0-y0_0*(x1_0-x0_0)/(y1_0-y0_0)
    x0_1 = x[zero_crossings[1]]
    x1_1 = x[zero_crossings[1]+1]
    y0_1 = matdiff[zero_crossings[1]]
    y1_1 = matdiff[zero_crossings[1]+1]
    xval_1 = x0_1-y0_1*(x1_1-x0_1)/(y1_1-y0_1)
    FWHM_val = xval_1-xval_0
    return FWHM_val

@conditional_numba(skip_numba=False)
def point_samp(npoint,point_arr,d_point):
    f_array = np.zeros((3,npoint*len(point_arr)))
    nr = int(np.sqrt(npoint)/np.sqrt(3))
    dr = (d_point/2)/nr
    r_arr = np.linspace(0., (d_point/2.)-dr, nr)
    # print(r_arr)
    r_arr = r_arr +dr/2
    da = (np.pi*(d_point**2)/4.)/npoint
    # print("da =" +str(da))
    # print("dr = "+ str(dr))
    f_arr_i = 0
    for p in point_arr:
        # f_array.append([p[0],p[1],p[2]])
        i = 0
        for r in r_arr:
            dphi = da/(dr*(r))
            # print("dphi ="+ str(dphi*r))
            nphi = round(2*np.pi/dphi)
            theta = np.linspace(0,2*np.pi-dphi, nphi)
            for t in theta:
                t= t+dphi/2
                f_array[:,f_arr_i]= [p[0]+np.cos(t)*r,p[1]+np.sin(t)*r,p[2]]
                f_arr_i = f_arr_i + 1
            i = i + 1
    return f_array



def setup_trevor_spectrum(orbital_num, nbi_params, _spec, lambda_min = None, lambda_max = None, calc_wavel = True):

    if nbi_params['ab'] == 1.0:
        beam_type = "H"
    elif nbi_params['ab'] == 2.0:
        beam_type = "D"
    else:
        raise ValueError("trevor spectrum only valid for H or D beam (nbiparams['ab'] = 1.0 or 2.0)")
    
    filename = beam_type+"_n"+str(orbital_num)+"_2.ncdf"
    import pyfidasim
    root_path = Path(pyfidasim.__file__).parent
    cdfvars,lambda0,trans,l_to_dwp = read_netCDF_stark(os.path.join(root_path,'tables/orbital_ncdf_data/'+filename))
    ncdf = {"cdfvars":cdfvars, "lambda0": lambda0, "trans": trans, "l_to_dwp":l_to_dwp, "active": True}
    print(lambda0)
    _spec['lambda0']=lambda0
    if not lambda_min:
        _spec["lambda_min"] = lambda0-5
    else:
        _spec["lambda_min"] = lambda_min
        
    if not lambda_max:
        _spec["lambda_max"] = lambda0+5
    else:
        _spec["lambda_max"] = lambda_max
    if calc_wavel:
        _spec['wavel'] = np.arange(_spec['lambda_min'], _spec['lambda_max'], _spec['dlam'])
        _spec['nlam'] = len(_spec['wavel'])

    return _spec, ncdf 

@conditional_numba(skip_numba=False)
def calc_intens(los_vec,B_vec,para_interp):
    #Determine p,t (and ct2,cp2) from los_Rot=[cos(p)*sin(t),sin(p)*sin(t),cos(t)]
    if np.linalg.norm(np.cross(los_vec,B_vec)) == 0.0:
        #checks if los is same direction as B-field
        #system implies t=0 from 3rd element's equation
        #elements 1 and 2 are equal to zero implying sin(p)=cos(p)
        #p,t found independently --> unique solution to system
        p = 0
        t = 0
        raise ValueError("los in direction of B-field: Case undefined (phi undefined)")
        
    else:
        #p obtained by dividing 2nd element's equation by first, then solving
        #t obtained by solving 3rd element's equation
        #p,t found independently --> unique solution to system
        p = np.arctan(los_vec[1]/los_vec[0])
        t = np.arccos(los_vec[2])

    cp2 = np.cos(p)**2
    ct2 = np.cos(t)**2
    
    
    # determine line strengths  
    intens = (1/2 * (para_interp[2]+para_interp[3]) * (1+ct2)) + \
        (para_interp[1]+para_interp[4] - 2*para_interp[4]*cp2) * (1-ct2)
        #L = 1/2*(p2+m2)*(1+ct2)+[z2+pm-2*pm*cp2]*(1-ct2)
        #[0]=wvl_shift, [1]=z2, [2]=p2, [3]=m2, [4]=pm
    return intens

def read_netCDF_stark(filename):
    #associated text from netCDF file
    #obtained using "ncdump" terminal command (on linux system)
    #describes geometric orientation, interpolation procedure, and line strength equation
    # =========================================================================
    # // global attributes:
    # :title = "Dipol Matrix Elements and Wavelength Shifts" ;
    # :atom_info = "No fine structure considered!" ;
    # :geom_info = "B-field shows in z-direction, E-field shows in x-direction, observation direction is k=[cos(p)*sin(t),sin(p)*sin(t),cos(t)]" ;
    # :usage_step_1 = "interpolate for each transition the variables wvl_shift, z2, p2, m2, and pm onto the requested perpendicular velocity" ;
    # :usage_step_2 = "multiply the the obtained line shifts with the actual B-field in Tesla and add Doppler shift" ;
    # :usage_step_3 = "the line strengths are L = 1/2*(p2+m2)*(1+ct2)+[z2-pm+2*pm*cp2]*(1-ct2) with cp2 = cos(p)^2, ct2 = cos(t)^2" ;
    # =========================================================================
    from scipy.io import netcdf
    import copy as cp
    #read and extract data from netCDF files
    file = netcdf.netcdf_file(filename,'r')
    
    v_perp = cp.deepcopy(file.variables['v_perp'].data)
    wvl_shift = cp.deepcopy(file.variables['wvl_shift'].data)
    
    #gets number of transitions
    trans = wvl_shift.shape[0]
    z2 = cp.deepcopy(file.variables['z2'].data)
    p2 = cp.deepcopy(file.variables['p2'].data)
    m2 = cp.deepcopy(file.variables['m2'].data)
    pm = cp.deepcopy(file.variables['pm'].data)
    lambda0 = cp.deepcopy(file.wavelength)
    l_to_dwp = cp.deepcopy(file.l_to_dwp)
    
    
    #collects parameters into list
    cdfvars = np.zeros((6,trans,201))
    cdfvars[0,:,:] = v_perp
    cdfvars[1,:,:] = wvl_shift
    cdfvars[2,:,:] = z2
    cdfvars[3,:,:] = p2
    cdfvars[4,:,:] = m2
    cdfvars[5,:,:] = pm

    file.close()
    
    return cdfvars,lambda0,trans, l_to_dwp


@conditional_numba(skip_numba=False)
def exp_fun_rejection_method(tau):
    ## use a rejection method to determine random time points following an 
    ## exponential distribution 
    for i in range(10000):
        t  = np.random.uniform(0,1)*(5*tau)
        f  = np.exp(-t/tau)
        if(f > np.random.uniform(0,1)): 
          break
    
    if(i > 9999):
        print('rejection method found no solution!')
        t=0
    return(t)



@conditional_numba(skip_numba=False)
def gaussian_hermite_approx(x, x0, sigma, coef):
    # This function returns 'y', which is a numpy array if x is an array, and is a float if x is a float.
    # The equation this satisfies is:
    # y = 1/(sqrt(2*pi)*sigma) * e**(-(x-x0)**2 / (2*sigma**2)) * SUM(m=0->N){a_vector[m]*H_m((x-x0)/(sqrt(2)*sigma))
    # H_m() in the above equation is the m_th physics hermite polynomial.

    herm_func = hermite.hermval((x-x0)/(np.sqrt(2)*sigma), coef)
    herm_func = np.array(herm_func)
    y = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-1*(x-x0)**2 / 2 / sigma**2) * herm_func
    if len(y) == 1:
        return y[0]
    else:
        return y


@conditional_numba(skip_numba=False)
def calc_hermite_coef(y1, x1, x0, sigma, N, verbose):
    # Calculates the hermite coefficients to order N for a hermite-modified gaussian approximation of y1(x1)
    # x1, y1 should be numpy arrays or equal length
    # x0 and sigma are the values in the gaussian:  np.exp(-1*(x1-x0)**2/(2*sigma**2))
    coef = np.zeros(N+1)
    for i in range(0, N+1):
        coef_temp = np.zeros(i+1)
        coef_temp[i] = 1
        hermite_temp = hermite.hermval((x1-x0)/(np.sqrt(2)*sigma), coef_temp)
        coef[i] = 1. / 2**i / np.math.factorial(i) * integrate.trapezoid(y1*hermite_temp, x1)
        if verbose:
            print('hermite temp = ', hermite_temp)
            print('coef = ', coef)
            print()

    return coef


@conditional_numba(skip_numba=False)
def mod_gaus_fit_resids(x, *args):
    # x = [x0, sigma]
    # args = (y1, x1, N, return_coef)
    # x0 = x offset of base gaussian. float.
    # sigma = standard deviation of base gaussian. float.
    # y1 = function to use for residuals. numpy array.
    # x1 = x-axis of y1, equal in length to y1. numpy array.
    # N = integer, order of approximation.
    #       N=0 referres to an un-modified gaussian, i.e. g(u)*(a0). N=1 includes g(u)*(a0 + a1*H_1(U)), etc.
    # return_coef = Boolean: whether or not to return the coef as an additional argument.

    x0 = x[0]
    sigma = x[1]

    y1 = args[0]
    x1 = args[1]
    N = args[2]
    return_coef = args[3]
    verbose = args[4]

    coef = calc_hermite_coef(y1, x1, x0, sigma, N, verbose)
    y_gaus = gaussian_hermite_approx(x1, x0, sigma, coef)
    resids = y1-y_gaus

    if return_coef is False:
        return resids
    else:
        return resids, coef



def fit_modified_gaussian(x1, y1, N, verbose=False):
    # fit a function to a modified gaussian to order N.
    # N=0 refferes to an un-modified gaussian, i.e. g(u)*(a0). N=1 includes g(u)*(a0 + a1*H_1(U)), etc.
    # x1 and y1 should be the axis and intensities of your function, respectively.
    # x1 and y1 SHOULD EACH BE A NUMPY ARRAY!!!!

    # calculate starting values
    was_inf = False
    norm = integrate.trapezoid(y1, x1)
    x0_start = integrate.trapezoid(y1*x1, x1)/norm
    if np.isnan(x0_start):
        x0_start = np.mean(x1)
        if verbose:
            print('x0_start was infinite')
        was_inf = True
    sigma_start = np.sqrt(integrate.trapezoid((np.power((x1 - x0_start), 2) * y1 / norm), x1))
    if np.isnan(sigma_start):
        sigma_start = np.abs(x1[-1] - x1[0])/5.
        if verbose:
            print('sigma_start was infinite')
        was_inf = True

    try:
        results = optimize.least_squares(fun=mod_gaus_fit_resids, x0=[x0_start, sigma_start], args=(y1, x1, N, False, verbose))
    except:
        print('least_squares failed, exiting.')
        print('x0 = ', [x0_start, sigma_start])
        raise RuntimeError

    xfinal = results['x']
    resids_final, coef_final = mod_gaus_fit_resids(xfinal, y1, x1, N, True, verbose)

    sum_resid_sq = np.sum(resids_final**2)
    mean_y = np.mean(y1)
    total_sum_sqrs = np.sum((y1-mean_y)**2)
    Rsqrd = 1 - sum_resid_sq/total_sum_sqrs

    output = [xfinal[0], xfinal[1], coef_final]
    if was_inf and verbose:
        print('was inf: output = ', output)

    return output, Rsqrd


@conditional_numba(skip_numba=False)
def get_energy_component(wavel, Coef):
    # wavel = np.array([all wavelengths needed])
    # Coef => Coef[line_num] = [x0, sigma, a0, a1,... an]
    wavel_0 = wavel
    sqrt2 = np.sqrt(2)
    yfinal = np.zeros((len(Coef), len(wavel)), dtype=np.float64)
    for i, coef_0 in enumerate(Coef):
        wavel = (wavel_0 - coef_0[0])/sqrt2/coef_0[1]
        n0 = len(coef_0) - 2
        y1 = np.zeros(len(wavel))
        y0 = 1/(np.sqrt(2*np.pi)*coef_0[1]) * np.exp(-1*wavel**2)
        if n0 > 0:
            y1 += coef_0[2]*np.ones(len(wavel))
        if n0 > 1:
            y1 += coef_0[3]*2*wavel
        if n0 > 2:
            y1 += coef_0[4]*(4*wavel**2 - 2)
        if n0 > 3:
            y1 += coef_0[5]*(8*wavel**3 - 12*wavel)
        if n0 > 4:
            y1 += coef_0[6]*(16*wavel**4 - 48*wavel**2 + 12)
        if n0 > 5:
            y1 += coef_0[7]*(32*wavel**5 - 160*wavel**3 + 120*wavel)
        if n0 > 6:
            y1 += coef_0[8]*(64*wavel**6 - 480*wavel**4 + 720*wavel**2 - 120)
        if n0 > 7:
            y1 += coef_0[9]*(128*wavel**7 - 1344*wavel**5 + 3360*wavel**3 - 1680*wavel)
        if n0 > 8:
            y1 += coef_0[10]*(256*wavel**8 - 3584*wavel**6 + 13440*wavel**4 - 13440*wavel**2 + 1680)
        if n0 > 9:
            y1 += coef_0[11]*(512*wavel**9 - 9216*wavel**7 + 48384*wavel**5 - 80640*wavel**3 + 30240*wavel)
        if n0 > 10:
            y1 += coef_0[12]*(1024*wavel**10 - 23040*wavel**8 + 161280*wavel**6 - 403200*wavel**4 + 302400*wavel**2 - 30240)

        yfinal[i][:] = y0*y1

    return yfinal





