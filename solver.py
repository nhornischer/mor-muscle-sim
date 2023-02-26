import warnings
warnings.filterwarnings("ignore")


import numpy as np
import scipy.sparse.linalg as linalg
import scipy
from scipy.sparse import spdiags, diags
from numpy import reshape, ones, full, transpose
import time
import sys
import math
import gc

import base
from minidihu import extract_channels, extractX

# Note: np.reshape from matrix to vector stores row-wise 

from minidihu import Conductivity, Am, Cm
Cm=0.58
Am=500.0
Conductivity=3.828

C = Conductivity / Am / Cm

shift, hash_log= None, None

# Placeholder for precalculated matrices
pre_A_hat , pre_M_hat , pre_B_hat = None , None , None
pre_bool = False 


# =============================================================================
# Setup
# =============================================================================
def load_reduced_data(hash_log : str , data_hashes : list , discretization : dict):
    """Loads the parameters of the reduced ansatz space.

    Parameters
    ----------
    hash_log : str
        Hash of the log entry refering to the current decomposition task.
    data_hashes : list
        List of hashes to the data to be loaded.
    discretization : dict
        Discretization of the initial values and time discretization.

    Returns
    -------
    (tuple,dict)
        Tuple of the reduced ansatz space (phi,paths) and discretization.
    """
    decomposed_phi,decomposed_paths,decomposed_R=None,None,0
    for i, data_hash in enumerate(data_hashes):
        print(f"Loaded reduced data from {data_hash}")
        data=base.load_np_data(data_hash, task='DECOMPOSITION')
        discretization_decomp,_=base.read_log(data_hash)

        M_decomp,R,N,GAMMA=discretization_decomp['M'],discretization_decomp['R'],discretization_decomp['N'],discretization_decomp['Gamma']
        assert N==discretization['N'], 'Spatial discretization does not match reduced ansatz space'

        # extract the values from the loaded data
        phi = reshape(data[GAMMA*M_decomp*R:GAMMA*M_decomp*R+R*N], [R, N])
        paths = data[GAMMA*M_decomp*R+R*N:]

        assert np.size(paths) == R, "No valid shape for the paths"

        decomposed_R+=R

        if i==0:
            decomposed_phi = phi
            decomposed_paths=paths
        else:
            decomposed_phi=np.concatenate((decomposed_phi,phi))
            decomposed_paths=np.concatenate((decomposed_paths,paths))
    
        base.write_log([f"{'':<15} loaded decomposition hash: {data_hash}\n"], hash_log)
   
    discretization['R']=decomposed_R

    decomposed=(decomposed_phi,decomposed_paths)
    print(f"Parameters of the reduced space:")
    print(f"  p: {np.min(decomposed_paths):>+.2e} -- {np.max(decomposed_paths):>+.2e}")
    print(f"  φ: {np.min(decomposed_phi):>+.2e} -- {np.max(decomposed_phi):>+.2e}")

    return decomposed, discretization

def load_initial_data(hash_log : str, initial_value_file = ['standard'], waves = None):
    """Loads the initial values and creates the discretization in space

    Parameters
    ----------
    hash_log : str
        Hash of the log entry refering to the current decomposition task.
    initial_value_file : list, optional
        List of instructions for the initial values, by default ['standard']
    waves : str, optional
        Instruction for separating the waves, by default None

    Returns
    -------
    (np.array, dict)
        Tuple containing the initial values and the discretization.
    """
    if initial_value_file[0] == 'extract':
        assert len(initial_value_file)>=3,'No data for extraction given'
            
        hash=initial_value_file[1]
        time=initial_value_file[2]
        extracted_discretization,_=base.read_log(hash)

        assert time <= extracted_discretization['T'], 'Can not extract initial values at a not simulated time.'
        
        states=base.load_np_data(hash,'SOLVER')[int(time/extracted_discretization['HT'])]
        
        z_0=np.reshape(states,[4,int(len(states)/4)]).T
        N=np.shape(z_0)[0]
        xs = np.linspace(0,11.9, N)
        hxs = xs[1:] - xs[:-1]
        print(f'Extracted fiber from {hash} at {time} ms')
        initial_value_file=str(f'type=extract , origin={hash} , time={time} ')

    elif initial_value_file[0] == 'file':
        xyz = extractX(initial_value_file[1], 'geometry', 'xyz')['val'] # fiber location in space
        z_0 = extract_channels(initial_value_file[1])['val'] # initial values
        hxs = np.sum((xyz[1:,:] - xyz[:-1,:])**2, axis=1)**.5
        xs = np.zeros(hxs.shape[0] + 1) # 1D location
        xs[1:] = np.cumsum(hxs)
        print(f"Loaded fiber from {initial_value_file[1]}")
        initial_value_file=str('type=file , name={initial_value_file[1]} ')
    
    else:
        N=1190+1
        xs = np.linspace(0,11.9, N)
        hxs = xs[1:] - xs[:-1]
        z_0 = np.zeros((N, 4))
        z_0[:,0] = -75.0,
        z_0[:,1] =   0.05,
        z_0[:,2] =   0.6,
        z_0[:,3] =   0.325,
        # initial acivation

        z_0[N//2 - 3 : N//2 + 3, 0] = 50
        from scipy.stats import norm
        gaus=norm.pdf(xs,11.9/2,0.02)
        z_0[:,0]=gaus*((35+75)/np.max(gaus))-75

        if initial_value_file[0] != 'standard': print('No valid initial instruction given')
        print("Created standard fiber")
        initial_value_file='type=standard'

    # Use only the desired side of the wavefronts
    if waves=='left':
        z_0[int(np.shape(z_0)[0]/2):,:]=z_0[-1,:]
        print(f'Extract left part')
        initial_value_file=str(initial_value_file + ' , wave=left')
    elif waves=='right':
        z_0[:int(np.shape(z_0)[0]/2),:]=z_0[-1,:]
        print(f'Extract right part')
        initial_value_file=str(initial_value_file + ' , wave=right')
    else:
        initial_value_file=str(initial_value_file + ' , wave=dual')


    HS=np.mean(np.unique(np.round(hxs,4)))
    N=xs.shape[0]
    print(f"  length:              {xs[-1]:>5.2f} cm")
    print(f"  spatial step-width:  {HS:>4} cm")
    print(f"  nodes:               {N:>4}")
    print(f"Initial values:")
    print(f"  V: {np.min(z_0[:,0]):>+.2e} -- {np.max(z_0[:,0]):>+.2e}")
    print(f"  m: {np.min(z_0[:,1]):>+.2e} -- {np.max(z_0[:,1]):>+.2e}")
    print(f"  h: {np.min(z_0[:,2]):>+.2e} -- {np.max(z_0[:,2]):>+.2e}")
    print(f"  n: {np.min(z_0[:,3]):>+.2e} -- {np.max(z_0[:,3]):>+.2e}")
    print(f"Model parameters:")
    print(f"   σ:    {Conductivity:>7.3f}")
    print(f"  Am:    {Am:>7.3f}")
    print(f"  Cm:    {Cm:>7.3f}")

    discretization = dict({'N' : N, 'HS' : HS, 'Gamma':1})

    base.write_log([f"{'':<15} Initialization: {initial_value_file}\n", f"{'':<15} Spatial discretization: N={N} , HS={HS} cm\n"], hash_log)

    return z_0, discretization

# =============================================================================
# Definition of the differential equation
# =============================================================================
def hodgkin_huxley(v : np.array, y : np.array):
    """Hodgkin Huxley Model

    Parameters
    ----------
    v : np.array
        Voltage over the spatial domain. Shape = (N,)
    y : np.array
        Ion states over the spatial domain. [[m,h,n]]*N : shape=(N,3)

    Returns
    -------
    (np.array, np.array)
        F : shape=(N,) for the voltage part of the PDE and G : shape=(N,3) for the system of ODEs of the ion states
    """

    # init constants

    CONSTANTS_0 = -75
    CONSTANTS_1 = 1
    CONSTANTS_2 = 0
    CONSTANTS_3 = 120
    CONSTANTS_4 = 36
    CONSTANTS_5 = 0.3
    CONSTANTS_6 = CONSTANTS_0 + 115
    CONSTANTS_7 = CONSTANTS_0 - 12
    CONSTANTS_8 = CONSTANTS_0 + 10.613

    # compute rates 
    ALGEBRAIC_1 = ( - 0.1*(v+50))/(np.exp(- (v+50)/10) - 1)
    ALGEBRAIC_5 =  4*np.exp(- (v+75)/18)
    RATES_1 =  ALGEBRAIC_1*(1 - y[:,0]) -  ALGEBRAIC_5*y[:,0]
    ALGEBRAIC_2 =  0.07*np.exp(- (v+75)/20)
    ALGEBRAIC_6 = 1/(np.exp(- (v+45)/10)+1)
    RATES_2 =  ALGEBRAIC_2*(1 - y[:,1]) -  ALGEBRAIC_6*y[:,1]
    ALGEBRAIC_3 = ( - 0.01*(v+65))/(np.exp(- (v+65)/10) - 1)
    ALGEBRAIC_7 =  0.125*np.exp((v+75)/80)
    RATES_3 =  ALGEBRAIC_3*(1 - y[:,2]) -  ALGEBRAIC_7*y[:,2]
    ALGEBRAIC_0 =  CONSTANTS_3*np.power(y[:,0], 3)*y[:,1]*(v - CONSTANTS_6)
    ALGEBRAIC_4 =  CONSTANTS_4*np.power(y[:,2], 4)*(v - CONSTANTS_7)
    ALGEBRAIC_8 =  CONSTANTS_5*(v - CONSTANTS_8)

    F = - (- CONSTANTS_2+ALGEBRAIC_0+ALGEBRAIC_4+ALGEBRAIC_8)/CONSTANTS_1
    G = np.array([RATES_1, RATES_2, RATES_3])

    return F.T, G.T 

def RHS(z : np.array, Mass_inv : np.array, Stiff : np.array, projection=()):
    """Constructs the right-hand side of the differential equation, euqivalent to the terms on the right side
    of the equality sign of equation 4.5 (definition 4.1) or equation 2.4 (definition 2.2) in the thesis

    Parameters
    ----------
    z : np.array
        states for one time step shape=(R,) or shape=(R+3*N,)
    Mass_inv : np.array
        Inverted mass matrix shape=(R,R)
    Stiff : np.array
        Stiffness matrix shape=(R,R)
    projection : tuple, optional
        Projection Matrices for the reduced case, by default ()

    Returns
    -------
    np.array
        Returns the right hand side with same dimensions as the input state z
    """
    dim_first=np.shape(Mass_inv)[0]
    dim_others=int((np.shape(z)[0]-dim_first)/3)
    v=z[:dim_first]
    y=np.reshape(z[dim_first:],[3,dim_others]).T

    linear_term=Mass_inv @ Stiff @ v

    if dim_first == dim_others:
        F,G=hodgkin_huxley(v,y)
        nonlinear_term=F
    else:
        # Project the reduced solution to the original space and back
        v_projected = up_projection(v,projection)+shift

        F,G=hodgkin_huxley(v_projected,y)
        
        nonlinear_term = down_projection(F,projection)
    RHS=np.reshape(np.concatenate([linear_term+nonlinear_term ,G[:,0],G[:,1],G[:,2]]),[dim_first+dim_others*3])

    return RHS

# =============================================================================
# Time-Stepping Methods
# =============================================================================
def explicit_euler(z_0 : np.array, discretization : dict, bounds = 'Neumann', decomposed = ()):
    """Defines the explicit euler method zᵢ₊₁ = zᵢ + h * f(zᵢ), where f is the right-hand side RHS.

    Parameters
    ----------
    z_0 : np.array
        Initial values
    discretization : dict
        Discretization paramters in space and time
    bounds : str, optional
        Instruction for bound of the FOM, by default 'Neumann'
    decomposed : tuple, optional
        Parameters for the reduced ansatz space, by default ()

    Returns
    -------
    (np.array,float)
        Result as numpy array and computation time.
    """
    # the solver takes a one-dimensional vector of the variables
    z=(z_0.T).flatten()

    N,HS,M,HT,T=discretization['N'],discretization['HS'],discretization['M'],discretization['HT'], discretization['T']
    
    # Define time-discretization
    ts=np.arange(0,T+HT/2,HT)[1:]

    assert ts[1]-ts[0]==HT, 'Time-discretization destroys given time-step-width'

    # Checks if enough memory is avaiable to store 64-bit objects, if not stores in 32-bit
    try:
        result=np.empty((len(ts)+1,len(z)))
    except:
        result=np.empty((len(ts)+1,len(z)),dtype=np.float32)
    result[0,:]=z[:]

    try: 
        R=discretization['R']
    
    except:
        # Non-Reduced Method
        startTime = time.time()
        Stiff,Mass_inv=FOM(N, HS, bounds)
        
        for i,t in enumerate(ts):
            z_=np.array(z)
            z_ += HT*RHS(z_,Mass_inv,Stiff)
            z=z_
            print("Time-Stepping {:4} / {:4} [{}{}] {:3} %".format(i+1, M, "#"*int((i+1)/(M-1)*20), "-"*(20-int((i+1)/(M-1)*20)),int((i+1)/(M-1)*100)), end='\r')
            result[i+1,:]=z
        exec_time = time.time() - startTime

    else:
        # Reduced Method
        print('Reduced Method')
        assert decomposed != (), 'No valid decomposition data.'
        (phi, p)=decomposed

        startTime = time.time()

        _,Mass_bar_inv=FOM(N, HS, bounds)

        for i,t in enumerate(ts):
            Stiff,Mass_hat_inv=ROM_tuned(t, R,N, HS,  phi, p)
            P_bar=ROM_projection(t,R,N,HS,phi,p)
            z_=np.array(z)
            z_ = z_+ HT*RHS(z_,Mass_hat_inv,Stiff,(Mass_bar_inv, Mass_hat_inv,P_bar))
            z=z_
            print("Time-Stepping {:4} / {:4} [{}{}] {:3} %".format(i+1, M, "#"*int((i+1)/(M-1)*20), "-"*(20-int((i+1)/(M-1)*20)),int((i+1)/(M-1)*100)), end='\r')
            result[i+1,:]=z
        
        exec_time = time.time() - startTime

    print(" "*100, end='\r')
    print('Computation time Time-Stepping: {:.4f}s'.format(exec_time))
    return result, exec_time

def implicit_euler(z_0 : np.array, discretization : dict, bounds = 'Neumann', decomposed = ()):
    """Defines the implicit euler method zᵢ₊₁ = zᵢ + h * f(zᵢ₊₁), where f is the right-hand side RHS.

    Parameters
    ----------
    z_0 : np.array
        Initial values
    discretization : dict
        Discretization paramters in space and time
    bounds : str, optional
        Instruction for bound of the FOM, by default 'Neumann'
    decomposed : tuple, optional
        Parameters for the reduced ansatz space, by default ()

    Returns
    -------
    (np.array,float)
        Result as numpy array and computation time.
    """

    # Defines the implicit euler method uᵢ₊₁ = uᵢ + h * f(uᵢ₊₁), where f is the right-hand side RHS. 
    # Solves the equation by finding the root  0 = uᵢ - uᵢ₊₁ + h * f(uᵢ₊₁).


    # the solver takes a one-dimensional vector of the variables
    z=(z_0.T).flatten()
    result = [z]

    N,HS,M,HT=discretization['N'],discretization['HS'],discretization['M'],discretization['HT']

    from scipy.optimize import fsolve

    # Iterates over an implicit euler-method by finding the root of the minimizing function
    def implicit_min(u_1,*params):
        return params[0]-u_1+HT*RHS(u_1,params[1],params[2],projection=params[3])

    # Define time-discretization
    ts=np.linspace(0,T,M+1)[:-1]

    assert ts[1]-ts[0]==HT, 'Time-discretization destroys given time-step-width'

    try:
        R=discretization['R']

    except:
        # Calculate matrices ones
        startTime = time.time()
        Stiff,Mass_inv=FOM(N, HS, bounds)
        
        for i,t in enumerate(ts):
            z_=np.array(z)
            z_=fsolve(implicit_min,z_,(z_,Mass_inv,Stiff),maxfev=100)
            z=z_
            print("Time-Stepping {:4} / {:4} [{}{}] {:3} %".format(i+1, M, "#"*int((i+1)/(M-1)*20), "-"*(20-int((i+1)/(M-1)*20)),int((i+1)/(M-1)*100)), end='\r')
            result.append(z)
        exec_time = time.time() - startTime
    
    else:
        print('Reduced Method')
        assert decomposed != (), 'No valid decomposition data.'
        (phi, p)=decomposed

        startTime = time.time()

        _,Mass_bar_inv=FOM(N, HS, bounds)

        if decomposed == {}:
            raise Exception('No valid decomposition data.')
        

        for i,t in enumerate(ts):
            Stiff,Mass_hat_inv=ROM_tuned(t, R, N,HS,  phi, p)
            P_bar=ROM_projection(t,R,N,HS,phi,p)
            z_=np.array(z)
            z_=fsolve(implicit_min,z_,(z_,Mass_hat_inv,Stiff,(Mass_bar_inv, Mass_hat_inv,P_bar)),maxfev=100)
            z=z_
            print("Time-Stepping {:4} / {:4} [{}{}] {:3} %".format(i+1, M, "#"*int((i+1)/(M-1)*20), "-"*(20-int((i+1)/(M-1)*20)),int((i+1)/(M-1)*100)), end='\r')
            result.append(z)
        exec_time = time.time() - startTime

    print(" "*100, end='\r')
    print('Computation time Time-Stepping: {:.4f}s'.format(exec_time))
    return np.asarray(result), exec_time

# =============================================================================
# Definition of the FOM and ROM_tuned
# =============================================================================
def FOM(N : int, HS : float, bounds = 'Neumann', lumped = False, disp = False):
    """Defines the mass and stiffness matrix for the full order model, see chapter A.1 in the Thesis

    Parameters
    ----------
    N : int
        Spatial dimension
    HS : float
        Spatial step-width.
    bounds : str, optional
        Declaration of the bound of the model problem, by default 'Neumann'
    lumped : bool, optional
        Instruction if lumped mass matrix should be used, by default False
    disp : bool, optional
        Instruction if matrices should be displayed, by default False

    Returns
    -------
    (np.array,np.array)
        Stiffness matrix and inverted mass matrix.
    """
    diags_laplace=transpose(full((N,3),[-1/HS, 2/HS, -1/HS]))
    if bounds == 'Neumann':
        diags_laplace[1,0]=1/HS
        diags_laplace[1,-1]=1/HS
    if bounds == 'Dirichlet':
        diags_laplace[1,0]=-1/HS
        diags_laplace[2,0]=0
        diags_laplace[0,-2]=0
        diags_laplace[1,-1]=-1/HS
    if bounds == 'skip':
        diags_laplace[1,0]=0
        diags_laplace[2,0]=0
        diags_laplace[0,-2]=0
        diags_laplace[1,-1]=0
    laplace = diags(diags_laplace, [-1,0,1], shape=(N, N))
    
    Stiff = - C * laplace

    diags_mass=transpose(full((N,3),[1/6*HS,2/3*HS , 1/6*HS]))
    Mass= diags(diags_mass, [-1,0,1], shape=(N, N))

    if lumped:
        diags_mass_lumped=Mass @ ones(N)
        Mass_lumped= diags(diags_mass_lumped, 0, shape=(N, N))
        Mass_inv=linalg.inv(Mass_lumped)
    else:
        Mass_inv=linalg.inv(Mass)    

    
    if disp:
        print(f"Laplace \033[93m[FOM] \033[m \033[1;92m [{bounds}] \033[0m")
        print('  '+base.arr2str(laplace.todense(), prefix='  '))
        print(f"Stiffness \033[93m[FOM] \033[m \033[1;92m [{bounds}] \033[0m")
        print('  '+base.arr2str(Stiff.todense(), prefix='  '))    
        print("Mass \033[93m[FOM] \033[m")
        print('  '+base.arr2str(Mass.todense(), prefix='  '))
        if lumped:        
            print("Mass⁻¹ (lumped) \033[93m[FOM] \033[m")
            print('  '+base.arr2str(Mass_inv.todense(), prefix='  '))
        else:
            print("Mass⁻¹ \033[93m[FOM] \033[m")
            print('  '+base.arr2str(Mass_inv.todense(), prefix='  '))

    return Stiff,Mass_inv

# Definition of the scalar products
def C_Matrix(p,N,HS):
    """Reduced C-Matrix (C(p))ᵢⱼ=⟨ψᵢ,𝒯(p)ψⱼ⟩ , see Lemma A.5 in the Thesis

    Parameters
    ----------
    path : float
        Path value
    N : int
        Number of spatial nodes
    HS : float
        Spatial step width

    Returns
    -------
    C-Matrix : sparse
        sparse C-matrix
    """
    q=math.floor(p/HS)
    p_tilde=p-q*HS
    diags = transpose(full((N, 4), [(4*HS**3-6*HS*p_tilde**2+3*p_tilde**3)/(6*HS**2),(HS-p_tilde)**3/(6*HS**2),(HS**3+3*HS**2*p_tilde+3*HS*p_tilde**2-3*p_tilde**3)/(6*HS**2),p_tilde**3/(6*HS**2)]))
    return spdiags(diags, [-q,-q+1,-q-1,-q-2], N, N)

def Q_hat_Matrix(p,N,HS):
    """Reduced Q̂-Matrix (Q̂(p))ᵢⱼ=⟨∇ψᵢ,∇𝒯(p)ψⱼ⟩ , see Lemma A.5 in the Thesis

    Parameters
    ----------
    path : float
        Path value
    N : int
        Number of spatial nodes
    HS : float
        Spatial step width

    Returns
    -------
    Q̂-Matrix : sparse
        sparse Q̂-matrix
    """
    q=math.floor(p/HS)
    p_tilde=p-q*HS
    diags = transpose(full((N, 4), [(2*HS-3*p_tilde)/(HS**2),(p_tilde-HS)/(HS**2),(3*p_tilde-HS)/(HS**2),p_tilde**2/(HS**2)]))
    return spdiags(diags, [-q,-q+1,-q-1,-q-2], N, N)

def R_hat_Matrix(p,N,HS):
    """Reduced R̂-Matrix (R̂(p))ᵢⱼ=⟨∇ψᵢ,𝒯(p)ψⱼ⟩ , see Lemma A.5 in the Thesis

    Parameters
    ----------
    path : float
        Path value
    N : int
        Number of spatial nodes
    HS : float
        Spatial step width

    Returns
    -------
    R̂-Matrix : sparse
        sparse R̂-matrix
    """
    q=math.floor(p/HS) 
    p_tilde=p-q*HS
    diags = transpose(full((N, 4), [(3*p_tilde**2-4*p_tilde*HS)/(2*HS**2),-(HS-p_tilde)**2/(2*HS**2),(HS**2+2*HS*p_tilde-3*p_tilde**2)/(2*HS**2),p_tilde**2/(2*HS**2)]))
    return spdiags(diags, [-q,-q+1,-q-1,-q-2], N, N)  

def ROM(t : float, R : int, N : int, HS : float,  phi : np.array, p : np.array, lumped = False, disp = False):
    """Defines the mass and stiffness matrix for the reduced order model, according to definition 4.1.

    Parameters
    ----------
    t : float
        Time in ms
    R : int
        Dimension of reduced ansatz space
    N : int
        Spatial Dimension
    HS : float
        Spatial step-width
    phi : np.array
        Parameter for the coefficients of the linear combination to create the reduced basis functions.
    p : np.array
        Time-discrete shifting parameter
    lumped : bool, optional
        Instruction if lumped mass matrix should be used, by default False
    disp : bool, optional
        Instruction if matrices should be displayed, by default False

    Returns
    -------
    (np.array,np.array,np.array)
        Stiffness matrix, inverted mass matrix and projection matrix 
    """
     
    A_hat=np.empty([R,R])
    B_hat=np.empty([R,R])
    M_hat=np.empty([R,R])

    p_hats=np.empty([R,R])
    for i in range(R): p_hats[i,:]=(p[i]-p[:])*t

    for i in range(R):
        for j in range(R):
            p_hat=p_hats[i,j]
            A_hat[i,j]= phi[i,:] @ (phi[j,:] @ Q_hat_Matrix(p_hat,N,HS)).T
            M_hat[i,j]= phi[i,:] @ (phi[j,:] @ C_Matrix(p_hat,N,HS)).T
            B_hat[i,j]= phi[i,:] @ (phi[j,:] @ R_hat_Matrix(p_hat,N,HS)).T        
        B_hat[:,i]= -p[i]*B_hat[:,i]
    if lumped:
        diags_mass_lumped=M_hat @ ones(N)
        Mass_lumped= diags(diags_mass_lumped, 0, shape=(N, N))
        Mass_inv=linalg.inv(Mass_lumped)
    else:
        Mass_inv=scipy.linalg.inv(M_hat)  
            
    Stiff = - C * A_hat - B_hat

    if disp:
        print("Stiffness \033[91m[reduced method] \033[m")
        print('  '+base.arr2str(Stiff, prefix='  '))
        print("Mass \033[91m[reduced method] \033[m")
        print('  '+base.arr2str(M_hat, prefix='  '))
        if lumped:        
            print("Mass⁻¹ (lumped) \033[91m[reduced method] \033[m")
            print('  '+base.arr2str(Mass_inv, prefix='  '))
        else:
            print("Mass⁻¹ \033[91m[reduced method] \033[m")
            print('  '+base.arr2str(Mass_inv, prefix='  '))

    return Stiff,Mass_inv

def ROM_tuned(t : float, R : int, N : int, HS : float,  phi : np.array, p : np.array, lumped = False, disp = False):
    """Defines the mass and stiffness matrix for the reduced order model, according to definition 4.1.
    Tuned version using global pre-calculated matrices

    Parameters
    ----------
    t : float
        Time in ms
    R : int
        Dimension of reduced ansatz space
    N : int
        Spatial Dimension
    HS : float
        Spatial step-width
    phi : np.array
        Parameter for the coefficients of the linear combination to create the reduced basis functions.
    p : np.array
        Time-discrete shifting parameter
    lumped : bool, optional
        Instruction if lumped mass matrix should be used, by default False
    disp : bool, optional
        Instruction if matrices should be displayed, by default False

    Returns
    -------
    (np.array,np.array,np.array)
        Stiffness matrix, inverted mass matrix and projection matrix 
    """
     
    A_hat=np.empty([R,R])
    B_hat=np.empty([R,R])
    M_hat=np.empty([R,R])

    p_hats=np.empty([R,R])
    for i in range(R): p_hats[i,:]=(p[i]-p[:])*t


    global pre_bool
    global pre_A_hat,pre_M_hat,pre_B_hat
    if pre_bool==False:
        A_hat=np.empty([R,R])
        B_hat=np.empty([R,R])
        M_hat=np.empty([R,R])

        for i in range(R):
            for j in range(R):
                p_hat=0
                A_hat[i,j]= phi[i,:] @ (phi[j,:] @ Q_hat_Matrix(p_hat,N,HS)).T
                M_hat[i,j]= phi[i,:] @ (phi[j,:] @ C_Matrix(p_hat,N,HS)).T
                B_hat[i,j]= phi[i,:] @ (phi[j,:] @ R_hat_Matrix(p_hat,N,HS)).T        

        pre_bool=True
        pre_A_hat,pre_M_hat,pre_B_hat=np.copy(A_hat),np.copy(M_hat),np.copy(B_hat)
    
    A_hat=np.copy(pre_A_hat)
    B_hat=np.copy(pre_B_hat)
    M_hat=np.copy(pre_M_hat)
    for i in range(R):
        for j in range(R):
            if p_hats[i,j]!=0:
                p_hat=p_hats[i,j]
                A_hat[i,j]= phi[i,:] @ (phi[j,:] @ Q_hat_Matrix(p_hat,N,HS)).T
                M_hat[i,j]= phi[i,:] @ (phi[j,:] @ C_Matrix(p_hat,N,HS)).T
                B_hat[i,j]= phi[i,:] @ (phi[j,:] @ R_hat_Matrix(p_hat,N,HS)).T   
        B_hat[:,i]= -p[i]*B_hat[:,i]


    if lumped:
        diags_mass_lumped=M_hat @ ones(N)
        Mass_lumped= diags(diags_mass_lumped, 0, shape=(N, N))
        Mass_inv=linalg.inv(Mass_lumped)
    else:
        Mass_inv=scipy.linalg.inv(M_hat)  
            
    Stiff = - C * A_hat - B_hat

    if disp:
        print("Stiffness \033[91m[reduced method] \033[m")
        print('  '+base.arr2str(Stiff, prefix='  '))
        print("Mass \033[91m[reduced method] \033[m")
        print('  '+base.arr2str(M_hat, prefix='  '))
        if lumped:        
            print("Mass⁻¹ (lumped) \033[91m[reduced method] \033[m")
            print('  '+base.arr2str(Mass_inv, prefix='  '))
        else:
            print("Mass⁻¹ \033[91m[reduced method] \033[m")
            print('  '+base.arr2str(Mass_inv, prefix='  '))

    return Stiff,Mass_inv

def ROM_projection(t : float, R : int, N : int, HS : float,  phi : np.array, p : np.array, disp = False):
    """Defines the projection matrix according to equation 4.6

    Parameters
    ----------
    t : float
        Time in ms
    R : int
        Dimension of reduced ansatz space
    N : int
        Spatial Dimension
    HS : float
        Spatial step-width
    phi : np.array
        Parameter for the coefficients of the linear combination to create the reduced basis functions.
    p : np.array
        Time-discrete shifting parameter
    disp : bool, optional
        Instruction if matrices should be displayed, by default False

    Returns
    -------
    P_bar : np.array
        Projection matrix
    """
    P_bar=np.empty([R,N])
    for i in range(R):
        P_bar[i,:] = C_Matrix(p[i]*t,N,HS).dot(phi[i,:])
    
    if disp:
        print("P̄  \033[91m[reduced method] \033[m")
        print('  '+base.arr2str(P_bar, prefix='  '))
    return P_bar

def up_projection(v : np.array , projection : tuple):
    """Performs a up-projection according to eqaution 4.6 in the Thesis

    Parameters
    ----------
    v : np.array
        Coefficients to be projected
    projection : tuple
        All necessarry matrices for the projection

    Returns
    -------
    v_up : np.array
        up-projected coefficients
    """
    (Mass_bar_inv, _,P_bar)=projection
    return Mass_bar_inv @  P_bar.T @ v

def down_projection(v : np.array , projection : tuple):
    """Performs a down-projection according to eqaution 4.7 in the Thesis

    Parameters
    ----------
    v : np.array
        Coefficients to be projected
    projection : tuple
        All necessarry matrices for the projection

    Returns
    -------
    v_up : np.array
        down-projected coefficients
    """

    (_, Mass_hat_inv,P_bar)=projection
    return Mass_hat_inv @ P_bar @ v
    
# =============================================================================
# Solving Algorithms
# =============================================================================

def solve_FOM(z_0 : np.array , discretization : dict ,  method = 'EE', bound = 'Neumann'):
    """Solves the Full Order Model by executing the time-stepping scheme

    Parameters
    ----------
    z_0 : np.array
        Initial values of all the states
    discretization : dict
        Dictionary containing the time and spatial discretization and all parameters
    method : str, optional
        Time-stepping scheme to solve the semi-discrete FOM, by default 'EE'
    bound : str, optional
        Boundary for the FOM, by default 'Neumann'

    Returns
    -------
    (states, time) : (np.array,float)
        The solution of the FOM and the computation time

    Raises
    ------
    Exception
        If the time-stepping scheme is not defined
    """
    z=(z_0.T).flatten()

    FOM(discretization['N'],discretization['HS'], disp=True)

    if method == 'EE':
        print(f"  Method:           Explicit Euler")
        states, time = explicit_euler(z, discretization, bound)
    elif method=='IE':
        print(f"  Method:           Implicit Euler")
        states, time = implicit_euler(z, discretization, bound)
    else:
        raise Exception("No matching method was found")

    
    base.write_log([f"{'':<15} Problem type: Original \n",f"{'':<15} Boundary: {bound} \n",f"{'':<15} Method: {method} \n",f"{'':<15} Computation time: {time:.4f} s\n"],hash_log)
    if hash_log!=None:
        np.save(str("solutions/"+hash_log+"_states.npy"),states)
    return states, time

def solve_ROM(z_0 : np.array , discretization : dict, decomp_vars : tuple ,  method='EE'):
    """Solves the Reduced Order Model by executing the time-stepping scheme, see algorithm 1 in the Thesis

    Parameters
    ----------
    z_0 : np.array
        Initial values for all states
    discretization : dict
        Dictionary containing the time and spatial discretization and all parameters
    decomp_vars : tuple
        Tuple containing the values for the reduced ansatz space (phi , p)
    method : str, optional
        Time-stepping scheme, by default 'EE'

    Returns
    -------
    states_full : np.array
        Full-dimensional solution
    states : np.array
        Solution in the reduced ansatz space for the voltage and full-dimensions for other states
    exec_time : float
        Computation time in seconds
    proj_time : float
        Time used to project the solution into high dimensions in seconds

    Raises
    ------
    Exception
        If the time-stepping scheme is not defined
    """
    N,R,HS,M,T,HT=discretization['N'],discretization['R'],discretization['HS'],discretization['M'],discretization['T'],discretization['HT']

    (phi, p)=decomp_vars
    _,Mass_bar_inv=FOM(N,HS,disp=True)

    _,Mass_hat_inv=ROM_tuned(0,R,N,HS,phi,p,disp=True)
    P_bar=ROM_projection(0,R,N,HS,phi,p)

    z_org=(z_0.T).flatten()
    global shift
    shift=z_org[0]

    v_0=np.subtract(z_org[:N],shift)

    projection=(Mass_bar_inv, Mass_hat_inv,P_bar)
    v_down=np.asarray(down_projection(v_0, projection))

    z=np.concatenate([v_down, z_org[N:]])

    if method == 'EE':
        states, exec_time = explicit_euler(z, discretization, decomposed=(phi,p))
    elif method=='IE':
        states, exec_time = implicit_euler(z, discretization, decomposed=decomp_vars)

    else:
        raise Exception("No matching method was found")
    

    states_full=np.empty((discretization['M']+1,N*4))
    states_full[:,N:]=states[:,R:]

    states_full[0,:]=z_org
    
    # Define time-discretization
    ts=np.linspace(0,T,M+1)

    assert ts[1]-ts[0]==HT, 'Time-discretization destroys given time-step-width'
   
    start_time=time.time()
    for i,t in enumerate(ts):
        _,Mass_hat_inv=ROM_tuned(t,R,N,HS,phi,p)
        P_bar=ROM_projection(t,R,N,HS,phi,p)
        projection=(Mass_bar_inv, Mass_hat_inv, P_bar)
        states_full[i,:N]=up_projection(states[i,:R],projection)+shift
        print("Projection {:5} / {:5} [{}{}] {:3} %".format(i+1, M, "#"*int((i+1)/(M)*20), "-"*(20-int((i+1)/(M)*20)),int((i+1)/(M)*100)), end='\r')
    
    proj_time=time.time()-start_time
    print(" "*100, end='\r')
    print('Computation time projection: {:.4f}s'.format(proj_time))
  
    base.write_log([f"{'':<15} Problem initialization: Reduced , shift={shift}\n",f"{'':<15} Reduced discretization: R={R}\n",f"{'':<15} Method: {method} \n",f"{'':<15} Computation time: {exec_time:.4f} s\n"],hash_log)
    if hash_log!=None:
        print(np.shape(states[:,:R]),np.shape(phi),np.shape(p))
        np.save(str("solutions/"+hash_log+"_decomposed.npy"),np.concatenate([np.asarray([states[:,:R]]).flatten(), phi.flatten(), p.flatten()]))
        np.save(str("solutions/"+hash_log+"_states_reduced.npy"),states)
        np.save(str("solutions/"+hash_log+"_states.npy"),states_full)
    return states_full,states,exec_time,proj_time

# =============================================================================
# Main-Method
# =============================================================================
def main(T : float , HT : float , problem : str , method : str , initial_values : list , reduced_data = 'None' , waves = 'Both' , save = True):
    """_summary_

    Parameters
    ----------
    T : float
        Simulation time in milliseconds
    HT : float
        Time-step width in milliseconds
    problem : str
        High-dimensional or reduced problem
    method : str
        Time-stepping scheme
    initial_values : list
        Information about the initial values, see line method "load_initial_data" for more information
    reduced_data : str, optional
        Hash to the reduced data, by default 'None'
    waves : str, optional
        Defines which side of the wavefront should be used, by default 'Both'
    save : bool, optional
        Control if the results should be saved or not, by default True

    Returns
    -------
    hash, time : str, float
        Hash under which the data is stored and the computation time
    """
    global hash_log
    if save:
        hash_log=base.create_hash()
        with open("logs.txt",'a') as logs:
            logs.write("\n{:<15}SOLVER\n".format(hash_log))
        print("Used hash: {}".format(hash_log))
    else:
        hash_log=None

    z_0, discretization = load_initial_data(hash_log, initial_values, waves=waves)

    if len(initial_values)>1:
        if initial_values[0]=='extract':
            T=T-initial_values[2]
    # Construct discretization in time
    M=max(1, int((T)/HT + 0.5))
    discretization['T']=T
    discretization['HT']=HT
    discretization['M']=M

    base.write_log([f"{'':<15} Time discretization: M={M+1} , HT={HT} cm, T={T}\n"], hash_log)

    print('Solver parameters:')
    print(f"  simulation time:  {T} ms")
    print(f"  time step-width:  {HT} ms")
    print(f"  time steps:       {M}")
    if method=='EE':
        print(f"  Method:           Explicit Euler")
    elif method=='IE':
        print(f"  Method:           Implicit Euler")

    if problem=='FOM':
        _,time=solve_FOM(z_0,discretization,method=method)
        time=[time]
    elif problem=='ROM':
        decomposed, discretization = load_reduced_data(hash_log, [reduced_data], discretization)
        _, _, time, proj_time=solve_ROM(z_0, discretization, decomposed, method=method)
        time=[time,proj_time]

    gc.collect()
    return hash_log, time

# =============================================================================
# Call to Main
# =============================================================================
if __name__ == '__main__':

    T=22
    HT=1e-3
    problem='FOM'
    method='EE'
    initial_values=['standard']

    if len(sys.argv)>1:
        T=float(sys.argv[1])
        HT=float(sys.argv[2])
        problem=sys.argv[4]

    main(T, HT,  problem, method, initial_values)