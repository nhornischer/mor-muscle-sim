# Imports
import numpy as np
from scipy.optimize import minimize
from numpy import reshape, full, transpose
from scipy.sparse import spdiags, kron, eye, block_diag
import math

import base

# All vectors are collumn vectors and need to be transposed to match the PDF
# Also the matrices have to be transposed.

# =============================================================================
# Problem and Datageneration
# =============================================================================
cost_values=[]
def load_problem(hash : str , hash_data : str , shift = False , tasks = ['SOLVER','NONLINEAR'] , time_range = 4 , wave='Both'):
    """Loads the snapshots matrices stored under the given hash_data.

    Parameters
    ----------
    hash : str
        Hash of the log entry refering to the current decomposition task.
    hash_data : str
        Hash of the data to be loaded.
    shift : bool, optional
        Controls if the snapshot data of the solver is shifted or not, by default False
    tasks : list, optional
        List of string of the different tasks to be loaded under the given hash_data, by default ['SOLVER','NONLINEAR']
    time_range : double, optional
        Selection of the time range to reduce computation time in ms, by default 4

    Returns
    -------
    (list, dict, list)
        List of snapshots of numpy arrays and the discretization of the problem and list of shift values.
    """
    discretization,_=base.read_log(hash_data)
    
    data=[]
    shifts=[]
    for current_task in tasks:
        try:
            loaded_data=base.load_np_data(hash_data,task=current_task)
            time_selection=min(time_range,discretization['T'])
        except:
            print('No data avaiable for hash {} and task {}'.format(hash_data,current_task))
            continue
        
        M=max(1, int((time_selection)/discretization['HT'] + 0.5))
        if current_task=='SOLVER':
            # Extract voltage
            snapshot=loaded_data[:(M+1),:discretization['N']]
            
            if shift:
                # Transfer to new zero-value
                shift_value=snapshot[-1,-1]
                print(f'Shifted input by value {shift_value}')
                snapshot=np.subtract(snapshot,shift_value)
                shifts.append(shift_value)
            else:
                shifts.append(0)
        elif current_task=='NONLINEAR':
            snapshot=loaded_data[:(M+1),:]
            shifts.append(0)
        elif current_task=='LINEAR':
            snapshot=loaded_data[:(M+1),:]
            shifts.append(0)
        if wave=='Left':
            snapshot[:,int(np.shape(snapshot)[1]/2):]=np.full([int(np.shape(snapshot)[0]),int(np.shape(snapshot)[1]/2)+1],snapshot[-1,-1])
            print(f'Extract left part')
        elif wave=='Right':
            snapshot[:,:int(np.shape(snapshot)[1]/2)]=np.full([int(np.shape(snapshot)[0]),int(np.shape(snapshot)[1]/2)],snapshot[-1,-1])
            print(f'Extract right part')
        data.append(snapshot)
    discretization['M']=np.shape(snapshot)[0]
    discretization['T']=time_selection
    assert len(data)>0, 'No data avaiable.'



    base.write_log([f"{'':<15} Original data: {hash_data} for tasks: {tasks} \n",f"{'':<15} Spatial discretization: N={discretization['N']} , HS={discretization['HS']} cm , Gamma={len(data)}\n",f"{'':<15} Time discretization: T={discretization['HT']*M} ms , HT={discretization['HT']} ms , M={discretization['M']} \n"],hash)
    if shift:
        base.write_log([f"{'':<15} Data shifts: {shifts}\n"],hash)
        
    print('Loaded solution')
    print(f"  simulation time:      {discretization['T']} ms")
    print(f"  decomposition time:   {discretization['HT']*M} ms")
    print(f"  length:               {discretization['N']*discretization['HS']} cm")
    print(f"  spatial-nodes:        {discretization['N']}")
    print(f"  time samples:         {discretization['M']}")
    print(f"  spatial step-width:   {discretization['HS']} cm")
    print(f"  time step-width:      {discretization['HT']} ms")  
    return data, discretization, shifts
   
def initial_values(data : list , discretization : dict , hash : str , initial = None):
    """Creates the initial values for the decomposition task. 

    Parameters
    ----------
    data : list
        List of snapshot data to create the initial values.
    discretization : dict
        Discretization of the snapshot data.
    hash : str
        Hash of the log entry refering to the current decomposition task.
    initial : str, optional
        Specifing the way the initial values are created, by default None

    Returns
    -------
    (np.array,np.array,np.array)
        Intial values (alpha,phi,paths)
    """
    N,M,T,HS,R,Gamma,HT=discretization['N'],discretization['M'],discretization['T'],discretization['HS'],discretization['R'],discretization['Gamma'],discretization['HT']
    # Guessing initial transport for all snapshots, see Algorithm 2 in the Thesis
    slopes=[]
    for gamma in range(Gamma):
        slopes.append((np.argmax(data[gamma][-1, :])-np.argmax(data[gamma][0, :]))*HS/T)  
    slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
    slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
    alpha = np.ones([Gamma, M, R])
    phi = np.zeros([R,N])
    paths = np.empty([R])

    # Creating initial variables
    if initial=='snapshots':
        # Initialisation with complete snapshot
        for gamma in range(Gamma):
            alpha[gamma,:,int(R/Gamma)*gamma:int(R/Gamma)*(gamma+1)]=1/int(R/Gamma)
            phi[int(R/Gamma)*gamma:int(R/Gamma)*(gamma+1),:]=data[gamma][:int(R/Gamma),:]
        paths = full(R, slopes[:R])

    elif initial=='dualSnapshots':
        # Initialisation with snapshot for each side separately
        slopes=[]
        for gamma in range(Gamma):
            slopes.append((np.argmax(data[gamma][-1, :int(N/2)])-np.argmax(data[gamma][0, :int(N/2)]))*HS/T)  
            slopes.append((np.argmax(data[gamma][-1, int(N/2):])-np.argmax(data[gamma][0, int(N/2):]))*HS/T)
        slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
        paths = 1 * full(R, slopes[:R])
        alpha[:,:,:] = np.ones([Gamma, M, R]) / (R / 2) 
        for i in range(R):
            if i%2==0:
                phi[i,:int(N/2)]=data[0][0,:int(N/2)]
            else:
                phi[i,int(N/2):]=data[0][0,int(N/2):]

    elif initial=='dualSnapshotsOptimized':
        slopes=[]
        for gamma in range(Gamma):
            slopes.append((np.argmax(data[gamma][-1, :int(N/2)])-np.argmax(data[gamma][0, :int(N/2)]))*HS/T)  
            slopes.append((np.argmax(data[gamma][-1, int(N/2):])-np.argmax(data[gamma][0, int(N/2):]))*HS/T)
        slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
        paths = 1*full(R, slopes[:R])
        optValues = base.load_np_data("cc01273f333e", 'DECOMPOSITION')
        optDiscr,_ = base.read_log("cc01273f333e")
        opt_R = optDiscr['R']
        paths[:opt_R] = optValues[Gamma*M*opt_R + opt_R*N:]
        alpha[:,:,:opt_R] = np.reshape(optValues[:Gamma*M*opt_R],(Gamma,M,opt_R))
        phi[:opt_R,:] = np.reshape(optValues[Gamma*M*opt_R:Gamma*M*opt_R + opt_R*N],(opt_R,N))

    elif initial=='SimpleLeft':
        # Initialisation with zero values for left side of the wave
        slopes=[]
        for gamma in range(Gamma):
            slopes.append((np.argmax(data[gamma][-1, :int(N/2)])-np.argmax(data[gamma][0, :int(N/2)]))*HS/T)  
            slopes.append((np.argmin(data[gamma][-1, :int(N/2)])-np.argmin(data[gamma][0, :int(N/2)]))*HS/T)
        
        slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
        print(slopes)
        alpha=np.ones([Gamma, M, R])
        paths=full(R,slopes[:R])

    elif initial=='SimpleRight':
        # Initialisation with zero values for right side of the wave
        slopes=[]
        for gamma in range(Gamma):
            slopes.append((np.argmax(data[gamma][-1, int(N/2):])-np.argmax(data[gamma][0, int(N/2):]))*HS/T)
            slopes.append((np.argmin(data[gamma][-1, int(N/2):])-np.argmin(data[gamma][0, int(N/2):]))*HS/T)
        
        slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
        print(slopes)
        alpha=np.ones([Gamma, M, R])
        paths=full(R,slopes[:R])

    elif initial=='SimpleOptimized':
        slopes=[]
        for gamma in range(Gamma):
            slopes.append((np.argmax(data[gamma][-1, :int(N/2)])-np.argmax(data[gamma][0, :int(N/2)]))*HS/T)  
            slopes.append((np.argmax(data[gamma][-1, int(N/2):])-np.argmax(data[gamma][0, int(N/2):]))*HS/T)
            slopes.append((np.argmin(data[gamma][-1, :int(N/2)])-np.argmin(data[gamma][0, :int(N/2)]))*HS/T)
            slopes.append((np.argmin(data[gamma][-1, int(N/2):])-np.argmin(data[gamma][0, int(N/2):]))*HS/T)
        optValues = base.load_np_data("78feb399fce0", 'DECOMPOSITION')
        optDiscr,_ = base.read_log("78feb399fce0")
        opt_R = optDiscr['R']
        slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
        alpha=np.ones([Gamma, M, R])
        phi=np.zeros([R,N])
        paths=full(R,slopes[:R])
        paths[:opt_R] = optValues[Gamma*M*opt_R + opt_R*N:]
        alpha[:,:,:opt_R] = np.reshape(optValues[:Gamma*M*opt_R],(Gamma,M,opt_R))
        phi[:opt_R,:] = np.reshape(optValues[Gamma*M*opt_R:Gamma*M*opt_R + opt_R*N],(opt_R,N))

    else:
        # Initialisation with zero values for both sides of the wave
        slopes=[]
        for gamma in range(Gamma):
            slopes.append((np.argmax(data[gamma][-1, :int(N/2)])-np.argmax(data[gamma][0, :int(N/2)]))*HS/T)  
            slopes.append((np.argmax(data[gamma][-1, int(N/2):])-np.argmax(data[gamma][0, int(N/2):]))*HS/T)
            slopes.append((np.argmin(data[gamma][-1, :int(N/2)])-np.argmin(data[gamma][0, :int(N/2)]))*HS/T)
            slopes.append((np.argmin(data[gamma][-1, int(N/2):])-np.argmin(data[gamma][0, int(N/2):]))*HS/T)
        
        slopes=np.concatenate((np.asarray(slopes),np.asarray(slopes)))
        print(slopes)
        alpha=np.ones([Gamma, M, R])
        paths=full(R,slopes[:R])

    print(f"Initial values:")
    print(f"  p: {np.min(paths):>+.2e} -- {np.max(paths):>+.2e} {np.shape(paths)}")
    print(f"  α: {np.min(alpha):>+.2e} -- {np.max(alpha):>+.2e} {np.shape(alpha)}")
    print(f"  φ: {np.min(phi):>+.2e} -- {np.max(phi):>+.2e} {np.shape(phi)}")
    
    return (alpha,phi,paths)

# =============================================================================
# Definition of the cost function
# =============================================================================

def cost(x : np.array , *params):
    """Discretized cost function, see Equation 3.17 and 3.13 in the Thesis

    Parameters
    ----------
    x : np.array
        Optimization variables (alpha,phi,paths)
    params : list
        Parameters of the cost functional (true_data,M,N,R,T,HT,HS,Gamma)
    Returns
    -------
    float
        Value of the cost function
    """
    # Extraction of the variables
    data = params[0]
    [M,N,R,T,HT,HS,Gamma]=params[1:]
    alpha_full = reshape(x[0:Gamma*M*R], [Gamma,M, R])
    phi = reshape(x[Gamma*M*R:Gamma*M*R+R*N], [R, N])
    p = x[Gamma*M*R+R*N:]

    # Definition of the weights
    weights = full(M, HT)
    weights[[0, -1]] = weights[[0, -1]]/2

    # Define time-discretization
    ts=np.linspace(0,T,M)


    assert ts[1]-ts[0]==HT, 'Time-discretization destroys given time-step-width'

    # Pre calculated matrices over all time-steps
    Mass=M_Matrix(N,HS)
    C_i=np.empty([R,M],dtype=object)
    B_i_j=np.empty([R,R,M],dtype=object)
    for time_step,t in enumerate(ts):
        for i in range(R):
            C_i[i,time_step]=C_Matrix(p[i]*t,N,HS)
            for j in range(R):
                B_i_j[i,j,time_step]=C_Matrix(p[i]*t-p[j]*t,N,HS)

    # Cost function J
    J_gamma=np.empty([Gamma])
    for gamma in range(Gamma):

        alpha=alpha_full[gamma,:,:]
        snapshot=data[gamma]

        J_gamma[gamma] = np.dot(weights,[(
            (snapshot[time_step, :]@(snapshot[time_step, :] @ Mass).T)
            -2*sum([alpha[time_step, i]*(phi[i, :]@(snapshot[time_step, :]@C_i[i,time_step]).T) for i in range(R)])
            +sum([alpha[time_step, i]*alpha[time_step, j]*(phi[i, :]@ (phi[j, :]@B_i_j[i,j,time_step]).T) for i in range(R) for j in range(R)])
        )for time_step in range(M)]) #Integration over all timesteps using the trapezoidal rule
    J=0.5*np.sum(J_gamma)
    global cost_values
    cost_values.append(abs(J))
    return abs(J)

# =============================================================================
# Definition of the derivatives
# =============================================================================

def jacobian(x : np.array , *params):
    """Calculation of the jacobian, see Equations 3.21-3.23 in the Thesis

    Parameters
    ----------
    x : np.array
        Optimization variables (alpha,phi,paths)
    params : list
        Parameters of the cost functional (true_data,M,N,R,T,HT,HS,Gamma)
    Returns
    -------
    np.array()
        Flattened jacobian
    """
    # Extraction of the variables
    data = params[0]
    [M,N,R,T,HT,HS,Gamma]=params[1:]
    alpha_full = reshape(x[0:Gamma*M*R], [Gamma,M, R])
    phi = reshape(x[Gamma*M*R:Gamma*M*R+R*N], [R, N])
    p = x[Gamma*M*R+R*N:]

    my_hat = np.empty([R, M])
    ny_i = np.empty([M, N])
    xi_hat_gamma = np.empty([R, M])

    # Precalculated matrices
    C_i=np.empty([R,M],dtype=object)
    B_i_j=np.empty([R,R,M],dtype=object)
    E_i=np.empty([R,M],dtype=object)
    D_i_j=np.empty([R,R,M],dtype=object)

    # Define time-discretization
    ts=np.linspace(0,T,M)

    assert ts[1]-ts[0]==HT, 'Time-discretization destroys given time-step-width'

    for time_step,t in enumerate(ts):
        for i in range(R):
            C_i[i,time_step]=C_Matrix(p[i]*t,N,HS)
            E_i[i,time_step]=E_Matrix(p[i]*t,N,HS)
            for j in range(R):
                B_i_j[i,j,time_step]=C_Matrix(p[i]*t-p[j]*t,N,HS)
                D_i_j[i,j,time_step]=E_Matrix(p[i]*t-p[j]*t,N,HS)
    
    weights = full(M, HT)
    weights[[0, -1]] = weights[[0, -1]]/2
    weight_matrix = kron(eye(R), spdiags(weights, 0, M, M))

    dcost_dphi=np.zeros([N*R])
    dcost_dp=np.zeros([R])
    for gamma in range(Gamma):
        alpha=alpha_full[gamma,:,:]
        snapshot=data[gamma]
        for i in range(R):
            for time_step in range(M):
                my_hat[i, time_step] = (
                    sum((alpha[time_step, j]*phi[i,:]@(phi[j, :]@B_i_j[i,j,time_step]).T for j in range(R)))
                    -(phi[i, :]@(snapshot[time_step,:]@C_i[i,time_step]).T)
                )
                ny_i[time_step, :] = alpha[time_step, i]*sum((alpha[time_step, j] * (phi[j,:] @ B_i_j[i,j,time_step]).T-(snapshot[time_step,:]@C_i[i,time_step]).T) for j in range(R))

                xi_hat_gamma[i, time_step] = alpha[time_step, i]*(
                    sum((alpha[time_step, j]*phi[i, :]@(phi[j, :]@D_i_j[i,j,time_step]).T for j in range(R)))
                    -(phi[i, :]@(snapshot[time_step, :]@E_i[i,time_step]).T)
                )
            if i == 0:
                ny_hat_gamma = ny_i
            else:
                ny_hat_gamma = block_diag((ny_hat_gamma, ny_i))

        if gamma == 0:
            dcost_dalpha = my_hat.flatten() @ weight_matrix
        else:
            dcost_dalpha=np.concatenate([dcost_dalpha, my_hat.flatten() @ weight_matrix])
        dcost_dphi =dcost_dphi + np.ones(R*M) @ (weight_matrix.dot(ny_hat_gamma))
        dcost_dp = dcost_dp + xi_hat_gamma @ weights

    return np.concatenate([dcost_dalpha, dcost_dphi, dcost_dp])

# =============================================================================
# Functions for cost and derivative
# =============================================================================

def M_Matrix(N : int , HS : float):
    """Full-dimensional mass matrix (M)ᵢⱼ=⟨ψᵢ,ψⱼ⟩, see Lemma A.1 in the Thesis

    Parameters
    ----------
    N : int
        Number of spatial nodes
    HS : float
        Spatial step width

    Returns
    -------
    M-Matrix : sparse
        sparse mass matrix
    """
    diags = transpose(full((N, 3), [2/3*HS, HS/6, HS/6]))
    return spdiags(diags, [0, -1, 1], N, N)

def C_Matrix(path : float , N : int , HS : float):
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
    q=math.floor(path/HS)
    p_tilde=path-q*HS
    diags = transpose(full((N, 4), [(4*HS**3-6*HS*p_tilde**2+3*p_tilde**3)/(6*HS**2), 1/6/(HS**2)*(HS-p_tilde)**3, 1/(HS**2)*(1/6*(HS-p_tilde)**3-1/3*(p_tilde**3)+(HS**2)*p_tilde), 1/6/(HS**2)*(p_tilde**3)]))
    return spdiags(diags, [-q, -q+1, -q-1, -q-2], N, N)

def E_Matrix(path : float , N : int , HS : float):
    """Reduced E-Matrix (E(p))ᵢⱼ=⟨ψᵢ , ∂ₚ𝒯(p) ψⱼ⟩ , see Lemma A.5 in the Thesis

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
    E-Matrix : sparse
        sparse E-matrix
    """
    q=math.floor(path/HS)
    p_tilde=path-q*HS
    diags = transpose(full((N, 4), [(-1/(HS**2)*(2*p_tilde*HS-3/2*p_tilde**2)), (-1/2/(HS**2)*(HS-p_tilde)**2), (-1/(HS**2)*(2*p_tilde**2-2*HS*p_tilde-1/2*(HS-p_tilde)**2)), (1/2/(HS**2)*(p_tilde)**2)]))
    return spdiags(diags, [-q, -q+1, -q-1, -q-2], N, N)

def omega(path : float , N : int , HS : float):
    """Technical matrix to evaluate the reduced ansatz space

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
    Matrix : sparse
        Technical matrix in sparse format
    """
    q=math.floor(path/HS)
    p_tilde=path-q*HS
    diags = transpose(full((N, 2), [(-p_tilde+HS)/HS, p_tilde/HS]))
    return spdiags(diags, [-q, -q-1], N, N)

# =============================================================================
# Definition of the high dimensional L2 error
# =============================================================================
def L_2_error(true_data : np.array , test_data : np.array , M : int , N : int , HT : float , HS : float , T : float , Gamma : int):
    """Evaluates the L-2 norm between two full-dimensional data sets

    Parameters
    ----------
    true_data : np.array
        True array with dimensions [M,N]
    test_data : np.array
        Test array with dimensions [M,N]
    M : int
        Number of time nodes
    N : int
        Number of spatial nodes
    HT : float
        Time-step width
    HS : float
        Spatial step width
    T : float
        Simulation time
    Gamma : int
        Number of samples

    Returns
    -------
    L2 : float
        Value of the L2 norm
    """
    Mass=M_Matrix(N,HS)

    # Definition of the weights
    weights = full(M, HT)
    weights[[0, -1]] = weights[[0, -1]]/2

    # Define time-discretization
    ts=np.linspace(0,T,M)

    assert ts[1]-ts[0]==HT, 'Time-discretization destroys given time-step-width'

    # Cost function J
    J_gamma=np.empty([Gamma])
    for gamma in range(Gamma):
        z=true_data[gamma]
        z_hat=test_data[gamma]

        J_gamma[gamma] = np.dot(weights,[(
            (z[time_step, :]@(z[time_step, :] @ Mass).T)
            -2*(z[time_step, :]@(z_hat[time_step, :] @ Mass).T)
            +(z_hat[time_step, :]@(z_hat[time_step, :] @ Mass).T)
        )for time_step in range(M)]) #Integration over all timesteps using the trapezoidal rule
    J=0.5*np.sum(J_gamma)
    global cost_values
    cost_values.append(abs(J))
    return abs(J)

# =============================================================================
# Definition of the optimization method
# =============================================================================
def minimization(data : list , x_initial : np.array , discretization : dict , hash : str):
    """Minimization algorithm to determine the reduced ansatz space.

    Parameters
    ----------
    data : list
        List of snapshot data.
    x_initial : np.array
        Initial values of the minimization.
    discretization : dict
        Discretization of the snapshot data.
    hash : str
        Hash of the log entry refering to the current decomposition task.

    Returns
    -------
    (tuple,list)
        Tuple of the optimized variables (alpha,phi,paths) and the list of snapshots.
    """
    N,M,T,HS,HT,R,Gamma=discretization['N'],discretization['M'],discretization['T'],discretization['HS'],discretization['HT'],discretization['R'],discretization['Gamma']

    (alpha_full,phi,paths)=x_initial

    x_initial_flat = np.concatenate([alpha_full.flatten(), phi.flatten(), paths.flatten()])
    # Visualization of the solution
    initial_snapshots=[]
    for gamma in range(Gamma):
        alpha=alpha_full[gamma,:,:]
        initial_vis = np.empty([M, N])
        for i,t in enumerate(np.linspace(0, T, M)):
            initial_vis[i, :] = sum([alpha[i, k]*((omega(paths[k]*t,N,HS)@phi[k, :])) for k in range(R)])
        initial_snapshots.append(initial_vis)
   
    print(f'Optimizer paramters:')
    print(f'  variables: {len(x_initial_flat)}')

    import time
    start_time=time.time()
    initial_cost=cost(x_initial_flat,data,M,N,R,T,HT,HS,Gamma)
    cost_time=time.time()-start_time
    print(f'  Initial cost: {initial_cost:.2f} evaluated in {cost_time:.4f} s')
    

    start_time=time.time()
    initial_jacobian=jacobian(x_initial_flat,data,M,N,R,T,HT,HS,Gamma)
    jacobian_time=time.time()-start_time
    print(f'  Initial jacobian: {np.mean(initial_jacobian):.2f} evaluated in {jacobian_time:.4f} s')

    base.write_log([f"{'':<15} Initial cost: {initial_cost:.2f} ({cost_time:.4f} s)\n",f"{'':<15} Initial jacobian: {np.mean(initial_jacobian):.2f} ({jacobian_time:.4f} s)\n"],hash)
    
    # Minimization Method
    res = minimize(cost, x_initial_flat, args=(data,M,N,R,T,HT,HS,Gamma), method='L-BFGS-B', jac=jacobian,
     options={'disp':True, 'maxiter':100, 'iprint':99})
    
    x = res.x
    # Extraction of the variables
    alpha_full = reshape(x[0:Gamma*M*R], [Gamma,M, R])
    phi = reshape(x[Gamma*M*R:Gamma*M*R+R*N], [R, N])
    p = x[Gamma*M*R+R*N:]

    # Visualization of the solution
    snapshots=[]
    for gamma in range(Gamma):
        alpha=alpha_full[gamma,:,:]
        sol_vis = np.empty([M, N])
        for i,t in enumerate(np.linspace(0, T, M)):
            sol_vis[i, :] = sum([alpha[i, k]*((omega(p[k]*t,N,HS)@phi[k, :])) for k in range(R)])
        snapshots.append(sol_vis)
        
    base.write_log([f"{'':<15} Optimizer success: {res.success} with cost {res.fun:.2f} \n",f"{'':<15} Iterations: {res.nit}\tEvaluations of function: {res.nfev} \n",f"{'':<15} Termination cause: {res.message}\n"],hash)
    if hash!=None:
        np.save("solutions/"+hash+"_decomposed",x)
        np.save("solutions/"+hash+"_decomp_initial",x_initial_flat)
    return (alpha_full,phi,p), snapshots

# =============================================================================
# Main-Method
# =============================================================================
def main(hash_data : str , R : int , data_shift : bool , initial_data : str , data_wave = 'Both' , save = True):
    """Main decomposition method, constructing the reduced ansatz space based on chapter 3 of the Thesis

    Parameters
    ----------
    hash_data : str
        Hash to the full-dimensional data
    R : int
        Dimensions of the reduced ansatz space
    data_shift : bool
        Control if the equilibrium shift should be applied
    initial_data : str
        _description_
    data_wave : str, optional
        Control which wavefront should be used, by default 'Both'
    save : bool, optional
        Control if the results should be saved, by default True

    Returns
    -------
    hash : str
        Hash under which the results are stored
    """
    if save:
        hash=base.create_hash()
        base.write_log([f"\n{hash:<15}DECOMPOSITION\n"],hash)
        print("Used hash: {}".format(hash))
    else:
        hash=None

    DATA, discretization, shifts = load_problem(hash,hash_data,shift=data_shift,tasks=['SOLVER'],wave=data_wave)
    
    discretization['Gamma']=np.shape(DATA)[0]
    discretization['R']=R

    base.write_log([f"{'':<15} Reduced discretization: R={R}\n",f"{'':<15} Initialization: init={initial_data} , shift={shifts[0]}\n"],hash)
    X_INITIAL=initial_values(DATA, discretization,hash,initial=initial_data)
    
    _, snapshots=minimization(DATA, X_INITIAL, discretization,hash)
    
    # Backshifts of the snapshot data
    for gamma in range(len(snapshots)):
        snapshots[gamma]=np.add(snapshots[gamma],shifts[gamma])

    if hash!=None:
        for gamma in range(discretization['Gamma']):
            np.save(f"solutions/{hash}_{str(gamma)}_states",snapshots[gamma])
    return hash

# =============================================================================
# Call to Main
# =============================================================================
if __name__ == '__main__':
    import sys

    assert len(sys.argv)>1, 'No input hash is given'

    hash_data=sys.argv[1]

    # Standard values
    R=2
    save = True
    data_shift=True
    initial_data='gausDual'

    if len(sys.argv)>2:
        R=int(sys.argv[2])
        save = sys.argv[3] in ['True', 'true', '1', 'SAVE', 'Save']
        data_shift= sys.argv[4] in ['True', 'true', '1', 'SAVE', 'Save']
        initial_data=sys.argv[5]


    main(hash_data, R, data_shift, initial_data, save)