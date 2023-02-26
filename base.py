import numpy as np

def create_hash():
    """Creates an unique hash, that is not already in use.

    Returns
    -------
    hash : str
        The created unique hash as a string.
    """
    import uuid
    # Reads existing hashes
    hashes=[]
    with open('hashes.txt', 'r') as f:
        lines=f.readlines()
        for line in lines:
            hashes.append(line.split('\n')[0])

    # Creates new unique hash
    while True:
        hash=''.join(str(uuid.uuid4()).split('-')[0:2])
        if hash not in hashes:
            with open('hashes.txt','a') as f:
                f.write('\n'+hash)
            break
    
    return hash

def remove_log(hash : str):
    """Deletes all files and log entries stored under the given hash.

    Parameters
    ----------
    hash : str
        Hash to the log entry, that should be deleted.
    """   
    # Delete hash from list of hashes 
    with open('hashes.txt','r') as fr:
        lines=fr.readlines()
        for i,line in enumerate(lines):
            if hash in line:
                index=i
                break
        if 'index' not in locals():
            print('No hash entry found')
        else:
            del lines[index]
    with open('hashes.txt','w') as fw:
        for line in lines:
            fw.write(line)

    # Delete log entry 
    with open('logs.txt','r') as lr:
        lines=lr.readlines()
        found=False
        for i,line in enumerate(lines):
            if found and not line.startswith(" "):
                end=i
                break
            elif line.startswith(hash):
                start=i-1
                found=True
        if 'end' not in locals():
            end=len(lines)

        if not found:
            print('No log entry found')
        else:
            del lines[start:end]
    with open('logs.txt','w') as lw:
        for line in lines:
            lw.write(line) 

    # Delete all plots and files refering to the hash
    import os
    files = os.listdir('solutions')
    for file in files:
        if file.startswith(hash):
            os.remove('/'.join(['solutions', file]))
    files = os.listdir('plots')
    for file in files:
        if file.startswith(hash):
            os.remove('/'.join(['plots', file]))
    print(f'Deleted hash: {hash}')

def write_log(text : list, hash : str):
    """Writes a given list of strings into the log file under the given hash.
    Writing to the log file can be obmitted, if the hash is not defined.

    Parameters
    ----------
    text : list
        List of strings representing the lines to be written into the log file.
    hash : str or None
        Hash refering to the entry in the log file. Necessary to determine if something should be inserted or not.
        If None is given, nothing is written to the log file.
    """
    # Read all hashes to determine if the given hash is valid.
    hashes=[]
    with open('hashes.txt', 'r') as f:
        lines=f.readlines()
        for line in lines:
            hashes.append(line.split('\n')[0])

    if hash !=None:
        assert (hash in hashes),'Given hash is not valid and not existent.'
        with open("logs.txt",'a') as logs:
            for line in text:
                logs.write(line)

def read_log(hash : str):
    """Extracts all information about the discretization and initialization
    from the log-file for the given hash.

    Parameters
    ----------
    hash : str
        The hash for the information to be extracted.

    Returns
    -------
    dicts : tuple of dictionaries
        Returns two dictionaries, one for the discretization and the other 
        one for the initialization parameters.
    """
    with open('logs.txt', 'r') as f:
        lines=f.readlines()
    
    # Get the part refering to the hash
    found=False
    for i,line in enumerate(lines):
        if found and not line.startswith(" "):
            end=i
            break
        elif line.startswith(hash):
            start=i+1
            found=True
    if 'end' not in locals():
        end=len(lines)

    assert found==True, "No entry in the log file for the given hash"

    log_entry=lines[start:end]
    discretization,initialization=dict(),dict()
    # Extract the values from the log_file
    for line in log_entry:
        if 'discretization' in line:
            values=line.split()
            for value in values:
                if '=' in value:
                    [char,number]=value.split("=")
                    if '.' in number:
                        discretization[char]=float(number)
                    else:
                        discretization[char]=int(number)
        if 'Initialization' in line or 'initialization' in line:
            values=line.split()
            for value in values:
                if '=' in value:
                    [char,number]=value.split("=")
                    if '.' in number:
                        initialization[char]=float(number)
                    else:
                        initialization[char]=str(number)
    dicts=(discretization, initialization)
    return dicts

def delete_last_logs(amount : int):
    """Deletes the n-latest logs and the corresponding files.

    Parameters
    ----------
    amount : int
        Amount of log entries to delete.
    """
    hashes=[]
    with open('hashes.txt', 'r') as f:
        lines=f.readlines()
        for line in lines:
            hashes.append(line.split('\n')[0])
    for i in range(1,amount+1):
        remove_log(hashes[-i])

def delete_first_logs(amount : int):
    """Deletes the n-first logs and the corresponding files.

    Parameters
    ----------
    amount : int
        Amount of log entries to delete.
    """
    hashes=[]
    with open('hashes.txt', 'r') as f:
        lines=f.readlines()
        for line in lines:
            hashes.append(line.split('\n')[0])
    for i in range(0,amount):
        remove_log(hashes[i])

def load_np_data(hash : str, task : str):
    """Loads the numpy data stored at the given hash matching the given task.

    Parameters
    ----------
    hash : str
        Hash under which the desired numpy data is stored.
    task : str
        Task that was assigned to the hash or specification of desired data.
        Possible tasks are:
            SOLVER:         Returns all the states of a simulation.
            DECOMPOSITION:  Returns the parameters of reduced ansatz space.
            NONLINEAR:      Returns the nonlinear snapshot in the simulation.
            SOLUTION:       Returns the voltage snapshot of the simulation.
            LINEAR:         Returns the linear snapshot in the simulation.
            ERROR:          Returns the absolute error snapshot.
            COST:           Returns the cost decay for the optimizer.

    Returns
    -------
    data : np.array
        Loaded numpy array.

    """
    # Define file endings depending on the task
    if task=='SOLVER':
        file_ending='_states.npy'
    elif task=='DECOMPOSITION':
        file_ending='_decomposed.npy'
    elif task=='NONLINEAR':
        file_ending='_nonlinear.npy'
    elif task=='SOLUTION':
        file_ending='_snapshot.npy'
    elif task=='LINEAR':
        file_ending='_linear.npy'
    elif task=='ERROR':
        file_ending='_error_abs.npy'
    elif task=='COST':
        file_ending='_cost.npy'
    
    # Load the data
    import os
    files = os.listdir('solutions')
    found = False
    for file in files:
        if file.endswith(file_ending) and file.startswith(hash):
            data = np.load('/'.join(['solutions', file]), allow_pickle=True)
            found=True
            break
    assert found==True, f"No files were stored under the hash {hash} and task {task}"

    return data

def arr2str(arr : np.array, **kwargs): 
    """Converts a numpy array to a string in order to display matrices.

    Parameters
    ----------
    arr : np.array
        Array to be converted.

    Returns
    -------
    str
        Converted array to printable string.
    """
    return np.array2string(arr, formatter={'float_kind': lambda x: '{:+.2e}'.format(x)}, max_line_width=100, edgeitems=4,**kwargs).replace('+0.00e+00', '    -    ').replace('-0.00e+00', '    -    ')

def plot_snapshots(hash : str):
    """Plots the snapshot matrices for the different states and saves them.

    Parameters
    ----------
    hash : str
        Hash to load the snapshot data.
    """
    print(f'Plotting snapshots for {hash}')
    states_flat=load_np_data(hash,'SOLVER')
    discretization,init=read_log(hash)
    
    xs=np.linspace(0,11.9,discretization['N'])
    try: zero=init['time']
    except: zero=0
    ts=np.linspace(zero,zero+discretization['HT']*(discretization['M']-1),discretization['M'])

    states=np.array([np.reshape(states_flat[i,:],[int(np.shape(states_flat)[1]/discretization['N']),discretization['N']]).T for i in range(discretization['M'])])
    labels=['$u$ [mV]','$y_1$','$y_2$','$y_3$']
    import matplotlib.pyplot as plt
    # Plot each snapshot separately
    for i in range(np.shape(states)[-1]):
        fig = plt.figure(tight_layout=True)
        plt.imshow(states[:, :,i], aspect='auto',extent=[0,xs[-1],ts[0],ts[-1]], origin='lower')
        plt.xlabel('$x$ [cm]')
        plt.ylabel('$t$ [ms]')
        plt.savefig(f"plots/{hash}_snapshot_{i}.pdf")
        plt.savefig(f"plots/{hash}_snapshot_{i}.png")
        plt.close(fig)
    
    # Plot all states in one figure
    fig = plt.figure(tight_layout=True,num=f"{hash}_snapshots")
    for i in range(np.shape(states)[-1]):

        ax=fig.add_subplot(int((np.shape(states)[-1]+1)/2),int((np.shape(states)[-1]+1)/2),i+1)
        ax.imshow(states[:, :,i], aspect='auto',extent=[0,xs[-1],ts[0],ts[-1]], origin='lower')
        plt.xlabel('$x$ [cm]')
        plt.ylabel('$t$ [ms]')
        plt.title(labels[i])

    plt.savefig(f"plots/{hash}_snapshots.pdf")
    plt.savefig(f"plots/{hash}_snapshots.png")

def plot_states(hash : str , t_res = 22):
    """Plots the states over the spatial domain for selected
    times.

    Parameters
    ----------
    hash : str
        Hash to load the different states.
    t_res : int, optional
        Number of selected timevalues, by default 22.
    """

    states_flat=load_np_data(hash,'SOLVER')
    discretization,_=read_log(hash)
    
    labels=['$u$ [mV]','$y_1$','$y_2$','$y_3$']

    xs=np.linspace(0,11.9,discretization['N'])

    states=np.array([np.reshape(states_flat[i,:],[int(np.shape(states_flat)[1]/discretization['N']),discretization['N']]).T for i in range(discretization['M'])])
    import matplotlib.pyplot as plt
    t_res=min(t_res,discretization['M'])
    t_stride=int(discretization['M']/t_res)
    print(f'Plotting states for {hash} with {t_stride*discretization["HT"]} ms between the curves')
    cs = np.linspace(0,1, discretization['M'] // t_stride + 1)
    # Plot each state separately
    for i in range(np.shape(states)[-1]):
        fig = plt.figure(tight_layout=True)
        ax=plt.axes()
        ax.plot(xs, states[::-t_stride, :,i].T)
        for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
        ax.plot(xs, states[0, :,i], '--', color='black')
        ax.plot(xs, states[-1, :,i], color='black')
        plt.xlabel('$x$ [cm]')
        plt.ylabel(labels[i])

        plt.savefig(f"plots/{hash}_state_{i}.pdf")
        plt.savefig(f"plots/{hash}_state_{i}.png")
        plt.close(fig)
    
    # Plot all states in one figure
    fig = plt.figure(tight_layout=True,num=f"{hash}_states")
    for i in range(np.shape(states)[-1]):
        ax=fig.add_subplot(np.shape(states)[-1],1,i+1)
        ax.plot(xs, states[::-t_stride, :,i].T)
        for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
        ax.plot(xs, states[0, :,i], '--', color='black')
        ax.plot(xs, states[-1, :,i], color='black')
        plt.ylabel(labels[i])
    plt.xlabel('$x$ [cm]')
    plt.savefig(f"plots/{hash}_states.pdf")
    plt.savefig(f"plots/{hash}_states.png")

def plot_initial(hash : str):
    """Plots the initial values for the states of the given hash.

    Parameters
    ----------
    hash : str
        Hash to load the initial values.
    """
    states_flat=load_np_data(hash,'SOLVER')
    discretization,_=read_log(hash)
    
    labels=['$u$ [mV]','$y_1$','$y_2$','$y_3$']
    xs=np.linspace(0,11.9,discretization['N'])

    states=np.array([np.reshape(states_flat[i,:],[int(np.shape(states_flat)[1]/discretization['N']),discretization['N']]).T for i in range(discretization['M'])])
    import matplotlib.pyplot as plt
    for i in range(np.shape(states)[-1]):
        fig = plt.figure(tight_layout=True)
        ax=plt.axes()
        ax.plot(xs, states[0, :,i], color='black')
        plt.xlabel('$x$ [cm]')
        plt.ylabel(labels[i])

        plt.savefig(f"plots/{hash}_initial_{i}.pdf")
        plt.savefig(f"plots/{hash}_initial_{i}.png")
        plt.close(fig)
    # Plot all states in one figure
    fig = plt.figure(tight_layout=True,num=f"{hash}_states")
    for i in range(np.shape(states)[-1]):
        ax=fig.add_subplot(np.shape(states)[-1],1,i+1)
        ax.plot(xs, states[0, :,i], color='black')
        plt.ylabel(labels[i])
    plt.xlabel('$x$ [cm]')
    plt.savefig(f"plots/{hash}_initials.pdf")
    plt.savefig(f"plots/{hash}_initials.png")

def load_true_data(discretization : dict , initialization : dict , solution_hash : str):
    """Loads the states and data from the solution hash and matches it with
    the given discretization and initialization. Necessary to calculate
    the errors.

    Parameters
    ----------
    discretization : dict
        Time and Spatial discretization to match th solution to.
    initialization : dict
        Initialization to extract specific wavefronts and shifts,
        to match the solutio data.
    solution_hash : str
        Hash of the true data or solution.

    Returns
    -------
    states : np.array
        Returns the states matched to the given setting.
    """
    data=load_np_data(solution_hash,'SOLVER')
    true_discretization,_=read_log(solution_hash)
    step_width= discretization['HT']/true_discretization['HT']
    
    # Constructs an array of indices to extract the necessary time steps.
    try: 
        if initialization['type']=='extract':
            time_selection=np.arange(int(initialization['time']/true_discretization['HT']),int(initialization['time']/true_discretization['HT'])+step_width*(discretization['M']),int(step_width),dtype=int)
        else:
            time_selection=np.arange(0,step_width*(discretization['M']),int(step_width),dtype=int)
    except:
        time_selection=np.arange(0,step_width*(discretization['M']),int(step_width),dtype=int)
    

    # Match time_interval
    states_flat=data[time_selection,:]

    # Divided into each states
    states_full=np.array([np.reshape(states_flat[i,:],[int(np.shape(states_flat)[1]/discretization['N']),true_discretization['N']]).T for i in range(discretization['M'])])
    
    # Match spatial discretization
    extract_step_widhts=discretization['HS']/true_discretization['HS']
    extract_steps=np.arange(0,true_discretization['N'],extract_step_widhts,dtype=int)
    states=np.empty([discretization['M'],discretization['N'],np.shape(states_full)[-1]])
    for i in range(4):
        states[:,:,i]=states_full[:,extract_steps,i]

    # Extract the waves
    try:
        if initialization['wave']!='dual':
            for i in range(4):
                if initialization['wave']=='left':
                    states[:,int(discretization['N']/2):,i]=np.full([np.shape(states)[0],int(discretization['N']/2)+1],states[0,-1,i])
                elif initialization['wave']=='right':
                    states[:,:int(discretization['N']/2),i]=np.full([np.shape(states)[0],int(discretization['N']/2)],states[0,1,i])
    except:
        None
    
    return states

def plot_error(hash : str , t_res=22):
    """Plots the absolute error snapshot and states for selected times.
    Only the error stored under the given hash is plotted not calculated.

    Parameters
    ----------
    hash : str
        Hash to the absolute error data. 
    t_res : int, optional
        Number of selected timevalues, by default 22.
    """
    print(f'Plotting error for {hash}')
    error_abs=load_np_data(hash,'ERROR')
    discretization,init=read_log(hash)

    try: zero=init['time']
    except: zero=0
    xs=np.linspace(0,11.9,discretization['N'])
    ts=np.linspace(zero,zero+discretization['HT']*(discretization['M']-1),discretization['M'])


    labels=['$e_{abs}(u)$','$e_{abs}(y_1)$','$e_{abs}(y_2)$','$e_{abs}(y_3)$']
    import matplotlib.pyplot as plt
    # Plot error snapshot for each state separately
    for i in range(np.shape(error_abs)[-1]):
        fig=plt.figure(tight_layout=True)
        plt.imshow(error_abs[:, :,i], aspect='auto',extent=[0,xs[-1],ts[0],ts[-1]], cmap='Reds', origin='lower')
        cbar=plt.colorbar()
        cbar.set_label('$e_{abs}$')
        plt.xlabel('$x$ [cm]')
        plt.ylabel('$t$ [ms]')

        plt.savefig(f"plots/{hash}_error_abs_snapshot_{i}.pdf")
        plt.savefig(f"plots/{hash}_error_abs_snapshot_{i}.png")
        plt.close(fig)
    
    # Plot error snapshots for all states in one figure
    fig = plt.figure(tight_layout=True,num=f"{hash}_errors_abs_snapshots")
    subplots=[411,412,413,414]
    for i in range(np.shape(error_abs)[-1]):
        plt.subplot(int((np.shape(error_abs)[-1]+1)/2),int((np.shape(error_abs)[-1]+1)/2),i+1)
        plt.imshow(error_abs[:, :,i], aspect='auto',extent=[0,xs[-1],ts[0],ts[-1]], cmap='Reds', origin='lower')
        plt.xlabel('$x$ [cm]')
        plt.ylabel('$t$ [ms]')
        plt.title(labels[i])

    plt.savefig(f"plots/{hash}_errors_abs_snapshots.pdf")
    plt.savefig(f"plots/{hash}_errors_abs_snapshots.png")

    t_stride=int(discretization['M']/t_res)
    cs = np.linspace(0,1, discretization['M'] // t_stride + 1)
    # Plot error for each state separately
    for i in range(np.shape(error_abs)[-1]):
        fig = plt.figure(tight_layout=True)
        ax=plt.axes()
        ax.plot(xs, error_abs[::-t_stride, :,i].T)
        for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
        ax.plot(xs, error_abs[0, :,i], '--', color='black')
        ax.plot(xs, error_abs[-1, :,i], color='black')
        plt.xlabel('$x$ [cm]')
        plt.ylabel(labels[i])

        plt.savefig(f"plots/{hash}_errors_abs__state_{i}.pdf")
        plt.savefig(f"plots/{hash}_errors_ab_state_{i}.png")
        plt.close(fig)
    
    # Plot errors for all states in one figure
    fig = plt.figure(tight_layout=True,num=f"{hash}_errors_abs_states")
    for i in range(np.shape(error_abs)[-1]):
        ax=fig.add_subplot(np.shape(error_abs)[-1],1,i+1)
        ax.plot(xs, error_abs[::-t_stride, :,i].T)
        for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
        ax.plot(xs, error_abs[0, :,i], '--', color='black')
        ax.plot(xs, error_abs[-1, :,i], color='black')
        plt.ylabel(labels[i])
    plt.xlabel('$x$ [cm]')
    plt.savefig(f"plots/{hash}_errors_abs_states.pdf")
    plt.savefig(f"plots/{hash}_errors_abs_states.png")

def plot_decomposition(hash : str):
    """Plots the parameters of the reduced ansatz space defined by the
    decomposition.

    Parameters
    ----------
    hash : str
        Hash of the decomposition results.
    """
    print(f'Plotting decompostion for {hash}')

    decomposed=load_np_data(hash,'DECOMPOSITION')
    discretization,_=read_log(hash)

    M,R,N,GAMMA=discretization['M'],discretization['R'],discretization['N'],discretization['Gamma']

    xs=np.linspace(0,11.9,discretization['N'])
    ts=np.linspace(0,discretization['HT']*(discretization['M']-1),discretization['M'])

    # extract the values from the loaded data
    alpha_full = np.reshape(decomposed[0:GAMMA*M*R], [GAMMA,M, R])
    phi = np.reshape(decomposed[GAMMA*M*R:GAMMA*M*R+R*N], [R, N])
    paths = decomposed[GAMMA*M*R+R*N:]

    import matplotlib.pyplot as plt

    fig=plt.figure(tight_layout=True)
    for i in range(min(R,10)):
        plt.plot(xs,phi[i,:], label=f'$\phi_{i}$')
    plt.xlabel('$x$ [cm]')
    plt.legend()

    plt.savefig(f"plots/{hash}_decomposition_phi.pdf")
    plt.savefig(f"plots/{hash}_decomposition_phi.png")

    plt.close(fig)
    fig=plt.figure(tight_layout=True)
    for i in range(min(R,10)):
        plt.plot(ts,alpha_full[0,:,i], label=f'$ \\alpha_{i}$')
    plt.xlabel('$t$ [ms]')
    plt.legend()

    plt.savefig(f"plots/{hash}_decomposition_alpha.pdf")
    plt.savefig(f"plots/{hash}_decomposition_alpha.png")

    plt.close(fig)
    fig=plt.figure(tight_layout=True)
    plt.subplot(121)
    for i in range(min(R,10)):
        plt.plot(xs,phi[i,:], label=f'$\phi_{i}$')
    plt.xlabel('$x$ [cm]')
    plt.legend()
    plt.subplot(122)
    for i in range(min(R,10)):
        plt.plot(ts,alpha_full[0,:,i], label=f'$ \\alpha_{i}$')
    plt.xlabel('$t$ [ms]')
    plt.legend()

    plt.savefig(f"plots/{hash}_decomposition.pdf")
    plt.savefig(f"plots/{hash}_decomposition.png")

def calculate_nonlinear_term(hash : str):
    """Evaluates the Hodgkin-Huxley model for the given statees of the hash.
    Stores the snapshot and plots it.

    Parameters
    ----------
    hash : str
        Hash of the states as input to the Hodgkin-Huxley model.

    Returns
    -------
    nonlinear_snapshot : np.array
        Snapshot of the nonlinear term for the voltage state.
    """
    states=load_np_data(hash,task='SOLVER')
    discretization,_=read_log(hash)
    M,N=discretization['M'],discretization['N']
    u=states[:,:N]
    y=np.array([np.reshape(states[i,N:],[3,N]).T for i in range(M+1)])
    nonlinear_terms=np.empty([M+1,N])

    import solver
    for i in range(M+1):
        nonlinear_terms[i,:],_=solver.hodgkin_huxley(u[i,:],y[i,:,:])
    import matplotlib.pyplot as plt
    plt.figure(tight_layout=True)
    xs=np.linspace(0,11.9,discretization['N'])
    ts=np.linspace(0,discretization['HT']*(discretization['M']-1),discretization['M'])
    plt.imshow(nonlinear_terms,aspect='auto',extent=[0,xs[-1],0,ts[-1]], origin='lower')
    plt.xlabel('x [cm]')
    plt.ylabel('t [ms]')
    cbar=plt.colorbar()
    cbar.set_label('Voltage [mV]')

    plt.savefig("plots/"+hash+"_nonlinear.svg")
    plt.savefig("plots/"+hash+"_nonlinear.png")
    np.save(str("solutions/"+hash+"_nonlinear.npy"),np.asarray(nonlinear_terms))

    return nonlinear_terms

def calculate_linear_term(hash : str):
    """Evaluates the diffusion term for the given statees of the hash.
    Stores the snapshot and plots it.

    Parameters
    ----------
    hash : str
        Hash of the states as input to the diffusion term.

    Returns
    -------
    nonlinear_snapshot : np.array
        Snapshot of the diffusion term for the voltage state.
    """
    states=load_np_data(hash,task='SOLVER')
    discretization,_=read_log(hash)
    M,N,HS=discretization['M'],discretization['N'],discretization['HS']
    u=states[:,:N]
    y=np.array([np.reshape(states[i,N:],[3,N]).T for i in range(M+1)])
    import solver
    linear_terms=np.empty([M+1,N])

    Stiff,Mass_inv=solver.FOM(N, HS)
    
    for i in range(M+1):
        linear_terms[i,:]=Mass_inv @ Stiff @ u[i,:]
    import matplotlib.pyplot as plt
    plt.figure(tight_layout=True)
    xs=np.linspace(0,11.9,discretization['N'])
    ts=np.linspace(0,discretization['HT']*discretization['M'],discretization['M']+1)
    plt.imshow(linear_terms,aspect='auto',extent=[0,xs[-1],0,ts[-1]], origin='lower')
    plt.xlabel('x [cm]')
    plt.ylabel('t [ms]')
    cbar=plt.colorbar()
    cbar.set_label('Voltage [mV]')

    plt.savefig("plots/"+hash+"_linear.svg")
    plt.savefig("plots/"+hash+"_linear.png")
    np.save(str("solutions/"+hash+"_linear.npy"),np.asarray(linear_terms))

    return linear_terms

def psy(k : int, x : float, HS : float):
    """Returns the value of the k-th hat function at position x.

    Parameters
    ----------
    k : int
        Index of the hat function.
    x : float
        Spatial positon.
    HS : float
        Spatial step-width.

    Returns
    -------
    float
        Value of the hat function
    """
    # k = index of the center node
    # x = spatial variable
    if (k-1)*HS < x <= k*HS:
        return (x-(k-1)*HS)/HS
    if k*HS < x <= (k+1)*HS:
        return ((k+1)*HS-x)/HS
    return 0

def merge_spaces(hash_1 : str , hash_2 : str):
    """Merges the reduced spaces of two different hashes into one new space.

    Parameters
    ----------
    hash_1 : str
        Hash of the data at the front position of the resulting space.
    hash_2 : str
        Hash of the data at the end position of the resulting space.

    Returns
    -------
    hash : str
        Hash under which the data of the merged spaces is stored
    """
    # Creates an unique hash
    hash=create_hash()
    with open("logs.txt",'a') as logs:
        logs.write("\n{:<15}MERGED\n".format(hash))
    print("Used hash: {}".format(hash))

    # Loads the data
    decomposed_1=load_np_data(hash_1,'DECOMPOSITION')
    discretization_1,initial_1=read_log(hash_1)
    decomposed_2=load_np_data(hash_2,'DECOMPOSITION')
    discretization_2,_=read_log(hash_2) 


    M,R,N,GAMMA=discretization_1['M'],discretization_1['R'],discretization_1['N'],discretization_1['Gamma']

    alpha_full_1 = np.reshape(decomposed_1[0:GAMMA*M*R], [GAMMA,M, R])
    phi_1 = np.reshape(decomposed_1[GAMMA*M*R:GAMMA*M*R+R*N], [R, N])
    paths_1 = decomposed_1[GAMMA*M*R+R*N:]

    M,R,N,GAMMA=discretization_2['M'],discretization_2['R'],discretization_2['N'],discretization_2['Gamma']
    
    alpha_full_2 = np.reshape(decomposed_2[0:GAMMA*M*R], [GAMMA,M, R])
    phi_2 = np.reshape(decomposed_2[GAMMA*M*R:GAMMA*M*R+R*N], [R, N])
    paths_2 = decomposed_2[GAMMA*M*R+R*N:]

    paths=np.array([paths_1,paths_2]).flatten()

    phi=np.concatenate([phi_1,phi_2],axis=0)

    R=np.shape(paths)[0]

    alpha_full=np.concatenate([alpha_full_1,alpha_full_2],axis=2)

    x=np.concatenate([alpha_full.flatten(), phi.flatten(), paths.flatten()])

    write_log([f"{'':<15} Merged hashes:{hash_1} , max={hash_2} \n",f"{'':<15} Spatial discretization: N={discretization_1['N']} , HS={discretization_1['HS']} cm , Gamma=1\n",f"{'':<15} Time discretization: T={discretization_1['HT']*M} ms , HT={discretization_1['HT']} ms , M={discretization_1['M']} \n", f"{'':<15} Reduced discretization: R={R}\n",f"{'':<15} Initialization: init={initial_1['init']} , shift={initial_1['shift']}\n"],hash)
    np.save("solutions/"+hash+"_decomposed",x)
    return hash
      
def calculate_error(hash_test_data : str , hash_true_data : str):
    """Calculates all the errors (local absolute, global mean, global max) between test-data and true-data.

    Parameters
    ----------
    hash_test_data : str
        Hash of the test-data.
    hash_true_data : str
        Hash of the true-data.

    Returns
    -------
    error_abs : np.array
        Array of the local absolute errors.
    """
    # Load test snapshot data
    test_discretization,test_initialization=read_log(hash_test_data)
    test_data_flat=load_np_data(hash_test_data,'SOLVER')
    test_data=np.array([np.reshape(test_data_flat[i,:],[int(np.shape(test_data_flat)[1]/test_discretization['N']),test_discretization['N']]).T for i in range(test_discretization['M'])])

    # Load true data
    true_data=load_true_data(test_discretization,test_initialization,hash_true_data)

    assert np.shape(test_data)[:2]==np.shape(true_data)[:2] and np.shape(test_data)[2]<=np.shape(true_data)[2], 'Test data not included in true data due to shape mismatch'

    error_abs=np.empty(np.shape(test_data))
    
    for i in range(np.shape(test_data)[-1]):
        error_abs[:,:,i]=np.abs(np.subtract(true_data[:,:,i],test_data[:,:,i]))

    np.save(str("solutions/"+hash_test_data+"_error_abs.npy"),error_abs)

    import decomp
    N,M,T,HS,HT,R,Gamma=test_discretization['N'],test_discretization['M'],test_discretization['T'],test_discretization['HS'],test_discretization['HT'],test_discretization['R'],1


    norm_true=np.sqrt(decomp.L_2_error([true_data[:,:,0]],[np.zeros([M,N])],M,N,HT,HS,T,1))
    try:
        shift=test_initialization['shift']
    except:
        shift=None

    if shift!=None:
        true_data[:,:,0]=np.subtract(true_data[:,:,0],shift)

    try: 
        test_data_decomp=load_np_data(hash_test_data,'DECOMPOSITION')
        L2=np.sqrt(decomp.cost(test_data_decomp,[true_data[:,:,0]],M,N,R,T,HT,HS,Gamma))
        print(f'Errors for {hash_test_data}: max_abs={np.max(error_abs):.5f} , mean_abs={np.mean(error_abs):.5f} , min_abs={np.min(error_abs):.5f} , L2={L2:.5f}, , L2_rel={L2/norm_true:.10f}')
        error_info=np.asarray([np.max(error_abs),np.mean(error_abs),np.min(error_abs),L2,L2/norm_true])
    except:
        print(f'Errors for {hash_test_data}: max_abs={np.max(error_abs):.5f} , mean_abs={np.mean(error_abs):.5f} , min_abs={np.min(error_abs):.5f}')
        error_info=np.asarray([np.max(error_abs),np.mean(error_abs),np.min(error_abs)])

    np.save(str("solutions/"+hash_test_data+"_error_info.npy"), error_info)

    return error_abs

def animation(hash : str):
    """Animates the propagation of the action potential.

    Parameters
    ----------
    hash : str
        Hash of the solution to be visualised.
    """
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation 
    
    data=load_np_data(hash,'SOLVER')
    voltage=data[:,:int(np.shape(data)[1]/4)]

    # initializing a figure in 
    # which the graph will be plotted
    fig = plt.figure(dpi=600,figsize=(12,3),tight_layout=True) 
    
    # marking the x-axis and y-axis
    axis = plt.axes(xlim =(0, 11.9), 
                    ylim =(-90, 40))
    plt.xlabel('$x$ [cm]') 
    plt.ylabel('$u$ [mV]')
    
    # initializing a line variable
    line, = axis.plot([], [], lw = 3) 
    
    # data which the line will 
    # contain (x, y)
    def init(): 
        line.set_data([], [])
        return line,
    
    def animate(i):
        x = np.linspace(0, 11.9, np.shape(voltage)[1])
    
        # plots a sine graph
        y = voltage[i,:]
        line.set_data(x, y)
        
        return line,
    
    anim = FuncAnimation(fig, animate, init_func = init,
                        frames = range(0,int(np.shape(voltage)[0]),100), interval = 10, blit = True)
    
    
    anim.save('plots/Voltage.mp4', 
            writer = 'ffmpeg', fps = 30)
    plt.close()
    fig_2 = plt.figure(dpi=600,figsize=(12,3),tight_layout=True)
    axis = plt.axes(xlim =(0, 11.9), 
                    ylim =(-90, 40))
    plt.xlabel('$x$ [cm]') 
    x = np.linspace(0, 11.9, np.shape(voltage)[1])

    plt.ylabel('$u$ [mV]')
    plt.plot(x,voltage[5000,:])
    plt.savefig('plots/Voltage_shot.png')

def plot_comparison(hash_test_data : str , hash_true_data : str , t_res=5):
    """Plots the true data and the test data for selected time values.

    Parameters
    ----------
    hash_test_data : str
        Hash to the data to be tested.
    hash_true_data : str
        Hash to the true data to compare the test data to.
    t_res : int, optional
        Number of selected times to be plotted, by default 5
    """
    print(f'Plotting comparison for {hash_test_data}')
    # Load test data
    test_discretization,test_initialization=read_log(hash_test_data)
    test_data_flat=load_np_data(hash_test_data,'SOLVER')
    test_data=np.array([np.reshape(test_data_flat[i,:],[int(np.shape(test_data_flat)[1]/test_discretization['N']),test_discretization['N']]).T for i in range(np.shape(test_data_flat)[0])])

    try: zero=test_initialization['time']
    except: zero=0

    # Load true data
    true_data=load_true_data(test_discretization,test_initialization,hash_true_data)

    assert np.shape(test_data)[:2]==np.shape(true_data)[:2] and np.shape(test_data)[2]<=np.shape(true_data)[2], 'Test data not included in true data due to shape mismatch'

    xs=np.linspace(0,11.9,test_discretization['N'])
    ts=np.linspace(zero,zero+test_discretization['HT']*(test_discretization['M']-1),test_discretization['M'])

    labels=['$u$ [mV]','$y_1$','$y_2$','$y_3$']

    import matplotlib.pyplot as plt

    colors=['Black','Red','Green','Blue','Orange']
    # Plot comparison for each state separately
    for i in range(np.shape(test_data)[-1]):
        fig = plt.figure(tight_layout=True)
        for idx,j in enumerate(np.linspace(0,test_discretization['M']-1,t_res,dtype=int)):
            plt.plot(xs,true_data[j,:,i],color=colors[idx],label=f'$t={ts[j]}$ ms')
            plt.plot(xs,test_data[j,:,i],color=colors[idx],linestyle='None',marker='o',markersize=1)
            plt.xlabel('$x$ [cm]')
        plt.legend()
        plt.savefig(f"plots/{hash_test_data}_comparison_{i}.pdf")
        plt.savefig(f"plots/{hash_test_data}_comparison_{i}.png")
        plt.close(fig)

    # Plot comparison for all states in one figrure
    fig = plt.figure(tight_layout=True,num=f"{hash_test_data}_comparisons")
    for i in range(np.shape(test_data)[-1]):
        plt.subplot(np.shape(test_data)[-1],1,i+1)
        for idx,j in enumerate(np.linspace(0,test_discretization['M']-1,t_res,dtype=int)):
            plt.plot(xs,true_data[j,:,i],color=colors[idx],label=f'$t={ts[j]}$ ms')
            plt.plot(xs,test_data[j,:,i],color=colors[idx],linestyle='None',marker='o')
            plt.xlabel('$x$ [cm]')
            plt.ylabel(labels[i])
        plt.legend()

    plt.savefig(f"plots/{hash_test_data}_comparisons.pdf")
    plt.savefig(f"plots/{hash_test_data}_comparisons.png")
    # Plot comparison for voltage
    fig = plt.figure(tight_layout=True,num=f"{hash_test_data}_comparison_0")
    for idx,j in enumerate(np.linspace(0,test_discretization['M']-1,t_res,dtype=int)):
        plt.plot(xs,true_data[j,:,0],color=colors[idx],label=f'$t={ts[j]}$ ms')
        plt.plot(xs,test_data[j,:,0],color=colors[idx],linestyle='None',marker='o')
        plt.xlabel('$x$ [cm]')
        plt.ylabel(labels[i])
    plt.legend()

def plot_cost_convergence(hash : str):
    """Plots the cost decay for the minimization algorithm at differet scales.

    Parameters
    ----------
    hash : str
        Hash to decomposition task
    """
    cost=load_np_data(hash,'COST')
    import matplotlib.pyplot as plt

    # Plot cost
    fig = plt.figure(tight_layout=True)
    plt.plot(cost)
    plt.ylabel(r'$J( \alpha ,\phi ,p) $')
    plt.xlabel('iterations')
    plt.ylim(min(cost[:]), cost[0]+cost[0]*0.5)
    plt.savefig(f"plots/{hash}_cost.pdf")
    plt.savefig(f"plots/{hash}_cost.png")
    plt.close(fig)

    # Plot cost logarithmic
    fig=plt.figure(tight_layout=True)
    ax=plt.axes()
    ax.plot(cost)
    ax.set_yscale('log')
    plt.ylabel(r'$J( \alpha ,\phi ,p) $')
    plt.ylim(min(cost[:]), cost[0]+cost[0]*0.5)
    plt.xlabel('iterations')
    plt.savefig(f"plots/{hash}_cost_log.pdf")
    plt.savefig(f"plots/{hash}_cost_log.png")
    plt.close(fig)


    # Plot cost normal and logarithmic in one Plot
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel(r'$J( \alpha ,\phi ,p) $', color = color)
    ax1.plot(cost, color = color)
    ax1.tick_params(axis ='y', labelcolor = color)
    ax1.set_ylim(min(cost[:]), cost[0]+cost[0]*0.5)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel(r'$J( \alpha ,\phi ,p) $', color = color)
    ax2.plot(cost, color = color)
    ax2.set_yscale('log')
    ax2.set_ylim(min(cost[:]), cost[0]+cost[0]*0.5)
    ax2.tick_params(axis ='y', labelcolor = color)
    plt.savefig(f"plots/{hash}_costs_combined.pdf")
    plt.savefig(f"plots/{hash}_costs_combined.png")
    plt.close(fig)

    # Plot cost normal and logarithmic as subplots in one figure
    fig = plt.figure(tight_layout=True,num=f"{hash}_costs")
    ax=fig.add_subplot(121)
    ax.plot(cost)
    ax.set_ylim(min(cost[:]), cost[0]+cost[0]*0.5)
    plt.ylabel(r'$J( \alpha ,\phi ,p) $')
    plt.xlabel('iterations')
    ax=fig.add_subplot(122)
    ax.plot(cost)
    ax.set_yscale('log')
    ax.set_ylim(min(cost[:]), cost[0]+cost[0]*0.5)
    plt.ylabel(r'$J( \alpha ,\phi ,p) $')
    plt.xlabel('iterations')
    plt.savefig(f"plots/{hash}_costs.pdf")
    plt.savefig(f"plots/{hash}_costs.png")

def time_test(T : float , HT : float, problem : str , method : str , initial_values , reduced_data = 'None' , test_num = 5 , save = True):
    """Measures the computation time of a FOM or ROM simulation.

    Parameters
    ----------
    T : float
        Simulation time in ms
    HT : float
        Time-step width in ms
    problem : str
        Problemtype of the simulation (FOM or ROM)
    method : str
        Time-stepping method (EE or IE)
    initial_values : list
        Information of the desired initial values, for more information llok at "solver.py"
    reduced_data : str, optional
        Hash to the data to construct the reduced ansatz space, by default 'None'
    test_num : int, optional
        Number of test samples, by default 5
    save : bool, optional
        Control if the results should be stored or not, by default True

    Returns
    -------
    (hash , results) : (str , list)
        Hash to the results and list containing the measurements.
    """
    import solver
    
    times=[]
    if save:
        hash=create_hash()
        with open("logs.txt",'a') as logs:
            logs.write("\n{:<15}TIME\n".format(hash))
        print("Used hash: {}".format(hash))
    else:
        hash=None

    write_log([f"{'':<15} Test parameters: num={test_num} \n",f"{'':<15} Problem parameters: T={T} , HT={HT} , problem={problem} , init={initial_values} , red={reduced_data}\n"], hash)

    for _ in range(test_num):
        _,time=solver.main(T, HT, problem,method, initial_values, reduced_data, save=False)
        times.append(time)

    times=np.asarray(times).T
    mean,max,min=np.mean(times[0,:]),np.max(times[0,:]),np.min(times[0,:])
    write_log([f"{'':<15} Results: mean={mean} , max={max} , min={min} \n"], hash)
    if np.shape(times)[0]==2:
        mean_pro,max_pro,min_pro=np.mean(times[1,:]),np.max(times[1,:]),np.min(times[1,:])
        write_log([f"{'':<15} Projection results: mean={mean_pro} , max={max_pro} , min={min_pro} \n"], hash)
    if hash!=None:
        np.save(str("solutions/"+hash+"_times.npy"),times)


    return hash,[mean,max,min]