import pythonMFS as mfs
import numpy as np
import matplotlib.pyplot as plt

#NOTE: -ALL variables non-dimensionalised with viscoity and a length scale
#      -If you want to change viscosity, scale A matrix by 1/mu. Default here, mu=1
 
example = 5 #input example number to run

if example == 1:
    #EXAMPLE 1: force and torque on a unit sphere, normalised by f = 6\pi\mu R|V| and t = 8\pi\mu R\omega
    R = 1
    rb = mfs.sphereMaker(200,R)                          #distribute N=200 points around a sphere with radius=R
    rs = mfs.sphereMaker(160,0.5*R)                      #distribute M=160 points around a sphere with radius=R
    v,omega = [1,0,0], [1,0,0]                           #define the wall velocity and angular velocity
    f,t = mfs.findForceAndTorque(rb,rs,v,omega)          #calculate force and torque from pythonMFS file
    f = np.sum(f,0)/(6*np.pi*R*np.linalg.norm(v))        #total force = sum of forces on all sites, normalised by 6\pi\mu|v|R
    t = np.sum(t,0)/(8*np.pi*R**3*np.linalg.norm(omega)) #total torque = sum of torque on all sites, normalised by 8\pi\mu|\omega|R^3
    print(f"Normalised net force: {np.round(f,5)}")      #display force result   (should =1)
    print(f"Normalised net torque: {np.round(t,5)}")     #display torque result  (should =-1)

elif example == 2:
    #EXAMPLE 2: resistance tensor of a spheroid, normalised by the force on a single sphere of radius R=a=b=0.5c
    rb = mfs.spheroidMaker(200,1,1,2)
    rs = mfs.spheroidMaker(160,0.5,0.5,1)
    Re = np.zeros([6,6]) #initialise
    v =np.eye(3) 
    for dir in range(3): #three orthogonal directions
        f,t = mfs.findForceAndTorque(rb,rs,v[dir,:],[0,0,0]) #find force and torque from translational velocity
        Re[dir,0:3], Re[dir,3:7] = np.sum(f,0)/(6*np.pi), np.sum(t,0)/(8*np.pi) #fill first three rows of Re with f&t from translational velocity
    
        f,t = mfs.findForceAndTorque(rb,rs,[0,0,0],v[dir,:]) #find force and torque from translational velocity
        Re[dir+3,0:3], Re[dir+3,3:7] = np.sum(f,0)/(6*np.pi), np.sum(t,0)/(8*np.pi) #fill last three rows of Re from f&t from angular velocity

    Re = 0.5*(Re+Re.T) #ensure symmetric
    print(f"Normalised resistance tensor: \n {np.round(Re,2)}")

elif example == 3: 
    #EXAMPLE 3: plot streamlines around a sphere (crude, not as easily changeable as the others)
    fig = plt.figure() #initialise matplotlib plotter
    ax = fig.add_subplot(projection='3d')

    u, v = np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,100) #this block is just plotting a sphere
    u, v = np.meshgrid(u,v)
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    s = ax.plot_surface(x,y,z,color=(0.635,0.341,0.671))

    rb, rs = mfs.sphereMaker(200,1), mfs.sphereMaker(160,0.5)
    A = mfs.matrixConstruct(rb,rs) #find A matrix of the sphere
    pinva = np.linalg.pinv(A)
    V, om = [0,0,-1], [0,0,0] #define velocity of the fluid around the sphere
    v = mfs.rhsConstruct(rb,V,om)
    f = np.matmul(pinva,v) #find forces on each site
    
    numberOfStreamlines = 3 
    start = 3 #point in the z-direction to start; ensure start>1 to be outside the sphere 
    theta = np.linspace(0,2*np.pi,numberOfStreamlines+1)
    radius = 0.25 #radius of starting streamline circle ##NOTE: can change this easily to a square, cross, single point by changing this line and the ones above/below, just dont start at [0,0,start] due to the stagnation point 
    streamlineStart = np.array([radius*np.sin(theta), radius*np.cos(theta), 0*theta + start]).T #create circle of streamlines at z=start
    dt = 0.075 #time step size 

    for n in range(numberOfStreamlines): #loop over each streamline
        streamline = streamlineStart[n,:]
        streamlineTotal = streamline
        slz = start #to check when to finish the while loop
        it = 0
        slVelTot = 0 #for colouring the streamlines 
        while slz > -start:
            it += 1
            A = mfs.matrixConstruct(streamline,rs) #create A matrix of all sites within sphere, and the start of the streamline
            vStreamline = np.matmul(A,f).T 
            vStreamline = vStreamline[0:3] #i dont know why this is needed but its a 'feature' i haven't fixed yet 
            streamline = streamline - dt*(vStreamline-V) #velocity at the point given by v=A.f - free-stream velocity*time step 
            streamlineTotal = np.vstack((streamlineTotal,streamline))
            slz = streamline[2]
            slVel = dt*(vStreamline-V)
            slVel = (slVel[0]**2 + slVel[1]**2 + slVel[2]**2)**0.5
            slVelTot = np.vstack((slVelTot,slVel)) #for colouring the lines
        print(f'Done streamline #{n+1}, in {it} calculations')
        slVelTot = slVelTot.T
        streamlineTotal = streamlineTotal.T
        x,y,z = streamlineTotal[0,:], streamlineTotal[1,:], streamlineTotal[2,:]
        x,y,z,slVelTot = x.ravel(),y.ravel(),z.ravel(),slVelTot.ravel() #another 'feature' (please email me why)
        ax.plot(x,y,z,color='k')
    ax.axis('equal')
    ax.set_axis_off()
    plt.show()
    #NOTE: in this example, increasing the number of streamlines and/or decreasing dt can require very long computational times due to the number of computations required

elif example == 4:
    #EXAMPLE 4: plot filled contour field of velocities around a sedimenting sphere, and superimpose streamlines on top
    ### Find force on sites of sphere ###
    N = 150
    M = np.floor(0.8*N)                
    rb = mfs.sphereMaker(N,1)        #create N nodes on sphere
    rs = mfs.sphereMaker(M,0.5)      #create M sites in sphere (radius smaller than that of the nodes, so outside the flow)
    A = mfs.matrixConstruct(rb,rs)
    V = [0,-1,0]                        #velocity of the wall of the sphere (each node)
    v = mfs.rhsConstruct(rb,V,[0,0,0])
    f = np.matmul(np.linalg.pinv(A),v)        #find the force on each site in the sphere

    ### Generate grid of points to find the velocity of fluid ###
    n = 100                                                     #number of grid points, n^2 calculations so be careful
    x = np.linspace(-2.25, 2.25, n)
    y = np.linspace(-2.25, 2.25, n)
    X, Y = np.meshgrid(x, y)                                    #create grid
    Z = np.zeros_like(X)                                        #create grid
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel())) #(n^2 x 3) array of points to find the velocity at
    vGrid = np.zeros_like(points)                               #initialise

    ### Find velocities at each grid point ###
    for ii in range(len(points)):                                    #loop over all points in mesh
        if points[ii,0]**2 + points[ii,1]**2 + points[ii,2]**2 >= 1: #if the point is in the fluid, not in the sphere, then...
            A = mfs.matrixConstruct(points[ii,:],rs)                 #A matrix between point in fluid and all sites within sphere
            v = np.matmul(A,f).T                                     #velocity of point in fluid
            vGrid[ii,:] = v[0:3] - V                                 #remove imposed wall velocity
        else:                                                        #inside sphere, so do not calculate and set to 0
            vGrid[ii,:] = [0,0,0]          
    u,v = vGrid[:, 0].reshape(n, n), vGrid[:, 1].reshape(n, n)       #grid-ify
    vGridMagnitude = np.sqrt(u**2 + v**2)                            


    ### Plotting ###
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contourf(X,Y,vGridMagnitude,cmap='plasma',levels=250) #Plot filled contours of velocity grid
    ax.streamplot(X,Y,u,v,density=1.125,color='k')           #Plot streamlines around the sphere (careful with the stagnation point)
    phi = np.linspace(0,2*np.pi,100)                         #Plot sphere
    x,y = np.cos(phi),np.sin(phi)                            #Plot sphere
    ax.fill(x,y,color=(0.6,0.6,0.6),alpha=0.75)              #Plot sphere
    ax.axis('equal')
    ax.set_axis_off()
    plt.show()

if example == 5:
    #EXAMPLE 5: calculate error and matrix conditioning with different size rs and different N
    #This is for information, if you are interested in why the site location is important. Condition number and error are inversely proportional for the most part
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(1,3, layout='constrained', figsize = (15,5))
    
    numNodeList = [50,100,150,200,250]
    sitePosRadius = np.linspace(0.4,0.95,20)
    forceError, matrixCondition = np.zeros([len(numNodeList),len(sitePosRadius)]), np.zeros([len(numNodeList),len(sitePosRadius)])
    for n in range(len(numNodeList)):
        N = numNodeList[n]
        M = int(np.floor(0.8*N))
        rb = mfs.sphereMaker(N,1)
        v = mfs.rhsConstruct(rb,[1,0,0],[0,0,0])
        for s in range(len(sitePosRadius)):
            Rs = sitePosRadius[s]
            rs = mfs.sphereMaker(M,Rs)
            A = mfs.matrixConstruct(rb,rs)

            f = np.matmul(np.linalg.pinv(A),v)
            f = np.reshape(f,[int(np.size(f)/3),3])
            f = np.sum(f,0)/(6*np.pi)

            forceError[n,s] = np.absolute(1-f[0])
            matrixCondition[n,s] = np.linalg.cond(A)
        ax[0].plot(sitePosRadius,100*forceError[n,:], label=f'M={M}')                  #Plot error between MFS and alnalytical solution, smaller the better
        ax[1].plot(sitePosRadius,matrixCondition[n,:], label=f'M={M}')                 #Plot matrix conditioning number, smaller the better
        ax[2].plot(sitePosRadius,matrixCondition[n,:]*forceError[n,:], label=f'M={M}') #Plot product of the two. if not on O(1), then it is a bad combination

    ax[0].set_yscale('log')
    ax[0].set_xlabel('radius of sphere of sites')
    ax[0].set_ylabel('absolute error, %')
    ax[0].legend()

    ax[1].set_yscale('log')
    ax[1].set_xlabel('radius of sphere of sites')
    ax[1].set_ylabel('matrix conditioning number')
    ax[1].legend()

    ax[2].set_yscale('log')
    ax[2].set_xlabel('radius of sphere of sites')
    ax[2].set_ylabel('product of error and condition number')
    ax[2].legend()
    
    plt.show()

    

else:
    print("Enter valid example number")
