import pythonMFS as mfs
import numpy as np
import matplotlib.pyplot as plt

#NOTE: -ALL variables non-dimensionalised with viscoity and a length scale
#      -If you want to change viscosity, scale A matrix by 1/mu. Default here, mu=1
#      -Pretty much everything is a numpy array
 
example = 6 #input example number to run

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
    rb, nb = mfs.spheroidMaker(200,1,1,2)
    rs = mfs.rsFinder(rb,nb)
    #rs = mfs.spheroidMaker(160,0.5,0.5,1)
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

elif example == 5:
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

elif example == 6:
    fg = [0,0,-0.01*9.81/(6*np.pi)] #gravity in -ve z direction
    nSteps = 100
    dt = 0.0005
    
    N = 150
    rb = mfs.sphereMaker(N,1)
    rsO = mfs.rsFinder(rb,rb)
    rsO = rsO[np.random.rand(N)<0.83]
    rs = rsO
    M = rsO.shape[0]
    print(f"Number of sites = {M}")

    #A = mfs.matrixConstruct(rb,rs)
    #pinva = np.linalg.pinv(A)
    #maxl = np.max(np.abs(np.linalg.eigvals(pinva)))
    #print(f"Convergence criteria = {maxl*dt/2}. Should be much less than 1.")
    #if dt > 2/maxl:
    #    print("Unstable initial conditions")

    fg = np.repeat(fg,M)
    v = np.zeros([3*N])
    cHist = np.zeros([nSteps,3])
    tHist = np.linspace(1,nSteps,nSteps)
    for t in range(nSteps):
        A = mfs.matrixConstruct(rb,rs)
        pinva = np.linalg.pinv(A)
        fd = np.matmul(pinva,v)
        ft = fg - fd
        ft = sum(np.reshape(ft,[M,3]),0)
        v = np.reshape(v,[N,3]) + ft*dt
        rb = rb + v*dt
        v = np.reshape(v,[3*N])
        c = np.mean(rb,axis=0)
        cHist[t,:] = c
        rs = rsO + c
        print(f"Time step {t}, force magnitude = {ft[2]}")
    fig, ax = plt.subplots()
    ax.plot(tHist,cHist[:,2])
    plt.show()  





elif example == 7:
    #EXAMPLE 6: plot animation of two spheres falling together
    c1 = np.array([2,0,-1]) #centres of each sphere
    c2 = np.array([-2,0,0])
    fg = np.array([0,0,9.81])
    I = 0.4 #Moment of inertia of a sphere with mass = 1 and radius = 1
    nSteps = 25 #Number of time steps
    dt = 0.01 #Time discretisation
    ### Find force on sites of sphere ###
    N = 100
    M = int(np.floor(0.8*N))                
    rbO = mfs.sphereMaker(N,1)        #create N nodes on sphere
    rsO = mfs.sphereMaker(M,0.5)      #create M sites in sphere (radius smaller than that of the nodes, so outside the flow)
    rb = np.vstack([rbO + c1, rbO + c2]) 
    rs = np.vstack([rsO + c1, rsO + c2])
    N = len(rb)
    M = len(rs) 
    fg = fg/(2*M)
    A = mfs.matrixConstruct(rb,rs)
    f = np.tile(fg,M)
    v = np.matmul(A,f)
    w1, w2 = np.zeros([3,]), np.zeros([3,])
    fPerStep = np.zeros(nSteps)
    step = 0
    c1Hist, c2Hist = np.zeros([nSteps,3]), np.zeros([nSteps,3])
    tHist = np.linspace(1,nSteps,nSteps)
    for t in range(nSteps):
        c1Hist[step,:],c2Hist[step,:] = c1, c2

        #v = u + at, m=1, a=f/m => v = u + f*dt
        rb1, rb2 = np.split(rb,2)
        rs1, rs2 = np.split(rs,2)

        #Calculate force
        A = mfs.matrixConstruct(rb,rs)
        pinva = np.linalg.pinv(A)
        f = np.matmul(pinva,v)
        f = np.reshape(f,[M,3]) - fg
        v = np.reshape(v,[N,3])
        f1, f2 = np.split(f,2)
        
        #Calculate torque
        t1, t2 = np.zeros([M//2,3]), np.zeros([M//2,3])
        for m in range(np.size(t1,0)):
            t1[m,:] = np.cross(rs1[m,:] - c1,f1[m,:]) 
            t2[m,:] = np.cross(rs2[m,:] - c2,f2[m,:]) 
        
        #Calculate angular velocities 
        w1 = w1 + np.sum(t1,0)*dt/I
        w2 = w2 + np.sum(t2,0)*dt/I
        #Calculate linear velocities
        f1 = sum(f1,0)
        f2 = sum(f2,0)
        v1,v2 = np.split(v,2)
        for n in range(np.size(v1,0)):
            v1[n,:] = v1[n,:] + np.cross(w1,rb[n,:]-c1)
            v2[n,:] = v2[n,:] + np.cross(w2,rb[n,:]-c2)
        v1 = v1 + f1*dt
        v2 = v2 + f2*dt

        v = np.vstack([v1,v2])
        v = np.reshape(v,[3*N,])
        f = np.reshape(f,[3*M,])

        rb1, rb2 = rb1 + v1*dt, rb2 + v2*dt
        rb1, rb2 = mfs.rotVec(w1[0]*dt,w1[1]*dt,w1[2]*dt,(rb1-c1).T) + c1, mfs.rotVec(w2[0]*dt,w2[1]*dt,w2[2]*dt,(rb2-c2).T) + c2
        rs1, rs2 = mfs.rotVec(w1[0]*dt,w1[1]*dt,w1[2]*dt,(rs1-c1).T) + c1, mfs.rotVec(w2[0]*dt,w2[1]*dt,w2[2]*dt,(rs2-c2).T) + c2
        c1, c2 = np.mean(rb1,axis=0), np.mean(rb2,axis=0)
        rb = np.vstack([rb1,rb2])
        rs = np.vstack([rs1,rs2])
        
        fPerStep[step] = np.linalg.norm(np.sum(f,0))/(6*np.pi*2)
        print(f"Step {step}, average drag force = {fPerStep[step]}")
        step += 1
    print(fPerStep)
    fig, ax = plt.subplots()
    ax.plot(tHist,c1Hist[:,2])
    ax.plot(tHist,c2Hist[:,2])
    plt.show()  


    

else:
    print("Enter valid example number")
