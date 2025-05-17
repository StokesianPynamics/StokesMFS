import numpy as np
from numpy import linalg as la

################################
######---- STOKES MFS ----######
######- this file defines -#####
######- all the functions -#####
######-- to be  used for --#####
######--- the MFS  code ---#####
################################

#CONTRUCT OSEEN TENSOR FOR EACH SITE/NODE PAIR TO CREATE A MATRIX (Eq. 9)
#Input: rb: Nx3 list of node positions, rs: Mx3 list of site positions (M<N)
#Output: A: matrix of Oseen tensors of each site/node pair
def matrixConstruct(rb,rs):
    rb,rs = np.array(rb), np.array(rs)
    N, M = len(rb), len(rs)
    A = np.zeros([3*N,3*M]) #Initialise
    for n in range(N):
        if N == 3:
            N = 1
            bx, by, bz = rb[0], rb[1], rb[2]
        elif N != 1:
            bx, by, bz = rb[n,0], rb[n,1], rb[n,2]
        for m in range(M):
            if M == 3:
                M = 1
                sx, sy, sz = rs[0], rs[1], rs[2]
            elif M != 1:
                sx, sy, sz = rs[m,0], rs[m,1], rs[m,2]
            
            rbsx, rbsy, rbsz = bx - sx, by - sy, bz - sz #Eq. 6
            r = np.sqrt(rbsx**2+rbsy**2+rbsz**2)
            r3 = r**3

            A[3*n,  3*m], A[3*n,  3*m+1], A[3*n,  3*m+2] = 1/r + rbsx*rbsx/r3, rbsx*rbsy/r3, rbsx*rbsz/r3 # instead of explicitly creating J,            #
            A[3*n+1,3*m], A[3*n+1,3*m+1], A[3*n+1,3*m+2] = rbsy*rbsx/r3, 1/r + rbsy*rbsy/r3, rbsy*rbsz/r3 # this calculated each component individually, #
            A[3*n+2,3*m], A[3*n+2,3*m+1], A[3*n+2,3*m+2] = rbsz*rbsx/r3, rbsz*rbsy/r3, 1/r + rbsz*rbsz/r3 # I've found it to be about 4x faster.         #  
    A = A/(8*np.pi) #Change to (8*np.pi*mu) if using a non-unit viscosity
    return A

#CONTRUCT VELOCITY SUPERVECTOR FOR EACH NODE (Eq. 10)
#Input: rb: Nx3 list of nodes, V: 1x3 translational velocity vector, om: 1x3 angular velocity vector
#output: v: 3Nx1 supervector of velocities of each node (rhs in Eq. 8)
def rhsConstruct(rb,V,om):
    rb, V, om = np.array(rb), np.array(V), np.array(om)
    N = np.size(rb,0)
    v = np.zeros([3*N])
    for n in range(N):
        vTot = np.cross(rb[n,:],om)
        v[3*n:(3*n+3)] = V + vTot
    return v


#FUNCTION TO FIND FORCE AND TORQUE
#Input: rb: Nx3 list of nodes, Mx3 list of sites, V: 1x3 translational velocity vector, om: 1x3 angular velocity vector
#Output: f: Mx3 force vector, t: Mx3 torque vector. ----- F = np.sum(f,0), T = np.sum(t,0) are the force and torque on the body, respectively
#note, forces and torques are non-dimensional
def findForceAndTorque(rb,rs,V,om):
    rb, rs, V, om = np.array(rb), np.array(rs), np.array(V), np.array(om)
    A = matrixConstruct(rb,rs)  # Eq. 9
    v = rhsConstruct(rb,V,om)   # Eq. 10
    f = np.matmul(la.pinv(A),v) # Eq. 8
    f = np.reshape(f,[np.size(rs,0),3])
    t = np.zeros([np.size(rs,0),3]) # Eq. 12a
    for m in range(np.size(rs,0)):
        t[m,:] = np.cross(rs[m,:],f[m,:]) # Eq. 12b
    return f,t

#####################################
######--- GENERAL FUNCTIONS ---######
######- these are  not needed -######
######-- for the MFS but are --######
######------ very useful ------######
######################################

#FUNCTION TO FIND SITES GIVEN NODES & NORMALS
#Input: rb: node positions, nb: node normal (nb[i,:] is the outward-facing normal of the ith node (rb[i,:]))
#Output: rs: site positons. note, prints MQ, mesh quality -- anything <80% means the result may be unreliable
#Useful for .obj or .ply etc. files, or non-trivial shapes
def rsFinder(rb,nb):
    N=len(rb)
    rs = np.zeros((N,3))
    beta=1.05
    maxcosangle=0
    cosAngleSum=0
    for II in range(N):
        ri=rb[II,:]
        ni=-nb[II,:]
        lmin=1e14 
        for IJ in range(N):      
            if II != IJ:
                rj=rb[IJ,:]
                a=rj-ri
                m=a/np.linalg.norm(a)
                qb=2*np.linalg.norm(a)*np.dot(ni,m)
                qa=beta**2-1
                qc=-np.linalg.norm(a)**2
                l=(-qb+(qb**2-4*qa*qc)**0.5)/(2*qa)
                if l < lmin:
                    lmin = l
                    rjmin = rj
    
        rs[II,:]=ri+ni*lmin
    
        a1=(rjmin-rs[II,:])/np.linalg.norm(rjmin-rs[II,:])
        a2=(ri-rs[II,:])/np.linalg.norm(ri-rs[II,:])
        cosAngleSum=cosAngleSum+np.absolute(np.dot(a1,a2))
    
    avCosAngle=cosAngleSum/N
    MQ = 100*avCosAngle
    print(f"Number of nodes: N={N}; mesh-quality measure: {round(100*avCosAngle,2)}%")
    return rs

#FUNCTION TO FIND nPoints POINTS DISTRIBUTED AROUND A UNIT SPHERE
#Input: nPoints: number of points on a unit sphere, r: radius
#Output: points: points on a sphere 
def sphereMaker(nPoints,r):
    indices = np.arange(0, nPoints, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/nPoints)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = r*np.cos(theta) * np.sin(phi), r*np.sin(theta) * np.sin(phi), r*np.cos(phi)

    points = np.array([x,y,z]).T
    return points  

#FUNCTION TO FIND nPoints POINTS DISTRIBUTED AROUND A SPHEROID
#Input: nPoints: number of points on a unit sphere, a,b,c: radii in the x,y,z directions, respectively
#Output: points: points on a sphere 
def spheroidMaker(nPoints,a,b,c):
    indices = np.arange(0, nPoints, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/nPoints)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = a*np.cos(theta) * np.sin(phi), b*np.sin(theta) * np.sin(phi), c*np.cos(phi)

    points = np.array([x,y,z]).T
    return points  

#FUNCTION BECAUSE I GOT TIRED OF CONTINOUSLY WRITING IT
def printForce(f):
    f = np.reshape(f,[int(np.size(f)/3),3])
    f = np.sum(f,0)/(6*np.pi)
    print(f"Normalised net force: {np.round(f,5)}")