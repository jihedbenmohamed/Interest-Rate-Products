
import numpy as np
import enum 
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.interpolate import splrep, splev, interp1d

# This class defines puts and calls
class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0
    
def IRSwap(CP,notional,K,t,Ti,Tm,n,P0T):
    # CP- payer of receiver
    # n- notional
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # r_t -interest rate at time t
    ti_grid = np.linspace(Ti,Tm,int(n))
    tau = ti_grid[1]- ti_grid[0]
    
    # overwrite Ti if t>Ti
    prevTi = ti_grid[np.where(ti_grid<t)]
    if np.size(prevTi) > 0: #prevTi != []:
        Ti = prevTi[-1]
    
    # Now we need to handle the case when some payments are already done
    ti_grid = ti_grid[np.where(ti_grid>t)]           

    temp= 0.0
        
    for (idx,ti) in enumerate(ti_grid):
        if ti>Ti:
            temp = temp + tau * P0T(ti)
            
    P_t_Ti = P0T(Ti)
    P_t_Tm = P0T(Tm)
    
    if CP==OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp
    elif CP==OptionTypeSwap.RECEIVER:
        swap = K * temp - (P_t_Ti - P_t_Tm)
    
    return swap * notional

def P0TModel(t,ti,ri,method):
    rInterp = method(ti,ri)
    r = rInterp(t)
    return np.exp(-r*t)

def YieldCurve(instruments, maturities, r0, method, tol):
    r0 = deepcopy(r0) 
    ri = MultivariateNewtonRaphson(r0, maturities, instruments, method, tol=tol)
    return ri

def MultivariateNewtonRaphson(ri, ti, instruments, method, tol):
    err = 10e10
    idx = 0
    while err > tol:
        idx = idx +1 # iteration number
        values = EvaluateInstruments(ti,ri,instruments,method) # ti time for spine points, ri our spine points
        # values represent : PV(df(idx)) 
        J = Jacobian(ti,ri, instruments, method)
        J_inv = np.linalg.inv(J)
        err = - np.dot(J_inv, values)  # df(idx)-df(idx-1) "difference between tow consecutive set of error"
        ri[0:] = ri[0:] + err  # here we sefine or next set of spine points
        err = np.linalg.norm(err) # on prend la norme pour trouver l'erreur
        print('index in the loop is',idx,' Error is ', err)
    return ri

def Jacobian(ti, ri, instruments, method):
    eps = 1e-05
    swap_num = len(ti)
    J = np.zeros([swap_num, swap_num])
    val = EvaluateInstruments(ti,ri,instruments,method)
    ri_up = deepcopy(ri) # deepcopy : If we choc ri_up, on garde ri the same
    
    for j in range(0, len(ri)): # do this for every colonnes
        ri_up[j] = ri[j] + eps # We will choc every each individual spine points
        val_up = EvaluateInstruments(ti,ri_up,instruments,method) # We evaluate swaps of the choced spine points
        ri_up[j] = ri[j]
        dv = (val_up - val) / eps
        J[:, j] = dv[:]
    return J

def EvaluateInstruments(ti,ri,instruments,method):
    P0Ttemp = lambda t: P0TModel(t,ti,ri,method) # ZCB curve
    val = np.zeros(len(instruments))
    for i in range(0,len(instruments)):
        val[i] = instruments[i](P0Ttemp) # evaluate every Swap "Instrument" as a function of ZCB 
    return val # the output is a vector

def linear_interpolation(ti,ri):
    interpolator = lambda t: np.interp(t, ti, ri)
    return interpolator

def mainCode():
    
    # Convergence tolerance "Stopping cretiria for the optimization"
    tol = 1.0e-15
    
    # Initial guess for the spine points
    r0 = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])   
    
    # Interpolation method
    method = linear_interpolation
    
    # Construct swaps that are used for building of the yield curve
    K   = np.array([0.04/100.0,	0.16/100.0,	0.31/100.0,	0.81/100.0,	1.28/100.0,	1.62/100.0,	2.22/100.0,	2.30/100.0])
     # for K every thing is devised by 100 because slide 25/47 every thing is in %
     # K is the strike of the swap which give us price of zero for the swap at t=0    
    mat = np.array([1.0,2.0,3.0,5.0,7.0,10.0,20.0,30.0]) # vector of maturity of the Swaps in Years
    
    # Here we define Swaps to be function on ZCB 
    swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[0],0.0,0.0,mat[0],4*mat[0],P0T)
    swap2 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[1],0.0,0.0,mat[1],4*mat[1],P0T)
    swap3 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[2],0.0,0.0,mat[2],4*mat[2],P0T)
    swap4 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[3],0.0,0.0,mat[3],4*mat[3],P0T)
    swap5 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[4],0.0,0.0,mat[4],4*mat[4],P0T)
    swap6 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[5],0.0,0.0,mat[5],4*mat[5],P0T)
    swap7 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[6],0.0,0.0,mat[6],4*mat[6],P0T)
    swap8 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[7],0.0,0.0,mat[7],4*mat[7],P0T)
    instruments = [swap1,swap2,swap3,swap4,swap5,swap6,swap7,swap8]
    
    # determine optimal spine points
    ri = YieldCurve(instruments, mat, r0, method, tol) # This function return our spine points
    print('\n Spine points are',ri,'\n')
    
    plt.plot(mat,ri)
    
    # Build a ZCB-curve/yield curve from the spine points
    P0T_Initial = lambda t: P0TModel(t,mat,r0,method)
    P0T = lambda t: P0TModel(t,mat,ri,method)
    
    # price back the swaps
    swapsModel = np.zeros(len(instruments))
    swapsInitial = np.zeros(len(instruments))
    for i in range(0,len(instruments)):
        swapsInitial[i] = instruments[i](P0T_Initial)
        swapsModel[i] = instruments[i](P0T)
    
    print('Prices for Pas Swaps (initial) = ',swapsInitial,'\n')
    print('Prices for Par Swaps = ',swapsModel,'\n') # if the prices here are zero so the model price back the swaps used
                                                     # to build the yield curve
    
    return 0.0

mainCode()
    
    