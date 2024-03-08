
import numpy as np
import matplotlib.pyplot as plt

def f0T(t,P0T):  # r(0)=f(0,0)
    # time-step needed for differentiation
    dt = 0.01    
    expr = - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    return expr

def GeneratePathsHoLeeEuler(NoOfPaths,NoOfSteps,T,P0T,sigma):    #Generate paths   #P0T is in input 
    
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.01,P0T)   
    theta = lambda t: (f0T(t+dt,P0T)-f0T(t-dt,P0T))/(2.0*dt) + sigma**2.0*t 
     
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    M = np.zeros([NoOfPaths, NoOfSteps+1])
    M[:,0]= 1.0 # initialisation de M
    R[:,0]=r0  # initialisation de R "disant c'est le point de sortie à t=t0 on a r(t0)=r0" "Spot maintenant"
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)  
        for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        R[:,i+1] = R[:,i] + theta(time[i]) * dt + sigma* (W[:,i+1]-W[:,i])  # discretisation de R
        M[:,i+1] = M[:,i] * np.exp((R[:,i+1]+R[:,i])*0.5*dt) # pour le pricing du bonds
        time[i+1] = time[i] +dt

        
    # Outputs
    paths = {"time":time,"R":R,"M":M}
    return paths  
        
def mainCalculation():
    NoOfPaths = 25000  # nbre of monticarlo paths
    NoOfSteps = 500    # nbre of steps
       
    sigma = 0.007      # volatility, whatever sigma is I get the ZCB Yield calibrated 
        
    # We define a ZCB curve (obtained from the market)
    
    P0T = lambda T: np.exp(-0.1*T)  # à t=0 market price of zero coupon bond
    # P0T = lambda T: 100*np.exp(-0.1*T) 
    
    # In this experiment we compare ZCB from the Market and Monte Carlo
    "Pricing with Monte Carlo part"
    "En faite le principe mathalan bich n5arej il r(t) na3mal mean 3la kol colonne ya3ni fi kol dt, r(t) bich ykoun moyenne mta3"
    " les valeurs li lguithom fi kol sénario fi kol path ya3ni kif kif binesba lil M"
    T = 40
    paths= GeneratePathsHoLeeEuler(NoOfPaths,NoOfSteps,T,P0T,sigma)
    M = paths["M"] # ici 5arajet les M mta3i lkol 
    ti = paths["time"]
        
    # Here we compare the price of an option on a ZCB from Monte Carlo and Analytical expression    
    P_t = np.zeros([NoOfSteps+1])
    for i in range(0,NoOfSteps+1):     # içi o  calcule l'expected value 
        P_t[i] = np.mean(1.0/M[:,i])   #  100*np.mean(1.0/M[:,i])
    

    # objectif verifier que P_t equiale to P0T
    
    plt.figure(1)
    plt.grid()
    plt.xlabel('T')
    plt.ylabel('P(0,T)')
    plt.plot(ti,P0T(ti))
    plt.plot(ti,P_t,'--r')
    plt.legend(['P(0,t) market','P(0,t) Monte Carlo'])
    plt.title('ZCBs from Ho-Lee Model')
    
mainCalculation()



# Dans ce Notebook nous avons fait la comparaison entre deux expressions de P(t,T)  

