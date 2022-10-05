#Kostenko KM-93 lab 1
#part 1: Classic Neuron Training

import numpy as np
from matplotlib import pyplot as plt


#additional list for process visualisation
y_list=[]
dn_list=[]

#training function
def training(x, yr, w, dd):
    iter=0
    
    #set initial dn > than dd
    dn=0.5

    #training untill condition is met
    while dn>dd:
        xsum=0
        #calculate xs
        for i in range(len(x)):
            xsum+=x[i]*w[i]
        #calculate y 
        y=1/(1+np.exp(-1*xsum))
        y_list.append(y)
        #calculate dn
        dn=np.abs((yr-y)/y)
        dn_list.append(dn)
        #print every 5th iteration
        if iter%5==0:
            print(f"iteration {iter}")
            print(f"y = {y}")
            print(f"dn = {dn}") 

        if dn<=dd:
            break
        else:   
            #adjust weights
            q=y*(1-y)*(yr-y)
            for i in range(len(w)):
                dw=x[i]*q
                w[i]=w[i]+dw
            iter+=1
    return w, y_list, dn_list

#testing function
def testing(x, w):
    xsum=0
    for i in range(len(x)):
        xsum+=x[i]*w[i]
    y=1/(1+np.exp(-1*xsum))
    return y


def classic_neuron():
    #input values
    x=[1, 3, 5, 7]
    yr=.3
    dd=.1
    w=np.random.rand(len(x))
    #training neuron
    w, y_list, dn_list=training(x, yr, w, dd)

    print(f"Conclusive y: {y_list[-1]}")
    print(f"Conclusive w: {w}")

    #visualizing training process
    iteration=[]
    for i in range(len(y_list)):
        iteration.append(i)

    plt.plot(iteration, y_list, color="blue", label="y")
    plt.plot(iteration, dn_list, color="orange", label="dn")

    plt.title("Classic Neuron Training Process")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    #testing
    x=input("Input 4 x devided by 'space': ")
    x=x.split(" ")
    for i in range(len(x)):
        x[i]=float(x[i])
    print(x)
    print(testing(x, w))

classic_neuron()