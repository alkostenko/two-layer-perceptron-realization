#Kostenko KM-93 lab 1
#part 3: Elementary Two-Layer Perceptron with 2-3-1 Structure Training

import numpy as np
from matplotlib import pyplot as plt

#training function
def training(x, dd=0.01):
    yr=np.sum(x)
    # print(yr)
    y_list=[]
    dn_list=[]
    x0=np.random.rand(4)
    #set initial dn
    dn=0.5
    #initialize wieghts
    w=[np.random.rand(2,3), np.random.rand(3), np.random.rand(3), np.random.rand(1)]
    iter=0
    #train until condition is met
    while dn>dd:
        xsums=[]
        ys=[]
        #calculate xs
        for i in range(len(w[1])):
            xsum=w[1][i]*x0[i]+w[0][0][i]*x[0]+w[0][1][i]*x[1]
            xsums.append(xsum)
            ys.append(1/(1+np.exp(-1*xsum)))
        xsum=w[1][0]*x0[3]+w[2][0]*ys[0]+w[2][1]*ys[1]+w[2][2]*ys[2]
        #calculate output value
        y=1/(1+np.exp(-1*xsum))
        y_list.append(y)
        #calculate dn
        dn=np.abs((yr-y)/yr)
        dn_list.append(dn)

        if iter%10==0:
            print(f"iteration {iter}")
            print(f"y = {y}")
            print(f"dn = {dn}") 

        #check if codition is met
        if dn<=dd:
            break

        else:
            #adjust weights
            q=y*(1-y)*(yr-y)
            b=[q*x0[3], q*ys[0], q*ys[1], q*ys[2]]
            w[3][0]=w[3][0]+b[0]
            w[2][0]=w[2][0]+b[1]
            w[2][1]=w[2][1]+b[2]
            w[2][2]=w[2][2]+b[3]

            for i in range(len(ys)):
                q1=ys[i]*(1-ys[i])*(q*w[2][0])
                b=[q1*x0[i], q1*x[0], q1*x[1]]
                w[1][i]=w[1][i]+b[0]
                w[0][0][i]=w[0][0][i]+b[1]
                w[0][1][i]=w[0][1][i]+b[2]
            
            iter+=1
        
    return w, x0, y_list, dn_list

#testing function
def testing(x, x0, w):
    xsums=[]
    ys=[]

    for i in range(len(w[1])):
        xsum=w[1][i]*x0[i]+w[0][0][i]*x[0]+w[0][1][i]*x[1]
        xsums.append(xsum)
        ys.append(1/(1+np.exp(-1*xsum)))

    xsum=w[1][0]*x0[3]+w[2][0]*ys[0]+w[2][1]*ys[1]+w[2][2]*ys[2]
    y=1/(1+np.exp(-1*xsum))
    return y
    
def elementary_perceptron_231():
    #input initial data
    x=np.random.randn(2)
    print(f"x={x}")
    #training
    w, x0, y_list, dn_list=training(x)
    print(f"X0: {x0}")
    print(f"Conclusive y: {y_list[-1]}")
    print(f"Conclusive w: {w}")

    #visualizing training process
    iteration=[]
    for i in range(len(y_list)):
        iteration.append(i)

    plt.plot(iteration, y_list, color="blue", label="y")
    plt.plot(iteration, dn_list, color="orange", label="dn")

    plt.title("Elementary Two-Layer Perceptron Training Process")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    #testing 
    x+=np.random.rand(2)
    print(f"x={x}")
    print(f"y={testing(x, x0, w)}")

elementary_perceptron_231()