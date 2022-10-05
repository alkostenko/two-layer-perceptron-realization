#Kostenko KM-93 lab 1
#part 2: Elementary Two-Layer Perceptron with 1-1-1 Structure Training

import numpy as np
from matplotlib import pyplot as plt

#training function
def training(x, yr, dd=0.1):
    y_list=[]
    dn_list=[]
    #set initial dn
    dn=0.5
    #initialize wieghts
    w=np.random.rand(2)
    iter=0
    #training until condition is met
    while dn>dd:
        #calculate y2 and y3
        y2=1/(1+np.exp(-1*w[0]*x))
        y3=1/(1+np.exp(-1*w[1]*(1/(1+np.exp(-1*w[0]*x)))))
        y_list.append(y3)
        #calculate dn
        dn=np.abs((yr-y3)/yr)
        dn_list.append(dn)
        #print y and dn
        if iter%10==0:
            print(f"iteration {iter}")
            print(f"y = {y3}")
            print(f"dn = {dn}") 

        if dn<=dd:
            break
        else:
            #adjust weights
            q3=y3*(1-y3)*(yr-y3)
            q2=y2*(1-y3)*(q3*w[1])
            dw2=q3*y2
            dw1=q2*x
            w[1]=w[1]+dw2
            w[0]=w[0]+dw1

            iter+=1
    return w, y_list, dn_list


#testing function
def testing(x, w):
    return(1/(1+np.exp(-1*w[1]*(1/(1+np.exp(-1*w[0]*x))))))

def elementary_perceptron_111():
    #input initial data
    x=np.random.rand(1)[0]
    yr=np.random.rand(1)[0]
    print(f"x={x}\ny={yr}")
    #training
    w, y_list, dn_list=training(x, yr)
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
    x+=np.random.rand(1)[0]
    print(f"x={x}")
    print(f"y={testing(x, w)}")

elementary_perceptron_111()