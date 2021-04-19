#!/usr/bin/env python
# coding: utf-8

# # Assignment A2 [40 marks]
# 
# The assignment consists of 3 exercises. Each exercise may contain coding and/or discussion questions.
# - Type your **code** in the **code cells** provided below each question.
# - For **discussion** questions, use the **Markdown cells** provided below each question, indicated by 📝. Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 tutorial notebook for Markdown syntax.

# ---
# ## Question 1: Numerical Differentiation [10 marks]
# 
# A general $N$-point finite difference approximation of the derivative $F' \left( x \right)$ of a sufficiently smooth function $F \left( x \right)$ can be written as
# 
# $$
# F' \left( x \right) \approx \frac{1}{\Delta x} \sum_{i = 1}^N \alpha_i F \left( x + \beta_i \Delta x \right),
# \qquad \qquad \qquad (1)
# $$
# 
# with step size $\Delta x > 0$, and $\alpha_i, \beta_i \in \mathbb{Q}$, with $\beta_i \neq \beta_j$ for $i\neq j$. For example, the centred difference approximation $D_C(x)$ seen in the course has $N = 2$, and
# 
# $$
# \begin{cases}
# \alpha_1 = \frac{1}{2}, &\alpha_2 = -\frac{1}{2}, \\
# \beta_1 = 1, &\beta_2 = -1,
# \end{cases}
# \qquad
# \text{giving} \quad
# F'(x) \approx \frac{1}{2\Delta x} \left(F\left(x + \Delta x\right) - F\left(x - \Delta x\right)\right).
# $$
# 
# **1.1** Consider another finite difference approximation defined as in $(1)$, this time with $N=3$, and
# 
# $$
# \begin{cases}
# \alpha_1 = -\frac{4}{23}, &\alpha_2 = -\frac{9}{17}, &\alpha_3 = \frac{275}{391} \\
# \beta_1 = -\frac{3}{2}, &\beta_2 = -\frac{1}{3}, &\beta_2 = \frac{4}{5}
# \end{cases}.
# $$
# 
# Investigate the accuracy of this approximation.
# 
# **[5 marks]**

# In[23]:


import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):                   #Take sinx as a smooth function
    return math.sin(x)

def f_prime(x):             #Take cosx as derivative of sinx
    return math.cos(x)

def f_approx(f, dx, x):     #Simplified finite difference approximation
    D = 1/(391*dx) * (275*f(x + (4/5)*dx) - 68*f(x - (3/2)*dx) - 207*f(x - (1/3)*dx))
    return D
                    
x = np.linspace(0, 2*math.pi, 1000)     #Values for x
dx = [0.5, 0.25, 0.1, 0.05]             #Values for delta x
for i in range(len(dx)):
    err_list = []
    for j in range(len(x)):
        err = abs(f_approx(f, dx[i], x[j]) - f_prime(x[j]))         #Difference between approx and actual values
        err_list.append(err)
    plt.plot(x,err_list)

"Display and plot the graph"
        
plt.xlabel('Values for x')
plt.ylabel('Difference between actual and approximate value')
plt.title('Error of Approximation')
plt.legend(dx)

plt.show()


# ***📝 Discussion for question 1.1***
# 
# As can be seen from the code and the graph above, the accuracy of the approximation depends on the size of delta x. The smaller the value of delta x is, the more accurate the function gets. With a sufficiently small value of x as can be seen, the approximation gives the same value as the actual derivative for a sufficiently smooth function which in this particular case, is sin(x).

# **1.2** For an arbitrary choice of $\beta_i$ values, what is the minimum number of points $N_{p}$ required to find an approximation $(1)$ which is at least $p$th order accurate?
# 
# *Hint:* consider the Taylor expansion of $F \left( x + \beta_i \Delta x \right)$ around $x$.
# 
# **[3 marks]**

# ***📝 Discussion for question 1.2***

# **1.3** Using your reasoning from **1.2**, write a function `FD_coefficients()` which, given $N_p$ values $\beta_i$, returns $N_p$ coefficients $\alpha_i$ such that the approximation $(1)$ is at least $p$th order accurate.
# 
# Use your function to obtain the coefficients $\alpha_i$ from **1.1**.
# 
# **[2 marks]**

# In[2]:


print('this is working')


# ---
# ## Question 2: Root Finding [10 marks]
# 
# Consider the following polynomial of cubic order,
# 
# $$
# p(z) = z^3 + (c-1)z - c,
# $$
# where $c \in \mathbb{C}$.
# 
# This polynomial is complex differentiable, and we can apply Newton's method to find a complex root $z_\ast$, using a complex initial guess $z_0 = a_0 + ib_0$. In this problem, we seek to map the values of $z_0$ which lead to convergence to a root of $p$.
# 
# **2.1** Write a function `complex_newton(amin, amax, bmin, bmax, c, N, eps, target_roots)` which implements Newton's method to find roots of $p(z)$ using $N^2$ initial guesses $z_0 = a_0 + ib_0$. The input arguments are as follows:
# 
# - The real part $a_0$ of the initial guess should take `N` linearly spaced values between `amin` and `amax` (inclusive).
# - The imaginary part $b_0$ of the initial guess should take `N` linearly spaced values between `bmin` and `bmax` (inclusive).
# - `c` is the parameter $c \in \mathbb{C}$ in $p(z)$.
# - `eps` is the tolerance $\varepsilon > 0$.
# - `target_root` takes one of the following values:
#     - if `target_root` is given as `None`, then convergence should be considered as achieved if Newton's method has converged to any root of $p$.
#     - if `target_root` is given as a number, then convergence should only be considered as achieved if Newton's method has converged to the specific root $z_\ast =$ `target_root`.
# 
# Your function should return an array `kmax` of size $N \times N$, containing the total number of iterations required for convergence, for each value of $z_0$. You should decide what to do in case a particular value of $z_0$ doesn't lead to convergence.
#     
# Up to 2 marks will be given to solutions which iterate over each value of $z_0$. To obtain up to the full 4 marks, your implementation should be vectorised -- i.e. use a single loop to iterate Newton's method for all values of $z_0$ at once.
# 
# **[4 marks]**

# In[31]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def P(z,c):                     #the p function
  return z**3 + (c - 1)*z - c

def p(z,c):
  return 3*z**2 + (c - 1)       #the derivative of the p function

def G(z,c):                     #implementing Newton's method
  return z - P(z,c)/p(z,c)

def complex_newton(amin, amax, bmin, bmax, c, N, eps, target_roots):
  a = np.linspace(amin, amax, N)        #find each real point
  b = np.linspace(bmin, bmax, N)        #find each imaginary point
  tol = eps                             #let tolerance = epsilon
  kmax = np.zeros((N,N))                #create initial matrix
  for i in range(N):
    for j in range(N):
      z0 = a[i] + b[j]*1j               #start initial complex guess
      count = 0
      while True:                       #Newton iteration
        z_new = G(z0,c)
        count += 1
        
        'Not sure what to do if it fails to converge' 
        'so if exceeds over 100 loops, I set the count to zero.'
        
        if count > 100:                 
            kmax[i][j] = 0
            break
        
        if abs(z_new - z0) < tol:       #check if it reaches convergence
          if target_roots == "None":
            kmax[i][j] = count
          elif target_roots == z_new:
            kmax[i][j] = count
          else:
            kmax[i][j] = 0
          break
        z0 = z_new
  kmax = kmax.transpose()

  return kmax


# **2.2** For $c = 0$, $a_0 \in [-5,5]$ and $b_0 \in [-5,5]$, with at least $N = 200$ values for each (you can increase $N$ if your computer allows it), use your function `complex_newton()` to calculate, for each $z_0 = a_0 + ib_0$, the total number of iterates needed to reach a disk of radius $\varepsilon$ around the root at $z = 1$. Present your results in a heatmap plot, with $a_0$ on the abscissa, $b_0$ on the ordinate and a colour map showing the total number of iterates. 
# 
# **[3 marks]**

# In[32]:


kmax = complex_newton(-5,5,-5,5,0,200,1.0e-14,1)

ax = sns.heatmap(kmax)
ax.set_xticklabels(np.linspace(-5,5,25))
ax.set_yticklabels(np.linspace(-5,5,20))

plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Heat Map")


# **2.3** For $c = 0.32 + 1.64i$, map out the points $z_0$ for which Newton's method does not converge to any root. What does it do instead?
# 
# *Hint:* Pick a point $z_0$ in a region where Newton's method does not converge and print out, say, 50 iterates. What do you observe?
# 
# **[3 marks]**

# In[34]:


kmax = complex_newton(-5,5,-5,5,0.32 + 1.64*1j,200,1.0e-14,"None")

ax = sns.heatmap(kmax)
ax.set_xticklabels(np.linspace(-5,5,25))
ax.set_yticklabels(np.linspace(-5,5,20))

plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Heat Map")


# ***📝 Discussion for question 2.3***
# 
# We can see that if for that point it doesn't converge, it's stuck in a loop between two values and keeps on going back and forth between the two.
# 

# ---
# ## Question 3: Numerical Integration of an ODE [20 marks]
# 
# Cardiac tissue is an example of an excitable medium, where a small stimulus can lead to a large response (a heart beat). The FitzHugh-Nagumo model describes the electrical activity of a single cardiac cell in terms of the transmembrane potential $u$ and a recovery variable $v$
# 
# \begin{align}
#         \dot u & = f(u,v) = \frac{1}{\varepsilon} \left( u - \frac{u^3}{3} - v + I \right) \ , \\
#         \dot v & = g(u,v) = \varepsilon \left( u - \gamma v + \beta \right) \ ,
# \end{align}
# 
# where $I$ (a stimulus), $\varepsilon$, $\gamma$, and $\beta$ are known parameters.
# 
# The equation for $u$ leads to fast dynamics with the possibility of excitation, while the linear term proportional to $-v$ in the equation for the recovery variable produces slower dynamics and negative feedback. The FitzHugh-Nagumo model is an example of a stiff differential equation, where the stiffness becomes more pronounced for smaller $\varepsilon$.
# 
# In questions **3.1**, **3.2**, and **3.3**, we take $\varepsilon = 0.2$, $\gamma = 0.8$, and $\beta = 0.7$.
# 
# 
# **3.1** The fixed points, defined by $\dot u = f(u, v) = 0$ and $\dot v = g(u, v) = 0$, correspond to the state of a cell at rest. Write a function `resting_state()` to determine the values $(u_I^*, v_I^*)$ for the cell in its resting state for a given value of $I$ and a given initial guess $(u_{I, 0}, v_{I, 0})$, using Newton's method.
# 
# Use your function to compute $(u_I^*, v_I^*)$ for $I=0$ and $I = 0.5$, with initial guess $(u_{I, 0}, v_{I, 0}) = (0.2, 0.2)$.
# 
# 
# **[5 marks]**

# In[1]:


import numpy as np

def f(u,v,I):                       #function for transmembrane potential
    return (1/0.2)*(u - (u**3)/3 - v + I)

def g(u,v):                         #function for recovery variable
    return 0.2*(u - 0.8*v + 0.7)

def Jac(u, v):                      #function for Jacobian matrix
    J = np.zeros([2, 2])
    J[0, 0] = (1/0.2)*(1 - u**2)    
    J[0, 1] = (1/0.2)*(-1)          
    J[1, 0] = 0.2*(1)
    J[1, 1] = 0.2*(-0.8)
    return J

def F(u, v, I):                     #function to put solutions in an array
    return np.array([f(u, v, I), g(u, v)])

def resting_state(u, v, I):         #function to find the resting state
    tol = 1e-12
    x = np.array([u,v])
    rootx = [u]                     #list of u solutions
    rooty = [v]                     #list of v solutions
    count = 0
    while np.linalg.norm(F(u, v, I)) >= tol:         # Newton iteration
        e = -np.linalg.solve(Jac(x[0], x[1]), F(x[0], x[1], I))        
        x += e
        count += 1
        rootx.append(round(x[0], 12))
        rooty.append(round(x[1], 12))
        if rootx[count] == rootx[count - 1] and rooty[count] == rooty[count - 1]:                                                                                    
            'ensure there is no reptition of same solutions'
            break
    return x

print(resting_state(0.2,0.2,0))
print(resting_state(0.2,0.2,0.5))


# **3.2** Using the method of your choice **\***, compute the numerical solution $(u_n, v_n) \approx (u(n\Delta t), v(n\Delta t)), n=0, 1, 2, \dots$ for the FitzHugh-Nagumo model.
# 
# You should compute the solution for both $I = 0$ and $I = 0.5$, starting at time $t = 0$ until at least $t = 100$, with $(u_0 = 0.8, v_0 = 0.8)$ as the initial condition.
# 
# Present your results graphically by plotting
# 
# (a) $u_n$ and $v_n$ with **time** (not time step) on the x-axis,  
# (b) $v_n$ as a function of $u_n$. This will show what we call the solution trajectories in *phase space*.
# 
# You should format the plots so that the data presentation is clear and easy to understand.
# 
# Given what this mathematical model describes, and given that $I$ represents a stimulus, how do you interpret your results for the two different values of $I$? Describe your observations in less than 200 words.
# 
# 
# **\*** You may use e.g. the forward Euler method seen in Week 7 with a small enough time step, or use one of the functions provided by the `scipy.integrate` module, as seen in Quiz Q4.
# 
# 
# **[7 marks]**

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def f(u,v,I):                       #function for transmembrane potential
    return (1/0.2)*(u - (u**3)/3 - v + I)

def g(u,v):                         #function for recovery variable
    return 0.2*(u - 0.8*v + 0.7)

def sol(dt,I,u,v):
    
    "function to compute the numerical solution for the FitzHugh-Nagumo model"                      
    
    N = int(100/dt)
    psiu = np.zeros(N + 1)
    psiv = np.zeros(N + 1)
    psiu[0] = u
    psiv[0] = v
    for n in range(N):
        psiu[n + 1] = psiu[n] + dt * f(psiu[n],psiv[n],I)
        psiv[n + 1] = psiv[n] + dt * g(psiu[n],psiv[n])
    return psiu, psiv, N

dt = 0.1                            # time step used
psiu, psiv, N = sol(dt,0, 0.8.0.8)           # solution when I = 0
psiu2, psiv2, N = sol(dt,0.5, 0.8, 0.8)       # solution when I = 0.5

"Following code is to plot the graph"

t = np.linspace(0.0, N * dt, N + 1)
fig, ax = plt.subplots(2,2)
ax[0,0].plot(t, psiu, label = "transmembrane potential")
ax[0,0].plot(t, psiv, label = "recovery variable")
ax[1,0].plot(t, psiu2, label = 'transmembrane potential')
ax[1,0].plot(t, psiv2, label = "recovery variable")
ax[0,1].plot(psiu, psiv)
ax[1,1].plot(psiu2, psiv2)

"Following code is to label the graph"

ax[0, 0].set_title('U and V over time when I = 0')
ax[0, 1].set_title('U against V when I = 0')
ax[1, 0].set_title('U and V over time when I = 0.5')
ax[1, 1].set_title('U against V when I = 0.5')
#for i in ax.flat:
#    i.set(xlabel='x-label', ylabel='y-label')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for i in ax.flat:
    i.label_outer()

ax[0,0].set(ylabel = 'u and v')
ax[1,0].set(xlabel = 'time', ylabel = 'u and v')
ax[0,1].set(ylabel = 'v')
ax[1,1].set(xlabel = 'u', ylabel = 'v')
plt.show()


# ***📝 Discussion for question 3.2***
# 
# We can see that when there is no stimulus, the values of both the transmembrane potential and the recovery variable both converges to a single point over a short period of time. However when there is a stimulus (which was 0.5), we can see that neither the potential or variable converges and they both behave like a continuous wave with no limit. However, regardless of the stimulus, by looking on the graphs on the left hand side, we can see that for left half of the graph, diagonal from the top right to bottom left, with or without stimulus, they share the same values.

# **3.3** Compute the eigenvalues of the Jacobi matrix
#         
# $$
# \large
# \begin{pmatrix}
#     \frac{\partial f}{\partial u} & \frac{\partial f}{\partial v} \\ 
#     \frac{\partial g}{\partial u} & \frac{\partial g}{\partial v}
# \end{pmatrix}_{u = u_I^*, v = v_I^*}
# $$
# 
# evaluated at the fixed points $u = u_I^*, v = v_I^*$, for $I = 0$ and $I = 0.5$. What do you observe?
# 
# *You may use NumPy built-in functions to compute eigenvalues.*
# 
# 
# **[3 marks]**

# In[7]:


import scipy.linalg as la
evals, evecs = la.eig(Jac(0.2,0.2))
print(evals)


# ***📝 Discussion for question 3.3***
# I believe the value for I is made redundant since it is a known parameter. Since the Jacobi matrix diffrentiates the f function with respect to u and v, I simply becomes 0 and the eigenvalues for both values of I remain the same.

# **3.4** For this question, we set $I = 0$ and $\gamma = 5$.
# 
# (a) Use the function `resting_state()` you wrote for **3.1** to find three fixed points, $(u_{(0)}^*, v_{(0)}^*)$, $(u_{(1)}^*, v_{(1)}^*)$ and $(u_{(2)}^*, v_{(2)}^*)$, using the initial conditions provided in the NumPy array `uv0` below (each row of the array constitutes a pair of initial conditions for one of the three fixed points).
# 
# (b) Compute the numerical solution $(u_n, v_n), n=0, 1, 2, \dots$ using the initial condition $(u_{(0)}^* + \delta, v_{(0)}^* + \delta)$, for $\delta \in \{0, 0.3, 0.6, 1.0\}$. This simulates the activity of a cell starting at a small perturbation $\delta$ of one of its resting states, in this case $(u_{(0)}^*, v_{(0)}^*)$.
# 
# Plot your results in a similar way as in question **3.2**, and discuss your observations in less than 150 words. In particular, does the solution always return to the same fixed point, i.e. the same resting state?
# 
# **[5 marks]**

# In[20]:


import numpy as np

def g(u,v):                         #function for recovery variable
    return 0.2*(u - 5*v + 0.7)

def Jac(u, v):                      #function for Jacobian matrix
    J = np.zeros([2, 2])
    J[0, 0] = (1/0.2)*(1 - u**2)    
    J[0, 1] = (1/0.2)*(-1)          
    J[1, 0] = 0.2*(1)
    J[1, 1] = 0.2*(-5)
    return J

def sol(dt,I,u,v):
    
    "function to compute the numerical solution for the FitzHugh-Nagumo model"                      
    
    N = int(100/dt)
    psiu = np.zeros(N + 1)
    psiv = np.zeros(N + 1)
    psiu[0] = u
    psiv[0] = v
    for n in range(N):
        psiu[n + 1] = psiu[n] + dt * f(psiu[n],psiv[n],I)
        psiv[n + 1] = psiv[n] + dt * g(psiu[n],psiv[n])
    return psiu, psiv, N

# Initial conditions
uv0 = np.array([[0.9, 0.6], [0., 0.4], [-1.7, -0.3]])

ans = []

for i in uv0:
    ans.append(resting_state(i[0],i[1],0))
    
dt = [0.3, 0.6, 1.0]
fig, ax = plt.subplots(3)
for i in ans:
    count = 0
    for j in range(len(dt)):
       psiu, psiv, N = sol(dt[j],0, i[0], i[1])
       ax[count].plot(psiu,psiv)
    count += 1   
    
print(ans)
plt.show()


# ***📝 Discussion for question 3.4***
# 
# An attempt was made to plot the graphs but from what I can see, the graph does not always return to the same resting state. For the bigger values for dt, it leads to an overflow error but for when dt is 0.3, it converges to a single point.

# In[ ]:




