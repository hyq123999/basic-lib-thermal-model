# /bin/python3
# (c) Paul Hansel 2020

import numpy as np
import matplotlib.pyplot as plt

# uniform heat transfer model
# assumes infinite internal thermal conductivity

a = area = 0.1 # rough guess of pack exposed surface area in m2

# real surface area of all cells is about 0.7 m2
actual_area = 130 * (0.070 * 0.021 * np.pi + 2 * np.pi * 0.0105**2) #m2

# heat transfer coefficient
# randomly assigned by hand
h = 50 #w/K/m2

# centigrade
tout = outside_temp = 25

# joules/kg-K
h_sp = 900

# kg mass of pack
m = 9.2

# test currents
currents = [5,10,20,40,50,60]

# pack resistance
r = 0.03 * 13 / 10

# amp hours
q = 50.0

# time unit: seconds
def time_step_heat(i_chg,r,h,a,t0,tout=tout,dt=60,m=m,h_sp=h_sp):
    p_resistive = i_chg**2 * r
    p_radiated = -1 * h * a * (t0 - tout)
    dq = dt * (p_radiated + p_resistive)
    dtemp = dq / (h_sp * m)
    tend = t0 + dtemp
    print(i_chg,t0,tend,h,h_sp,m)
    return tend

def test_charge80pct(i_chg,r,h,a,t0,tout=tout,q=q,m=m,h_sp=h_sp):
    t_80pct = q / i_chg * 0.8 * 60 * 60
    steps = 1000
    t = np.linspace(0,t_80pct,steps)
    dt = t_80pct / steps # find the error
    print("dt here at", i_chg, "is ", dt)
    temp = np.zeros_like(t)
    temp[0] = t0
    i = 0
    while(i < len(t) - 1):
        temp[i+1] = time_step_heat(i_chg,r,h,a,temp[i],tout=tout,dt=dt)
        i += 1
    return (t,temp)

t0 = tout

for i_chg in iter(currents):
    x,y = test_charge80pct(i_chg,r,h,a,t0,tout)
    plt.plot(x/60,y,label=str(i_chg)+"A or "+str(i_chg/q)+"C")

plt.legend(loc=1,fontsize="small")

plt.title("temperature rise vs charge rate 48V50Ah h=" + str(h) + " W/m2-K" )
plt.xlabel("time since charge start (minutes)")
plt.ylabel("pack average temperature (degrees C)")

plt.show()
plt.clf()


# plot end temperature against charge rate

for i_chg in iter(currents):
    x,y = i_chg, np.max(test_charge80pct(i_chg,r,h,a,t0,tout)[1])
    plt.scatter(x,y)

plt.title("final temperature vs charge rate 48V50Ah h=" + str(h) + " W/m2-K" )
plt.xlabel("charge rate (A)")
plt.ylabel("final temperature (deg C)")

plt.show()

# could do more interesting or info-dense plots, but they're less intuitive
