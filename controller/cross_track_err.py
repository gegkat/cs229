#open files waypoints.csv and drive.out to calculate the cross track errors.
import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
import math

f1= 'lake_track_waypoints.csv'
with open(f1) as f:
	content= f.readlines()
	content=[m.strip() for m in content]
	#print(content)
	#print("\n")
	x=[i.split(',',1)[1] for i in content]
	z=[i.split(',',1)[0] for i in content]
	#print(x)
	#print(z)
f.close()
#plots for waypoints

x_plot=np.asarray(x,dtype=float)
z_plot=np.asarray(z,dtype=float)
x_plot=np.fliplr([x_plot])[0]
z_plot=np.fliplr([z_plot])[0]
#print(np.size(x_plot))
#print(np.size(z_plot))
plt.plot(z_plot,x_plot)
#plt.show()

f2= 'drive.out'
with open(f2) as f:
	content= f.readlines()
	content=[m.strip() for m in content]
	#print(content)
	#print("\n")
	x=[i.split(',',6)[4] for i in content]
	z=[i.split(',',6)[0] for i in content]
	speed=[i.split(',',6)[2] for i in content]

f.close()
x_o=np.asarray(x,dtype=float)
z_o=np.asarray(z,dtype=float)
speed_o=np.asarray(speed,dtype=float)

#TO GET THE AVERAGE SPEED OF DRIVING DURING THE WHOLE CAR RUN
avg_dist=trapz(speed_o,dx=5)
avg_speed=avg_dist/(np.size(speed_o))
print(avg_dist)
print(avg_speed)

#IDEA: divide the train data into those many number of chunks as the waypoints csv matrix has, and I've reverted it since the csv matrix has clockwise data and the drive.out has counter clock-wise data.
#After the data is segmented into those many number of chunks as the csv matrix (if oyou look closely, the x and z coordinates don't change much during that whole segment so it's safe to move on to the next segment)
sample= np.size(x_plot)
#print("samples: "+ str(sample)+"\n")
jump=math.ceil(np.size(x_o)/sample)
x_out=x_o[1:np.size(x_o):jump]
z_out=z_o[1:np.size(x_o):jump]
x_ref=x_plot[0:math.ceil(np.size(x_plot))]
z_ref=z_plot[0:math.ceil(np.size(x_plot))]
print("x_out size: " + str(np.size(x_out))+"z_out size:" + str(np.size(z_out)))
print("x_out: " + str(x_out)+"z_out:" + str(z_out))
print("x_ref size" + str(np.size(x_ref))+"z_ref size:" + str(np.size(z_ref)))
print("x_ref " + str(x_ref)+"z_ref:" + str(z_ref))

#finding out the max distance to get idea of where the CAR WENT OFF-TRACK
dist=np.square(x_ref-x_out)+np.square(x_ref-x_out) 
max_dist=np.amax(dist)
max_arg=np.argmax(dist)
#zdel=np.sqaure(z_ref-z_out)
#xdel=np.sqaure(x_ref-x_out)
print("max dist: "+str(max_dist)+"at trackpoint index: "+str(max_arg))

#This part is just playing with x, z norms to get the norms of difference vectors sqaured--need to figure out a way to use
x_norm= np.linalg.norm(np.square(x_ref-x_out))
z_norm= np.linalg.norm(np.square(z_ref-z_out))
print("x_ref norm: " + str(x_norm)+"z_ref norm:" + str(z_norm))
#print(np.absolute(x_ref)-np.absolute(x_out))
#print(np.absolute(z_ref)-np.absolute(z_out))

