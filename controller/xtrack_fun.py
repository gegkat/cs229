import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
import math
import pylab

def xtrack(x,z,x_ref,z_ref):
	#finding the nearest waypoint for each given x,z
	near=np.zeros(np.size(x_ref))
	for i in range(0, np.size(x_ref)):
		near[i]= np.square(float(x)-x_ref[i])+ np.square(float(z)-z_ref[i])
	arg=np.argmin(near)	
	if(arg==(np.size(x_ref)-1)):
		fwarg=0
	else:
		fwarg=arg+1
	print("fwarg="+str(fwarg))
	vec1=np.array([x-x_ref[arg], z-z_ref[arg]])
	vec2=np.array([x_ref[fwarg]-x_ref[arg],z_ref[fwarg]-z_ref[arg]])
	#dp=np.dot(vec1,vec2)
	if(np.dot(vec1,vec2)>0):
		#narg=arg+1
		narg=fwarg
		print("next")
	else:
		narg=arg-1
		print("previous")
	print("arg:"+str(arg)+"nextarg:"+str (narg))
	print("x:"+str(x)+"z:"+str (z))
	print("point arg:"+str(x_ref[arg])+","+str(z_ref[arg])+"point nextarg:"+str (x_ref[narg])+","+str(z_ref[narg]))
	h=(x-x_ref[arg])**2+(z-z_ref[arg])**2
	v1=np.array([x-x_ref[arg],z-z_ref[arg]])
	v2=np.array([x_ref[narg]-x_ref[arg],z_ref[narg]-z_ref[arg]])
	b=((np.dot(v1,v2))/(np.linalg.norm(v2)))**2
	num=(abs((z_ref[narg]-z_ref[arg])*(x)-(x_ref[narg]-x_ref[arg])*(z)+ (x_ref[narg]*z_ref[arg]-z_ref[narg]*x_ref[arg])))**2
	den=((z_ref[narg]-z_ref[arg])**2+(x_ref[narg]-x_ref[arg])**2)
	#d_sq=(num/den)**(0.5)
	d_sq=(h-b)**(0.5)
	print("dist from center:"+str(d_sq))
	return(d_sq,b)

#OPENING WAYPOINT FILE

f1= 'lake_track_waypoints.csv'
with open(f1) as f:
	content= f.readlines()
	content=[m.strip() for m in content]
	#print(content)
	#getting x and z from the lines
	x=[i.split(',',1)[0] for i in content]
	z=[i.split(',',1)[1] for i in content]
f.close()

#np array creation
x_plot=np.asarray(x,dtype=float)
z_plot=np.asarray(z,dtype=float)
#print(x_plot)
#print(z_plot)
#plt.figure(1)
pylab.plot(z_plot,x_plot, label='Track waypoints')


#for quick purpose didn't implement list access so just have four file handles
f2= './NVIDIA/learning_curve/NVIDIA_frac_100/model_NVIDIA_2017-11-19_13-45-50.h5_12mph.telem'
f3= './cnn/learning_curve/cnn_frac_100/model_cnn_2017-11-19_13-36-39.h5_12mph.telem'
f4= './complex/learning_curve/complex_frac_100/model_complex_2017-11-19_13-42-23.h5_12mph.telem'
f5= './simple/learning_curve/simple_frac_100/model_simple_2017-11-19_13-31-33.h5_12mph.telem'
with open(f2) as f:
	content= f.readlines()
	content=[m.strip() for m in content]
	#getting x,z steering angle 
	z=[i.split(',',6)[4] for i in content]
	x=[i.split(',',6)[3] for i in content]
	steer=[i.split(',',6)[0] for i in content]
	#speed=[i.split(',',6)[2] for i in content]

f.close()

#NVIDIA architecture x, z, steer np arrays
x_o=np.asarray(x,dtype=float)
z_o=np.asarray(z,dtype=float)
steer_o=np.asarray(steer,dtype=float)

with open(f3) as f:
	content= f.readlines()
	content=[m.strip() for m in content]
	z_cnn=[i.split(',',6)[4] for i in content]
	x_cnn=[i.split(',',6)[3] for i in content]

f.close()
#cnn architecture x,z np arrays
xcnn_o=np.asarray(x_cnn,dtype=float)
zcnn_o=np.asarray(z_cnn,dtype=float)

with open(f4) as f:
	content= f.readlines()
	content=[m.strip() for m in content]
	z_cmplx=[i.split(',',6)[4] for i in content]
	x_cmplx=[i.split(',',6)[3] for i in content]

f.close()
#complex architecture x,z np arrays 
xcmplx_o=np.asarray(x_cmplx,dtype=float)
zcmplx_o=np.asarray(z_cmplx,dtype=float)

with open(f5) as f:
	content= f.readlines()
	content=[m.strip() for m in content]
	z_simple=[i.split(',',6)[4] for i in content]
	x_simple=[i.split(',',6)[3] for i in content]

f.close()
#simple architecture x,z array
xsimple_o=np.asarray(x_simple,dtype=float)
zsimple_o=np.asarray(z_simple,dtype=float)

#plot for track vs car run
pylab.plot(z_o,x_o, label='Car run')
pylab.legend(loc='upper left')
pylab.xlabel('x')
pylab.ylabel('z')
pylab.title('Original track vs Car run')
pylab.show()

#plot fo steering angle
plt.figure()
plt.plot(steer_o)
plt.ylabel('steering angle(rad)')
plt.title('Steering angle during the run using NVIDIA architecture')
plt.show()

#distannce array - dist_x
#along track distance array - b_x
#where x=nvidia,cnn,cpmplex,simple
dist= np.zeros(np.size(x_o))
b= np.zeros(np.size(x_o))
dist_cnn= np.zeros(np.size(xcnn_o))
b_cnn= np.zeros(np.size(xcnn_o))
dist_cmplx= np.zeros(np.size(xcmplx_o))
b_cmplx= np.zeros(np.size(xcmplx_o))
dist_simple= np.zeros(np.size(xsimple_o))
b_simple= np.zeros(np.size(xsimple_o))

#each have different sizes, so 4 calls to xtrack function which returns the distance from the line joining waypoints[dist_x] and along the track distance[b_x]
for j in range(0, np.size(x_o)):
	dist[j],b[j]=xtrack(x_o[j],z_o[j],x_plot,z_plot)

for j in range(0, np.size(xcnn_o)):
	dist_cnn[j],b_cnn[j]=xtrack(xcnn_o[j],zcnn_o[j],x_plot,z_plot)
for j in range(0, np.size(xcmplx_o)):
	dist_cmplx[j],b_cmplx[j]=xtrack(xcmplx_o[j],zcmplx_o[j],x_plot,z_plot)
for j in range(0, np.size(xsimple_o)):
	dist_simple[j],b_simple[j]=xtrack(xsimple_o[j],zsimple_o[j],x_plot,z_plot)
#2 lines not use now
max_d= np.amax(dist)
max_d_at= np.argmax(dist)

#plot the four architectures distances from the line joining waypoints
pylab.plot(dist,'-b', label='NVIDIA architecture')
pylab.plot(dist_cnn,'-r', label='CNN architecture')
pylab.plot(dist_cmplx,'-g', label='Complex architecture')
pylab.plot(dist_simple,'-k',label='Simple architecture')
pylab.legend(loc='upper left')
pylab.ylabel('Cross track error')
pylab.xlabel('training data')
pylab.title('Comparision of Cross Track errors for different models')
pylab.show()
plt.plot(b)
plt.show()
#print("max distance :"+ str(max_d)+"near training number:"+str(max_d_at))

