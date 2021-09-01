# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:15:13 2019

@author: pineirosjm
"""

from mpl_toolkits.mplot3d import Axes3D  #needed for 3d plots
import numpy as np 
from matplotlib import pyplot as plt
from scipy import signal
from scipy import stats 
from matplotlib import colors
import matplotlib.pylab as pl
import math
import numpy.ma as ma
import copy
from functools import reduce
from operator import concat
from tempfile import TemporaryFile #to save and load np arrays
from matplotlib.ticker import PercentFormatter

import pandas as pd
from scipy import spatial
from itertools import chain
import collections  #to count occurences of elements in an array

import ctypes
import pickle #to save and load dataframes
#%%

### EDIT THESE PARAMETERS ###
'''load data and define border parameter, unit cell and interatomic distances'''
plt.close("all")
data=np.loadtxt("raw_data_file.txt", skiprows=4,dtype=float)  #load the raw data file containing the 3D coordinates of the crystal surface as a txt file


#border and gradient mark need to be chosen per image, these are test values
border = 1e-14 # test value, usually in the order of 1e-14
gradient_mark= 1e-12 # test value, usually in the order of 1e-12


'''unit_cell_a, scale_atom (x), scale_atom_y, step_height and kink_angle need to be defined per crystal'''
unit_cell_a= "unit cell in [m]" #the parameter should be a number, remove the quotation marks (string) before running the script
scale_atom = "interatomic distance in x [m]"  #the parameter should be a number, remove the quotation marks (string) before running the script
scale_atom_y= "interatomic distance in x [m]"  #the parameter should be a number, remove the quotation marks (string) before running the script
step_height= "step height in [m]" #the parameter should be a number, remove the quotation marks (string) before running the script
print(scale_atom, scale_atom_y,step_height)

kink_angle= 'valid kink angle in degrees' #the parameter should be a number, remove the quotation marks before running the script

temp= 'temperature at which the STM images were taken [K]' #the parameter should be a number, remove quotation marks (string) before running the script

'''load atomic grid'''
atomic_grid_xy = pickle.load( open( "atomic_grid_file.p", "rb" ) )   #loads atomic grid as data frames

### END OF EDITABLE PARAMETERS ###

#%%
### START OF SCRIPT ###
'''to remove first or last rows if needed (in case there are deffects on the edges of the STM image )'''
unique_y=np.unique(data[:,1])
out= [np.append(unique_y[:20],unique_y[-1:]) ]  #values to remove, separated by comas, or section of unique
rem=np.isin(data[:,1],out[0],invert=True)
#data=data[rem]  #this step is needed to remove rows defined in 'out'

#%%
'''Bins defined as atom-atom distance, for Ag(100) is 2.88e-10m'''
#first histogram, before correcting, in Angstroms
bins_i=[]
for i in range(0,500):
    b=(scale_atom/2)/(1e-10) +((scale_atom/2)/(1e-10))*(2*i)
    bins_i.append(b)
#defining bins as a function of interatomic distance in x

#second histogram, after masking errors, in atoms
bins_c=[]
for i in range(0,501):
    b=0.5 +0.5*(2*i)
    bins_c.append(b)
#print(bins_c[0:-1])
print(len(bins_c))



#%%
'''plots font size'''
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 16
BIGGER_SIZE = 20
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#%%
'''FUNCTIONS '''
def array_like(array):
    as_array= np.asarray(array)
    return as_array

Angstrom = 1e-10
def Angstrom_scale(array):
    A_scaled_array= np.array(array)/Angstrom
    return A_scaled_array
    
print(scale_atom )
def array_atom_scaled(array):
    atom_scaled_array= np.array(array)/scale_atom
    return atom_scaled_array

def array_atom_scaled_y(array):  #different scale atom for 'y'
    atom_scaled_array_y= np.array(array)/scale_atom_y
    return atom_scaled_array_y

#%%
'''Separation of terraces based on the gradient'''
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
    
ax1.set_xlabel("X [m]" , labelpad=25)
ax1.set_ylabel("Y [m]" , labelpad=25)
ax1.set_zlabel("Z [m]" , labelpad=20)
#ax.set_title("raw data")
ax1.plot(data[:,0],data[:,1],data[:,2],linestyle="",marker=".",color="r",markersize=1)  #raw data

#sort data, according to z
sortlist = np.argsort(data,axis=0)
data = data[sortlist[:,2]]  #data is sorted in z here

#check if properly sorted acording to z
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title("Data sorted by z")
ax2.plot(range(len(data[:,2])),data[:,2])

#lateral view of terraces (blue plot)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_title("Gradient of z")
ax3.plot(range(len(data[:,2])),np.gradient(data[:,2]))

'''Border'''
# define border, which eventually determines which points are considered as steps, and which are terraces
    
grad = np.gradient(data[:,2])
grad_s = signal.savgol_filter(grad,51,3)

    
ax3.plot(range(len(data[:,2])),grad_s) #orange filtered plot
ax3.plot(np.linspace(0,len(data[:,2]),int(1e3)),border*np.ones(int(1e3)), color="black") #green plot

#green terraces
#adds terrace points on green on top of the raw data plot
terpoint = np.where(grad_s < border)[0]#terpoint is 1darray, containing indices of datapoints that make up the terraces
ax1.plot(data[terpoint,0],data[terpoint,1],data[terpoint,2],linestyle="",marker=".",color="green",markersize=1)


#terraces separated as vectors(columns)
terrace_x = data[terpoint,0]
terrace_y = data[terpoint,1]
terrace_z = data[terpoint,2]
print('x',len(terrace_x))
print('y',len(terrace_y))
print('z',len(terrace_z))


fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.set_title("Lateral view of the terraces")
ax5.plot(terrace_z,"b.",markersize=10)

fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.set_title('grad')
grad=np.append([0],np.gradient(terrace_z))
ax6.plot(grad)


'''transition indexes calculation'''
#find all local maxima in grad
local_max_arg = signal.argrelextrema(grad,np.greater_equal)[0]
ax6.plot(local_max_arg,grad[local_max_arg],"rx")
#use mask to find all maxima above a threshold, to filter noise   
mask = grad[local_max_arg]>gradient_mark
ax6.plot(local_max_arg[mask],grad[local_max_arg][mask],"gx")
ax6.plot(np.linspace(0,len(grad),int(1e3)),gradient_mark*np.ones(int(1e3)), color='black')
#print(grad[local_max_arg])
grad_local_max=(grad[local_max_arg])
grad_local_m_iqr=stats.iqr(grad_local_max,axis=0)
print(gradient_mark)


'''index terraces'''
#array with indices indicating where one terrace ends, and transitions to next
index_nextstep = local_max_arg[mask]
#print(index_nextstep)
index_diff=np.append([0],np.diff(index_nextstep))
#print(index_diff)
indices_valid_terraces=np.logical_and(index_diff>=1,index_diff<=100) #discard indexes that are too close (not real start of a terrace)
#print(indices_valid_terraces)
index_nextstep=index_nextstep[~indices_valid_terraces]
print(index_nextstep)


'''separation of steps and terraces'''
lengths = {}
step_edge_max={}
step_edge_min={}
step_edge_min_y={}
step_edge_max_y={}
step_edge_min_z={}
step_edge_max_z={}
terrace_x_s=[] #used to plot terraces in cyan
terrace_y_s=[]
terrace_z_s=[]


#in this loop, terace_(x,y,z)_i contains the (x,y,z) coordinate of terrace i
for s in range(len(index_nextstep)):
    if s == 0:
        startind = 0
    else:
        startind = index_nextstep[s-1]
    
    terrace_x_i = terrace_x[startind:index_nextstep[s]]
    terrace_y_i = terrace_y[startind:index_nextstep[s]]
    terrace_z_i = terrace_z[startind:index_nextstep[s]]

    #removing outliers on z
    q25_terrace_z_i=np.percentile(terrace_z_i, 25, axis=0)
    q75_terrace_z_i=np.percentile(terrace_z_i, 75, axis=0)
    iqr_terrace_z_i=stats.iqr(terrace_z_i,axis=0)
    pos=q75_terrace_z_i + 1.25*iqr_terrace_z_i
    neg=q25_terrace_z_i - 1.25*iqr_terrace_z_i
    mask2=np.logical_and(terrace_z_i > neg, terrace_z_i< pos)
    #I need to apply the mask to the entire array, not just z column, so I make the array terraces_i
    terraces_i=np.array((terrace_x_i,terrace_y_i,terrace_z_i)).T
    terraces_i=terraces_i[mask2]
    terrace_x_i=terraces_i[:,0]
    terrace_y_i=terraces_i[:,1]
    terrace_z_i=terraces_i[:,2]
    ax5.plot(terrace_z_i,markersize=3,linestyle="",marker=".")
 
    #sort according to y:
    sortlist = np.argsort(terrace_y_i)
    terrace_x_i = terrace_x_i[sortlist]
    terrace_y_i = terrace_y_i[sortlist]
    terrace_z_i = terrace_z_i[sortlist]
    terrace_x_s.append(terrace_x_i)
    terrace_y_s.append(terrace_y_i)
    terrace_z_s.append(terrace_z_i)
  
    #for each terrace, we need to find max(x) and min(x) for constant y value. 
    #this gives an approximation of terrace length but doesn't take into account the convolution of the tip on the step edges
    ind_split = signal.argrelextrema(np.diff(terrace_y_i),np.greater)[0]
    #print('ind_split', ind_split)    
       
    
    list_of_lengths_at_each_y = []
    list_max_x_per_y=[]
    list_min_x_per_y=[]
    list_min_y_step_edge=[]
    list_max_y_step_edge=[]
    list_min_z_step_edge=[]
    list_max_z_step_edge=[]
    
    for j in range(len(ind_split)):
        if j == 0:
            start = 0
        else:
            start = ind_split[j-1]
                
        max_x_at_set_y = np.max(terrace_x_i[start:ind_split[j]])
        min_x_at_set_y = np.min(terrace_x_i[start:ind_split[j]])
        argument_min_x_at_set_y = np.argmin(terrace_x_i[start:ind_split[j]])
        argument_max_x_at_set_y = np.argmax(terrace_x_i[start:ind_split[j]])
        ymin_s_e = terrace_y_i[argument_min_x_at_set_y+start]
        ymax_s_e = terrace_y_i[argument_max_x_at_set_y+start]
        zmin_s_e = terrace_z_i[argument_min_x_at_set_y+start]
        zmax_s_e = terrace_z_i[argument_max_x_at_set_y+start]
        #correct deffinition of TW should be from max to max or from min to min, not from max to min
        length_at_y = max_x_at_set_y - min_x_at_set_y 
      
        
        list_of_lengths_at_each_y.append(length_at_y)
        list_max_x_per_y.append(max_x_at_set_y)
        list_min_x_per_y.append(min_x_at_set_y)
        list_min_y_step_edge.append(ymin_s_e)
        list_max_y_step_edge.append(ymax_s_e)
        list_min_z_step_edge.append(zmin_s_e)
        list_max_z_step_edge.append(zmax_s_e)
        
   
        lengths[str(s)] = list_of_lengths_at_each_y
        step_edge_max[str(s)]= list_max_x_per_y
        step_edge_min[str(s)]= list_min_x_per_y
        step_edge_min_y[str(s)]=list_min_y_step_edge
        step_edge_max_y[str(s)]=list_max_y_step_edge
        step_edge_min_z[str(s)]=list_min_z_step_edge
        step_edge_max_z[str(s)]=list_max_z_step_edge
        
#to plot terrace by terrace change the number
#ax1.plot(terrace_x_s[5],terrace_y_s[5],terrace_z_s[5],linestyle="-",marker=".",color="darkviolet",markersize=1)  
        


#length of all the terraces

order_terraces_0 = [v for v in lengths.keys()] #gives the 'name' of each terrace
print(order_terraces_0) 
length_terraces_0 = [len(v) for v in lengths.values()] #gives the lengths of each terrace
print(length_terraces_0)   
max_len_list_lengths=np.amax(length_terraces_0)
print(max_len_list_lengths)
#this finds terraces with length  smaller than  (max terrace length -50)
invalid_terraces_lengths=np.asarray(np.where( length_terraces_0<(max_len_list_lengths-50)))
print((invalid_terraces_lengths)) #these are the terraces I want to remove
ind_inv_t=np.reshape(invalid_terraces_lengths.tolist(), (len(invalid_terraces_lengths[0]),)) 
print(ind_inv_t)
print(np.shape(ind_inv_t))

length_step_edge_max = [len(v) for v in step_edge_max.values()] #gives the lengths of each step_edge_max
print(length_step_edge_max)

length_step_edge_min = [len(v) for v in step_edge_min.values()] #gives the lengths of each step_edge_min
print(length_step_edge_min)

'''this erases invalid(50 elements or shorter than max len) from all the lists'''
for k in ind_inv_t:
    del lengths[str(k)] 
    del step_edge_max[str(k)]
    del step_edge_min[str(k)]
    del step_edge_min_y[str(k)]
    del step_edge_max_y[str(k)]
    del step_edge_min_z[str(k)]
    del step_edge_max_z[str(k)]


order_terraces = [v for v in lengths.keys()] # erases first and last elements after removing incomplete terraces
print(order_terraces) 
length_terraces = [len(v) for v in lengths.values()] #gives the lengths of each terrace
print(length_terraces) 
first_and_last=[order_terraces[0],order_terraces[-1]]  #this erases first and last terraces, which are usually invalid
print(first_and_last)
for k in first_and_last:
    del lengths[str(k)] 
    del step_edge_max[str(k)]
    del step_edge_min[str(k)]
    del step_edge_min_y[str(k)]
    del step_edge_max_y[str(k)]
    del step_edge_min_z[str(k)]    
    del step_edge_max_z[str(k)]#
    
number_terraces = [v for v in lengths.keys()] #need to redefine it here so it erases first and last elements after removing incomplete terraces
print(number_terraces) 
    

# concatenate dictionary values
terrazas_new_method=reduce(concat, lengths.values())

'''IMPORTANT: gives information about the orientation of the crystal: stepping down to the left or right'''
condition_side=np.subtract(np.max(step_edge_max[number_terraces[-1]]),np.max(step_edge_max[number_terraces[0]]))
print(condition_side)
#%%
'''trying to find border line automatically'''
##
print(np.min(grad_s))
print(border)   
print(np.mean(grad_s))
print(np.max(grad_s))

grad_s_25=np.percentile(grad_s, 25, axis=0)
grad_s_75=np.percentile(grad_s, 75, axis=0) 
print('25', grad_s_25, '75', grad_s_75)  #border line close to 3rd quartile (np.percentile 75) for some images, this maybe a way to define the border line automatically

'''trying to find gradient_marke automatically'''
print(np.max(grad_local_max)-0.5*np.max(grad_local_max)) #this might work to define gradient_mark automatically
grad_l_25=np.percentile(grad_local_max, 25, axis=0)
grad_l_75=np.percentile(grad_local_max, 90, axis=0)
print('25', grad_l_25, '75', grad_l_75)
print(grad_local_m_iqr+1000*grad_l_75)  #this might work for the  gradient mark, check other crystals
##
#%%

'''checking that the terraces are actually properly plotted'''
#print('lon ', len(terrace_x_s))
for item in range(len(terrace_x_s)):  #to plot on the original plot
    ax1.plot(terrace_x_s[item] ,terrace_y_s[item] ,terrace_z_s[item],linestyle="",marker=".",color="darkturquoise",markersize=1)
#to plot terrace by terrace change the number
#ax.plot(terrace_x_s[11],terrace_y_s[11],terrace_z_s[11],linestyle="-",marker=".",color="darkviolet",markersize=1)    


fig8=plt.figure()
ax8=fig8.add_subplot(111)
ax8.set_xlabel("X")
ax8.set_ylabel("Y")
ax8.set_title("Step edges [m]")
for key in step_edge_min.keys():
    if condition_side>0:
        ax8.plot(step_edge_min[key] ,step_edge_min_y[key],linestyle="-",marker=".",markersize=1)
        ax8.plot(step_edge_max[key] ,step_edge_max_y[key],linestyle="-",color='gray',marker=".",markersize=1)
    else:
        ax8.plot(step_edge_min[key] ,step_edge_min_y[key],linestyle="-",marker=".",markersize=1,color='gray')
        ax8.plot(step_edge_max[key] ,step_edge_max_y[key],linestyle="-",marker=".",markersize=1)


'''histograms with approximate but not precise definition of TW'''
terraces_A_list=[]

for item in terrazas_new_method:
    terraces_A=Angstrom_scale(item)
    terraces_A_list.append(terraces_A)

    

fig9 = plt.figure()
ax9 = fig9.add_subplot(111) 
ax9.set_xlim(-20,np.max(terraces_A_list)+5)
ax9.hist(terraces_A_list, bins_i, density=False, facecolor='green', alpha=0.5)
ax9.set_xlabel("Terrace width ($\AA$) aprox")
ax9.set_ylabel("Counts")

print(np.max(terraces_A_list))


#to plot stepedges on purple on the 3D plot
for key in step_edge_min.keys():  
    if condition_side>0:
        ax1.plot(step_edge_min[key] ,step_edge_min_y[key] ,step_edge_min_z[key],linestyle="",marker=".",color="darkviolet",markersize=1)
    else:
        ax1.plot(step_edge_max[key] ,step_edge_max_y[key] ,step_edge_max_z[key],linestyle="",marker=".",color="darkviolet",markersize=1)
    
#%%
    
'''minimizing step height, for Ag(100) should be around 2.04265e-10'''
step_edge_max_z_as_arrays = [np.array(v) for v in step_edge_max_z.values()] #makes step edges from dictionaries as arrays into a list
print(np.shape(step_edge_max_z_as_arrays[0]))
max_z_list=step_edge_max_z_as_arrays

max_z_same_length=[]
length_x=[]
max_z_list_same_length=[]

#here I calculate the length of every sted edge(max_z)      
for i in range(len(max_z_list)):
    lengths_max_z=len(max_z_list[i])
    length_x.append(lengths_max_z)
    #print(len(length_x))
    
#here I calculate the max length of step edges and calculate the index (n=max_len-len(array)) 
#of every terrace with respect to max_len. Then I repeat the last value of the array "n"times, so all the arrays have the same length   
#usually just a couple op values but needed if I want to subtract arrays of different steps
for k in range(len(max_z_list)): 
    max_len=np.amax(length_x)
    #print(max_len)
    max_len_a=np.tile(max_len,len(max_z_list))
    n_lengths=np.subtract(max_len_a,length_x)
    #print(n_lengths)
    #print(len(n_lengths))    
    extension=[max_z_list[k][-1]]*n_lengths[k]
    same_length=np.append(max_z_list[k],extension)
    max_z_list_same_length.append(same_length)
       
print(len(max_z_list_same_length))

step_edge_min_z_as_arrays = [np.array(v) for v in step_edge_min_z.values()] ##makes step edges from dictionaries as arrays into a list
print(len(step_edge_min_z_as_arrays))
print(np.shape(step_edge_min_z_as_arrays))
min_z_list=step_edge_min_z_as_arrays
print(len(step_edge_min_z_as_arrays))

min_z_same_length=[]
length_m=[]
min_z_list_same_length=[]

#here I calculate the length of every sted edge(min_z)  
print(len(min_z_list))
for j in range(len(min_z_list)):
    lengths_min_z=len(min_z_list[j])
    length_m.append(lengths_min_z)
    #print(len(length_m)) 

#here I calculate the max length of step edges and calculate the index (n=max_len-len(array)) 
#of every terrace with respect to max_len. Then I repeat the last value of the array "n"times 
for h in range(len(min_z_list)):    
    min_len=np.amax(length_m)
    #print(min_len)
    min_len_a=np.tile(min_len,len(min_z_list))
    n_lengths_m=np.subtract(min_len_a,length_m)
    #print(len(n_lengths_m))
    extension_m=[min_z_list[h][-1]]*n_lengths_m[h]
    same_length_m=np.append(min_z_list[h],extension_m)
    min_z_list_same_length.append(same_length_m)



'''step height is calculated as the difference between z values from the upper and the lower par of each step'''
#step height is calculated as min1 -max0...until min_n  - max(n-1) : for right to left crystal slope,
# step height is calculated as max1 -min0...until max_n  - min(n-1) : for left to right crystal slope'''


#I need one element less is this list, because step heighs has n-1 terraces
step_height_list=[]
for items in range(len(min_z_list_same_length)-1):
    if condition_side>0:
        step_z=np.subtract(min_z_list_same_length[items+1],max_z_list_same_length[items]) 
        step_height_list.append(step_z)
    else:
        step_z=np.subtract(max_z_list_same_length[items+1],min_z_list_same_length[items]) 
        step_height_list.append(step_z)
    #print(len(step_height_list))


mean_steps_list=[]
std_steps_list=[]
scale_list=[]

for item in range(len(step_height_list)):
    mean_steps=np.mean(step_height_list[item])
    mean_steps_list.append(mean_steps)
    std_steps=np.std(step_height_list[item])
    std_steps_list.append(std_steps)
    scale_list=np.tile(step_height,len(step_height_list))
    
s_h_m_diff=np.subtract(scale_list,mean_steps_list)
min_diff_s_h_m=np.argmin(abs(s_h_m_diff)) #best step (mean)
print(min_diff_s_h_m)
min_s_h_std=np.argmin(std_steps_list) #best step (std)
print(min_s_h_std)
mean_step_height=np.mean(mean_steps_list)
std_step_height=np.mean(std_steps_list) 
  
print('mean_steps', mean_steps_list)
print('std_steps',std_steps_list)
print('best_step_mean',min_diff_s_h_m)
print('best_step_std',min_s_h_std)
print('mean_step_height',mean_step_height)
print('std_step_height',std_step_height)

print('step height ', step_height)
print(np.abs(step_height-mean_step_height))
print('percentage', np.abs(step_height-mean_step_height)*(100/step_height))

if np.isclose(mean_step_height ,step_height,  atol=((20/100)*step_height)):
    print('APPROPRIATE BORDER LINE')
    ctypes.windll.user32.MessageBoxW(0, "APPROPRIATE BORDER LINE", "Border line", 1)
else:
    #plt.close("all")
    print('CHECK BORDER LINE')
    ctypes.windll.user32.MessageBoxW(0, "CHECK BORDER LINE", "Border line", 1)  #click if accept to continue or close to redefine border value, 
                                                                                #if close stop (red box) on console


#%%
'''STEP EDGES as arrays and scaled in atomic units'''
#make dictionary elements as arrays in a list instead of a dictionary
if condition_side>0:
    step_edge_min_x_as_arrays = [np.array(v) for v in step_edge_min.values()]
    min_x_as_list=step_edge_min_x_as_arrays
else: 
    step_edge_min_x_as_arrays = [np.array(v) for v in step_edge_max.values()]  #keep 'min'name but for other side (R to L)crystal is max
    min_x_as_list=step_edge_min_x_as_arrays
#print(np.unique(np.subtract(min_x_as_list[0],step_edge_min['1'])))

min_x_same_length=[]
length_x_min=[]
min_x_list_same_length=[]

print(len(min_x_as_list))
for j in range(len(min_x_as_list)):
    lengths_min_x=len(min_x_as_list[j])
    length_x_min.append(lengths_min_x)
    
for h in range(len(min_x_as_list)):    
    min_len_x=np.amax(length_x_min)
    min_len_x_a=np.tile(min_len_x,len(min_x_as_list))
    n_lengths_m_x=np.subtract(min_len_x_a,length_x_min)
    extension_m_x=[min_x_as_list[h][-1]]*n_lengths_m_x[h]  #copy last element at the end
    same_length_m_x=np.append(min_x_as_list[h],extension_m_x)
    min_x_list_same_length.append((same_length_m_x))


'''mask atoms further apart than 2.3 atomic distances'''
list_copy_step_min_x_scaled=[]
list_copy_step_min_x_scaled_diff=[]
list_copy_masked_min_x=[]

for l in  min_x_list_same_length:
        sc_a=array_atom_scaled(l)
        list_copy_step_min_x_scaled.append(sc_a)
        diff_s_e_m_x=np.append([0],np.diff(sc_a))
        list_copy_step_min_x_scaled_diff.append(diff_s_e_m_x)
        masked_s_e_m_x=ma.masked_where(abs( diff_s_e_m_x) >= 2.2, sc_a) 
        list_copy_masked_min_x.append(masked_s_e_m_x)

print(len(sc_a))
print(len(diff_s_e_m_x))
print(len(list_copy_masked_min_x))


'''in Y'''
#makes step edges from dictionaries as arrays into a list
if condition_side>0:
    step_edge_min_y_as_arrays = [np.array(v) for v in step_edge_min_y.values()]
    min_y_as_list=step_edge_min_y_as_arrays
else:
    step_edge_min_y_as_arrays = [np.array(v) for v in step_edge_max_y.values()] #keep 'min'name but for other side (R to L)crystal is max
    min_y_as_list=step_edge_min_y_as_arrays
#print(np.unique(np.subtract(min_y_as_list[0],step_edge_min_y['1'])))
print(len(min_y_as_list))


min_y_same_length=[]
length_y_min=[]
min_y_list_same_length=[]


for j in range(len(min_y_as_list)):
    lengths_min_y=len(min_y_as_list[j])
    length_y_min.append(lengths_min_y)   
for h in range(len(min_y_as_list)):    
    min_len_y=np.amax(length_y_min)
    min_len_y_a=np.tile(min_len_y,len(min_y_as_list))
    n_lengths_m_y=np.subtract(min_len_y_a,length_y_min)
    extension_m_y=[min_y_as_list[h][-1]]*n_lengths_m_y[h]
    same_length_m_y=np.append(min_y_as_list[h],extension_m_y)
    min_y_list_same_length.append(same_length_m_y)


list_copy_step_min_y_scaled=[]  

for g in min_y_list_same_length:
    sc_a_y=array_atom_scaled_y(g)
    list_copy_step_min_y_scaled.append(sc_a_y)
    
print(np.shape(min_y_as_list[1]))
print(np.shape(list_copy_step_min_y_scaled[1]))
print(len(list_copy_step_min_y_scaled))

list_copy_step_min_y_scaled=[]  

for g in min_y_list_same_length:
    sc_a_y=array_atom_scaled_y(g)
    list_copy_step_min_y_scaled.append(sc_a_y)


fig18=plt.figure()
ax18=fig18.add_subplot(111)
ax18.set_xlabel("X")
ax18.set_ylabel("Y")
ax18.set_title("Step edges masked [atomic units]")
ax18.grid(True)
ax18.set_ylim(-1,np.max(list_copy_step_min_y_scaled)+5)
ax18.set_xlim(0,np.max(list_copy_step_min_x_scaled)+5)
for items in range(len(list_copy_masked_min_x)):
    ax18.plot(list_copy_step_min_x_scaled[items] ,list_copy_step_min_y_scaled[items],linestyle="-",marker="o",color="darkviolet",markersize=2)
    ax18.plot(list_copy_masked_min_x[items],list_copy_step_min_y_scaled[items], linestyle="-",
              marker=">", markersize=4)

#%%
    '''correction of the drift'''

first=0
last=-1
sin_list=[]
arctan_list=[]
arctan_sign_condition=[]


for l in range(len(list_copy_masked_min_x)):
    for m in range(len(list_copy_masked_min_x)): #I need a double loop here because I put break
        if np.ma.is_masked(list_copy_masked_min_x[m][first]) == True:
            first = first + 1
        if np.ma.is_masked(list_copy_masked_min_x[m][last]) == True:
            last = last - 1
        else:
            break
    diff_x=np.subtract(np.mean(list_copy_masked_min_x[l][(last-51):last]),np.mean(list_copy_masked_min_x[l][first:(first+50)]))
    diff_y=np.subtract(np.mean(list_copy_step_min_y_scaled[l][(last-51):last]),np.mean(list_copy_step_min_y_scaled[l][first:(first+50)]))
    arctan_y_x=np.arctan(diff_y/diff_x)
    sin_v=math.sin((arctan_y_x))
    sin_list.append(sin_v)
    arctan_list.append(arctan_y_x)
    pos_arctan= (np.array(arctan_list) > 0)
    neg_arctan= (np.array(arctan_list) < 0)
    pos_count=(pos_arctan).sum()
    neg_count=(neg_arctan).sum()
    if np.logical_or(len(arctan_list)==neg_count, len(arctan_list)==pos_count):
        arctan_sign_condition= True
    else:
        arctan_sign_condition= False


print(arctan_sign_condition)  #tells if all the step edges go to the same direction (right or left)
print((arctan_list))
#print(math.tan(arctan_list[6]))
print(np.all(arctan_sign_condition))#gives boolean value of the sign condition

#mean value of the angles
arctan_mean_v=np.nanmean((np.array(arctan_list)) )   
print(arctan_mean_v)
print(math.cos(arctan_mean_v))
print(np.degrees(arctan_mean_v)) #value of the angles in degrees
#print(distw*np.cos(85*np.pi/180))
print(np.tan(70*np.pi/180))
print(np.tan(87*np.pi/180))

'''rotation of the step edge -  (drift correction in x)'''
#first create a 2D vector of all the coordinates
vectors_x_y_s_e=np.stack((list_copy_masked_min_x, list_copy_step_min_y_scaled), axis=-1)  #this works
new_x_list=[]
x_s_e_rotated_list=[]
list_x_rot_diff=[]

if np.logical_and(arctan_sign_condition==True, 2.74<abs(math.tan(arctan_mean_v))<19.1): #to limit the angle between cos(87) and cos(70)
#define a new reference system for each step edge, subtract the drift (defined as a function of cosine ot arctan_mean_v)
    for b in range(len(list_copy_masked_min_x)):
        x_s_e_new_ref= abs(vectors_x_y_s_e[b][:,0] - vectors_x_y_s_e[b][0,0] )
        new_x_list.append(x_s_e_new_ref)
        x_s_e_drift= np.sqrt(np.square(new_x_list[b])+np.square(vectors_x_y_s_e[b][:,1]))*math.cos(arctan_mean_v)
        if arctan_mean_v>0: #or arctan_mean_v arctan_list[b]>0
            x_s_e_rotated=np.subtract(vectors_x_y_s_e[b][:,0], x_s_e_drift ) #positive slope , I subtract the drift
            diff_x_rot=np.append([0],np.diff(x_s_e_rotated))
            list_x_rot_diff.append( diff_x_rot)
            masked_x_rot=ma.masked_where(abs(diff_x_rot) >= 10,x_s_e_rotated) 
            diff_x_rot_2=np.append([0],np.diff(masked_x_rot))
            mask_x_rot_2=ma.masked_where(abs(diff_x_rot_2) >= 5,masked_x_rot) 
            diff_x_rot_3=np.append([0],np.diff(mask_x_rot_2))
            mask_x_rot_3=ma.masked_where(abs(diff_x_rot_3) >= 4,mask_x_rot_2) #to remove consecutive outliers
            diff_x_rot_4=np.append([0],np.diff(mask_x_rot_3))
            mask_x_rot_4=ma.masked_where(abs(diff_x_rot_4) >= 3,mask_x_rot_3) 
            x_s_e_rotated_list.append(mask_x_rot_4)
        else:
            x_s_e_rotated=np.sum((vectors_x_y_s_e[b][:,0], x_s_e_drift ), axis=0)  #negative slope, I add the drift
            diff_x_rot=np.append([0],np.diff(x_s_e_rotated))
            list_x_rot_diff.append( diff_x_rot)
            masked_x_rot=ma.masked_where(abs(diff_x_rot) >= 10,x_s_e_rotated) 
            diff_x_rot_2=np.append([0],np.diff(masked_x_rot))
            mask_x_rot_2=ma.masked_where(abs(diff_x_rot_2) >= 5,masked_x_rot) 
            diff_x_rot_3=np.append([0],np.diff(mask_x_rot_2))
            mask_x_rot_3=ma.masked_where(abs(diff_x_rot_3) >= 4,mask_x_rot_2) #to remove consecutive outliers
            diff_x_rot_4=np.append([0],np.diff(mask_x_rot_3))
            mask_x_rot_4=ma.masked_where(abs(diff_x_rot_4) >= 3,mask_x_rot_3) 
            x_s_e_rotated_list.append(mask_x_rot_4)

#mask applied here to show the procedure of removing outliers, mask procedure repeated for df


fig24=plt.figure()
ax24=fig24.add_subplot(111)
ax24.set_xlabel("X")
ax24.set_ylabel("Y")
ax24.set_title("Step edges rotated")
ax24.grid(True)
for items in range(0,len(x_s_e_rotated_list)):
    #ax24.plot(list_copy_step_min_x_scaled[items] ,list_copy_step_min_y_scaled[items],linestyle="-",marker="o",color="darkviolet",markersize=1)
    ax24.plot(x_s_e_rotated_list[items],list_copy_step_min_y_scaled[items], linestyle="-",marker=">", markersize=2)

print(np.shape(x_s_e_rotated_list))
print(np.shape(list_copy_masked_min_x))

#checking the drift correction
first2=0
last2=-1
arctan_list2=[]

#I need a double loop here because I put break
for l in range(len(x_s_e_rotated_list)):
    for m in range(len(x_s_e_rotated_list)): 
        if np.ma.is_masked(x_s_e_rotated_list[m][first2]) == True:
            first2 = first2 + 1
        if np.ma.is_masked(x_s_e_rotated_list[m][last2]) == True:
            last2 = last2 - 1
        else:
            break
    diff_x2=np.subtract(np.mean(x_s_e_rotated_list[l][(last2-51):last2]),np.mean(x_s_e_rotated_list[l][first2:(first2+50)]))
    diff_y2=np.subtract(np.mean(list_copy_step_min_y_scaled[l][(last2-51):last2]),np.mean(list_copy_step_min_y_scaled[l][first2:(first2+50)]))
    arctan_y_x2=np.arctan(diff_y2/diff_x2)
    arctan_list2.append(arctan_y_x2)

print(arctan_list2)
print(np.mean(arctan_list2), 'close enough to zero? ')

#%%
'''correct TW definition, from min to min, including thermal drift correction'''

print(len(list_copy_masked_min_x))
subtraction_TWD_list=[]

for r in range(len(list_copy_masked_min_x)-1):
    if np.logical_and(arctan_sign_condition==True, 2.74<abs(math.tan(arctan_mean_v))<19.1):
        subtraction_TWD= np.absolute(np.subtract(x_s_e_rotated_list[(r+1)] ,x_s_e_rotated_list[r]))
        subtraction_TWD_list.append(subtraction_TWD)
    else:
        subtraction_TWD= np.absolute(np.subtract(list_copy_masked_min_x[(r+1)] ,list_copy_masked_min_x[r]))
        subtraction_TWD_list.append(subtraction_TWD)

concatenation_TWD=np.concatenate(subtraction_TWD_list, axis=0)  #after correcting for the drift

fig20=plt.figure()
ax20=fig20.add_subplot(111)
ax20.set_xlabel("Terrace width [atomic units]")
ax20.set_ylabel("Counts")
ax20.set_title("TWD corrected")
ax20.set_xlim(0,np.max(concatenation_TWD)+5)
ax20.hist(concatenation_TWD, bins_c,density=False, facecolor='blue', alpha=0.5)


'''sigma vs mean for all the terraces''' 
mean_TW_atoms_list=[]
for v in subtraction_TWD_list:
    mean_TW=np.mean(v)
    mean_TW_atoms_list.append(mean_TW)
    
print(mean_TW_atoms_list)
mean_TW_atoms=np.reshape(mean_TW_atoms_list,(len(mean_TW_atoms_list),))
print(mean_TW_atoms)
average_mean_TW_atoms=np.mean(mean_TW_atoms)
print(average_mean_TW_atoms)

std_TW_atoms_list=[]
for v in subtraction_TWD_list:
    std_TW=np.std(v)
    std_TW_atoms_list.append(std_TW)

print(std_TW_atoms_list)
std_TW_atoms=np.reshape(std_TW_atoms_list,(len(std_TW_atoms_list),))
average_std_TW_atoms=np.mean(std_TW_atoms_list)
print(std_TW_atoms)
print(average_std_TW_atoms)
    
mean_TW_atoms_sortlist= np.argsort(mean_TW_atoms,axis=0)
print(mean_TW_atoms_sortlist)
mean_TW_atoms_sorted_ascendent = mean_TW_atoms[mean_TW_atoms_sortlist]
print(mean_TW_atoms_sorted_ascendent)
mean_TW_A=(scale_atom/1e-10)*mean_TW_atoms_sorted_ascendent
print(mean_TW_A)
mean_2=np.mean(mean_TW_A) #mean of all the terraces, one by one in [A]
print(' TWmean :', mean_2, '[A]')

std_TW_atoms_sortlist= np.argsort(std_TW_atoms_list,axis=0)
std_TW_atoms_sorted_ascendent = std_TW_atoms[std_TW_atoms_sortlist]
print(std_TW_atoms_sorted_ascendent)
std_TW_A= (scale_atom/1e-10)*std_TW_atoms_sorted_ascendent
print(std_TW_A)
std_2=np.mean(std_TW_A) # mean std for all the terraces, one by one in [A]

mean_1=(np.mean(concatenation_TWD))*(scale_atom/1e-10) #average and std for the entire image (concatenated terraces)
std_1=(np.std(concatenation_TWD))*(scale_atom/1e-10)
print('mean entire histogram',mean_1)
print('std entire histogram',std_1)

'''calculate error bars'''
#The standard error is calculated by dividing the standard deviation by the 
#square root of number of measurements that make up the mean
s_e_mean_TW_A=np.std(mean_TW_A)/(np.sqrt(len(mean_TW_A)))
s_e_std_TW_A=np.std(std_TW_A)/(np.sqrt(len(std_TW_A)))
#black point, for the entire image
std_e_i=np.std(concatenation_TWD)*(scale_atom/1e-10)
mean_e_i=np.mean(concatenation_TWD)*(scale_atom/1e-10)
#Angstrom ($\AA$)
fig22=plt.figure()
ax22=fig22.add_subplot(111)
#ax22.set_xlabel("Mean terrace width (Atoms)")
ax22.set_xlabel("Mean terrace width ($\AA$)")
#ax22.set_ylabel("$\sigma$ (Atoms)")
ax22.set_ylabel("$\sigma$ ($\AA$)")
#ax22.set_title("$\sigma$ vs mean")
#sigma vs mean in atomic distance units
#ax22.plot(mean_TW_atoms_sorted_ascendent,std_TW_atoms_sorted_ascendent,linestyle="",marker="o",color='blue',markersize=2)
'''sigma vs mean in A'''
ax22.plot(mean_TW_A,std_TW_A, linestyle="",marker="o",color='blue',markersize=7)
ax22.plot(mean_2,std_2, linestyle="",marker="o",color='red',markersize=10)
ax22.plot(mean_2,std_e_i, linestyle="",marker="^",color='black',markersize=7)  #value calculated for the entire image

#ax22.errorbar(mean_2,std_2, yerr=s_e_std_TW_A, xerr=s_e_mean_TW_A, color='r', fmt='--o')# fmt='none'
print('mean av [A]',mean_2,'std av [A]' ,std_2)


#%%
'''displacement of odd step edges by a fractional atomic value before comparing to the atomic grid'''
#this is the case for a crystal that has a displacement of half an atom 

x_s_e_rotated_list_copy=copy.deepcopy(x_s_e_rotated_list)
list_copy_masked_min_x_copy=copy.deepcopy(list_copy_masked_min_x)   #deep copy to  not alter the original file

if np.logical_and(arctan_sign_condition==True, 2.74<abs(math.tan(arctan_mean_v))<14.3):
    for i in range(len(list_copy_masked_min_x_copy)):
        for m in range(len(x_s_e_rotated_list_copy[i])):
            if (i%2 == 0):                                      #even stays the same
                x_s_e_rotated_list_copy[i]=x_s_e_rotated_list_copy[i]
            else:                                               #odd subtract 0.5 to each element
                x_s_e_rotated_list_copy[i][m]=x_s_e_rotated_list_copy[i][m] -0.5
else:
    for i in range(len(list_copy_masked_min_x_copy)):
        for m in range(len(list_copy_masked_min_x_copy[i])):
            if (i%2 == 0):
                list_copy_masked_min_x_copy[i]=list_copy_masked_min_x_copy[i]
            else:
                list_copy_masked_min_x_copy[i][m]=list_copy_masked_min_x_copy[i][m] -0.5


#%%
'''finding limits of step edges in x and y'''

# minimum and maximum of step edges in x
minimum_x_s_e=[]
maximum_x_s_e=[]
minimum_x_s_e_extended=[]  #safe range from minimum (-2)
maximum_x_s_e_extended=[]  #safe range from minimum (+0-p2)

if np.logical_and(arctan_sign_condition==True, 2.74<abs(math.tan(arctan_mean_v))<14.3):
    for k in x_s_e_rotated_list_copy:
        min_x_s_e=np.amin(k)
        minimum_x_s_e_i=(min_x_s_e-0.5)
        minimum_x_s_e.append(min_x_s_e)
        max_x_s_e=np.amax(k)
        maximum_x_s_e_i=(max_x_s_e+0.5)
        maximum_x_s_e.append(max_x_s_e)
        minimum_x_s_e_extended.append( minimum_x_s_e_i)
        maximum_x_s_e_extended.append(maximum_x_s_e_i)
else:
    for k in list_copy_masked_min_x_copy:
        min_x_s_e=np.amin(k)
        minimum_x_s_e_i=(min_x_s_e-0.5)
        minimum_x_s_e.append(min_x_s_e)
        max_x_s_e=np.amax(k)
        maximum_x_s_e_i=(max_x_s_e+0.5)
        maximum_x_s_e.append(max_x_s_e)
        minimum_x_s_e_extended.append( minimum_x_s_e_i)
        maximum_x_s_e_extended.append(maximum_x_s_e_i)
    
# minimum and maximum of step edges in y
minimum_y_s_e=[]
maximum_y_s_e=[]
minimum_y_s_e_extended=[]  #safe range from minimum (-2)
maximum_y_s_e_extended=[]  #safe range from minimum (+0-p2)
for k in list_copy_step_min_y_scaled:
    min_y_s_e=np.amin(k)
    minimum_y_s_e_i=(min_y_s_e-1)
    minimum_y_s_e.append(min_y_s_e)
    max_y_s_e=np.amax(k)
    maximum_y_s_e_i=(max_y_s_e+1)
    maximum_y_s_e.append(max_y_s_e)
    minimum_y_s_e_extended.append( minimum_y_s_e_i)
    maximum_y_s_e_extended.append(maximum_y_s_e_i) 
    
print(minimum_x_s_e)
print(maximum_x_s_e)
print(maximum_y_s_e_extended)

print(condition_side)
if condition_side>0:
    s_e_limits_x=np.dstack((minimum_x_s_e_extended,maximum_x_s_e_extended)).flatten() #to keep intervals eventhough there is overlap
    s_e_limits_y=np.sort(np.append(minimum_y_s_e_extended,maximum_y_s_e_extended))
else:
    s_e_limits_x=np.flip((np.dstack((maximum_x_s_e_extended,minimum_x_s_e_extended)).flatten()),0) #to keep intervals eventhough there is overlap, flip used for dataframes
    s_e_limits_y=-np.sort(-np.append(minimum_y_s_e_extended,maximum_y_s_e_extended))
#dataframe require increasing bins,use condition side to compare step edges

#since ymin and ymax is more or less the same for all the step edges, better find one min and one max and limit the 
#atomic grid using just two values    
min_y_limit=np.min(s_e_limits_y)
max_y_limit=np.max(s_e_limits_y)
s_e_bins_y=np.append(min_y_limit,max_y_limit)
print(s_e_bins_y)


#%%
'''comparison of the atomic grid dataframe in intervals with step edges'''

#atomic grid as dataframe

sq_g_x_y_sorted=atomic_grid_xy.sort_values(by=[0,1]) #  this sort by x and then by y (within the same x)

sq_g_l=sq_g_x_y_sorted[sq_g_x_y_sorted[1].between(s_e_bins_y[0],s_e_bins_y[1] )]   #dataframe with y values between limits
print(np.shape(sq_g_l))

bins_s_e_limits=np.round(s_e_limits_x)
print(bins_s_e_limits)
print(s_e_limits_x)


if np.logical_and(arctan_sign_condition==True, 2.74<abs(math.tan(arctan_mean_v))<14.3):
    list_subarrays=[[[]] for i in range(len(x_s_e_rotated_list_copy))] #this gives an extra empty list, not needed
else:
    list_subarrays=[[[]] for i in range(len(list_copy_masked_min_x_copy))]
    
for i in range(len(list_copy_masked_min_x_copy)):
     sq_g_splits= sq_g_l[ sq_g_l[0].between(bins_s_e_limits[2*i],bins_s_e_limits[2*i+1])] #I need odd intervals of s_e_limits_x, to have points around the step edges but not mainly on the terraces
     list_subarrays[i].append(sq_g_splits)

#(list_subarrays[i][0]  ) is empty, only (list_subarrays[i][1] ) has data 
print(np.shape(list_subarrays))
print(np.shape(list_subarrays[0][0]))

#atomic grid is in the same shape than step edges, to compare
sq_g_df=pd.DataFrame(list_subarrays)[1]  #with this I take the lists with values


#going from list to array to dataframe with x and y stack together as columns
if np.logical_and(np.all(arctan_sign_condition)==True, 2.74<abs(math.tan(arctan_mean_v))<14.3):
    s_e_per_terrace=np.dstack((np.asarray(x_s_e_rotated_list_copy), np.asarray(list_copy_step_min_y_scaled))) #I need double (())
else:
    s_e_per_terrace=np.dstack((np.asarray(list_copy_masked_min_x_copy),np.asarray(list_copy_step_min_y_scaled)))
        
print((s_e_per_terrace[0][:10]))  #here  I have it as lists, I need to reshape or make it as a dataframe

#step edges as DataFrames
s_e_df=[]
for i in range(len(list_copy_masked_min_x_copy)):
    s_e_df_2=pd.DataFrame(s_e_per_terrace[i])
    s_e_df.append(s_e_df_2)


int_ln= np.round(np.linspace(0, len(s_e_df[0][0]), np.int(len(s_e_df[0][0])/26), endpoint=True)) # around 25 points interval
int_ln2=int_ln.astype(int) #I need to make integers indices, np.round gives float
print((int_ln2))

'''this masks outliers and values outside mean+- 1.6*std'''
s_e_df_b_l=[]
mean_i_l=[]
std_i_l=[]
mask1D_l=[]
mask_s_e=[]
list_se_df_diff=[]
se_df_mask1_list=[]

for m in range(len(s_e_df)):
    
    diff_se_df=np.append([0],np.diff(s_e_df[m][0]))
    list_se_df_diff.append(diff_se_df)
    masked_x_se=ma.masked_where(abs(diff_se_df) >= 5,s_e_df[m][0]) 
    diff_x_se_df_2=np.append([0],np.diff(masked_x_se))
    mask_x_se_2=ma.masked_where(abs(diff_x_se_df_2) >= 4,masked_x_se) 
    diff_x_se_df_3=np.append([0],np.diff(mask_x_se_2))
    mask_x_se_3=ma.masked_where(abs(diff_x_se_df_3) >= 3,mask_x_se_2) #to remove consecutive outliers
    se_df_mask1_list.append(mask_x_se_3)
    
    #second mask for outliers
    int_ln= np.round(np.linspace(0, len(s_e_df[m][0]), np.int(len(s_e_df[m][0])/26), endpoint=True)) # 25 points interval
    int_ln2=int_ln.astype(int)
    for j in range(len(int_ln2)-1):
        mean_in=np.mean(se_df_mask1_list[m][int_ln2[j]:int_ln2[j+1]])
        mean_i_l.append(mean_in)
        std_in=np.std(se_df_mask1_list[m][int_ln2[j]:int_ln2[j+1]])
        std_i_l.append(std_in)
        low_i=mean_in - 1.6*std_in
        high_i=mean_in + 1.6*std_in
        s_e_d_f_bt=ma.masked_outside(se_df_mask1_list[m][int_ln2[j]:int_ln2[j+1]], low_i , high_i)
        s_e_df_b_l.append(s_e_d_f_bt)
    #newlist = list(chain(*s_e_d_f_bt))    
    newlist = list(chain(*s_e_df_b_l))
    mask1D_l.append(newlist) # appends accumulated lists
   

print(len(mask1D_l[-1]))  # accumulated mask list

#generating index array to split the accumulated mask list (even parts)
bins_mask=np.round(np.linspace(0, len(mask1D_l[-1]), len(s_e_df)+1, endpoint=True))
bins_mask2=bins_mask.astype(int)
print(bins_mask2)

#here I'm splitting the mask in different (same lenght) arrays
mask_s_l=[]
for i in range(len(bins_mask2)-1):
    mask_splits= mask1D_l[-1][bins_mask2[i]:bins_mask2[i+1]] #with this I split the boolean mask in even lenght arrays
    mask_s_l.append(mask_splits)


'''Fitting step edges to an atomic grid’'''
index_k=[]
distances_k=[]
sq_x_k_t_list=[]
for m in range(len(s_e_df)):
    sq_g_df[m].index = range(len(sq_g_df[m])) #reset index according to length of dataframe, I need to do this !
    #s_e_df[m].index = range(len(s_e_df[m]))
    if condition_side>0:
         for c in range(len(s_e_df[m])):
             distanceq,indexq = spatial.KDTree(sq_g_df[m]).query((mask_s_l[m][c],s_e_df[m][1][c]), k=1)    
             index_k.append(indexq)
             distances_k.append(distanceq)
         indices0_k=np.unique(index_k) 
         index_k=[]
         sq_g_x_k= sq_g_df[m][0][indices0_k] #using indices to find the numeric values of the atomic grid
         sq_g_y_k=sq_g_df[m][1][indices0_k]
         sq_x_k=np.dstack((sq_g_x_k,sq_g_y_k))
         sq_x_k_t_list.append(sq_x_k)
    else:
        for c in range(len(s_e_df[-m-1])):
            distanceq,indexq = spatial.KDTree(sq_g_df[m]).query((mask_s_l[-m-1][c],s_e_df[-m-1][1][c]), k=1)
            index_k.append(indexq)
            distances_k.append(distanceq)
        indices0_k=np.unique(index_k) 
        index_k=[]
        sq_g_x_k= sq_g_df[m][0][indices0_k] #using indices to find the numeric values of the atomic grid
        sq_g_y_k=sq_g_df[m][1][indices0_k]
        sq_x_k=np.dstack((sq_g_x_k,sq_g_y_k))
        sq_x_k_t_list.append(sq_x_k)

print(len(index_k))  
print(len(indices0_k) )   
print(len(sq_x_k_t_list))     


sq_g_k_l=[]       #making sq_x_k_t_list as dataframes 
for d in range(len(sq_x_k_t_list)):
    sq_g_k=sq_x_k_t_list[d][0]
    sqg_k=pd.DataFrame(sq_g_k)
    sqg_k_sorted=sqg_k.sort_values(by=[1])  #this sort it by y to be able to plot it with linestyle
    sq_g_k_l.append(sqg_k_sorted)
print(len(sq_g_k_l))

fig26 = plt.figure()
ax26 = fig26.add_subplot(111)
ax26.set_title("Step edges, masks and atomic correspondence")
ax26.set_xlabel("X" , labelpad=15)
ax26.set_ylabel("Y" , labelpad=17)
ax26.grid(True) 
ax26.set_ylim(-1,np.max(list_copy_step_min_y_scaled)+5)
ax26.set_xlim(0,np.max(list_copy_step_min_x_scaled)+5)

for v in range(len(list_copy_step_min_x_scaled)):
    #ax26.plot(list_copy_step_min_x_scaled[v] ,list_copy_step_min_y_scaled[v],linestyle="--",marker="o",
     #     color="black",markersize=3,  alpha=0.7, label='original step edge')
    ax26.plot(se_df_mask1_list[v],s_e_df[v][1],marker="<",markersize=4, linestyle="-", color="red") #first mask
    ax26.plot(mask_s_l[v],s_e_df[v][1],marker="<",markersize=3, linestyle="-.", color="darkviolet", 
              label='step edge masked and odd-shifted')#second mask
    ax26.plot(sq_g_k_l[v][0],sq_g_k_l[v][1],marker="o",markersize=4, linestyle="-", color="blue",
              label='atomic correspondence')

#%%
''' correction of grid fitted step edges'''
# specific for a square atomic grid on Ag (100) single crystal with A-type step
n_even=np.arange(2,41,2)
m_odd=np.arange(1,21,2) #I need to exclude the case where m=1
leo=np.arange(2,15,1)    


A3_l=[]
B3_l=[]
A4_l=[]
B4_l=[]
diff_ex_x_l=[]
diff_ex_y_l=[]
for m in range(len(s_e_df)): #m is the number of the step edge
    A3 = sq_g_k_l[m][0].reset_index(drop=True)
    B3 = sq_g_k_l[m][1].reset_index(drop=True)
    A3_l.append(A3)
    B3_l.append(B3)
    A4 = (A3.copy()).values.tolist()
    B4 = (B3.copy()).values.tolist()
    A4_l.append(A4)
    B4_l.append(B4)
    
    diff_ex_x=np.append(0,np.diff(A3) ).tolist()
    diff_ex_y=np.append(0,np.diff(B3) ).tolist()
    diff_ex_x_l.append(diff_ex_x)
    diff_ex_y_l.append(diff_ex_y)
    
    for i in range(0,len(A3_l[m])):#I'm including particular cases of my example, one by one
        try: 
            
            if (diff_ex_x_l[m][i]==1 and diff_ex_x_l[m][i+1]==-1 and diff_ex_y_l[m][i]==1 and diff_ex_y_l[m][i+1]==0):  
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
            if (diff_ex_x_l[m][i]==-1 and diff_ex_x_l[m][i+1]==1 and diff_ex_y_l[m][i]==1 and diff_ex_y_l[m][i+1]==0): 
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
            
            if (diff_ex_x_l[m][i]==1 and diff_ex_x_l[m][i+1]==-1 and diff_ex_y_l[m][i]==0 and diff_ex_y_l[m][i+1]==1):  
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
            if (diff_ex_x_l[m][i]==-1 and diff_ex_x_l[m][i+1]==1 and diff_ex_y_l[m][i]==0 and diff_ex_y_l[m][i+1]==1): 
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
            
            if (diff_ex_x_l[m][i]==2 and diff_ex_x_l[m][i+1]==-1 and diff_ex_y_l[m][i]==1 and diff_ex_y_l[m][i+1]==0):  
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
            if (diff_ex_x_l[m][i]==-2 and diff_ex_x_l[m][i+1]==1 and diff_ex_y_l[m][i]==1 and diff_ex_y_l[m][i+1]==0): 
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
            
            if (diff_ex_x_l[m][i]==1 and diff_ex_x_l[m][i+1]==0 and diff_ex_y_l[m][i]==0 and diff_ex_y_l[m][i+1]==1
                and diff_ex_x_l[m][i-1]==0 and diff_ex_y_l[m][i-1]==1):  
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
            if (diff_ex_x_l[m][i]==-1 and diff_ex_x_l[m][i+1]==0 and diff_ex_y_l[m][i]==0 and diff_ex_y_l[m][i+1]==1
                and diff_ex_x_l[m][i-1]==0 and diff_ex_y_l[m][i-1]==1): 
                A4_l[m][i] =  A3_l[m][i+1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i+1] 
                
            if (diff_ex_y_l[m][i]==0 and diff_ex_x_l[m][i]>=2  and diff_ex_x_l[m][i+1]==0 ):  
                A4_l[m][i-1] =  A3_l[m][i] #makes it the same as the next point
                B4_l[m][i-1] =  B3_l[m][i] 
            if (diff_ex_y_l[m][i]==0  and diff_ex_x_l[m][i]<=-2 and diff_ex_x_l[m][i+1]==0 ): 
                A4_l[m][i-1] =  A3_l[m][i] #makes it the same as the next point
                B4_l[m][i-1] =  B3_l[m][i] 
            
            if (diff_ex_y_l[m][i]==0 and diff_ex_x_l[m][i]>=2  and diff_ex_x_l[m][i-1]==0 ):  
                A4_l[m][i] =  A3_l[m][i-1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i-1] 
            if (diff_ex_y_l[m][i]==0  and diff_ex_x_l[m][i]<=-2 and diff_ex_x_l[m][i-1]==0 ): 
                A4_l[m][i] =  A3_l[m][i-1] #makes it the same as the next point
                B4_l[m][i] =  B3_l[m][i-1] 
                
            if (diff_ex_y_l[m][i]==0 and diff_ex_x_l[m][i]>=2  and diff_ex_x_l[m][i-1]<0 and diff_ex_x_l[m][i+1]<0):  
                A4_l[m][i] =  A3_l[m][i-1] #makes it the same as the previous point
                B4_l[m][i] =  B3_l[m][i-1] 
            if (diff_ex_y_l[m][i]==0  and diff_ex_x_l[m][i]<=-2 and diff_ex_x_l[m][i-1]>0 and diff_ex_x_l[m][i+1]>0): 
                A4_l[m][i] =  A3_l[m][i-1] #makes it the same as the previous point
                B4_l[m][i] =  B3_l[m][i-1] 
            
            if (diff_ex_x_l[m][i]==1 and diff_ex_x_l[m][i-1]==0 and diff_ex_x_l[m][i+1]==0  and diff_ex_x_l[m][i+2]==-1 
                and diff_ex_x_l[m][i+3]==0 and diff_ex_x_l[m][i+4]==1 and diff_ex_x_l[m][i+5]==1 
                and diff_ex_y_l[m][i-1]==diff_ex_y_l[m][i+1]==diff_ex_y_l[m][i+3]==diff_ex_y_l[m][i+5]==1 
                and diff_ex_y_l[m][i]==diff_ex_y_l[m][i+2]==diff_ex_y_l[m][i+4]==0):  
                A4_l[m][i] =  A3_l[m][i-1] 
                B4_l[m][i] =  B3_l[m][i-1] 
                A4_l[m][i+1] =  A3_l[m][i+2] 
                B4_l[m][i+1] =  B3_l[m][i+2] 
                A4_l[m][i+3] =  A3_l[m][i+4] 
                B4_l[m][i+3] =  B3_l[m][i+4] 
                
            if (diff_ex_x_l[m][i]==-1 and diff_ex_x_l[m][i-1]==0 and diff_ex_x_l[m][i+1]==0  and diff_ex_x_l[m][i+2]==1 
                and diff_ex_x_l[m][i+3]==0 and diff_ex_x_l[m][i+4]==-1 and diff_ex_x_l[m][i+5]==-1 
                and diff_ex_y_l[m][i-1]==diff_ex_y_l[m][i+1]==diff_ex_y_l[m][i+3]==diff_ex_y_l[m][i+5]==1 
                and diff_ex_y_l[m][i]==diff_ex_y_l[m][i+2]==diff_ex_y_l[m][i+4]==0):  
                A4_l[m][i] =  A3_l[m][i-1] 
                B4_l[m][i] =  B3_l[m][i-1] 
                A4_l[m][i+1] =  A3_l[m][i+2] 
                B4_l[m][i+1] =  B3_l[m][i+2] 
                A4_l[m][i+3] =  A3_l[m][i+4] 
                B4_l[m][i+3] =  B3_l[m][i+4] 
                
        except IndexError:
            break
                
        
    for i in range(0,len(A3_l[m])):    
        for o in m_odd:
            try:
            #almost vertical lines
                if (diff_ex_x_l[m][i]==1 and diff_ex_y_l[m][i]==o+2 ) :# o+2 to start from 3 and not from 1
                     A4_l[m].append (  A3_l[m][i]  ) 
                     B4_l[m].append (  B3_l[m][i] - ((o+2)-1) )
                      
                if (diff_ex_x_l[m][i]==-1 and diff_ex_y_l[m][i]==o+2 ) : # o+2 to start from 3 and not from 1
                    A4_l[m].append (  A3_l[m][i]  ) 
                    B4_l[m].append (  B3_l[m][i] - ((o+2)-1) )
                    
         
                if (diff_ex_x_l[m][i]==2 and diff_ex_y_l[m][i]==o and diff_ex_x_l[m][i-1]==-1 and diff_ex_y_l[m][i-1]==1) : 
                    A4_l[m][i-1] =  A3_l[m][i-2] 
                    B4_l[m][i-1] =  B3_l[m][i-2]  
                    A4_l[m].append (  A3_l[m][i] -1 ) 
                    B4_l[m].append (  B3_l[m][i] -1 )
                     
                if (diff_ex_x_l[m][i]==-2 and diff_ex_y_l[m][i]==o and diff_ex_x_l[m][i-1]==1 and diff_ex_y_l[m][i-1]==1) : 
                    A4_l[m][i-1] =  A3_l[m][i-2] 
                    B4_l[m][i-1] =  B3_l[m][i-2] 
                    A4_l[m].append (  A3_l[m][i] - 1 ) 
                    B4_l[m].append (  B3_l[m][i] - 1 )
                    
                    
            except IndexError:
                break
            
    for i in range(0,len(A3_l[m])): 
        for n in n_even:
            try:
            #almost vertical lines
                if (diff_ex_x_l[m][i]==1 and diff_ex_y_l[m][i]==n ) :
                     A4_l[m].append (  A3_l[m][i]  ) 
                     B4_l[m].append (  B3_l[m][i] - (n-1) )
                      
                if (diff_ex_x_l[m][i]==-1 and diff_ex_y_l[m][i]==n ) :  
                    A4_l[m].append (  A3_l[m][i]  ) 
                    B4_l[m].append (  B3_l[m][i] - (n-1) )
                  
                if (diff_ex_x_l[m][i]==2 and diff_ex_y_l[m][i]==n ) :
                     A4_l[m].append (  A3_l[m][i]  ) 
                     B4_l[m].append (  B3_l[m][i] - (n-2) )
                      
                if (diff_ex_x_l[m][i]==-2 and diff_ex_y_l[m][i]==n ) :
                    A4_l[m].append (  A3_l[m][i]  ) 
                    B4_l[m].append (  B3_l[m][i] - (n-2) )
                
                if (diff_ex_x_l[m][i]==4 and diff_ex_y_l[m][i]==n ) :
                     A4_l[m].append (  A3_l[m][i]  ) 
                     B4_l[m].append (  B3_l[m][i] - (n-4) )
                      
                if (diff_ex_x_l[m][i]==-4 and diff_ex_y_l[m][i]==n ) :  
                    A4_l[m].append (  A3_l[m][i]  ) 
                    B4_l[m].append (  B3_l[m][i] - (n-4) )
                         
            except IndexError:
                break
            
    

fig27=plt.figure()
ax27=fig27.add_subplot(111)
ax27.set_xlabel("X")
ax27.set_ylabel("Y")
ax27.set_title("Step edges example")
ax27.set_ylim(np.min(list_copy_step_min_y_scaled),np.max(list_copy_step_min_y_scaled)+5)
ax27.set_xlim(0,np.max(list_copy_masked_min_x)+15)
ax27.grid(True)
ax27.plot(atomic_grid_xy[0],atomic_grid_xy[1], linestyle="",marker="o",color='green',markersize=14,alpha=0.2)

for m in range(len(s_e_df)):
    ax27.plot(sq_g_k_l[m][0],sq_g_k_l[m][1], linestyle="-",marker="o",color='blue',markersize=8,  label='atomic data')
    ax27.plot(A4_l[m],B4_l[m] ,marker=">",linestyle="",color='red',markersize=6, label='corrected data')  

#%%
'''kinks quantification'''
x_y_A_B_l=[]
diff_x_p_l=[]
diff_y_p_l=[]
m_p_list=[]
mean_x_edges=[]
dev_x_edges=[]
dev_x_se_sq_l=[]
kinks_v_l=[]  #kinks valid angles
kinks_p_l=[]  #positive kinks
kinks_n_l=[]  #negative kinks
kinks_v_0_l=[]  #kinsk and zeros
for m in range(len(s_e_df)):
    A4_B4_ds=np.dstack((A4_l[m],B4_l[m]))
    A4_B4_rs=np.reshape(A4_B4_ds, (len(A4_B4_ds[0]),2)) 
#A4_B4=pd.DataFrame({'col1':A4,'col2':B4})
    x_y_A_B=pd.DataFrame(A4_B4_rs).dropna().drop_duplicates().sort_values(by=[1]).reset_index(drop=True)  #sort by y
    x_mean_A_B=x_y_A_B[0].mean()#extra step to calculate the average x per step edge (kink formation energy calculation)
    dev_x_se=(x_y_A_B[0]- np.round(x_y_A_B[0].mean()))
    dev_x_se_sq=(dev_x_se)**2  #definition of mean squares displacement in x
    mean_x_edges.append(x_mean_A_B)
    dev_x_edges.append(dev_x_se)
    dev_x_se_sq_l.append(dev_x_se_sq)
    x_y_A_B_l.append(x_y_A_B)
    diff_x_p=np.diff(x_y_A_B_l[m][0])
    diff_y_p=np.diff(x_y_A_B_l[m][1]) 
    mask_diff_x_p=np.where(diff_x_p==0)
    diff_x_p1=np.delete(diff_x_p,mask_diff_x_p)  #zeros from diff_x deleted from both arrays
    diff_y_p1=np.delete(diff_y_p,mask_diff_x_p)  #zeros from diff_x deleted from both arrays
    diff_x_p_l.append(diff_x_p1)
    diff_y_p_l.append(diff_y_p1)
    m_p=np.divide(diff_y_p1,diff_x_p1)
    m_p_list.append(m_p)
    m_p_0=np.divide(diff_y_p,diff_x_p) #entire step edge ,without removing zeros
    m_p_w=np.logical_or(np.isclose(m_p, np.tan(kink_angle*np.pi/180),atol=0.001),np.isclose(m_p, -np.tan(kink_angle*np.pi/180),atol=0.001)) 
    kinks_v=diff_x_p1[m_p_w]
    kinks_v_l.append(kinks_v)
    kinks_v_0_l.append(diff_x_p)
    #now I need to separate positive from negative
    m_p_p=np.isclose(m_p, 1,atol=0.001)
    m_p_n=np.isclose(m_p, -1,atol=0.001)
#I apply diff_x_p to the m_p positive and negative to count the kinks in and out
    kinks_p=diff_x_p1[m_p_p]  
    kinks_n=diff_x_p1[m_p_n]  
    kinks_p_l.append(kinks_p)
    kinks_n_l.append(kinks_n)


zeros_and_kinks=np.concatenate(kinks_v_0_l, axis=0)  
#valid angle for Ag100
#(diffy/diffx)=1 = 1
angle_Ag100=np.arctan(1)   



#kinks sum
list_kinks_p=[[]for i in range(len(kinks_v_l))]    
list_kinks_n=[[] for i in range(len(kinks_v_l))]   


#kinks_s_p_m=[]
#kinks_s_n_m=[]
kinks_sum_p=0
kinks_sum_n=0

for m in range(len(s_e_df)):
    for j in range(len(kinks_v_l[m])):
        try:
            if kinks_v_l[m][j]>0 and (kinks_v_l[m][j+1]>0 or kinks_v_l[m][j+1]<0)  :  #diff==0 has been excluded previously
                kinks_sum_p+=kinks_v_l[m][j]
                #print('j ', j, 'kinkspos ',kinks_sum_p)
                #kinks_s_n_m.append(kinks_sum_n)
                list_kinks_n[m].append(kinks_sum_n)
                kinks_sum_n=0
            
            if kinks_v_l[m][j]<0 and (kinks_v_l[m][j+1]<0 or kinks_v_l[m][j+1]>0) :
                kinks_sum_n+=kinks_v_l[m][j]
                #kinks_s_p_m.append(kinks_sum_p)
                list_kinks_p[m].append(kinks_sum_p)
                kinks_sum_p=0
    
        except IndexError:
                break
            
for m in range(len(s_e_df)):  #I have to run this part twice to get the plot without the zeros
    mask_kinks_p=np.where(list_kinks_p[m]==0)
    mask_kinks_n=np.where(list_kinks_n[m]==0)
    list_kinks_p[m]=np.delete(list_kinks_p[m],mask_kinks_p)
    list_kinks_n[m]=np.delete(list_kinks_n[m],mask_kinks_n)
   
for m in range(len(s_e_df)):  
    mask_kinks_p=np.where(list_kinks_p[m]==0)
    mask_kinks_n=np.where(list_kinks_n[m]==0)
    list_kinks_p[m]=np.delete(list_kinks_p[m],mask_kinks_p)
    list_kinks_n[m]=np.delete(list_kinks_n[m],mask_kinks_n)

print(len(dev_x_edges))

all_kinks_p=np.concatenate(list_kinks_p, axis=0)
all_kinks_n=np.concatenate(list_kinks_n, axis=0)
print(np.count_nonzero(m_p_p))
all_kinks_t=np.concatenate((all_kinks_p,-all_kinks_n),axis=0)

kinks_p_c=collections.Counter(all_kinks_p)  #behaves like a dictionary
kinks_n_c=collections.Counter(all_kinks_n)
kinks_t_c=collections.Counter(all_kinks_t)
#all_kinks_n.count(-1)

#m_p_w_p=m_p[]
#m_p_w_n=#
bins_k=np.array([0-0.5,1-0.5,2-0.5,3-0.5,4-0.5,5-0.5,6-0.5,7-0.5,8-0.5])
#unique_k_n, counts_k_n = np.unique(all_kinks_n, return_counts=True)

fig28=plt.figure()
ax28=fig28.add_subplot(111)
ax28.set_xlabel("Kink length")
ax28.set_ylabel("Occurrence")
ax28.set_xticks(np.arange(0, 8, step=1))
#ax28.set_title("Kinks [atomic units]")
#ax28.set_xlim(0,np.max(bins_k))
#ax28.hist(all_kinks_p, bins_k,density=False, facecolor='red', alpha=0.5, weights=np.ones(len(all_kinks_p)) / len(all_kinks_p))
#ax28.hist(-all_kinks_n, bins_k,density=False, facecolor='blue', alpha=0.5, weights=np.ones(len(all_kinks_n)) / len(all_kinks_n))
ax28.hist(all_kinks_t, bins_k,density=False, facecolor='purple', alpha=0.5, weights=np.ones(len(all_kinks_t)) / len(all_kinks_t))
#ax28.legend(loc='upper right', fontsize= 'medium') 
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

#%%
'''kink formation energy calculation'''

print(number_terraces)  #valid terraces

#using step edges after corrections on atomic correspondence 
#mean_x_edges is  the list with the mean_x values per step edge
#dev_x_edges is the list with the subtraction of each value from the mean x value per step edge, after fitting step edges to an atomic grid


if np.logical_and(np.all(arctan_sign_condition)==True, 2.74<abs(math.tan(arctan_mean_v))<14.3):
    s_e_msd=x_s_e_rotated_list_copy
else:
    s_e_msd=list_copy_masked_min_x_copy

   
y_n_sq_l=[[[] for j in range(len(dev_x_edges))] for i in range(0,21)] #20 list (diff (x20-x0)), every list of those has m(step edges) lists
#every y_n_sq_l has m sublists

for m in range(len(dev_x_edges)):
    for i in range(0,len(dev_x_edges[m])):#dev_x_edges[m]
        for n in range(0,21):
            if (i-n)>=0:  #with this I avoid index errors for ever value of n used
                y_n_edge= (dev_x_edges[m][i] - dev_x_edges[m][i-n])**2 #this squares the subtraction of x0-xn
                y_n_sq_l[n][m].append(y_n_edge) 
                
mean_n_l=[[] for i in range(0,21)] 
mean_m_l=[]
corr_n=[]
mean_n_l2=[[] for i in range(0,21)] 
mean_m_l2=[]
corr_n2=[]
'''check this double mean calculation'''
for n in range(0,21): 
    for m in range(len(dev_x_edges)):
        #atomic correspondance
        mean_m=np.mean(y_n_sq_l[n][m]) #first I need to calculate the mean per step edge for every n (0,20)
        mean_m_l.append(mean_m) #here I should be appending the m values (averaged)
        mean_n_l[n].append(mean_m)
    mean_m_2=np.mean(mean_n_l[n])
    corr_n.append(mean_m_2)


slope0, intercept0, r_value0, p_value0, std_err0 = stats.linregress(range(0,21), corr_n)
y0=slope0*(range(0,21))+intercept0


fig39=plt.figure()
ax39=fig39.add_subplot(111)
ax39.set_xlabel("r")
ax39.set_ylabel("<(x0-xr)**2>")
ax39.set_title("Mean squares displacement")
#ax29.set_ylim(np.min(list_copy_step_min_y_scaled),np.max(list_copy_step_min_y_scaled)+5)
ax39.set_ylim(-0.1,30)
ax39.set_xlim(-0.1,21)
ax39.set_xticks(np.arange(0, 21, step=2))
ax39.set_yticks(np.arange(0, 31, step=5))
#ax39.set_yticks(np.arange(0, np.round(np.max(corr_n)+1), step=1))
#ax29.grid(True)
ax39.plot(range(0,21),corr_n, linestyle="",marker="o",color='blue',markersize=5)
ax39.plot(range(0,21),y0, linestyle="--",marker="o",color='green',markersize=2)


boltzman=1.38064852e-23 

print(boltzman*temp)


slope_k=slope0
energy=-(boltzman*temp*np.log((slope_k)/2))
energyeV=energy*(6.241509e18)

print('slope',slope0)
print('energy_eV', energyeV) 






