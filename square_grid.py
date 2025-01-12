import numpy as np
from matplotlib import pyplot as plt
from math import sqrt as sqrt
from tempfile import TemporaryFile


n=401
def square_grid(n):

    ones=(np.ones(n))


    #print(ones)
    '''X arrays'''
    x1=np.array(range(n))
    x_sq = np.tile(x1,(n,1)) 
    print(x_sq)
    #print(len(ones))
    print("x_sq", x_sq)
    print(np.shape(x_sq))



    '''Y arrays'''

    y1_list=[]
    for i in range(n):
        y_gen=i*ones
        y1_list.append(y_gen)
        y_sq=np.array(y1_list)


    print('y_sq',y_sq)
    print(np.shape(y_sq))

    fig_hg = plt.figure()
    ax_hg = fig_hg.add_subplot(111)

    ax_hg.set_xlabel("X")
    ax_hg.set_ylabel("Y")
    
    ax_hg.set_title("square grid")

    ax_hg.plot(x_sq, y_sq,linestyle="",marker="o",color="purple",markersize=3)
  



    return (x_sq, y_sq)
square_grid(401)

sq_x400 = TemporaryFile()
sq_y400 = TemporaryFile()


np.save('sq_x400.npy',x_sq) 
np.save('sq_y400.npy',y_sq)

sq_x400=np.load('sq_x400.npy')
sq_y400=np.load('sq_y400.npy')



print(x_sq)
print(y_sq)


#np.savetxt('even_x.txt', it_even, delimiter=' ',fmt='%d') didn't preserve the desired structure
#np.savetxt('even_y.txt', y_even_array,delimiter=' ', fmt='%d')
#np.savetxt('odd_x.txt', it_odd, delimiter=' ',fmt='%d')
#np.savetxt('odd_y.txt', y_odd_array,delimiter=' ', fmt='%d')
#np.save(even_x, it_even)
#np.save(even_y, y_even_array)
#np.save(odd_x, it_odd)
#np.save(odd_x, y_odd_array)

#even_x=it_even
#even_y=y_even_array
#odd_x=it_odd
#odd_y=y_odd_array
#print(np.shape(e_x))
#print(np.shape(e_y))
#print(np.shape(o_x))
print(np.shape(o_y))

fig_hg2 = plt.figure()
ax_hg2 = fig_hg2.add_subplot(111)

ax_hg2.set_xlabel("X")
ax_hg2.set_ylabel("Y")
    
ax_hg2.set_title("hexagonal grid")

ax_hg2.plot(e_x, e_y,linestyle="",marker="o",color="purple",markersize=3)
ax_hg2.plot(o_x,o_y,linestyle="",marker="o",color="b",markersize=3)
