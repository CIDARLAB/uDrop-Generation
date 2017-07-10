#Author: Christopher Rodriguez
#Usage: to be used with a front end that automatically splits frames into the frames folder and calls this with the correct arguments
#Arguments: fps, conversion factor (channel size in mm over channel size in pixels), startx of bounding box, starty, width, height

import warnings
warnings.filterwarnings("ignore") #Supresses warnings: useful for deploying, bad for debugging

import numpy
import math
import PIL
from PIL import ImageTk, Image
import os
import os.path
import shutil
import sys
import cv2



#Parse variables from our command arguments
fps = float(sys.argv[1])
conversion_factor = float(sys.argv[2])
start_x = int(sys.argv[3])
start_y = int(sys.argv[4])
width = int(sys.argv[5])
height = int(sys.argv[6])

#Load our frames from the frames directory
buf = []
for f in sorted(os.listdir("frames")):
        buf.append(cv2.imread("frames/"+f))

#Reset the output directory
if(os.path.exists("output/")):
	shutil.rmtree('output/')
os.mkdir('output/')
os.mkdir('output/cleaned/')
os.mkdir('output/raw/')



#Canny's edge detection on each frame
edges = [cv2.Canny(cv2.blur(x,(4,4)),40,10)[start_y:start_y+height, start_x:start_x+width] for x in buf]



datlist = []
#Find the average pixel value of a band of pixels at the nozzle exit
#This band should get higher as a drop passes through it
#We build a wave of values from this that should look like a sin wave if everything went well
count = 0
for arr in edges:
	running_sum = numpy.mean(arr);
	datlist.append(running_sum)
	with open("output/sin.csv","a") as fout:
		fout.write(str(count)+","+str(running_sum)+"\n")
	count+=1

#Smoothing (although the waves are so smooth with edge detection, it is hardly necessary anymore
newdatlist = []
smooth_interval = 3 #How many data points should we use to smooth? (Taken both left and right)
weights = [1,0.75,0.25] #What should be the weights of these data points?

#Basically, we assign every point a new value that is equal to the weighted average of its neighbors
for i in range(0,len(datlist)):
	if i>=smooth_interval and i<=len(datlist)-1-smooth_interval:
		val = datlist[i]
		for j in range(1,smooth_interval+1):
			val+=datlist[i+j]*weights[j-1]
			val+=datlist[i-j]*weights[j-1]
		val /= (float)(sum(weights)*2 + 1)
		newdatlist.append(val)
	else:
		newdatlist.append(datlist[i])	
	with open("output/sin_smooth.csv","a") as fout:
		fout.write(str(i)+","+str(newdatlist[i])+"\n")



#Function for calculating areas of droplets
def getArea(arr,index):
	#Find contours
	im2, contours, hierarchy = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#Concatenate all their verticies (useful for multiple islands)
	outer_contour = numpy.concatenate(numpy.asarray(contours))
	#Find the convex hull of the verticies
	hull = cv2.convexHull(outer_contour)
	#Fit a rectangle to these verticies (currently unused except for visualization purposes)
	rect = cv2.minAreaRect(hull)

	#Find the area of our droplet
	area = cv2.contourArea(hull)


	#Write out the images of the contours
	box = cv2.boxPoints(rect)
	box = numpy.int0(box)
	img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
	cv2.drawContours(img,[hull],0,(255,255,255),cv2.FILLED)
	cv2.drawContours(img,[box],0,(0,0,255),2)
	Image.fromarray(img).save("output/cleaned/"+str(index)+"bounds"+".jpg")
	Image.fromarray(arr).save("output/raw/"+str(index)+"regular"+".jpg")	

	return area




#Drop rate calculations (Basically calculating the period of the wave)

#We track the start valley and end valley so that we don't cut off calculations in the middle of a wave
#This shouldn't matter if we have sufficiently large data, but let's keep it in the spirit of accuracy
start_valley = 0
end_valley = 0

last_max = 0

drop_areas = [];


#Loop through our wave data and get the maxes
maxes_count = 0
for i in range(1,len(newdatlist)-1):
	if newdatlist[i] > newdatlist[i-1] and newdatlist[i] > newdatlist[i+1] and start_valley != 0: #Local max inside our range
		maxes_count += 1
		last_max = i
		drop_areas.append(getArea(edges[i],i))
	elif newdatlist[i] <= newdatlist[i-1] and newdatlist[i] <= newdatlist[i+1]: #Local min
		if(start_valley == 0):
			start_valley = i
		else:
			end_valley = i


if(last_max > end_valley):
	maxes_count -= 1 #We don't want to count maxes outside of the range of the data we are looking at



drop_areas_np = numpy.asarray(drop_areas)
drops_per_second = (maxes_count / (float)(end_valley - start_valley)) * fps #Peaks per frames * frames per second = Peaks/seconds


#stdout outputs
print("Drops per Second: " + str(drops_per_second))
print("Average Drop Area (pixels): " + str(drop_areas_np.mean()))
print("Average Drop Area (micrometers): " + str(drop_areas_np.mean()*(conversion_factor**2)))
print("Drop Area Standard Deviation (pixels): " + str(drop_areas_np.std()))
print("Drop Area Standard Deviation (micrometers): " + str(drop_areas_np.std()*(conversion_factor**2)))

#disk outputs
with open("output/drop_data.csv","w") as f:
	f.write(str(drop_areas_np.mean())+"," + str(drop_areas_np.mean()*conversion_factor**2) + "," + str(drop_areas_np.std()) + "," + str(drop_areas_np.std()*conversion_factor**2) + "," + str(drops_per_second) + "\n")


with open("output/drop_data_raw.csv","w") as f:
	f.write(",".join([str(x) for x in drop_areas_np]) + "\n")


