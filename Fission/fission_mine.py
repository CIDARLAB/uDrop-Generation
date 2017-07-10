#Author: Christopher Rodriguez
#Usage: Called by front end script with split frames in frames and arguments as follows
#Arguments: fps, channel size in mm, then for each bounding box, give the startx, starty, width, height as comma separated values

import warnings
warnings.filterwarnings("ignore") #Supresses warnings: useful for debugging, bad for deploying

import numpy
import math
import PIL
from PIL import ImageTk, Image
import os
import os.path
import shutil
import sys
import cv2
import sys

#Weights to use for Canny edge detection
#Not used in this version, but kept in just in case it becomes useful again
canny_weights = (80,0) #190 60  #260,190

#Describes which way the drop flows through our chamber
#(l)eft, (r)ight, (u)p, (d)own. The count of this list is how many bounding boxes we have.
#If it flows from left to right, its orientation is (r)ight
box_orientations = ["l","u","d"] 



#Variables passed into the script from the web app
start_x = []
start_y = []
width = []
height = []
fps = 0
channel_size_mm = 0
conversion_factor = 1


#Parse the inputs
fps = float(sys.argv[1])
channel_size_mm = float(sys.argv[2])

for i in range(3,len(sys.argv)):
	points = sys.argv[i].split(",")
	start_x.append(int(points[0]))
	start_y.append(int(points[1]))
	width.append(int(points[2]))
	height.append(int(points[3]))


#We load our buffer from the frames folder of png files
buf = []
for f in sorted(os.listdir("frames")):
	buf.append(cv2.imread("frames/"+f))



#Returns area, minor axis, and major axis (in that order)
def getDimensions(arr,index,box_index):
	noise_threshold = 75 #Areas larger than this are a piece of the drop, smaller are just noise

	#Contour detection
	im2, contours, hierarchy = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#Contour areas
	areas = [cv2.contourArea(x) for x in contours]
	#Only choose contours past the threshold
	choose_indexes = [x for x in range(0,len(areas)) if areas[x] > noise_threshold]
	#Concatenate all verticies together
	#Useful if the program reads two "islands" in a picture
	outer_contour = numpy.concatenate(numpy.asarray(contours)[choose_indexes])
	#Find the convex hull
	hull = cv2.convexHull(outer_contour)
	#Fit a rect to this hull (rotations allowed)
	rect = cv2.minAreaRect(hull)

	#Area of the hull
	area = cv2.contourArea(hull)

	#Get drop dimensions
	width = rect[1][0]
	height = rect[1][1]
	major = max(width,height)
	minor = min(width,height)


	#Save the bounding box and hull to output so we can see if anything went wrong
	box = cv2.boxPoints(rect)
	box = numpy.int0(box)
	img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
	cv2.drawContours(img,[hull],0,(255,255,255),cv2.FILLED)
	cv2.drawContours(img,[box],0,(0,0,255),2)
	Image.fromarray(img).save("output/cleaned" +str(box_index)+"/"+str(index)+"bounds"+".jpg")
	
	return area,minor,major

	
#Is there pixels around the edge of the image?
#If there are, the drop is probably moving across the image
detect_width = 5 #Padding around the image to check for (so check 5 in from either side)
def edgeNoise(arr, orientation):
	edge_noise_ret = {"opening":False,"closing":False} #Dict of where the edge noise is found (Opening is right/bottom, closing is left/top)
	if(orientation == "u"):
		edge_noise_ret["opening"] = numpy.max(arr[-detect_width:]) > 0
		edge_noise_ret["closing"] = numpy.max(arr[:detect_width]) > 0
	elif(orientation == "d"):
		edge_noise_ret["opening"] = numpy.max(arr[:detect_width]) > 0
		edge_noise_ret["closing"] = numpy.max(arr[-detect_width:]) > 0
	elif(orientation == "l"):
		 edge_noise_ret["opening"] = numpy.max(arr[:,-detect_width:]) > 0
		 edge_noise_ret["closing"] = numpy.max(arr[:,:detect_width]) > 0
	elif(orientation == "r"):
		 edge_noise_ret["opening"] = numpy.max(arr[:,:detect_width]) > 0
		 edge_noise_ret["closing"] = numpy.max(arr[:,-detect_width:]) > 0
	return edge_noise_ret
			


#Limit to threshold at
#TODO: Take into account image qualities when making this threshold
fin_limit = 120


#This commented out line shows how you can do edge detection instead of inverse thresholding
#edges = [cv2.Canny(cv2.blur(x,(4,4)),canny_weights[0],canny_weights[1]) for x in buf]
edges = []
for x in buf:
	#Grayscale the image
	x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	#Apply inverse binary thresolding (pixels below fin_limit go to 255, pixels above fin_limit go to 0)
	#This works off the assumption that the drops are darker than the background
	ret,thresh = cv2.threshold(x_gray,fin_limit,255,cv2.THRESH_BINARY_INV)
	#Find contours
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	empty = numpy.zeros_like(im2)
	#Filter out the small ones
	contours_lg = [x for x in contours if cv2.contourArea(x) > 500]
	#Draw the contours on a black background
	cv2.drawContours(empty,contours_lg,-1,(255,255,255),cv2.FILLED)
	
	#These contours are what we will do our image mining on
	edges.append(empty)



#I tested with background subtraction, to no avail.
#Perhaps you'll find more luck with it?

#fgbg = cv2.createBackgroundSubtractorMOG2()
#We run through twice, once as a train, and the next as a test
#TODO: We really only need to train up to the max history of fgbg
#for x in range(0,len(edges)):
#	fgmask = fgbg.apply(edges[x])

#background_sub = []
#for x in range(0,len(edges)):
#	fgmask = fgbg.apply(edges[x])
#	background_sub.append(fgmask)
#	Image.fromarray(fgmask).save("full_frames/"+str(x)+"back_sub.png")


	

focus_spots = []
#Focus spots are the just the part of the whole image that we want to focus on
#TODO: do the focus spots before the thresholding/edge detection to make stuff faster
for i in range(0,len(box_orientations)):
	focus_spots.append([x[start_y[i]:start_y[i]+height[i], start_x[i]:start_x[i]+width[i]] for x in edges])

#Main function
#frames argument is a numpy 3D array of a user selected bounding box over time
#box_index argument is used for storing files
def mine_box(box_index):
	global fps
	global channel_size_mm
	global conversion_factor

	frames = focus_spots[box_index]


	#Save all the focus spots 	
	[Image.fromarray(frames[x]).save("output/raw" + str(box_index) + "/" + str(x) +".png") for x in range(0,len(frames))]



	any_edge_noise = []
	closing_edge_noise = []
	#Loop through the frames in our focus and determine if the drop is traversing or fully centered
	#We create a binary wave based on the truth values of the edge noise
	for count,arr in enumerate(frames):
		edge_noise_results = edgeNoise(arr,box_orientations[box_index])
		
		#REMEMBER: 1 means no edge noise, 0 means edge noise
		if(edge_noise_results["opening"] or edge_noise_results["closing"]): #Transient
			any_edge_noise.append(0)

		else: #Either fully in frame or nothing in frame
			any_edge_noise.append(1)

		#REMEMBER: closing edge noise is essentially the opposite. True when there is noise, false when there is not
		#I did this solely to annoy you. (But for real, it has to do with what I feel is a positive result and what I feel is a negative result)
		closing_edge_noise.append(edge_noise_results["closing"])

		with open("output/" + str(box_index) + "sin.csv","a") as fout:
			fout.write(str(count)+","+str(any_edge_noise[-1])+"\n")





	#Drop rate calculations (Basically calculating the period of the wave)

	#We track the start valley and end valley so that we don't cut off calculations in the middle of a wave
	#This shouldn't matter if we have sufficiently large data, but let's keep it in the spirit of accuracy
	start_valley = 0
	end_valley = 0

	last_max = 0

	drop_areas = [] #Area of drop in pixels
	drop_major_axes = [] #Major (largest) axis size in pixels
	drop_minor_axes = [] #Minor (smallest) axis size in pixels
	drop_speeds = [] #In pixels per frame

	#This boolean switch is used so we don't double count frames
	#We only want to check out a high if the wave was low right before it
	switch_ready = False

	maxes_count = 0


	#dist_dicts is for calculating drop speeds
	#Filled up, it looks like [{"index":14,"size":156}, ...]
	#index is the frame where the drop was observed
	#size is the drop's major axis size + the detect_width (this is the total distance the drop traveled)
	dist_dicts = []

	#Loop through our wave points
	#Basically, this block picks out the points where the derivative of the wave is one or negative one and then analyzes the drop at that point
	for i in range(0,len(any_edge_noise)):
		if any_edge_noise[i] == 1 and switch_ready:
			maxes_count += 1
			last_max = i
			area, minor_axis, major_axis = getDimensions(frames[i],i,box_index)
			drop_areas.append(area)
			drop_minor_axes.append(minor_axis)
			drop_major_axes.append(major_axis)
			dist_dicts.append({"index":i,"size":major_axis+detect_width})
			switch_ready = False
		elif any_edge_noise[i] == 0 and not switch_ready:
			if(start_valley == 0):
				start_valley = i
			else:
				end_valley = i
			switch_ready = True
	
	#Now, let's find drop speeds
	for entry in dist_dicts:
		size = float(entry["size"]) #Size of major axis
		search_index = entry["index"] #Keeps our place while looking through frames. Starts at the perfectly separated analyzed drop
		found_start = False #Turns true when we hit the first spot of edge noise after the search index
		time_in_transition = 0 #Keep track of how long the edge noise persisted (this is also how long the drop took to enter and leave the detect zone)
		good_ending = False #If we just stopped before we truly got the full size of the drop (because we ran out of frames), don't log the result
		while(search_index < len(closing_edge_noise)):
			if not found_start:
				if closing_edge_noise[search_index]:
					found_start = True
			else:
				if not closing_edge_noise[search_index]:
					good_ending = True
					break
				time_in_transition += 1
			search_index+=1

		if good_ending:
			drop_speeds.append(size/float(time_in_transition))	

	
			

	print(drop_speeds)

	if(last_max > end_valley):
		maxes_count -= 1 #We don't want to count maxes outside of the range of the data we are looking at



	#Preliminary conversions so the actual data output part isn't freakishly longer than it already is
	drop_areas_np = numpy.asarray(drop_areas)
	drop_major_axes_np = numpy.asarray(drop_major_axes)
	drop_minor_axes_np = numpy.asarray(drop_minor_axes)
	drops_per_second = (maxes_count / (float)(end_valley - start_valley)) * fps #Peaks per frames * frames per second = Peaks/seconds
	drop_speeds_np = numpy.asarray(drop_speeds)

	if(box_index == 0): #We only want to set the conversion factor once.
		conversion_factor = channel_size_mm / drop_minor_axes_np.mean()
	
	#What should we call our bounding boxes?
	channel_names = ["Central droplet","Top chamber", "Bottom chamber"]
	#Where should we save our info for each bounding box?
	channel_file_names = ["central_droplet","top_chamber", "bottom_chamber"]

	#stdout outputs
	print(channel_names[box_index])
	print("Drop areas px: " + str(drop_areas))
	print("Drop areas mean mm: " + str(drop_areas_np.mean()*conversion_factor**2))
	print("Drop areas standard deviation mm: " + str(drop_areas_np.std()*conversion_factor**2))
	print("Major axes px: " + str(drop_major_axes))
	print("Major axes mean mm: " + str(drop_major_axes_np.mean()*conversion_factor))
	print("Major axes standard deviation mm: " + str(drop_major_axes_np.std()*conversion_factor))
	print("Minor axes px: " + str(drop_minor_axes))
	print("Minor axes mean mm: " + str(drop_minor_axes_np.mean()*conversion_factor))
	print("Minor axes standard deviation mm: " + str(drop_minor_axes_np.std()*conversion_factor))
	print("Drop Speed Mean (px/frame): " + str(numpy.asarray(drop_speeds).mean()))
	print()

	#File outputs
	with open("output/"+channel_file_names[box_index] + ".csv","w") as f:
		f.write(str(drop_areas_np.mean())+"," + str(drop_areas_np.mean()*conversion_factor**2) + "," + str(drop_areas_np.std()) + "," + str(drop_areas_np.std()*conversion_factor**2) + "\n")
		f.write(str(drop_major_axes_np.mean())+"," + str(drop_major_axes_np.mean()*conversion_factor) + "," + str(drop_major_axes_np.std()) + "," + str(drop_major_axes_np.std()*conversion_factor) + "\n")
		f.write(str(drop_minor_axes_np.mean())+"," + str(drop_minor_axes_np.mean()*conversion_factor) + "," + str(drop_minor_axes_np.std()) + "," + str(drop_minor_axes_np.std()*conversion_factor) + "\n")
		f.write(str(drop_speeds_np.mean())+"," + str(drop_speeds_np.mean()*conversion_factor) + "," + str(drop_speeds_np.std()) + "," + str(drop_speeds_np.std()*conversion_factor) + "\n")


	with open("output/"+channel_file_names[box_index] + "raw.csv","w") as f:
		f.write(",".join([str(x) for x in drop_areas_np]) + "\n")
		f.write(",".join([str(x) for x in drop_major_axes_np]) + "\n")
		f.write(",".join([str(x) for x in drop_minor_axes_np]) + "\n")
		f.write(",".join([str(x) for x in drop_speeds_np]) + "\n")


#Reset our output folder
if(os.path.exists("output/")):
	shutil.rmtree('output/')
os.mkdir('output/')

#Loop through each bounding box
for x in range(0,len(box_orientations)):
	os.mkdir('output/cleaned' + str(x) +'/')
	os.mkdir('output/raw' + str(x) + '/')
	mine_box(x)
