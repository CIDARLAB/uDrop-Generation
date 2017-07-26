#Author: Christopher Rodriguez
#Built to simplify compiling to a Windows Executable
#Before Line 270 is the generator_mine_GUI.py without the system call with sys args
#Line 271 onwards is just generator_mine.py without the parsing of sys args
#Feel free to just copy paste when an edit is made to either

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
import tkinter
from tkinter import ttk
import cv2




if(os.path.exists("frames/")):
	shutil.rmtree('frames/')
os.mkdir('frames/')
os.system("ffmpeg -i " + sys.argv[1] + " -vf mpdecimate,setpts=N/FRAME_RATE/TB frames/%03d.png")

buf = []
for f in sorted(os.listdir("frames")):
	buf.append(cv2.imread("frames/"+f))




pic_width = 0
pic_height = 0
bound_top_left = [-1,-1]
bound_bottom_right = [-1,-1]
channel_selections = [[0,0],[0,0]]
select_index = 0

def click(event):
	global bound_top_left
	global bound_bottom_right
	global select_index
	global resize_ratio
	modx = event.x * resize_ratio
	mody = event.y * resize_ratio
	if select_index == 0:
		bound_top_left = [modx,mody]
		select_index += 1
	elif select_index == 1:
		bound_bottom_right = [modx,mody]
		drawGUI()
		select_index += 1
	elif select_index == 2:
		channel_selections[0][0] = modx
		channel_selections[0][1] = mody
		select_index += 1
	elif select_index == 3:
		channel_selections[1][0] = modx
		channel_selections[1][1] = mody
		select_index += 1
	drawGUI()


		

start_x = 0
start_y = 0
width = 0
height = 0
fps = 0
conversion_factor = 1
def confirm():
	global start_x
	global start_y
	global width
	global height
	global fps
	global conversion_factor
	global wanted_start

	if(select_index != 4):
		return

	start_x = round(bound_top_left[0])
	start_y = round(bound_top_left[1])
	width = round(bound_bottom_right[0] - bound_top_left[0])
	height = round(bound_bottom_right[1] - bound_top_left[1])
	fps = float(fps_entry.get())
	channel_size_mm = float(channel_entry.get())
	channel_size_px = float(abs(channel_selections[1][1] - channel_selections[0][1]))
	conversion_factor = channel_size_mm/channel_size_px

	root.destroy()

	#os.system("python3 generator_mine.py " + " ".join([str(fps),str(conversion_factor),str(start_x),str(start_y),str(width),str(height)]))

def drawGUI():
	global select_index
	global bound_top_left
	global bound_bottom_right
	canvas.create_image(0,0, anchor="nw", image=img)


	mtext = "Press cancel to start over"
	if select_index < 2:
		mtext = "Draw bounding box"
	elif select_index < 3:
		mtext = "Draw channel top boundary"
	elif select_index < 4:
		mtext = "Draw channel bottom boundary"

	
	
	canvas.create_text(10,10,fill="white",anchor="nw",font="Times 12",text=mtext)	
	



	if(select_index>1):
		canvas.create_line(bound_top_left[0] / resize_ratio, bound_top_left[1] / resize_ratio, bound_top_left[0] / resize_ratio, bound_bottom_right[1] / resize_ratio, fill="green")
		canvas.create_line(bound_top_left[0] / resize_ratio, bound_top_left[1] / resize_ratio, bound_bottom_right[0] / resize_ratio, bound_top_left[1] / resize_ratio, fill="green")
		canvas.create_line(bound_top_left[0] / resize_ratio, bound_bottom_right[1] / resize_ratio, bound_bottom_right[0] / resize_ratio, bound_bottom_right[1] / resize_ratio, fill="green")
		canvas.create_line(bound_bottom_right[0] / resize_ratio, bound_top_left[1] / resize_ratio, bound_bottom_right[0] / resize_ratio, bound_bottom_right[1] / resize_ratio, fill="green")
	if(select_index>2):
		for i in range(0,select_index-2):
			canvas.create_line((channel_selections[i][0]/resize_ratio)-10, (channel_selections[i][1]/resize_ratio), (channel_selections[i][0]/resize_ratio)+10, (channel_selections[i][1]/resize_ratio), fill="red")


def cancel():
	global bound_top_left
	global bound_bottom_right
	global select_index
	select_index = 0
	bound_top_left = [-1,-1]
	bound_bottom_right = [-1,-1]
	drawGUI()


def nextImage():
        global pic_index
        global img
        global buf
        global resize_ratio
        pic_index+=1
        pic_index%=len(buf)
        raw_image = Image.fromarray(buf[pic_index])
        if(resize_ratio > 1):
                raw_image = raw_image.resize((round(pic_width / resize_ratio), round(pic_height / resize_ratio)))
        img = ImageTk.PhotoImage(raw_image)
        drawGUI()


def previousImage():
        global pic_index
        global img
        global buf
        global resize_ratio
        pic_index-=1
        if(pic_index < 0):
                pic_index=len(buf)-1
        raw_image = Image.fromarray(buf[pic_index])
        if(resize_ratio > 1):
                raw_image = raw_image.resize((round(pic_width / resize_ratio), round(pic_height / resize_ratio)))
        img = ImageTk.PhotoImage(raw_image)
        drawGUI()




#TODO: add in selection of channel bottom and channel top. Selecting should be done on indexes now. Text box to show where we are at.
 
root = tkinter.Tk()


pic_index = 0

pic_width = 0
pic_height = 0
raw_image = Image.fromarray(buf[pic_index])
pic_width, pic_height = raw_image.size


#Resize so our window isn't too big (for high res videos)
max_size = [1200,600] #Arbitrary
resize_ratio = 1
if(pic_width > max_size[0]):
	resize_ratio = pic_width / max_size[0]

if(pic_height > max_size[1]):
	resize_ratio = max(resize_ratio,pic_height/max_size[1])


if(resize_ratio > 1):
	raw_image = raw_image.resize((round(pic_width / resize_ratio), round(pic_height / resize_ratio)))

img = ImageTk.PhotoImage(raw_image)

canvas_frame = tkinter.Frame(root) 
canvas_frame.pack()


fps_frame = tkinter.Frame(root)
fps_frame.pack(side="top")
fps_label = tkinter.Text(fps_frame,width=30,height=1)
fps_label.pack(side="left")
fps_label.insert("end","Frames per Second:")
fps_label.configure(state="disabled")
fps_entry = tkinter.Entry(fps_frame)
fps_entry.pack(side="left")

channel_frame = tkinter.Frame(root)
channel_frame.pack(side="top")
channel_label = tkinter.Text(channel_frame,width=30,height=1)
channel_label.pack(side="left")
channel_label.insert("end","Channel Width in micrometers:")
channel_label.configure(state="disabled")
channel_entry = tkinter.Entry(channel_frame)
channel_entry.pack(side="left")

image_selection_frame = tkinter.Frame(root)
image_selection_frame.pack(side="top")
next_button = ttk.Button(image_selection_frame, text='Next Frame',command = nextImage)
next_button.pack(side="left")
previous_button = ttk.Button(image_selection_frame, text='Previous Frame',command = previousImage)
previous_button.pack(side="left")


command_frame = tkinter.Frame(root)
command_frame.pack(side="top")
confirm_button = ttk.Button(command_frame, text='Confirm',command = confirm)
confirm_button.pack(side="left")
cancel_button = ttk.Button(command_frame, text='Cancel',command = cancel)
cancel_button.pack(side="left")




canvas = tkinter.Canvas(canvas_frame,width = pic_width / resize_ratio,height=pic_height / resize_ratio)
canvas.pack(side="top")
canvas.bind('<Button-1>', click)
drawGUI()

root.mainloop()
























canny_weights = [100,40] #Default tolerances for edge detection
wave_weights = [0.5,0.25,0.1] #Wave smoothing default weights


#Reset the output directory
if(os.path.exists("output/")):
	shutil.rmtree('output/')
os.mkdir('output/')
os.mkdir('output/cleaned/')
os.mkdir('output/raw/')

#Read user parameters if they choose to change them
if(os.path.exists("params.cfg")):
	with open("params.cfg","r") as f:
		lines = f.readlines()
		canny_weights = [float(x.replace("\n","")) for x in lines[0].split(",")]
		wave_weights = [float(x.replace("\n","")) for x in lines[1].split(",")]

frame_names = sorted(os.listdir("frames"))

#Apply blurs and edge detection to frame
def cleanFrame(frame_index):
	f = frame_names[frame_index]
	full_image = full_image = cv2.imread("frames/"+f)
	return cv2.Canny(cv2.blur(full_image,(4,4)),canny_weights[0],canny_weights[1])[start_y:start_y+height, start_x:start_x+width]




datlist = []
#Find the average pixel value of a band of pixels at the nozzle exit
#This band should get higher as a drop passes through it
#We build a wave of values from this that should look like a sin wave if everything went well
count = 0
for f in frame_names:
	
	#Canny's edge detection on each frame
	arr = cleanFrame(count)

	Image.fromarray(arr).save("output/raw/"+str(count)+"regular"+".png")
	running_sum = numpy.mean(arr);
	datlist.append(running_sum)
	
	with open("output/sin.csv","a") as fout:
		fout.write(str(count)+","+str(running_sum)+"\n")
	count+=1





#Wave smoothing
newdatlist = []

#Basically, we assign every point a new value that is equal to the weighted average of its neighbors
for i in range(0,len(datlist)):
	if i>=len(wave_weights) and i<=len(datlist)-1-len(wave_weights):
		val = datlist[i]
		for j in range(1,len(wave_weights)+1):
			val+=datlist[i+j]*wave_weights[j-1]
			val+=datlist[i-j]*wave_weights[j-1]
		val /= (float)(sum(wave_weights)*2 + 1)
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

	#Find the area of our droplet
	area = cv2.contourArea(hull)


	img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
	cv2.drawContours(img,[hull],0,(255,255,255),cv2.FILLED)
	Image.fromarray(img).save("output/cleaned/"+str(index)+"bounds"+".png")
	Image.fromarray(arr).save("output/raw/"+str(index)+"regular"+".png")

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
		drop_areas.append(getArea(cleanFrame(i),i))
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


