#Author: Christopher Rodriguez
#Usage: python3 fission_mine_GUI.py name_of_video.mp4

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
import tkinter
from tkinter import ttk
import skvideo
from skvideo import io



if(os.path.exists("frames/")):
	shutil.rmtree('frames/')
os.mkdir('frames/')
os.system("ffmpeg -i " + sys.argv[1] + " -vf mpdecimate,setpts=N/FRAME_RATE/TB frames/%03d.png")

buf = []
for f in sorted(os.listdir("frames")):
	buf.append(cv2.imread("frames/"+f))




pic_width = 0
pic_height = 0
mouse_coords = []
select_index = 0


#Save our click locations
def click(event):
	global mouse_coords
	global select_index


	if(select_index<6):
		mouse_coords.append({"x":event.x,"y":event.y})
		select_index+=1
	drawGUI()


fps = 0
conversion_factor = 
#Actually start the backend script
def confirm():
	global resize_ratio
	global fps
	global conversion_factor
	global wanted_start

	if(select_index != 6 or fps_entry.get() == "" or channel_entry.get() == ""):
		return

	fps = float(fps_entry.get())
	channel_size_mm = float(channel_entry.get())
	
	coord_string = ''
	for i in range(0,len(mouse_coords),2):
		startx = min(mouse_coords[i]["x"],mouse_coords[i+1]["x"])
		starty = min(mouse_coords[i]["y"],mouse_coords[i+1]["y"])
		width = abs(mouse_coords[i]["x"] - mouse_coords[i+1]["x"])
		height = abs(mouse_coords[i]["y"] - mouse_coords[i+1]["y"])
		coord_string += ' ' + ",".join([str(round(startx*resize_ratio)),str(round(starty*resize_ratio)),str(round(width*resize_ratio)),str(round(height*resize_ratio))])
	
	root.destroy()
	os.system("python3 fission_mine.py " + str(fps) + " " + str(channel_size_mm) + coord_string)


#Draw the GUI based on user changes
def drawGUI():
	global select_index
	global bound_top_left
	global bound_bottom_right
	#Redraw image
	canvas.create_image(0,0, anchor="nw", image=img)


	#Instructional text
	mtext = "Press cancel to start over"
	if select_index < 2:
		mtext = "Draw center drop"
	elif select_index < 4:
		mtext = "Draw top drop"
	elif select_index < 6:
		mtext = "Draw bottom drop"
	
	canvas.create_text(10,10,fill="white",anchor="nw",font="Times 12",text=mtext)	





	#Draw all boxes
	for i in range(select_index//2):
		startx = mouse_coords[i*2]["x"]
		endx = mouse_coords[i*2 + 1]["x"]
		if(endx < startx):
			temp = startx
			startx = endx 
			endx = temp

		starty = mouse_coords[i*2]["y"]
		endy = mouse_coords[i*2 + 1]["y"]
		if(endy < starty):
			temp = starty
			starty = endy 
			endy = temp
			
		canvas.create_rectangle(startx,starty,endx,endy)
	


#Undo changes
def cancel():
	global bound_top_left
	global bound_bottom_right
	global select_index
	select_index = 0
	bound_top_left = [-1,-1]
	bound_bottom_right = [-1,-1]
	drawGUI()

#Move on to the next frame
#This is just so the user can get a comfortable, separated frame to draw their rects
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

#Go back to the previous frame
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



#Draws all our controls
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
