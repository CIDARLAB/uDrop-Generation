#Author:Christopher Rodriguez
#Usage: python3 generator_mine_GUI.py name_of_video.mp4


import warnings
warnings.filterwarnings("ignore") #Supresses warnings: useful for deploying, bad for debugging
import PIL
from PIL import ImageTk, Image
import os
import os.path
import shutil
import sys
import tkinter
from tkinter import ttk
import cv2
import subprocess



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

	subprocess.call("python generator_mine.py " + " ".join([str(fps),str(conversion_factor),str(start_x),str(start_y),str(width),str(height)]),shell=True)
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
