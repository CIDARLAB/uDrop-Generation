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
import numpy
from tqdm import tqdm




class SetupGUI:

	def __init__(self,vid_path):
		self.makeFrames(vid_path)

		self.bound_top_left = [-1,-1]
		self.bound_bottom_right = [-1,-1]
		self.channel_selections = [[0,0],[0,0]]
		self.select_index = 0

		self.root = tkinter.Tk()


		self.pic_index = 0

		raw_image = Image.fromarray(self.buf[self.pic_index])
		self.pic_width, self.pic_height = raw_image.size


		#Resize so our window isn't too big (for high res videos)
		max_size = [1200,600] #Arbitrary
		self.resize_ratio = 1
		if(self.pic_width > max_size[0]):
			self.resize_ratio = self.pic_width / max_size[0]

		if(self.pic_height > max_size[1]):
			self.resize_ratio = max(self.resize_ratio,self.pic_height/max_size[1])


		if(self.resize_ratio > 1):
			raw_image = raw_image.resize((round(self.pic_width / self.resize_ratio), round(self.pic_height / self.resize_ratio)))

		self.img = ImageTk.PhotoImage(raw_image)

		canvas_frame = tkinter.Frame(self.root) 
		canvas_frame.pack()


		fps_frame = tkinter.Frame(self.root)
		fps_frame.pack(side="top")
		fps_label = tkinter.Text(fps_frame,width=30,height=1)
		fps_label.pack(side="left")
		fps_label.insert("end","Frames per Second:")
		fps_label.configure(state="disabled")
		self.fps_entry = tkinter.Entry(fps_frame)
		self.fps_entry.pack(side="left")

		channel_frame = tkinter.Frame(self.root)
		channel_frame.pack(side="top")
		channel_label = tkinter.Text(channel_frame,width=30,height=1)
		channel_label.pack(side="left")
		channel_label.insert("end","Channel Width in micrometers:")
		channel_label.configure(state="disabled")
		self.channel_entry = tkinter.Entry(channel_frame)
		self.channel_entry.pack(side="left")

		image_selection_frame = tkinter.Frame(self.root)
		image_selection_frame.pack(side="top")
		next_button = ttk.Button(image_selection_frame, text='Next Frame',command = self.nextImage)
		next_button.pack(side="left")
		previous_button = ttk.Button(image_selection_frame, text='Previous Frame',command = self.previousImage)
		previous_button.pack(side="left")


		command_frame = tkinter.Frame(self.root)
		command_frame.pack(side="top")
		confirm_button = ttk.Button(command_frame, text='Confirm',command = self.confirm)
		confirm_button.pack(side="left")
		cancel_button = ttk.Button(command_frame, text='Cancel',command = self.cancel)
		cancel_button.pack(side="left")




		self.canvas = tkinter.Canvas(canvas_frame,width = self.pic_width / self.resize_ratio, height = self.pic_height / self.resize_ratio)
		self.canvas.pack(side="top")
		self.canvas.bind('<Button-1>', self.click)
		self.drawGUI()

		self.root.mainloop()



	def makeFrames(self,vid_path):
		if(vid_path!=""):
			if(os.path.exists("frames/")):
				shutil.rmtree('frames/')
			os.mkdir('frames/')
			os.system("ffmpeg -i " + vid_path + " -vf mpdecimate,setpts=N/FRAME_RATE/TB frames/%03d.png")

		frame_list = sorted(os.listdir("frames"))
		self.buf = numpy.empty((len(frame_list),),dtype = numpy.object_)
		self.buf.fill([])
		for i in tqdm(range(len(frame_list))):
			self.buf[i] = cv2.imread("frames/"+frame_list[i])





	def click(self,event):
		modx = event.x * self.resize_ratio
		mody = event.y * self.resize_ratio
		if self.select_index == 0:
			self.bound_top_left = [modx,mody]
			self.select_index += 1
		elif self.select_index == 1:
			self.bound_bottom_right = [modx,mody]
			self.drawGUI()
			self.select_index += 1
		elif self.select_index == 2:
			self.channel_selections[0][0] = modx
			self.channel_selections[0][1] = mody
			self.select_index += 1
		elif self.select_index == 3:
			self.channel_selections[1][0] = modx
			self.channel_selections[1][1] = mody
			self.select_index += 1
		self.drawGUI()


		

	def confirm(self):
		if(self.select_index != 4):
			return

		start_x = round(self.bound_top_left[0])
		start_y = round(self.bound_top_left[1])
		width = round(abs(self.bound_bottom_right[0] - self.bound_top_left[0]))
		height = round(abs(self.bound_bottom_right[1] - self.bound_top_left[1]))
		fps = float(self.fps_entry.get())
		channel_size_mm = float(self.channel_entry.get())
		channel_size_px = float(abs(self.channel_selections[1][1] - self.channel_selections[0][1]))
		conversion_factor = channel_size_mm/channel_size_px

		self.root.destroy()

		subprocess.call("python3 uDROP_Generation_Standalone.py " + " ".join([str(fps),str(conversion_factor),str(start_x),str(start_y),str(width),str(height)]),shell=True)
		#os.system("python3 generator_mine.py " + " ".join([str(fps),str(conversion_factor),str(start_x),str(start_y),str(width),str(height)]))

	def drawGUI(self):
		self.canvas.create_image(0,0, anchor="nw", image=self.img)


		mtext = "Press cancel to start over"
		if self.select_index < 2:
			mtext = "Draw bounding box"
		elif self.select_index < 3:
			mtext = "Draw channel top boundary"
		elif self.select_index < 4:
			mtext = "Draw channel bottom boundary"

		
		
		self.canvas.create_text(10,10,fill="white",anchor="nw",font="Times 12",text=mtext)	
		



		if(self.select_index>1):
			self.canvas.create_line(self.bound_top_left[0] / self.resize_ratio, self.bound_top_left[1] / self.resize_ratio, self.bound_top_left[0] / self.resize_ratio, self.bound_bottom_right[1] / self.resize_ratio, fill="green")
			self.canvas.create_line(self.bound_top_left[0] / self.resize_ratio, self.bound_top_left[1] / self.resize_ratio, self.bound_bottom_right[0] / self.resize_ratio, self.bound_top_left[1] / self.resize_ratio, fill="green")
			self.canvas.create_line(self.bound_top_left[0] / self.resize_ratio, self.bound_bottom_right[1] / self.resize_ratio, self.bound_bottom_right[0] / self.resize_ratio, self.bound_bottom_right[1] / self.resize_ratio, fill="green")
			self.canvas.create_line(self.bound_bottom_right[0] / self.resize_ratio, self.bound_top_left[1] / self.resize_ratio, self.bound_bottom_right[0] / self.resize_ratio, self.bound_bottom_right[1] / self.resize_ratio, fill="green")
		if(self.select_index>2):
			for i in range(0,self.select_index-2):
				self.canvas.create_line((self.channel_selections[i][0]/self.resize_ratio)-10, (self.channel_selections[i][1]/self.resize_ratio), (self.channel_selections[i][0]/self.resize_ratio)+10, (self.channel_selections[i][1]/self.resize_ratio), fill="red")


	def cancel(self):
		self.select_index = 0
		self.bound_top_left = [-1,-1]
		self.bound_bottom_right = [-1,-1]
		self.drawGUI()


	def nextImage(self):
		self.pic_index+=1
		self.pic_index%=len(self.buf)
		raw_image = Image.fromarray(self.buf[self.pic_index])
		if(self.resize_ratio > 1):
			raw_image = raw_image.resize((round(self.pic_width / self.resize_ratio), round(self.pic_height / self.resize_ratio)))
		self.img = ImageTk.PhotoImage(raw_image)
		self.drawGUI()


	def previousImage(self):
		self.pic_index-=1
		if(self.pic_index < 0):
			self.pic_index=len(self.buf)-1
		raw_image = Image.fromarray(self.buf[self.pic_index])
		if(self.resize_ratio > 1):
			raw_image = raw_image.resize((round(self.pic_width / self.resize_ratio), round(self.pic_height / self.resize_ratio)))
		self.img = ImageTk.PhotoImage(raw_image)
		self.drawGUI()


SetupGUI("")









 





















