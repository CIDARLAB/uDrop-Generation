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

		analysis = Analysis(fps,conversion_factor,start_x,start_y,width,height,self.buf)	

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











 

class Analysis:	

	def __init__(self,fps,conversion_factor,start_x,start_y,width,height,buf):
		self.fps = fps
		self.conversion_factor = conversion_factor
		self.start_x = start_x
		self.start_y = start_y
		self.width = width
		self.height = height
		self.raw_frames = [ b[start_y:start_y+height, start_x:start_x+width] for b in buf]
		self.runAnalysis()

	def runAnalysis(self):
		self.resetOutputDir()
		self.readParams()
		self.createWave()
		self.smoothWave()
		self.getDropRate()
		self.getAllAreas()
		self.writeOutputs()

	def resetOutputDir(self):
		#Reset the output directory
		if(os.path.exists("output/")):
			shutil.rmtree('output/')
		os.mkdir('output/')
		os.mkdir('output/cleaned/')
		os.mkdir('output/raw/')


	def readParams(self):
		self.canny_weights = [100,40] #Default tolerances for edge detection
		self.wave_weights = [0.5,0.25,0.1] #Wave smoothing default weights

		#Read user parameters if they choose to change them
		if(os.path.exists("params.cfg")):
			with open("params.cfg","r") as f:
				lines = f.readlines()
				self.canny_weights = [float(x.replace("\n","")) for x in lines[0].split(",")]
				self.wave_weights = [float(x.replace("\n","")) for x in lines[1].split(",")]


	#Apply blurs and edge detection to frame
	def cleanFrame(self,focus_frame):
		return cv2.Canny(cv2.blur(focus_frame,(4,4)),self.canny_weights[0],self.canny_weights[1])

	def createWave(self):
		self.avg_pixel_vals = []
		self.edge_detected_frames = []
		#Find the average pixel value of a band of pixels at the nozzle exit
		#This band should get higher as a drop passes through it
		#We build a wave of values from this that should look like a sin wave if everything went well
		count = 0
		for b in self.raw_frames:

			#Canny's edge detection on each frame
			arr = self.cleanFrame(b)
			self.edge_detected_frames.append(arr)

			this_avg = numpy.mean(arr);
			self.avg_pixel_vals.append(this_avg)
			
			with open("output/sin.csv","a") as fout:
				fout.write(str(count)+","+str(this_avg)+"\n")
			count+=1




	def smoothWave(self):
		newdatlist = []

		#Basically, we assign every point a new value that is equal to the weighted average of its neighbors
		for i in range(0,len(self.avg_pixel_vals)):
			if i>=len(self.wave_weights) and i<=len(self.avg_pixel_vals)-1-len(self.wave_weights):
				val = self.avg_pixel_vals[i]
				for j in range(1,len(self.wave_weights)+1):
					val+=self.avg_pixel_vals[i+j]*self.wave_weights[j-1]
					val+=self.avg_pixel_vals[i-j]*self.wave_weights[j-1]
				val /= (float)(sum(self.wave_weights)*2 + 1)
				newdatlist.append(val)
			else:
				newdatlist.append(self.avg_pixel_vals[i])	
			with open("output/sin_smooth.csv","a") as fout:
				fout.write(str(i)+","+str(newdatlist[i])+"\n")
		self.avg_pixel_vals = newdatlist








	#Function for calculating areas of droplets
	def getArea(self,arr):
		#Find contours
		im2, contours, hierarchy = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#Concatenate all their verticies (useful for multiple islands)
		outer_contour = numpy.concatenate(numpy.asarray(contours))

		#Find the convex hull of the verticies
		hull = cv2.convexHull(outer_contour)

		#Find the area of our droplet
		area = cv2.contourArea(hull)


		filled_img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
		cv2.drawContours(filled_img,[hull],0,(255,255,255),cv2.FILLED)

		return area,filled_img




	#Drop rate calculations (Basically calculating the period of the wave)
	def getDropRate(self):

		#We track the start valley and end valley so that we don't cut off calculations in the middle of a wave
		#This shouldn't matter if we have sufficiently large data, but let's keep it in the spirit of accuracy
		start_valley = 0
		end_valley = 0

		last_max = 0

		#indexes of all maxes in the data set
		self.max_list = []



		#Loop through our wave data and get the maxes
		maxes_count = 0
		for i in range(1,len(self.avg_pixel_vals)-1):
			if self.avg_pixel_vals[i] > self.avg_pixel_vals[i-1] and self.avg_pixel_vals[i] > self.avg_pixel_vals[i+1] and start_valley != 0: #Local max inside our range
				maxes_count += 1
				last_max = i
				self.max_list.append(i)
			elif self.avg_pixel_vals[i] <= self.avg_pixel_vals[i-1] and self.avg_pixel_vals[i] <= self.avg_pixel_vals[i+1]: #Local min
				if(start_valley == 0):
					start_valley = i
				else:
					end_valley = i


		if(last_max > end_valley):
			maxes_count -= 1 #We don't want to count maxes outside of the range of the data we are looking at



		self.drops_per_second = (maxes_count / (float)(end_valley - start_valley)) * self.fps #Peaks per frames * frames per second = Peaks/seconds
	
	def getAllAreas(self):
		self.drop_areas = []
		self.filled_frames = []
		for i in self.max_list:
			area,img = self.getArea(self.edge_detected_frames[i])	
			self.drop_areas.append(area)
			self.filled_frames.append(img)
	

	def writeOutputs(self):
		#stdout outputs
		drop_areas_np = numpy.asarray(self.drop_areas)
		print("Drops per Second: " + str(self.drops_per_second))
		print("Average Drop Area (pixels): " + str(drop_areas_np.mean()))
		print("Average Drop Area (micrometers): " + str(drop_areas_np.mean()*(self.conversion_factor**2)))
		print("Drop Area Standard Deviation (pixels): " + str(drop_areas_np.std()))
		print("Drop Area Standard Deviation (micrometers): " + str(drop_areas_np.std()*(self.conversion_factor**2)))

		for i,f in enumerate(self.raw_frames):
			Image.fromarray(f).save("output/raw/"+str(i)+"regular"+".png")

		for i,f in enumerate(self.filled_frames):
			Image.fromarray(f).save("output/cleaned/"+str(i)+"cleaned"+".png")


		#disk outputs
		with open("output/drop_data.csv","w") as f:
			f.write(str(drop_areas_np.mean())+"," + str(drop_areas_np.mean()*self.conversion_factor**2) + "," + str(drop_areas_np.std()) + "," + str(drop_areas_np.std()*self.conversion_factor**2) + "," + str(self.drops_per_second) + "\n")


		with open("output/drop_data_raw.csv","w") as f:
			f.write(",".join([str(x) for x in drop_areas_np]) + "\n")





#Commands to run when script is called
SetupGUI("")
