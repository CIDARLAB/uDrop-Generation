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
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import math





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
		fps_label = tkinter.Label(fps_frame,width=30,height=1)
		fps_label.pack(side="left")
		fps_label["text"]="Frames per Second:"
		self.fps_entry = tkinter.Entry(fps_frame)
		self.fps_entry.pack(side="left")

		channel_frame = tkinter.Frame(self.root)
		channel_frame.pack(side="top")
		channel_label = tkinter.Label(channel_frame,width=30,height=1)
		channel_label.pack(side="left")
		channel_label["text"]="Channel Width in micrometers:"
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
		analysis_gui = AnalysisGUI(analysis)

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
		self.defaultParams()
		self.runAnalysis()

	def defaultParams(self):
		self.canny_weights = [100,40] #Default tolerances for edge detection
		self.wave_weights = [0.5,0.25,0.1] #Wave smoothing default weights

		#Read user parameters defaults if they choose to change them
		if(os.path.exists("params.cfg")):
			with open("params.cfg","r") as f:
				lines = f.readlines()
				self.canny_weights = [float(x.replace("\n","")) for x in lines[0].split(",")]
				self.wave_weights = [float(x.replace("\n","")) for x in lines[1].split(",")]

	def runAnalysis(self):
		self.resetOutputDir()
		self.createWave()
		self.smoothWave()
		self.getWaveMaxes()
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

		if(len(contours) == 0):
			filled_img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
			area = 0
			return area,filled_img
		                

		#Concatenate all their verticies (useful for multiple islands)
		outer_contour = numpy.concatenate(numpy.asarray(contours))

		#Find the convex hull of the verticies
		hull = cv2.convexHull(outer_contour)

		#Find the area of our droplet
		area = cv2.contourArea(hull)


		filled_img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
		cv2.drawContours(filled_img,[hull],0,(255,255,255),cv2.FILLED)

		return area,filled_img




	#Used for calculating the period of the wave
	def getWaveMaxes(self):

		#We track the start valley and end valley so that we don't cut off calculations in the middle of a wave
		#This shouldn't matter if we have sufficiently large data, but let's keep it in the spirit of accuracy
		self.start_valley = 0
		self.end_valley = 0


		#indexes of all maxes in the data set
		self.max_list = []



		#Loop through our wave data and get the maxes
		maxes_count = 0
		for i in range(1,len(self.avg_pixel_vals)-1):
			if self.avg_pixel_vals[i] > self.avg_pixel_vals[i-1] and self.avg_pixel_vals[i] > self.avg_pixel_vals[i+1] and self.start_valley != 0: #Local max inside our range
				self.max_list.append(i)
			elif self.avg_pixel_vals[i] <= self.avg_pixel_vals[i-1] and self.avg_pixel_vals[i] <= self.avg_pixel_vals[i+1]: #Local min
				if(self.start_valley == 0):
					self.start_valley = i
				else:
					self.end_valley = i


		if(self.max_list[-1] > self.end_valley):
			self.max_list = self.max_list[:-1] #We don't want to count maxes outside of the range of the data we are looking at



	
	def getDropRate(self):
		self.drops_per_second = (len(self.max_list) / (float)(self.end_valley - self.start_valley)) * self.fps #Peaks per frames * frames per second = Peaks/seconds
		
	

	
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



class AnalysisGUI:
	def __init__(self,analysis_obj):

		#Arbitrary. Just for visualization. Keeps the graph reasonably small
		self.frame_limit = 25
		self.begin_frame = 0 #End frame is frame_limit away from this variable	

		self.ao = analysis_obj

		self.root = tkinter.Tk()

		self.frame_index = 0;


		maxes_frame = tkinter.Frame(self.root)
		maxes_frame.pack(side="top")
		maxes_label = tkinter.Label(maxes_frame,width=30,height=1)
		maxes_label.pack(side="left")
		maxes_label["text"]="List of Local Maxes: "
		self.maxes_entry = tkinter.Entry(maxes_frame)
		self.maxes_entry.pack(side="left")
		self.maxes_entry.insert("end",",".join([str(x) for x in self.ao.max_list]))
		maxes_apply_button = ttk.Button(maxes_frame, text='Apply',command = self.updateMaxes)
		maxes_apply_button.pack(side="left")
		maxes_recalculate_button = ttk.Button(maxes_frame, text='Recalculate',command = self.recalculateMaxes)
		maxes_recalculate_button.pack(side="left")

		valley_frame = tkinter.Frame(self.root)
		valley_frame.pack(side="top")
		valley_label = tkinter.Label(valley_frame,width=30,height=1)
		valley_label.pack(side="left")
		valley_label["text"]="Start and end valleys: "
		self.valley_entry = tkinter.Entry(valley_frame)
		self.valley_entry.pack(side="left")
		self.valley_entry.insert("end",",".join([str(self.ao.start_valley),str(self.ao.end_valley)]))
		valley_apply_button = ttk.Button(valley_frame, text='Apply',command = self.updateValleys)
		valley_apply_button.pack(side="left")
		valley_recalculate_button = ttk.Button(valley_frame, text='Recalculate',command = self.recalculateValleys)
		valley_recalculate_button.pack(side="left")

		rate_frame = tkinter.Frame(self.root)
		rate_frame.pack(side="top")
		self.rate_label = tkinter.Label(rate_frame,width=30,height=1)
		self.rate_label.pack(side="left")
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		area_frame = tkinter.Frame(self.root)
		area_frame.pack(side="top")
		area_label = tkinter.Label(area_frame,width=30,height=1)
		area_label.pack(side="left")
		area_label["text"]="Droplet areas (px): "
		self.area_entry = tkinter.Entry(area_frame)
		self.area_entry.pack(side="left")
		self.area_entry.insert("end",",".join([str(x) for x in self.ao.drop_areas]))
		area_apply_button = ttk.Button(area_frame, text='Apply',command = self.updateAreas)
		area_apply_button.pack(side="left")
		area_recalculate_button = ttk.Button(area_frame, text='Recalculate',command = self.recalculateAreas)
		area_recalculate_button.pack(side="left")

		areamean_frame = tkinter.Frame(self.root)
		areamean_frame.pack(side="top")
		self.areamean_label = tkinter.Label(areamean_frame,width=30,height=1)
		self.areamean_label.pack(side="left")
		self.areamean_label["text"]="Mean area: "+ str(numpy.asarray(self.ao.drop_areas).mean())

		waveweight_frame = tkinter.Frame(self.root)
		waveweight_frame.pack(side="top")
		waveweight_label = tkinter.Label(waveweight_frame,width=30,height=1)
		waveweight_label.pack(side="left")
		waveweight_label["text"]="Wave smoothing weights: "
		self.waveweight_entry = tkinter.Entry(waveweight_frame)
		self.waveweight_entry.pack(side="left")
		self.waveweight_entry.insert("end",",".join([str(x) for x in self.ao.wave_weights]))
		waveweight_reset_button = ttk.Button(waveweight_frame, text='Reset',command = self.resetWaveWeights)
		waveweight_reset_button.pack(side="left")

		canny_frame = tkinter.Frame(self.root)
		canny_frame.pack(side="top")
		canny_label = tkinter.Label(canny_frame,width=30,height=1)
		canny_label.pack(side="left")
		canny_label["text"]="Edge detection thresholds: "
		self.canny_entry = tkinter.Entry(canny_frame)
		self.canny_entry.pack(side="left")
		self.canny_entry.insert("end",",".join([str(x) for x in self.ao.canny_weights]))
		canny_reset_button = ttk.Button(canny_frame, text='Reset',command = self.resetCannyWeights)
		canny_reset_button.pack(side="left")

		rerun_frame = tkinter.Frame(self.root)
		rerun_frame.pack(side="top")
		rerun_button = ttk.Button(rerun_frame, text='Rerun with new params', command = self.rerunAnalysis)
		rerun_button.pack(side="left")

		fnav_frame = tkinter.Frame(self.root)
		fnav_frame.pack(side="top")
		fnav_previous_max = ttk.Button(fnav_frame, text='<< Previous Max',command = self.previousMax)
		fnav_previous_max.pack(side="left")
		fnav_previous_frame = ttk.Button(fnav_frame, text='< Previous Frame',command = self.previousFrame)
		fnav_previous_frame.pack(side="left")
		self.fnav_label = tkinter.Label(fnav_frame,width=5,height=1)
		self.fnav_label.pack(side="left")
		self.fnav_label["text"]="0"
		fnav_next_frame = ttk.Button(fnav_frame, text='> Next Frame',command = self.nextFrame)
		fnav_next_frame.pack(side="left")
		fnav_next_max = ttk.Button(fnav_frame, text='>> Next Max',command = self.nextMax)
		fnav_next_max.pack(side="left")



		figure = Figure(figsize=(5,2), dpi=100)
		self.wave_plot = figure.add_subplot(111)
		self.figure_canvas = FigureCanvasTkAgg(figure, self.root)
		self.figure_canvas.show()
		self.figure_canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

		self.image_frame = tkinter.Frame(self.root)
		self.image_frame.pack(side="bottom")


		self.raw_canvas = tkinter.Canvas(self.image_frame,width = 50,height=50)
		self.raw_canvas.pack(side="left")

		self.edge_canvas = tkinter.Canvas(self.image_frame,width=50,height=50)
		self.edge_canvas.pack(side="left")

		self.filled_canvas = tkinter.Canvas(self.image_frame,width=50,height=50)
		self.filled_canvas.pack(side="left")


		                
		self.redrawCanvases()
		self.root.mainloop()
	
	def nextMax(self):
		nextIndex = self.frame_index
		for m in sorted(self.ao.max_list):
			if m > nextIndex:
				nextIndex = m
				break
		self.frame_index = nextIndex
		self.redrawCanvases()

	def previousMax(self):
		nextIndex = self.frame_index
		for m in sorted(self.ao.max_list,reverse=True):
			if m < nextIndex:
				nextIndex = m
				break
		self.frame_index = nextIndex
		self.redrawCanvases()

	def nextFrame(self):
		self.frame_index += 1
		self.frame_index %= len(self.ao.avg_pixel_vals)
		self.redrawCanvases()

	def previousFrame(self):
		self.frame_index -= 1
		if(self.frame_index<0):
			self.frame_index = len(self.ao.avg_pixel_vals) - 1
		self.redrawCanvases()





	def updateMaxes(self):
		self.ao.max_list = [int(x) for x in self.maxes_entry.get().split(",")]

		self.ao.getDropRate()
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		self.redrawCanvases()
	
	def recalculateMaxes(self):
		old_start_val = self.ao.start_valley
		old_end_val = self.ao.end_valley

		self.ao.getWaveMaxes()

		#Don't reset the valleys when we just want to calculate maxes
		self.ao.start_valley = old_start_val
		self.ao.end_valley = old_end_val

		self.ao.getDropRate()
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		self.maxes_entry.delete(0,"end")
		self.maxes_entry.insert("end",",".join([str(x) for x in self.ao.max_list]))
		self.redrawCanvases()

	def updateValleys(self):
		valleys = [int(x) for x in self.valley_entry.get().split(",")]
		self.ao.start_valley = valleys[0] 
		self.ao.end_valley = valleys[1] 

		self.ao.getDropRate()
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		self.redrawCanvases()
	
	def recalculateValleys(self):
		old_max_list = self.ao.max_list[:]

		self.ao.getWaveMaxes()
	
		#Don't reset the maxes when we just want to calculate valleys
		self.ao.max_list = old_max_list[:]

		self.ao.getDropRate()
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		self.maxes_entry.delete(0,"end")
		self.maxes_entry.insert("end",",".join([str(x) for x in self.ao.max_list]))
		self.redrawCanvases()

	def updateAreas(self):
		self.ao.drop_areas = [float(x) for x in self.area_entry.get().split(",")]
		self.redrawCanvases()
		self.areamean_label["text"]="Mean area: "+ str(numpy.asarray(self.ao.drop_areas).mean())
	
	def recalculateAreas(self):
		self.ao.getAllAreas()
		self.area_entry.delete(0,"end")
		self.area_entry.insert("end",",".join([str(x) for x in self.ao.drop_areas]))
		self.areamean_label["text"]="Mean area: "+ str(numpy.asarray(self.ao.drop_areas).mean())
		self.redrawCanvases()

	def resetWaveWeights(self):
		old_canny_weights = self.ao.canny_weights[:]

		self.ao.defaultParams()

		#Don't reset the canny weights when we want to reset the wave weights
		self.ao.canny_weights = old_canny_weights[:]

		self.waveweight_entry.delete(0,"end")
		self.waveweight_entry.insert("end",",".join([str(x) for x in self.ao.wave_weights]))

	def resetCannyWeights(self):
		old_wave_weights = self.ao.wave_weights[:]

		self.ao.defaultParams()

		#Don't reset the wave weights when we want to reset the canny weights
		self.ao.wave_weights = old_wave_weights[:]

		self.canny_entry.delete(0,"end")
		self.canny_entry.insert("end",",".join([str(x) for x in self.ao.wave_weights]))

	def rerunAnalysis(self):
		self.ao.canny_weights = [float(x) for x in self.canny_entry.get().split(",")]
		self.ao.wave_weights = [float(x) for x in self.waveweight_entry.get().split(",")]
		self.ao.runAnalysis()
		self.redrawCanvases()


	def redrawCanvases(self):
		self.wave_plot.clear()


		self.wave_plot.plot([x for x in range(len(self.ao.avg_pixel_vals))],self.ao.avg_pixel_vals)
		self.wave_plot.plot(self.ao.max_list,[self.ao.avg_pixel_vals[x] for x in self.ao.max_list],"ro")
		self.wave_plot.plot([self.ao.start_valley,self.ao.end_valley],[self.ao.avg_pixel_vals[x] for x in [self.ao.start_valley,self.ao.end_valley]],"go")
		self.figure_canvas.draw()


		raw_arr = self.ao.raw_frames[self.frame_index]
		self.raw_img = ImageTk.PhotoImage(Image.fromarray(raw_arr))
		self.raw_canvas.create_image(0,0, anchor="nw", image=self.raw_img)

		edge_arr = self.ao.cleanFrame(raw_arr)
		self.edge_img = ImageTk.PhotoImage(Image.fromarray(edge_arr))
		self.edge_canvas.create_image(0,0, anchor="nw", image=self.edge_img)

		filled_arr = self.ao.getArea(edge_arr)[1]
		self.filled_img = ImageTk.PhotoImage(Image.fromarray(filled_arr))
		self.filled_canvas.create_image(0,0, anchor="nw", image=self.filled_img)


		self.fnav_label["text"] = str(self.frame_index)
		if(self.frame_index in self.ao.max_list):
			self.fnav_label.configure(foreground="red")
		else:
			self.fnav_label.configure(foreground="black")




#Commands to run when script is called
SetupGUI("")
