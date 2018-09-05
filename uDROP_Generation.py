"""GUI display for uDROP setup
Usage: python3 uDROP_Generation.py name_of_video.mp4
Note that you can omit the video path if you already have something in your frames folder that you'd like to evaluate.
"""


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
	"""Class to display uDROP setup"""

	def __init__(self,vid_path):
		"""Set up the GUI"""

		self.vid_path = vid_path
		#Run the ffmpeg command in console
		self.makeFrames(vid_path)

		#Init user selections
		self.bound_top_left = [-1,-1]
		self.bound_bottom_right = [-1,-1]
		self.channel_selections = [[0,0],[0,0],[0,0]]
		self.select_index = 0

		self.root = tkinter.Tk()
		self.root.title("µ-DROP Setup")


		self.pic_index = 0

		#Load the first frame
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


		#Display the first frame
		self.img = ImageTk.PhotoImage(raw_image)

		canvas_frame = tkinter.Frame(self.root) 
		canvas_frame.pack()


		#Pack in all our user entries

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
		channel_label["text"]="Blue line distance in µm: "
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




		#Pack in the canvas
		self.canvas = tkinter.Canvas(canvas_frame,width = self.pic_width / self.resize_ratio, height = self.pic_height / self.resize_ratio)
		self.canvas.pack(side="top")
		self.canvas.bind('<Button-1>', self.click)
		self.drawGUI()

		self.root.mainloop()



	def makeFrames(self,vid_path):
		"""Save frames to buf and write them to disk if applicable"""

		#If there is a video path, rewrite the frames directory
		if(vid_path!=""):
			if(os.path.exists("frames/")):
				shutil.rmtree('frames/')
			os.mkdir('frames/')
			os.system("ffmpeg -i " + vid_path + " -vf mpdecimate,setpts=N/FRAME_RATE/TB frames/%05d.png")

		#Save all frames to buf in order
		frame_list = sorted(os.listdir("frames"))
		self.buf = numpy.empty((len(frame_list),),dtype = numpy.object_)
		self.buf.fill([])
		for i in tqdm(range(len(frame_list))):
			self.buf[i] = cv2.imread("frames/"+frame_list[i])





	def click(self,event):
		"""Handle user input"""

		# Log the click location in it's appropiate spot
		# Bounding box top left first
		# Then bottom right
		# Then channel selection line begin
		# Then the line end
		# Then the line offset
		if self.select_index == 0:
			self.bound_top_left = [event.x,event.y]
			self.select_index += 1
		elif self.select_index == 1:
			self.bound_bottom_right = [event.x,event.y]
			self.drawGUI()
			self.select_index += 1
		elif self.select_index == 2:
			self.channel_selections[0][0] = event.x
			self.channel_selections[0][1] = event.y
			self.select_index += 1
		elif self.select_index == 3:
			self.channel_selections[1][0] = event.x
			self.channel_selections[1][1] = event.y
			self.select_index += 1
		elif self.select_index == 4:
			self.channel_selections[2][0] = event.x
			self.channel_selections[2][1] = event.y
			self.getIntersectPoint()
			self.select_index += 1
		self.drawGUI()

	def getIntersectPoint(self):
		"""Return the intersection of the line segment with the perpendicular line that runs through the third point"""
	
		#If vertical
		if self.channel_selections[0][0] == self.channel_selections[1][0]:
			#Return the y position of the third point and the x position of the line
			self.intersect_point = [self.channel_selections[0][0],self.channel_selections[2][1]]
		
		#If horizontal
		elif self.channel_selections[0][1] == self.channel_selections[1][1]:
			#Return the x position of the third point and the y position of the line
			self.intersect_point = [self.channel_selections[2][0],self.channel_selections[0][1]]

		#Otherwise, do a classic point slope intersection equation
		else:
			slope1 = (self.channel_selections[0][1] - self.channel_selections[1][1]) / (self.channel_selections[0][0] - self.channel_selections[1][0])
			coeff1 = slope1
			b1 = (slope1 * -self.channel_selections[0][0]) + self.channel_selections[0][1]

			slope2 = -1/slope1
			coeff2 = slope2
			b2 = (slope2 * -self.channel_selections[2][0]) + self.channel_selections[2][1]

			intersectionX = (b1-b2)/(coeff2-coeff1)
			intersectionY = coeff1*intersectionX + b1

			self.intersect_point = [intersectionX,intersectionY]


		

	def confirm(self):
		"""Submit input parameters to be analyzed"""

		#Don't do anything unless the user is finished drawing
		if(self.select_index != 5):
			return

		#Parse all the user inputs
		start_x = round(self.bound_top_left[0]*self.resize_ratio)
		start_y = round(self.bound_top_left[1]*self.resize_ratio)
		width = round(abs(self.bound_bottom_right[0] - self.bound_top_left[0])*self.resize_ratio)
		height = round(abs(self.bound_bottom_right[1] - self.bound_top_left[1])*self.resize_ratio)
		fps = float(self.fps_entry.get())
		channel_size_mm = float(self.channel_entry.get())
		channel_size_px = (((self.intersect_point[0]-self.channel_selections[2][0])**2 + (self.intersect_point[1]-self.channel_selections[2][1])**2)**0.5)*self.resize_ratio
		conversion_factor = channel_size_mm/channel_size_px

		#Destroy the GUI window
		self.root.destroy()

		#Run the analysis
		analysis = Analysis(fps,conversion_factor,start_x,start_y,width,height,self.buf,self.vid_path)	

		#Display the analysis GUI
		analysis_gui = AnalysisGUI(analysis)

	def drawGUI(self):
		"""Draw thee elements in the canvas that need to be displayed"""
		
		#Actually draw the current frame
		self.canvas.create_image(0,0, anchor="nw", image=self.img)


		#Helpful text
		mtext = "Press cancel to start over"
		if self.select_index < 2:
			mtext = "Draw bounding box"
		elif self.select_index < 3:
			mtext = "Draw start of line segment"
		elif self.select_index < 4:
			mtext = "Draw end of line segment"
		elif self.select_index < 5:
			mtext = "Draw point off of segment"
		
		self.canvas.create_text(10,10,fill="white",anchor="nw",font="Times 12",text=mtext)	
		

		
		#Draw the bounding box
		if(self.select_index>1):
			self.canvas.create_line(self.bound_top_left[0], self.bound_top_left[1], self.bound_top_left[0], self.bound_bottom_right[1], fill="green")
			self.canvas.create_line(self.bound_top_left[0], self.bound_top_left[1], self.bound_bottom_right[0], self.bound_top_left[1], fill="green")
			self.canvas.create_line(self.bound_top_left[0], self.bound_bottom_right[1], self.bound_bottom_right[0], self.bound_bottom_right[1], fill="green")
			self.canvas.create_line(self.bound_bottom_right[0], self.bound_top_left[1], self.bound_bottom_right[0], self.bound_bottom_right[1], fill="green")

		#Draw the user selected line segment points
		if(self.select_index>2):
			for i in range(0,self.select_index-2):
				#These 2 lines make an x
				self.canvas.create_line((self.channel_selections[i][0])-3, (self.channel_selections[i][1])-3, (self.channel_selections[i][0])+3, (self.channel_selections[i][1])+3, fill="red")
				self.canvas.create_line((self.channel_selections[i][0])-3, (self.channel_selections[i][1])+3, (self.channel_selections[i][0])+3, (self.channel_selections[i][1])-3, fill="red")

		#Draw the line segment
		if(self.select_index>3):
			self.canvas.create_line(self.channel_selections[0][0],self.channel_selections[0][1],self.channel_selections[1][0],self.channel_selections[1][1],fill="red")

		#Draw the intersection point, and the perpendicular distance to the third point
		if(self.select_index>4):
			self.canvas.create_line((self.intersect_point[0])-3, (self.intersect_point[1])+3, (self.intersect_point[0])+3, (self.intersect_point[1])-3, fill="blue")
			self.canvas.create_line((self.intersect_point[0])-3, (self.intersect_point[1])-3, (self.intersect_point[0])+3, (self.intersect_point[1])+3, fill="blue")

			self.canvas.create_line(self.intersect_point[0], self.intersect_point[1], self.channel_selections[2][0], self.channel_selections[2][1], fill="blue",dash=(3,2))

	def cancel(self):
		"""Reset all user inputs"""
		self.select_index = 0
		self.bound_top_left = [-1,-1]
		self.bound_bottom_right = [-1,-1]
		self.drawGUI()


	def nextImage(self):
		"""Display the next frame"""
		self.pic_index+=1
		self.pic_index%=len(self.buf)
		raw_image = Image.fromarray(self.buf[self.pic_index])
		if(self.resize_ratio > 1):
			raw_image = raw_image.resize((round(self.pic_width / self.resize_ratio), round(self.pic_height / self.resize_ratio)))
		self.img = ImageTk.PhotoImage(raw_image)
		self.drawGUI()


	def previousImage(self):
		"""Display the last frame"""
		self.pic_index-=1
		if(self.pic_index < 0):
			self.pic_index=len(self.buf)-1
		raw_image = Image.fromarray(self.buf[self.pic_index])
		if(self.resize_ratio > 1):
			raw_image = raw_image.resize((round(self.pic_width / self.resize_ratio), round(self.pic_height / self.resize_ratio)))
		self.img = ImageTk.PhotoImage(raw_image)
		self.drawGUI()










 

class Analysis:	
	"""Backend class for all our video processing"""
	def __init__(self,fps,conversion_factor,start_x,start_y,width,height,buf,vid_name):
		"""Initialize the analysis with the user inputted data"""		
		self.fps = fps
		self.conversion_factor = conversion_factor
		self.start_x = start_x
		self.start_y = start_y
		self.width = width
		self.height = height
		self.raw_frames = [ b[start_y:start_y+height, start_x:start_x+width] for b in buf]
		self.vid_name = vid_name
		self.defaultParams()
		self.runAnalysis()

	def defaultParams(self):
		"""Read the default params from params.cfg
		Uses hardcoded defaults if it fails to find params.cfg
		"""
	
		self.canny_weights = [100,40] #Default tolerances for edge detection
		self.wave_weights = [0.5,0.25,0.1] #Wave smoothing default weights

		#Read user parameters defaults if they choose to change them
		if(os.path.exists("params.cfg")):
			with open("params.cfg","r") as f:
				lines = f.readlines()
				self.canny_weights = [float(x.replace("\n","")) for x in lines[0].split(",")]
				self.wave_weights = [float(x.replace("\n","")) for x in lines[1].split(",")]

	def runAnalysis(self):
		"""Run the video processing algorithm"""
		self.resetOutputDir()
		self.createWave()
		self.smoothWave()
		self.getWaveMaxes()
		self.getDropRate()
		self.getAllDiameters()
		self.writeOutputs()

	def resetOutputDir(self):
		"""Clear the output directory if it exists and populate it with empty folders"""

		#Reset the output directory
		if(os.path.exists("output/")):
			shutil.rmtree('output/')
		os.mkdir('output/')
		os.mkdir('output/cleaned/')
		os.mkdir('output/raw/')
		os.mkdir('output/edge/')



	def cleanFrame(self,focus_frame):
		"""Apply blurs and edge detection to frame"""
		return cv2.Canny(cv2.blur(focus_frame,(4,4)),self.canny_weights[0],self.canny_weights[1])

	def createWave(self):
		"""Create a wave of data from the average pixel values of each edge detected frame"""

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
		"""Apply wave smoothing to a wave of data"""
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








	def getDiameter(self,arr):
		"""Calculate diameter of edge detected droplet"""
		
		#Find contours
		im2, contours, hierarchy = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if(len(contours) == 0):
			filled_img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
			diameter = 0
			return diameter,filled_img
		                

		#Concatenate all their verticies (useful for multiple islands)
		outer_contour = numpy.concatenate(numpy.asarray(contours))

		#Find the convex hull of the verticies
		hull = cv2.convexHull(outer_contour)

		#Fit a rectangle to the hull
		rect = cv2.minAreaRect(hull)

		#Get major axis of diameter
		diameter = max(rect[1])


		box = cv2.boxPoints(rect)
		box = numpy.int0(box)
		filled_img = numpy.zeros((arr.shape[0],arr.shape[1],3),dtype='uint8')
		cv2.drawContours(filled_img,[hull],0,(255,255,255),cv2.FILLED)
		cv2.drawContours(filled_img,[box],0,(0,0,255),2)

		return diameter,filled_img




	def getWaveMaxes(self):
		"""Get a list of all local maxes in a wave of data
		Used for calculating the period of a wave and analyzing each droplet size at the local max
		"""
		#We track the start valley and end valley so that we don't cut off calculations in the middle of a wave
		#This shouldn't matter if we have sufficiently large data, but let's keep it in the spirit of accuracy
		self.start_valley = 0
		self.end_valley = 0


		#indexes of all maxes in the data set
		self.max_list = []

		self.meanVal = sum(self.avg_pixel_vals)/len(self.avg_pixel_vals)
		tempValList = []
		#Loop through our wave data and get the maxes
		for i in range(1,len(self.avg_pixel_vals)-1):
			if self.avg_pixel_vals[i] < self.meanVal :

				#Log the one true max that occured above the mean
				if len(tempValList) > 0:
					self.max_list.append(max(tempValList)[1])
					tempValList = []

				#Find if the below average points are a minimum
				if self.avg_pixel_vals[i] <= self.avg_pixel_vals[i-1] and self.avg_pixel_vals[i] <= self.avg_pixel_vals[i+1]: #Local min
					if self.start_valley == 0:
						self.start_valley = i
					else:
						self.end_valley = i
			#Find if the above average points are a maximum
			elif self.avg_pixel_vals[i] >= self.avg_pixel_vals[i-1] and self.avg_pixel_vals[i] >= self.avg_pixel_vals[i+1] and self.start_valley != 0: #Local max inside our range
				#Log all local maxes between when the wave goes above the mean and when it dips below
				tempValList.append((self.avg_pixel_vals[i],i))


		if(self.max_list[-1] > self.end_valley):
			self.max_list = self.max_list[:-1] #We don't want to count maxes outside of the range of the data we are looking at



	
	def getDropRate(self):
		"""Calculate the droplet generation rate"""
		self.drops_per_second = (len(self.max_list) / (float)(self.end_valley - self.start_valley)) * self.fps #Peaks per frames * frames per second = Peaks/seconds
		
	

	
	def getAllDiameters(self):
		"""Calculate the diameters at each local max"""
		self.drop_diameters = []
		self.filled_frames = []
		for i in self.max_list:
			diameter,img = self.getDiameter(self.edge_detected_frames[i])	
			self.drop_diameters.append(diameter)
			self.filled_frames.append(img) #We save the filled frames as a side effect to getting the droplet diameters
	

	def writeOutputs(self):
		"""Write the analysis outputs to standard out and to disk"""
		#stdout outputs
		drop_diameters_np = numpy.asarray(self.drop_diameters)
		print("Drops per Second: " + str(self.drops_per_second))
		print("Average Drop Diameter (pixels): " + str(drop_diameters_np.mean()))
		print("Average Drop Diameter (µm): " + str(drop_diameters_np.mean()*(self.conversion_factor)))
		print("Drop Diameter Standard Deviation (pixels): " + str(drop_diameters_np.std()))
		print("Drop Diameter Standard Deviation (µm): " + str(drop_diameters_np.std()*(self.conversion_factor)))
		print("Conversion Factor: " + str(self.conversion_factor))
		print(self.vid_name + "," + str(self.drops_per_second)+","+str(drop_diameters_np.mean()*(self.conversion_factor)) + "," + str(drop_diameters_np.std()*(self.conversion_factor)) + "," + str(drop_diameters_np.size))

		for i,f in enumerate(self.raw_frames):
			Image.fromarray(f).save("output/raw/"+str(i)+"regular"+".png")

		for i,f in enumerate(self.edge_detected_frames):
			Image.fromarray(f).save("output/edge/"+str(i)+"edge"+".png")

		for i,f in enumerate(self.filled_frames):
			Image.fromarray(f).save("output/cleaned/"+str(i)+"cleaned"+".png")


		#disk outputs
		with open("output/drop_data.csv","w") as f:
			f.write(str(drop_diameters_np.mean())+"," + str(drop_diameters_np.mean()*self.conversion_factor) + "," + str(drop_diameters_np.std()) + "," + str(drop_diameters_np.std()*self.conversion_factor) + "," + str(self.drops_per_second) + "\n")


		with open("output/drop_data_raw.csv","w") as f:
			f.write(",".join([str(x) for x in drop_diameters_np]) + "\n")

		with open("./drop_data.csv","a") as f:
			f.write(self.vid_name + "," + str(self.drops_per_second)+","+str(drop_diameters_np.mean()*(self.conversion_factor)) + "," + str(drop_diameters_np.std()*(self.conversion_factor)) + "," + str(drop_diameters_np.size)+"\n")



class AnalysisGUI:
	"""GUI front end to our analysis"""
	def __init__(self,analysis_obj):
		"""Initialize the GUI
		Args:
			analysis_obj: Analysis instance obtained from first initial run of the data
		"""

		#The frame limit is the amount of frames the graph shows at once. 
		#Just for visualization. Keeps the graph reasonably small.
		self.frame_limit = 50
		self.begin_frame = 0 #End frame is frame_limit away from this variable	

		self.ao = analysis_obj

		self.root = tkinter.Tk()
		self.root.title("µ-DROP Analysis")

		self.frame_index = 0;

		#Package the entries together 
		maxes_frame = tkinter.Frame(self.root)
		maxes_frame.pack(side="top")
		maxes_label = tkinter.Label(maxes_frame)
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
		valley_label = tkinter.Label(valley_frame)
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

		diameter_frame = tkinter.Frame(self.root)
		diameter_frame.pack(side="top")
		diameter_label = tkinter.Label(diameter_frame)
		diameter_label.pack(side="left")
		diameter_label["text"]="Droplet diameters: "
		self.diameter_entry = tkinter.Entry(diameter_frame)
		self.diameter_entry.pack(side="left")
		self.diameter_entry.insert("end",",".join([str(x) for x in self.ao.drop_diameters]))
		diameter_apply_button = ttk.Button(diameter_frame, text='Apply',command = self.updateDiameters)
		diameter_apply_button.pack(side="left")
		diameter_recalculate_button = ttk.Button(diameter_frame, text='Recalculate',command = self.recalculateDiameters)
		diameter_recalculate_button.pack(side="left")

		waveweight_frame = tkinter.Frame(self.root)
		waveweight_frame.pack(side="top")
		waveweight_label = tkinter.Label(waveweight_frame)
		waveweight_label.pack(side="left")
		waveweight_label["text"]="Wave smoothing weights: "
		self.waveweight_entry = tkinter.Entry(waveweight_frame)
		self.waveweight_entry.pack(side="left")
		self.waveweight_entry.insert("end",",".join([str(x) for x in self.ao.wave_weights]))
		waveweight_reset_button = ttk.Button(waveweight_frame, text='Reset',command = self.resetWaveWeights)
		waveweight_reset_button.pack(side="left")

		canny_frame = tkinter.Frame(self.root)
		canny_frame.pack(side="top")
		canny_label = tkinter.Label(canny_frame)
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
		save_button = ttk.Button(rerun_frame, text='Save modified outputs', command = self.saveAnalysis)
		save_button.pack(side="left")


		#Package the frame display controls together
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



		#Package the frame displays together
		self.image_frame = tkinter.Frame(self.root)
		self.image_frame.pack(side="top")
		self.raw_canvas = tkinter.Canvas(self.image_frame,width = 50,height=50)
		self.raw_canvas.pack(side="left")
		self.edge_canvas = tkinter.Canvas(self.image_frame,width=50,height=50)
		self.edge_canvas.pack(side="left")
		self.filled_canvas = tkinter.Canvas(self.image_frame,width=50,height=50)
		self.filled_canvas.pack(side="left")


		#Package the wave display elements togther
		figure_frame = tkinter.Frame(self.root)
		figure_frame.pack(side="top")
		figure = Figure(figsize=(5,2), dpi=100)
		self.wave_plot = figure.add_subplot(111)
		scroll_left = ttk.Button(figure_frame, text='<',command = self.scrollPlotLeft)
		scroll_left.pack(side="left")
		scroll_right = ttk.Button(figure_frame, text='>',command = self.scrollPlotRight)
		scroll_right.pack(side="right")
		self.figure_canvas = FigureCanvasTkAgg(figure, figure_frame)
		self.figure_canvas.show()
		self.figure_canvas.get_tk_widget().pack(side="left")
		

		#Package the outputs together
		answers_frame = tkinter.Frame(self.root)
		answers_frame.pack(side="top")
		self.rate_label = tkinter.Label(answers_frame)
		self.rate_label.pack(side="top")
		self.diameterpx_label = tkinter.Label(answers_frame)
		self.diameterpx_label.pack(side="top")
		self.diametermm_label = tkinter.Label(answers_frame)
		self.diametermm_label.pack(side="top")
		self.stddevpx_label = tkinter.Label(answers_frame)
		self.stddevpx_label.pack(side="top")
		self.stddevmm_label = tkinter.Label(answers_frame)
		self.stddevmm_label.pack(side="top")
		                
		self.redrawCanvases()
		self.root.mainloop()
	
	def nextMax(self):
		"""Go to the location of the next max in the wave"""
		nextIndex = self.frame_index
		for m in sorted(self.ao.max_list):
			if m > nextIndex:
				nextIndex = m
				break
		self.frame_index = nextIndex
		self.redrawCanvases()

	def previousMax(self):
		"""Go to the location of the previous max in the wave"""
		nextIndex = self.frame_index
		for m in sorted(self.ao.max_list,reverse=True):
			if m < nextIndex:
				nextIndex = m
				break
		self.frame_index = nextIndex
		self.redrawCanvases()

	def nextFrame(self):
		"""Go to the next frame in the wave"""
		self.frame_index += 1
		self.frame_index %= len(self.ao.avg_pixel_vals)
		self.redrawCanvases()

	def previousFrame(self):
		"""Go to the previous frame in the wave"""
		self.frame_index -= 1
		if(self.frame_index<0):
			self.frame_index = len(self.ao.avg_pixel_vals) - 1
		self.redrawCanvases()

	
	def scrollPlotLeft(self):
		"""Shift the wave plot to the left"""
		mod_amt = int(self.frame_limit/3)
		if(self.begin_frame - mod_amt < 0):
			self.begin_frame = 0
		else:
			self.begin_frame -= mod_amt
		self.redrawCanvases()

	def scrollPlotRight(self):
		"""Shift the wave plot to the right"""
		mod_amt = int(self.frame_limit/3)
		if(self.begin_frame + self.frame_limit + mod_amt >= len(self.ao.avg_pixel_vals)):
			self.begin_frame = len(self.ao.avg_pixel_vals) - self.frame_limit
		else:
			self.begin_frame += mod_amt
		self.redrawCanvases()


	def updateMaxes(self):
		"""Redraw our local maxes to the user specified locations"""
		self.ao.max_list = [int(x) for x in self.maxes_entry.get().split(",")]

		self.ao.max_list = [x for x in self.ao.max_list if x > self.ao.start_valley and x < self.ao.end_valley]

		self.ao.getDropRate()
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		self.redrawCanvases()
	
	def recalculateMaxes(self):
		"""Programatically determine the local maxes"""
		old_start_val = self.ao.start_valley
		old_end_val = self.ao.end_valley

		self.ao.getWaveMaxes()

		#Don't reset the valleys when we just want to calculate maxes
		self.ao.start_valley = old_start_val
		self.ao.end_valley = old_end_val

		self.ao.max_list = [x for x in self.ao.max_list if x > self.ao.start_valley and x < self.ao.end_valley]

		self.ao.getDropRate()
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		self.maxes_entry.delete(0,"end")
		self.maxes_entry.insert("end",",".join([str(x) for x in self.ao.max_list]))
		self.redrawCanvases()

	def updateValleys(self):
		"""Redraw our local mins to the user specified locations"""
		valleys = [int(x) for x in self.valley_entry.get().split(",")]
		self.ao.start_valley = valleys[0] 
		self.ao.end_valley = valleys[1] 

		self.ao.max_list = [x for x in self.ao.max_list if x > self.ao.start_valley and x < self.ao.end_valley]

		self.ao.getDropRate()
		self.rate_label["text"]="Drops per second: "+str(round(self.ao.drops_per_second,4))

		self.redrawCanvases()
	
	def recalculateValleys(self):
		"""Programatically determine the local mins"""
		old_max_list = self.ao.max_list[:]

		self.ao.getWaveMaxes()
	
		#Don't reset the maxes when we just want to calculate valleys
		self.ao.max_list = old_max_list[:]

		self.ao.max_list = [x for x in self.ao.max_list if x > self.ao.start_valley and x < self.ao.end_valley]

		self.ao.getDropRate()

		self.maxes_entry.delete(0,"end")
		self.maxes_entry.insert("end",",".join([str(x) for x in self.ao.max_list]))
		self.redrawCanvases()

	def updateDiameters(self):
		"""Set the diameters to the user given values"""
		self.ao.drop_diameters = [float(x) for x in self.diameter_entry.get().split(",")]
		self.redrawCanvases()
	
	def recalculateDiameters(self):
		"""Programatically determine the droplet diameters"""
		self.ao.getAllDiameters()
		self.diameter_entry.delete(0,"end")
		self.diameter_entry.insert("end",",".join([str(x) for x in self.ao.drop_diameters]))
		self.redrawCanvases()

	def resetWaveWeights(self):
		"""Reset the wave smoothing weights to default"""
		old_canny_weights = self.ao.canny_weights[:]

		self.ao.defaultParams()

		#Don't reset the canny weights when we want to reset the wave weights
		self.ao.canny_weights = old_canny_weights[:]

		self.waveweight_entry.delete(0,"end")
		self.waveweight_entry.insert("end",",".join([str(x) for x in self.ao.wave_weights]))

	def resetCannyWeights(self):
		"""Reset the canny weights to default"""
		old_wave_weights = self.ao.wave_weights[:]

		self.ao.defaultParams()

		#Don't reset the wave weights when we want to reset the canny weights
		self.ao.wave_weights = old_wave_weights[:]
 
		self.canny_entry.delete(0,"end")
		self.canny_entry.insert("end",",".join([str(x) for x in self.ao.canny_weights]))

	def rerunAnalysis(self):
		"""Rerun the entire analysis with updated values"""
		self.ao.canny_weights = [float(x) for x in self.canny_entry.get().split(",")]
		self.ao.wave_weights = [float(x) for x in self.waveweight_entry.get().split(",")]
		self.ao.runAnalysis()
		self.redrawCanvases()

	def saveAnalysis(self):
		"""Save the outputs of the analysis"""
		self.ao.writeOutputs()


	def redrawCanvases(self):
		"""Draw our wave plots, frame displays, and outputs"""
		self.wave_plot.clear()


		#Draw the mean values
		self.wave_plot.plot([self.begin_frame, self.begin_frame+self.frame_limit],
				[self.ao.meanVal,self.ao.meanVal],"r--")

		#Draw the average pixel values
		self.wave_plot.plot([x for x in range(self.begin_frame,self.begin_frame + self.frame_limit)],
				[self.ao.avg_pixel_vals[x] for x in range(self.begin_frame,self.begin_frame + self.frame_limit)])

		#Draw the local maxes as red dots
		self.wave_plot.plot([x for x in self.ao.max_list if x > self.begin_frame and x < self.begin_frame+self.frame_limit],
				[self.ao.avg_pixel_vals[x] for x in self.ao.max_list if x > self.begin_frame and x < self.begin_frame+self.frame_limit],"ro")

		#Draw the local mins as green dots if they are in frame
		if(self.ao.start_valley > self.begin_frame and self.ao.start_valley < self.begin_frame + self.frame_limit):
			self.wave_plot.plot([self.ao.start_valley],[self.ao.avg_pixel_vals[self.ao.start_valley]],"go")

		if(self.ao.end_valley > self.begin_frame and self.ao.end_valley < self.begin_frame + self.frame_limit):
			self.wave_plot.plot([self.ao.end_valley],[self.ao.avg_pixel_vals[self.ao.end_valley]],"go")

		#Set our wave scale to the global range
		#This makes it less confusing when scrolling through the wave
		self.wave_plot.set_ylim([min(self.ao.avg_pixel_vals)-1,max(self.ao.avg_pixel_vals)+1])
		self.figure_canvas.draw()


		#Draw the frame displays
		raw_arr = self.ao.raw_frames[self.frame_index]
		self.raw_img = ImageTk.PhotoImage(Image.fromarray(raw_arr))
		self.raw_canvas.create_image(0,0, anchor="nw", image=self.raw_img)

		edge_arr = self.ao.cleanFrame(raw_arr)
		self.edge_img = ImageTk.PhotoImage(Image.fromarray(edge_arr))
		self.edge_canvas.create_image(0,0, anchor="nw", image=self.edge_img)

		filled_arr = self.ao.getDiameter(edge_arr)[1]
		self.filled_img = ImageTk.PhotoImage(Image.fromarray(filled_arr))
		self.filled_canvas.create_image(0,0, anchor="nw", image=self.filled_img)


		#Draw what frame we are on (in red if it is a local max)
		self.fnav_label["text"] = str(self.frame_index)
		if(self.frame_index in self.ao.max_list):
			self.fnav_label.configure(foreground="red")
		else:
			self.fnav_label.configure(foreground="black")


		#Write outputs
		self.rate_label["text"] = "Drops per second: "+str(round(self.ao.drops_per_second,4))
		drop_diameters_np = numpy.asarray(self.ao.drop_diameters)
		self.diameterpx_label["text"] = "Average Drop Diameter (pixels): " + str(drop_diameters_np.mean())
		self.diametermm_label["text"] = "Average Drop Diameter (µm): " + str(drop_diameters_np.mean()*(self.ao.conversion_factor**2))
		self.stddevpx_label["text"] = "Drop Diameter Standard Deviation (pixels): " + str(drop_diameters_np.std())
		self.stddevmm_label["text"] = "Drop Diameter Standard Deviation (µm): " + str(drop_diameters_np.std()*(self.ao.conversion_factor**2))




#Commands to run when script is called
SetupGUI(sys.argv[1] if len(sys.argv)>1 else "")
