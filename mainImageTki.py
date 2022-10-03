import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from Tkinter import Scale
import Tkinter as tk
import ttk
from tkFileDialog import askopenfilename
import tkFileDialog
from operator import*
import numpy as np
import cv2
import shutil
import os.path
from os import listdir
import imutils
import argparse
from PIL import Image 
from PIL import ImageTk

from itertools import cycle

from matplotlib import pyplot as plt



LARGE_FONT= ("Verdana", 12)


class SeaofBTCapp(tk.Tk):
   
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="")
        tk.Tk.wm_title(self, "HHFTAI Writer Identification")
        
        
        container = tk.Frame(self,width=500,height=500,bg="yellow")
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)
	

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.config(width=200)
        frame.tkraise()   
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Choose the Database for Evaluation", font=LARGE_FONT, fg='royal blue', bg='gold')
        label.pack(pady=10,padx=10)

        button = tk.Button(self, text="HHFTAI on IAM Database",
                            command=lambda: controller.show_frame(PageOne), fg='blue violet', bg='cyan')
        button.pack()

        button2 = tk.Button(self, text="HHFTAI on Tamil Database",
                            command=lambda: controller.show_frame(PageTwo), fg='blue violet', bg='cyan')
        button2.pack()

        button3 = tk.Button(self, text="HHFTAI between writers (CVL German Database)",
                            command=lambda: controller.show_frame(PageThree), fg='blue violet', bg='cyan')
        button3.pack()
        label = tk.Label(self, text="Author: R. Raja Subramanian             Guide: Karthick Seshadri", font=LARGE_FONT, fg='gold', bg='blue')
        label.pack(side=tk.BOTTOM)
	


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        global testName
        def open_file():
            global content
            global file_path
            filename = askopenfilename()
            infile = open(filename, 'r')
            content = infile.read()
            file_path = os.path.dirname(filename)
            testName = file_path+filename
            return content
        def process_file(content):
            print(content)
            imgtest=content
        self.outlist=[]
        self.namelist=[]
        def add():

        	MIN_MATCH_COUNT = 60

		#################################################################################
		##        Level 1 Feature Descriptor Hough Circle Transform                    ##
		#################################################################################
                def hough(img):
                        img = cv2.medianBlur(img,5)
			cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
			circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
			#n=len(circles)
			circles = np.uint16(np.around(circles))
			#n=len(circles)
			n=0;
			for i in circles[0,:]:
			    # draw the outer circle
			    #cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
			    # draw the center of the circle
			    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
			    n=n+len(circles)
			#print n
	
			clust_thres=50
			nm=n/clust_thres
			name=str(nm)
			print 'c='+name
			return nm

		#################################################################################
		##        Level 2 Feature Descriptor Vertical length of characters             ##
		#################################################################################

		def findHeight(img):
			ret,thresh = cv2.threshold(img,0,230, cv2.THRESH_BINARY)
			height, width = img.shape
			#print "height and width : ",height, width
			size = img.size
			#print "size of the image in number of pixels", size 

			# plot the binary image
			#imgplot = plt.imshow(img, 'gray')
			#plt.show()
			if(height<125):
				desc='c1'
				print 'c1'
			else:
				desc='c2'
				print 'c2'
			return desc


		################################################################################
		##                        Describe the test image                             ##
		################################################################################

		filename = askopenfilename()
     	        infile = open(filename, 'r')
		content = infile.read()
		file_path = os.path.dirname(filename)
		test = filename
	        print test
		#testimage = test
		print type(test)
		#imgtest = cv2.imread(filename,0)
		#cv2.imshow(imgtest)
		#print content
		#test='/home/raja/opencv/a01/a01-014/a01-014-00.png'
		#print 'compare starts'		
		#if (test==testimage):
		#	print 'hello world'
		#print (testimage)
		#print (test)
		#imgtest = cv2.imread('/home/raja/opencv/a01/a01-014/a01-014-00.png',0)
		imgtest = cv2.imread(test,0)
		print imgtest
		
		desc1 = hough(imgtest)
		desc2 = findHeight(imgtest)

		#################################################################################
		##        Level 3 Feature Descriptor Scale Invariant Feature Transform         ##
		#################################################################################

		def find(img1,img2,imgName):

			# find the keypoints an-d descriptors with SIFT
			kp1, des1 = surf.detectAndCompute(img1,None)
			kp2, des2 = surf.detectAndCompute(img2,None)

			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks = 50)

			flann = cv2.FlannBasedMatcher(index_params, search_params)

			matches = flann.knnMatch(des1,des2,k=2)

			# store all the good matches as per Lowe's ratio test.
			good = []
			for m,n in matches:
			    if m.distance < 0.7*n.distance:
				good.append(m)


			if len(good)>MIN_MATCH_COUNT:
			    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			    matchesMask = mask.ravel().tolist()

			    h,w = img1.shape
			    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			    dst = cv2.perspectiveTransform(pts,M)

			    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

			else:
			    #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			    matchesMask = None


			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = matchesMask, # draw only inliers
					   flags = 2)

			img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

			#plt.imshow(img3, 'gray'),plt.show()
			plt.imsave(imgName,img3)
			return len(good)




		#img1 =>  # queryImage
		#img2 =>  # trainImage

		# Initiate SIFT detector
		surf = cv2.xfeatures2d.SURF_create(3000)

		
		#################################################################################
		##                              Author Inference                               ##
		#################################################################################

	
		
		def loadImages(path):
		    # return array of images
		    #namelist=[]
		    self.outlist[:]=[]
		    imagesList = listdir(path)
		    for image in imagesList:
			#img = PImage.open(path + image)
			#loadedImages.append(img)
			img = (path + image)
			img1 = cv2.imread(img,0)	
			imgName = os.path.basename(img)
			#innerList.append(imgName)
			match = find(imgtest,img1,imgName)
			#innerList.append(match)
			#mydict = mydict+{'imgName':match}
			print 'hai'
			print "%s --> %d" % (imgName,match)
			var=(match,imgName)
			self.namelist.append(imgName)
			self.outlist.append(var)
			#outerList.append(innerList)
			#print outerList
			#print outlist
		    #outlist=sorted(outlist, key=lambda out: out[1])	
		    self.outlist.sort(reverse=True)
		    print self.outlist

		    a=iter(self.namelist)
		    #print(next(a))
		    photos = cycle(ImageTk.PhotoImage(file=image) for image in self.namelist)

		    def new():
			wind = tk.Toplevel()
			wind.geometry('3000x200')
			imageFile2 = Image.open("/home/nwlab/opencv/a01/a01-000u/a01-000u-05.png")
			#image2 = ImageTk.PhotoImage(imageFile2)
			im2 = next(photos)
		        #panelx=tk.Label(wind, text=next(a)).pack()
			panel2 = tk.Label(wind , image=im2)
			panel2.place(relx=0.0, rely=0.0)
			wind.title(next(a))
			wind.mainloop()

		    #master = tk.Tk()
		    #master.geometry('100x100')
		    K=tk.Label(self, text = 'Keypoint comparison between the test Image and its close matches', fg='chocolate').pack(pady=10,padx=10)

		    B = tk.Button(self, text = 'Previous picture', command = new, fg='blue', bg='cyan').pack()

		    #B = tk.Button(master, text = 'Quit', command = quit).pack()

		    B = tk.Button(self, text = 'Next picture', command = new, fg='blue', bg='cyan').pack()

		    fwriter=self.outlist[0][1]
		    fwriter=fwriter[:-7]
		    writerinf = tk.Label(self, text = "Writer is "+fwriter, bg='dark slate blue', fg='lavender').pack()

		    #master.mainloop()
 
		    #return outerList

		#################################################################################
		##                          Model for IAM database                             ##
		#################################################################################

		path0l = "/home/nwlab/opencv/model/0/l125/"
		path0g = "/home/nwlab/opencv/model/0/g125/"
		path1l = "/home/nwlab/opencv/model/1/l125/"
		path1g = "/home/nwlab/opencv/model/1/g125/"
		path2l = "/home/nwlab/opencv/model/2/l125/"
		path2g = "/home/nwlab/opencv/model/2/g125/"
		path3l = "/home/nwlab/opencv/model/3/l125/"
		path3g = "/home/nwlab/opencv/model/3/g125/"
		path4l = "/home/nwlab/opencv/model/4/l125/"
		path4g = "/home/nwlab/opencv/model/4/g125/"
		path5l = "/home/nwlab/opencv/model/5/l125/"
		path5g = "/home/nwlab/opencv/model/5/g125/"
		path6l = "/home/nwlab/opencv/model/6/l125/"
		path6g = "/home/nwlab/opencv/model/6/g125/"

		model=[]
		model.append(path0l)
		model.append(path0g)
		model.append(path1l)
		model.append(path1g)
		model.append(path2l)
		model.append(path2g)
		model.append(path3l)
		model.append(path3g)
		model.append(path4l)
		model.append(path4g)
		model.append(path5l)
		model.append(path5g)
		model.append(path6l)
		model.append(path6g)

		#print model[0]
		for i in range(8):
			if(desc1==i and desc2 is 'c1'):
				images=loadImages(model[2*i])
				#print model[i]
			elif(desc1==i and desc2 is 'c2'):
				images=loadImages(model[(2*i)+1])
				#print model[2*i]

	
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="HHFTAI on IAM Database", font=LARGE_FONT, fg='white', bg='magenta')
        label.pack(pady=10,padx=10)



        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage), fg='blue', bg='cyan')
        button1.pack()

	label = tk.Label(self, text="Select an image for evaluation", font=LARGE_FONT, fg='chocolate')
        #label.pack(pady=10,padx=10)
	label.pack(pady=5,padx=10)
        #browse = ttk.Button(self, text="Browse", command=open_file)
	#browse.pack(padx=5,pady=10,side=tk.LEFT)


	IAM = tk.Button(self, text="Infer Writer", command=add, fg='blue', bg='cyan').pack()#when clicked sends a call back for a +
        #IAM.pack(padx=5,pady=10,side=tk.LEFT)
	#IAM.pack()

	
	label = tk.Label(self, text="Author: R. Raja Subramanian             Guide: Karthick Seshadri", font=LARGE_FONT, fg='yellow', bg='blue')
        label.pack(side=tk.BOTTOM)

	print('from here')
	print self.outlist

	"""# set x, y position only
        self.geometry('+{}+{}'.format(x, y))
        self.delay = delay
        # allows repeat cycling through the pictures
        # store as (img_object, img_name) tuple
        self.pictures = cycle((tk.PhotoImage(file=image), image)
                              for image in image_files)
        self.picture_display = tk.Label(self)
        self.picture_display.pack()"""



class PageTwo(tk.Frame):

    def __init__(self, parent, controller):


	global testName
    	def open_file():
	    global content
	    global file_path

	    filename = askopenfilename()
	    infile = open(filename, 'r')
	    content = infile.read()
	    file_path = os.path.dirname(filename)
	    testName = file_path+filename
	    return content

    	def process_file(content):
	    print content
	    imgtest=content

	self.outlist=[]
	self.namelist=[]
    	def add():

        	MIN_MATCH_COUNT = 60

		#################################################################################
		##        Level 1 Feature Descriptor Hough Circle Transform                    ##
		#################################################################################

		def hough(img):
			img = cv2.medianBlur(img,5)
			cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
			circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
			#n=len(circles)
			circles = np.uint16(np.around(circles))
			#n=len(circles)
			n=0;
			for i in circles[0,:]:
			    # draw the outer circle
			    #cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
			    # draw the center of the circle
			    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
			    n=n+len(circles)
			#print n
	
			clust_thres=50
			nm=n/clust_thres
			name=str(nm)
			print 'c='+name
			return nm

		#################################################################################
		##        Level 2 Feature Descriptor Vertical length of characters             ##
		#################################################################################

		def findHeight(img):
			ret,thresh = cv2.threshold(img,0,230, cv2.THRESH_BINARY)
			height, width = img.shape
			#print "height and width : ",height, width
			size = img.size
			#print "size of the image in number of pixels", size 

			# plot the binary image
			#imgplot = plt.imshow(img, 'gray')
			#plt.show()
			if(height<125):
				desc='c1'
				print 'c1'
			else:
				desc='c2'
				print 'c2'
			return desc


		################################################################################
		##                        Describe the test image                             ##
		################################################################################

		filename = askopenfilename()
     	        infile = open(filename, 'r')
		content = infile.read()
		file_path = os.path.dirname(filename)
		test = filename
	        print test
		imgtest = cv2.imread(test,0)
		print imgtest
		
		desc1 = hough(imgtest)
		desc2 = findHeight(imgtest)
		
		
		#################################################################################
		##        Level 3 Feature Descriptor Scale Invariant Feature Transform         ##
		#################################################################################

		def find(img1,img2,imgName):

			# find the keypoints an-d descriptors with SIFT
			kp1, des1 = surf.detectAndCompute(img1,None)
			kp2, des2 = surf.detectAndCompute(img2,None)

			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks = 50)

			flann = cv2.FlannBasedMatcher(index_params, search_params)

			matches = flann.knnMatch(des1,des2,k=2)

			# store all the good matches as per Lowe's ratio test.
			good = []
			for m,n in matches:
			    if m.distance < 0.7*n.distance:
				good.append(m)


			if len(good)>MIN_MATCH_COUNT:
			    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			    matchesMask = mask.ravel().tolist()

			    h,w = img1.shape
			    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			    dst = cv2.perspectiveTransform(pts,M)

			    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

			else:
			    #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			    matchesMask = None


			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = matchesMask, # draw only inliers
					   flags = 2)

			img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

			#plt.imshow(img3, 'gray'),plt.show()
			plt.imsave(imgName,img3)
			return len(good)




		#img1 =>  # queryImage
		#img2 =>  # trainImage

		# Initiate SIFT detector
		surf = cv2.xfeatures2d.SURF_create(3000)

		
		#################################################################################
		##                              Author Inference                               ##
		#################################################################################

	
		
		def loadImages(path):
		    # return array of images
		    #namelist=[]
		    self.outlist[:]=[]
		    imagesList = listdir(path)
		    for image in imagesList:
			#img = PImage.open(path + image)
			#loadedImages.append(img)
			img = (path + image)
			img1 = cv2.imread(img,0)	
			imgName = os.path.basename(img)
			#innerList.append(imgName)
			match = find(imgtest,img1,imgName)
			#innerList.append(match)
			#mydict = mydict+{'imgName':match}
			print 'hai'
			print "%s --> %d" % (imgName,match)
			var=(match,imgName)
			self.namelist.append(imgName)
			self.outlist.append(var)
			#outerList.append(innerList)
			#print outerList
			#print outlist
		    #outlist=sorted(outlist, key=lambda out: out[1])	
		    self.outlist.sort(reverse=True)
		    print self.outlist
		    
		    a=iter(self.namelist)
		    photos = cycle(ImageTk.PhotoImage(file=image) for image in self.namelist)

		    def new():
			wind = tk.Toplevel()
			wind.geometry('3000x200')
			imageFile2 = Image.open("/home/nwlab/opencv/a01/a01-000u/a01-000u-05.png")
			#image2 = ImageTk.PhotoImage(imageFile2)
			im2 = next(photos)
			panel2 = tk.Label(wind , text=im2, image=im2)
			panel2.place(relx=0.0, rely=0.0)
			wind.title(next(a))
			wind.mainloop()

		    #master = tk.Tk()
		    #master.geometry('100x100')
		    K=tk.Label(self, text = 'Keypoint comparison between the test Image and its close matches', fg='chocolate').pack(pady=10,padx=10)

		    B = tk.Button(self, text = 'Previous picture', command = new, fg='blue', bg='cyan').pack()

		    #B = tk.Button(master, text = 'Quit', command = quit).pack()

		    B = tk.Button(self, text = 'Next picture', command = new, fg='blue', bg='cyan').pack()

		    
		    fwriter=self.outlist[0][1]
		    fwriter=fwriter[:-7]
		    writerinf = tk.Label(self, text = "Writer is "+fwriter, bg='dark slate blue', fg='lavender').pack()

		    #master.mainloop()
 
		    #return outerList

		#################################################################################
		##                          Model for Tamil database                             ##
		#################################################################################

		path0l = "/home/nwlab/opencv/model1/0/"
		path1l = "/home/nwlab/opencv/model1/1/"
		"""path1g = "/home/nwlab/opencv/model1/1/"
		path2l = "/home/nwlab/opencv/model1/2/"
		path2g = "/home/nwlab/opencv/model1/2/"
		path3l = "/home/nwlab/opencv/model1/3/"
		path3g = "/home/nwlab/opencv/model1/3/"
		path4l = "/home/nwlab/opencv/model1/4/"
		path4g = "/home/nwlab/opencv/model1/4/"
		path5l = "/home/nwlab/opencv/model1/5/"
		path5g = "/home/nwlab/opencv/model1/5/"
		path6l = "/home/nwlab/opencv/model1/6/"
		path6g = "/home/nwlab/opencv/model1/6/"""

		model=[]
		model.append(path0l)
		model.append(path1l)
		"""model.append(path1l)
		model.append(path1g)
		model.append(path2l)
		model.append(path2g)
		model.append(path3l)
		model.append(path3g)
		model.append(path4l)
		model.append(path4g)
		model.append(path5l)
		model.append(path5g)
		model.append(path6l)
		model.append(path6g)"""

		#print model[0]
		for i in range(1):
			if(desc2 is 'c1'):
				images=loadImages(model[0])
				#print model[i]
			elif(desc2 is 'c2'):
				images=loadImages(model[1])
				#print model[2*i]




        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="HHFTAI on Tamil Database!!!", font=LARGE_FONT, fg='white', bg='magenta')
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage), fg='blue', bg='cyan')
        button1.pack()

        """browse = ttk.Button(self, text="Browse", command=SeaofBTCapp.open_file)
	browse.pack(padx=5,pady=10,side=tk.LEFT)"""
	label = tk.Label(self, text="Select an image for evaluation", font=LARGE_FONT, fg='chocolate')
        #label.pack(pady=10,padx=10)
	label.pack(pady=5,padx=10)



	IAM = tk.Button(self, text="Infer Writer", command=add, fg='blue', bg='cyan' ).pack()#when clicked sends a call back for a +
        #IAM.pack()

	label = tk.Label(self, text="Author: R. Raja Subramanian             Guide: Karthick Seshadri", font=LARGE_FONT, fg='yellow', bg='blue')
        label.pack(side=tk.BOTTOM)



class PageThree(tk.Frame):

    def __init__(self, parent, controller):



	global testName
    	def open_file():
	    global content
	    global file_path

	    filename = askopenfilename()
	    infile = open(filename, 'r')
	    content = infile.read()
	    file_path = os.path.dirname(filename)
	    testName = file_path+filename
	    return content

    	def process_file(content):
	    print content
	    imgtest=content

	self.outlist=[]
    	def add():

        	MIN_MATCH_COUNT = 60

		#################################################################################
		##        Level 1 Feature Descriptor Hough Circle Transform                    ##
		#################################################################################

		def hough(img):
			img = cv2.medianBlur(img,5)
			cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
			circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
			#n=len(circles)
			circles = np.uint16(np.around(circles))
			#n=len(circles)
			n=0;
			for i in circles[0,:]:
			    # draw the outer circle
			    #cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
			    # draw the center of the circle
			    #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
			    n=n+len(circles)
			#print n
	
			clust_thres=50
			nm=n/clust_thres
			name=str(nm)
			print 'c='+name
			return nm

		#################################################################################
		##        Level 2 Feature Descriptor Vertical length of characters             ##
		#################################################################################

		def findHeight(img):
			ret,thresh = cv2.threshold(img,0,230, cv2.THRESH_BINARY)
			height, width = img.shape
			#print "height and width : ",height, width
			size = img.size
			#print "size of the image in number of pixels", size 

			# plot the binary image
			#imgplot = plt.imshow(img, 'gray')
			#plt.show()
			if(height<125):
				desc='c1'
				print 'c1'
			else:
				desc='c2'
				print 'c2'
			return desc


		################################################################################
		##                        Describe the test image                             ##
		################################################################################

		filename = askopenfilename()
     	        infile = open(filename, 'r')
		content = infile.read()
		file_path = os.path.dirname(filename)
		test = filename
	        print test
		print type(test)
		imgtest = cv2.imread(test,0)
		print imgtest
		

		filename1 = askopenfilename()
     	        infile1 = open(filename1, 'r')
		content1 = infile1.read()
		file_path1 = os.path.dirname(filename1)
		test1 = filename1
	        print test1
		print type(test1)
		imgtest1 = cv2.imread(test1,0)
		print imgtest1

		desc1 = hough(imgtest)
		desc2 = findHeight(imgtest)

		desc1x = hough(imgtest1)
		desc2x = findHeight(imgtest1)


		#################################################################################
		##        Level 3 Feature Descriptor Scale Invariant Feature Transform         ##
		#################################################################################

		def find(img1,img2,imgName):

			# find the keypoints an-d descriptors with SIFT
			kp1, des1 = surf.detectAndCompute(img1,None)
			kp2, des2 = surf.detectAndCompute(img2,None)

			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks = 50)

			flann = cv2.FlannBasedMatcher(index_params, search_params)

			matches = flann.knnMatch(des1,des2,k=2)

			# store all the good matches as per Lowe's ratio test.
			good = []
			for m,n in matches:
			    if m.distance < 0.7*n.distance:
				good.append(m)


			if len(good)>MIN_MATCH_COUNT:
			    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			    matchesMask = mask.ravel().tolist()

			    h,w = img1.shape
			    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			    dst = cv2.perspectiveTransform(pts,M)

			    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

			else:
			    #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			    matchesMask = None


			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = matchesMask, # draw only inliers
					   flags = 2)

			img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

			#plt.imshow(img3, 'gray'),plt.show()
			plt.imsave(imgName,img3)
			return len(good)




		#img1 =>  # queryImage
		#img2 =>  # trainImage

		# Initiate SIFT detector
		surf = cv2.xfeatures2d.SURF_create(3000)

		
		#################################################################################
		##                              Author Inference                               ##
		#################################################################################

	
		
		def loadImages(imgtest,img1,flag):
		    # return array of images
		    self.outlist[:]=[]
		    namelist=[]
		    imgName = "answer"
		    match = find(imgtest,img1,imgName)
		    print "%s --> %d" % (imgName,match)
		    var=(match,imgName)
		    namelist.append(imgName)
		    self.outlist.append(var)
		    print self.outlist
		    
		    photos = cycle(ImageTk.PhotoImage(file=image) for image in namelist)

		    def new():
			wind = tk.Toplevel()
			wind.geometry('3000x200')
			imageFile2 = Image.open("/home/nwlab/opencv/a01/a01-000u/a01-000u-05.png")
			#image2 = ImageTk.PhotoImage(imageFile2)
			im2 = next(photos)
			panel2 = tk.Label(wind , text=im2, image=im2)
			panel2.place(relx=0.0, rely=0.0)
			wind.mainloop()

		    #master = tk.Tk()
		    #master.geometry('100x100')
		    K=tk.Label(self, text = 'Keypoint comparison between the test Image and its close matches', fg='chocolate').pack(pady=10,padx=10)

		    B = tk.Button(self, text = 'Previous picture', command = new, fg='blue', bg='cyan').pack()

		    #B = tk.Button(master, text = 'Quit', command = quit).pack()

		    B = tk.Button(self, text = 'Next picture', command = new, fg='blue', bg='cyan').pack()
		    if((flag==1) and (match>12)):
		    	writerinf = tk.Label(self, text = "Writers are same", bg='dark slate blue', fg='lavender').pack()
		    else:
			writerinf = tk.Label(self, text = "Writers are different", bg='dark slate blue', fg='lavender').pack()

		    #master.mainloop()
 
		    return match

		#################################################################################
		##                          Model for CVL database                             ##
		#################################################################################

		"""path0l = "/home/raja/opencv/model/0/l125/"
		path0g = "/home/raja/opencv/model/0/g125/"
		path1l = "/home/raja/opencv/model/1/l125/"
		path1g = "/home/raja/opencv/model/1/g125/"
		path2l = "/home/raja/opencv/model/2/l125/"
		path2g = "/home/raja/opencv/model/2/g125/"
		path3l = "/home/raja/opencv/model/3/l125/"
		path3g = "/home/raja/opencv/model/3/g125/"
		path4l = "/home/raja/opencv/model/4/l125/"
		path4g = "/home/raja/opencv/model/4/g125/"
		path5l = "/home/raja/opencv/model/5/l125/"
		path5g = "/home/raja/opencv/model/5/g125/"
		path6l = "/home/raja/opencv/model/6/l125/"
		path6g = "/home/raja/opencv/model/6/g125/"

		model=[]
		model.append(path0l)
		model.append(path0g)
		model.append(path1l)
		model.append(path1g)
		model.append(path2l)
		model.append(path2g)
		model.append(path3l)
		model.append(path3g)
		model.append(path4l)
		model.append(path4g)
		model.append(path5l)
		model.append(path5g)
		model.append(path6l)
		model.append(path6g)

		#print model[0]
		for i in range(8):
			if(desc1==i and desc2 is 'c1'):
				images=loadImages(model[2*i])
				#print model[i]
			elif(desc1==i and desc2 is 'c2'):
				images=loadImages(model[(2*i)+1])
				#print model[2*i]"""

		if((desc1==desc1x) and (desc2==desc2x)):
			print desc1x
			print desc2x
			loadImages(imgtest,imgtest1,1)
		else:
			loadImages(imgtest,imgtest1,0)





        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Comparison between documents of German Writers", font=LARGE_FONT, fg='white', bg='magenta' )
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage), fg='blue', bg='cyan')
        button1.pack()

	"""browse = ttk.Button(self, text="Browse", command=SeaofBTCapp.open_file)
	browse.pack(padx=5,pady=10,side=tk.LEFT)"""
	label = tk.Label(self, text="Select an image for evaluation", font=LARGE_FONT, fg='chocolate')
        #label.pack(pady=10,padx=10)
	label.pack(pady=5,padx=10)


	IAM = tk.Button(self, text="Select two handwritings for comparison", command=add, fg='blue', bg='cyan' ).pack()#when clicked sends a call back for a +
        #IAM.pack(padx=5,pady=10,side=tk.LEFT)

	label = tk.Label(self, text="Author: R. Raja Subramanian             Guide: Karthick Seshadri", font=LARGE_FONT, fg='yellow', bg='blue')
        label.pack(side=tk.BOTTOM)


        """#addObj=SeaofBTCapp()
	#im=addObj.img3
	f = Figure(figsize=(5,5), dpi=100)
	#f=cv2.imread('/home/raja/Downloads/lines/a01/a01-000x/a01-000x-01.png',0)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
        #im = img3
	image = plt.imread('/home/raja/Downloads/lines/a01/a01-000x/a01-000x-01.png')
	fig = plt.figure(figsize=(5,4))
	ims = plt.imshow(image) # later use a.set_data(new_data)
	ax = plt.gca()
	ax.set_xticklabels([]) 
	ax.set_yticklabels([]) 

        

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)"""

        

app = SeaofBTCapp()
#ap=PageOne()
#ap.show_slides()
app["bg"] = "black"
app.mainloop()
