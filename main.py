import numpy as np
from PIL import ImageTk
from matplotlib import pyplot as plt
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
from tkinter import *
from tkinter import messagebox
import mysql.connector as con
import cv2
from PIL import Image, ImageFilter
global var
def main_screen():

    global screen
    screen=Tk()

    screen.configure(background='white')
    screen.geometry('1280x720')
    screen.title("PROJECT")
    Label(screen, text="Automatic Image Captioning System", bg="Grey", height=2, width=250, font=("Arial Bold", 30) ).pack()
    Label(screen, text="",bg="white").pack()
    Label(screen, text="",bg="white").pack()
    Label(screen, text="",bg="white").pack()
    Label(screen, text="",bg="white").pack()
    b1=Button(screen,text="Login",height=3,width=30,command=login,bg="black",fg="white",font=("Arial Bold", 13))
    b1.pack()
    #dashboard()
    Label(screen, text="",bg="white").pack()
    Label(screen, text="",bg="white").pack()
    b2=Button(screen,text="Register",height=3,width=30,command=register,bg="black",fg="white",font=("Arial Bold", 13))
    b2.pack()
    Label(screen, text="",bg="white").pack()
    Label(screen, text="",bg="white").pack()
    b3 = Button(screen, text="EXIT", height=3, width=30,command=screen.destroy,bg="black",fg="white",font=("Arial Bold", 13))
    b3.pack()
    screen.mainloop()

def register():
    global screen1
    global name_input
    global passw_input
    global email_input
    screen1=Toplevel(screen)

    screen1.configure(background='white')
    screen1.configure(background="white")
    screen1.geometry('1280x720')
    Label(screen1,text="REGISTRATION",font=("Arial Bold", 25),bg="grey",height=2,width=250).pack()

    Label(screen1,text="",bg="white").pack()
    Label(screen1, text="", bg="white").pack()
    name=Label(screen1,text="UserName *",height=2, bg="white",font=("Arial Bold", 11))
    name.pack()
    name_input=Entry(screen1,width=20)
    name_input.pack()
    Label(screen1, text="", bg="white").pack()
    Label(screen1, text="", bg="white").pack()
    passw=Label(screen1,text="Password *",height=2, bg="white",font=("Arial Bold", 11))
    passw.pack()
    passw_input=Entry(screen1,width=20)
    passw_input.pack()
    Label(screen1, text="", bg="white").pack()
    Label(screen1, text="", bg="white").pack()
    email=Label(screen1,text="Email *",height=2, bg="white",font=("Arial Bold", 11))
    email.pack()
    email_input=Entry(screen1,width=20)
    email_input.pack()
    Label(screen1, text="", bg="white").pack()
    Label(screen1, text="", bg="white").pack()
    submit=Button(screen1,text="SUBMIT",height=2,width=30,command=conn,bg="black",fg="white",font=("Arial Bold", 13))
    submit.pack()


def login():
    global screen2
    global name_log
    global passw_log
    screen2=Toplevel(screen)
    screen2.configure(background='white')
    screen2.geometry('1280x720')
    Label(screen2, text="LOGIN", font=("Arial Bold", 25),bg="grey",width=250,height=2).pack()
    Label(screen2, text="", bg="white").pack()
    Label(screen2, text="", bg="white").pack()
    name = Label(screen2, text="UserName", height=2, bg="white",font=("Arial Bold", 11))
    name.pack()
    name_log = Entry(screen2, width=20)
    name_log.pack()
    Label(screen2, text="", bg="white").pack()
    Label(screen2, text="", bg="white").pack()
    passw = Label(screen2, text="Password", height=2, bg="white",font=("Arial Bold", 11))
    passw.pack()
    passw_log = Entry(screen2, width=20)
    passw_log.pack()
    Label(screen2, text="", bg="white").pack()
    Label(screen2, text="", bg="white").pack()
    login = Button(screen2, text="LOGIN", height=2, width=30,command=log,bg="black",fg="white",font=("Arial Bold", 13))
    login.pack()
    #dashboard()

def conn():
    db = con.connect(host="localhost", user="root", password="root", database="fruitsorting", charset="utf8")
    cur=db.cursor()

    nname=name_input.get()
    npassw=passw_input.get()
    nmail=email_input.get()
    query="insert into registration(id,name,password,email) values(%s,%s,%s,%s)"
    value=[0,nname,npassw,nmail]
    cur.execute(query,value)
    db.commit()
    #tkMessageBox.showinfo("Information", "Registration Successfull")
    messagebox.showinfo("Information", 'Registration Successfull')

    screen1.destroy()

def log():
    flag=0
    db = con.connect(host="localhost", user="root", password="root", database="fruitsorting", charset="utf8")
    cur = db.cursor()
    lname=name_log.get()
    lpassw=passw_log.get()
    if(lname=="admin" and lpassw=="admin"):
         screen2.destroy()
         admin()
    
    
    query="select * from registration where name='"+lname+"' and password='"+lpassw+"'"
    cur.execute(query)
    names=cur.fetchall()
    db.commit()
    if(len(names)>0):
        flag=1
    else:
        flag=0

    if(flag==1):
        screen2.destroy()
        dashboard()
    if(flag==0):
        #tkMessageBox.showinfo("Information", "Login Unsuccessfull")
        messagebox.showinfo("Information", "Login Unsuccessfull")
        screen2.destroy()


def dashboard():
    global screen3
    screen3=Toplevel(screen)
    var = StringVar()
    screen3.configure(background='white')
    screen3.title("DASHBOARD")
    screen3.geometry('1280x720')
    Label(screen3, text="Automatic Image Captioning System", bg="Grey", height=2, width=250,font=("Arial Bold", 30)).pack()
    Label(screen3, text="", bg="white").pack()
    Label(screen3, text="", bg="white").pack()

   
    Label(screen3, text="", bg="white").pack()
    b1 =  Button(screen3,text="Click Picture",height=2,width=30,bg="black",fg="white",font=("Arial Bold", 13),command=openfilename_sketch)
    b1.pack()
    Label(screen3, text="", bg="white").pack()
    b7 =  Button(screen3,text="Calculate Histogram",height=2,width=30,bg="black",fg="white",font=("Arial Bold", 13),command=histogram)
    b7.pack()
    Label(screen3, text="", bg="white").pack()
    b2=Button(screen3,text="Edge Detection",height=2,width=30,bg="black",fg="white",font=("Arial Bold", 13),command=find_edge)
    b2.pack()
    Label(screen3, text="", bg="white").pack()
    b3= Button(screen3, text="Clear Background",height=2,width=30,bg="black",fg="white",font=("Arial Bold", 13),command=clr_back)
    b3.pack()
    Label(screen3, text="", bg="white").pack()
    b4 = Button(screen3, text="Classify Image", height=2, width=30,bg="black",fg="white",font=("Arial Bold", 13),command=data0)
    b4.pack()

    Label(screen3, text="", bg="white").pack()
    b6 = Button(screen3, text="Exit", height=2, width=30, bg="black", fg="white", font=("Arial Bold", 13),
                command=screen3.destroy)
    b6.pack()
    Label(screen3, textvariable =var, height=2,width=30,fd="black",bg="white",font=("Arial Bold", 13)).pack()

def admin():
    global screen4
    screen4=Toplevel(screen)
    var = StringVar()
    screen4.configure(background='white')
    screen4.title("ADMIN")
    screen4.geometry('1280x720')
    Label(screen4, text="Admin Menu", bg="Grey", height=2, width=250,font=("Arial Bold", 30)).pack()
    Label(screen4, text="", bg="white").pack()
    Label(screen4, text="", bg="white").pack()

   
    Label(screen4, text="", bg="white").pack()
    b1 =  Button(screen4,text="Click Picture",height=2,width=30,bg="black",fg="white",font=("Arial Bold", 13),command=snapshot)
    b1.pack()
    Label(screen4, text="", bg="white").pack()
    b7 =  Button(screen4,text="Add Image",height=2,width=30,bg="black",fg="white",font=("Arial Bold", 13),command=histogram)
    b7.pack()
    
    Label(screen4, text="", bg="white").pack()
    b6 = Button(screen4, text="Exit", height=2, width=30, bg="black", fg="white", font=("Arial Bold", 13),
                command=screen4.destroy)
    b6.pack()
    Label(screen4, textvariable =var, height=2,width=30,fd="black",bg="white",font=("Arial Bold", 13)).pack()
system_path="E:/JSM/caption_generation_p_gui/"
from tkinter import filedialog
import os
import shutil
global filename_
def openfilename_sketch(): 
    import time
    global time1
    global filename_
    time1 = time.time()
    
    print(time1)
    
    filenames = filedialog.askopenfilename(initialdir = system_path,title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    des = system_path
    if filenames !=des:
        shutil.copy(filenames, des)
       # shutil.copy(filenames, system_path)
    ss=os.path.basename(filenames)
    print(ss)
    #code.filename=ss
    #code.main()
    cv2.namedWindow("Input Image")
    image = cv2.imread(des+ss)
    cv2.imshow("Input Image",image )
    #cv2.waitkey(0)
    global imagename
    imagename=ss
    filename_=imagename
    print(imagename)

import edge_detection

def find_edge():
    global filename_
    edge_detection.edge_detection(filename_)

def clr_back():

    global filename_
    img = cv2.imread(filename_)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    cv2.imwrite("removed.jpg", img)
    plt.imshow(img), plt.colorbar(), plt.show()

def histogram():
        global filename_
        im = Image.open(filename_).convert('L')
        #Display image
        #im.show()
        print("The image content are as follow:\n")
        ss=im.histogram()
        print (ss);
        #Applying a filter to the image
        im_sharp = im.filter( ImageFilter.SHARPEN )
        #Saving the filtered image to a new file
        im_sharp.save( 'object.jpg', 'JPEG' )
        #im.show()
        im.save('sharp.jpg','png')



import time


def search_string_in_file(string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    string_image=string_to_search[8:]
    print(string_image)
    line_number = 0
    list_of_results = []
    file_name='tokens.txt'
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            #print(line)
            line_number += 1
            if str(string_image) in line:
                #print(string_image)
                # If yes, then add the line number & line as a tuple in the list
                ss=line.rstrip().split("#")
                #print(ss[1])
                list_of_results.append(ss[1])
    # Return list of tuples containing line numbers and lines where string is found
    speetch(list_of_results)
    return list_of_results
    
    
    


def data0():
    p=0
    import cv2
    import os
    import glob
    global Detected_fruit
    import tkinter as tk
    from tkinter import filedialog  as t
    import tkinter
    import tkinter.messagebox as msg

    var = StringVar()
    img_dir = "dataset"  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
 #36422830_55c844bc2d       
        img = cv2.imread(f1)
        data.append(img)
    final=[]
    img_names=[]
    
    for i in range(len(data)):
        original = cv2.imread(files[i])
        print(i)
        #print(files[i])
        image_to_compare = cv2.imread("removed.jpg")
        if original.shape == image_to_compare.shape:
           # print("The images have same size and channels")
            difference = cv2.subtract(original, image_to_compare)
            b, g, r = cv2.split(difference)

            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                #print("The images are completely Equal")
                p+=1
                
            
            else:
                #print("The images are NOT equal")
                p+=1
                

        
        sift = cv2.xfeatures2d.SIFT_create()
        #cv2.ORB_create()
        #sift=cv2.SIFT()
        #sift = cv2.SIFT_create()       
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        # Define how similar they are
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        #print("Keypoints 1ST Image: " + str(len(kp_1)))
        #print("Keypoints 2ND Image: " + str(len(kp_2)))
        n1=int(len(kp_1))
        n2=int(len(kp_2))

        percentage=len(good_points) #* 100/ number_keypoints
        final.append(percentage)
        img_names.append(str(files[i]))
        #result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        #print(good_points)

        #cv2.imwrite("result/feature_matching."+str(i)+".jpg", result)
        #cv2.imshow("result", cv2.resize(result, None, fx=1, fy=1))
        #cv2.waitKey(1000)

    #import numpy as np
    val=max(final)
    
    ind=0
    Detected_fruit=""
    print (final,val)
    
    
    ind = final.index(val)

    print (ind)
    print(img_names[ind])
    search_string_in_file(img_names[ind])


    #cv2.destroyAllWindows()
from gtts import gTTS
import os
import gtts  
from playsound import playsound  
def speetch(ss):
    # need gTTS and mpg123
# pip install gTTS
# apt install mpg123
    string_created="Possible Captions for given images may be as follow:"
    #print(string_created)
    for i in ss:
        string_created=string_created+str(i)
        print(str(i))
        

    t1 = gtts.gTTS(string_created)
    t1.save("welcome.mp3")
    playsound("welcome.mp3") 
# define variables
   
main_screen()
