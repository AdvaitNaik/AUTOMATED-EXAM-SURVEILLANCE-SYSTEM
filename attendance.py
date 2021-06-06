from tkinter import*
import tkinter as tk
import cv2
import os
import csv
import numpy as np
from PIL import Image
from PIL import ImageTk
import pandas as pd
import datetime
import time
from tkinter import messagebox
from twilio.rest import Client


from tkinter.ttk import *
import tkinter.font as font
from tkinter import ttk
from ttkthemes import ThemedTk


a = 0
window = tk.Tk()

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
# set width and height
canvas = Canvas(window, width=screen_width, height=screen_height)
canvas.pack(expand=YES, fill=BOTH)

image = PhotoImage(
    file="F:\ADVAIT\ADVAIT RESUME\LBS\Drishti_Cheating-detection-system\Bg_image.png")

canvas.create_image(0, 0, anchor=NW, image=image)
canvas.pack()

window.title("Surveillance System")

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

x_cord = 150
y_cord = 100
checker = 0
message = tk.Label(window, text="*corona precautionary norms followed", bg="darkblue",
                   fg="white", width=40, height=1, font=('Times New Roman', 13, 'normal'))
message.place(x=1000, y=5)

message = tk.Label(window, text="EXAM SURVEILLANCE SYSTEM", bg="darkblue",
                   fg="white", width=30, height=1, font=('Times New Roman', 25, 'bold underline'))
message.place(x=350, y=5)

lbl = tk.Label(window, text="Enter Your College ID  :", width=20, height=1,
               fg="white", bg="darkblue", font=('Times New Roman', 25, ' bold '))
lbl.place(x=230-x_cord, y=228-y_cord)

txt = tk.Entry(window, width=30, bg="white", fg="black",
               font=('Times New Roman', 15, ' bold '))
txt.place(x=650-x_cord, y=237-y_cord)

lbl2 = tk.Label(window, text="Enter Your Name   :", width=20, fg="white",
                bg="darkblue", height=1, font=('Times New Roman', 25, ' bold '))
lbl2.place(x=230-x_cord, y=348-y_cord)

txt2 = tk.Entry(window, width=30, bg="white", fg="black",
                font=('Times New Roman', 15, ' bold '))
txt2.place(x=650-x_cord, y=357-y_cord)

lbl2 = tk.Label(window, text="Enter Time Limit   :", width=20, fg="white",
                bg="darkblue", height=1, font=('Times New Roman', 25, ' bold '))
lbl2.place(x=230-x_cord, y=478-y_cord)

tm = tk.Entry(window, width=30, bg="white", fg="black",
              font=('Times New Roman', 15, ' bold '))
tm.place(x=650-x_cord, y=487-y_cord)

lbl3 = tk.Label(window, text="NOTIFICATION    :", width=20, fg="white",
                bg="darkblue", height=1, font=('Times New Roman', 25, ' bold '))
lbl3.place(x=230-x_cord, y=588-y_cord)

message = tk.Label(window, text="", bg="white", fg="blue", width=30, height=1,
                   activebackground="white", font=('Times New Roman', 15, ' bold '))
message.place(x=650-x_cord, y=597-y_cord)

lbl3 = tk.Label(window, text="ATTENDANCE", width=20, fg="white",
                bg="black", height=2, font=('Times New Roman', 30, ' bold '))
lbl3.place(x=120, y=700-y_cord)

message2 = tk.Label(window, text="", fg="white", bg="black",
                    activeforeground="green", width=40, height=4, font=('times', 15, ' bold '))
message2.place(x=700, y=700-y_cord)


def clear1():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def ExamWindow():
    def on_press(event):
        global a
        a+=1
        import tkinter
        from tkinter import messagebox
        print('{0} pressed'.format(event.keysym))
        print(a)
        if a>5:
            window.destroy()
        tkinter.messagebox.showwarning(title='Warning', message='Using keyboard is prohibited. You will be removed from the test.')
        if a>4:
            tkinter.messagebox.showwarning(title='Final Warning', message='❌❌❌❌❌❌This is your final warning. You will be removed from the test after this warning.❌❌❌❌❌❌')

            
    window = ThemedTk(theme="breeze")
    window.title("EXAM")

    window.geometry('1920x1080')

    myFont=font.Font(family='TIMES NEW ROMAN')
    lbla= Label(window, text='INTERNAL ASSESSMENT',font=("TIMES NEW ROMAN", 24)).place(x=650,y=10)
    lbl3= Label(window, text='1)When was the first computer game invented?',font=("TIMES NEW ROMAN", 16)).place(x=20,y=60)
    ttk.rad1 = Checkbutton(window,text=' 1927').place(x=20,y=100)

    rad2 = Checkbutton(window,text=' 1958').place(x=20,y=125)

    rad3 = Checkbutton(window,text=' 1945').place(x=20,y=150)

    rad4 = Checkbutton(window,text=' 1945').place(x=20,y=175)

    lbl4= Label(window, text="2)Finally, as of 2019, which of these was the world's biggest tech company?",font=("TIMES NEW ROMAN", 16)).place(x=20,y=220)
    rad5 = Checkbutton(window,text=' Apple ').place(x=20,y=260)

    rad6 = Checkbutton(window,text=' Sony ').place(x=20,y=285)

    rad7 = Checkbutton(window,text=' Microsoft ').place(x=20,y=310)

    rad7 = Checkbutton(window,text=' IBM ').place(x=20,y=335)

    lbl4= Label(window, text="3)How many websites are on the Internet (as of March 2021)?",font=("TIMES NEW ROMAN", 16)).place(x=20,y=380)
    rad9 = Checkbutton(window,text=' Just over 1 billion ').place(x=20,y=420)

    rad10 = Checkbutton(window,text=' Just over 5 billion ').place(x=20,y=445)

    rad11 = Checkbutton(window,text=' Around 10 billion ').place(x=20,y=470)

    rad12 = Checkbutton(window,text=' Almost 20 billion ').place(x=20,y=495)

    lbl5= Label(window, text="4)Where did the name 'Bluetooth' come from?",font=("TIMES NEW ROMAN", 16)).place(x=20,y=540)
    rad13 = Checkbutton(window,text=' A medieval Scandinavian king ').place(x=20,y=580)

    rad14 = Checkbutton(window,text=' An electric eel with blue teeth ').place(x=20,y=605)

    rad15 = Checkbutton(window,text=' A bear that loves blueberries ').place(x=20,y=630)

    rad16 = Checkbutton(window,text=' A Native American chieftain ').place(x=20,y=655)

    btn = Button(window, text = 'SUBMIT', command = window.destroy).place(x=650,y=700)

    window.bind('<KeyPress>', on_press)


    window.mainloop()



def start_exam():
    import numpy as np
    from keras.models import load_model
    from mtcnn.mtcnn import MTCNN
    from PIL import Image
    from sklearn.svm import SVC
    from faceidentify.SVMclassifier import model as svm
    from faceidentify.SVMclassifier import out_encoder

    import argparse
    import cv2
    import os.path as osp
    from detectheadposition import headpose
    from gaze_tracking import GazeTracking

    import pygame  # For play Sound
    import time  # For sleep
    import threading  # For multi thread


    


    
    def Fail(timee, redcard):
        if redcard >= 4:
            tk.messagebox.showinfo(
                title="BLACK-LIST", message=f"You have been caught cheating {redcard} times, You are responsible for punishment")
            window.destroy()

    # get the face embedding for one face

    def get_embedding(model, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = np.expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]

    # Print Result
    def PrintResult(x, y):
        print("-----------------RESULT-------------------")
        print("NO. OF DOUBTS:", x, "/ SUSPICIOUS ACTIVITY DETECTED", y)
        print("------------------------------------------")

    def notnegative(x):
        if x < 0:
            return 0
        else:
            return x

    def smsofsus(x, y):
        res = "NO. OF DOUBTS : " + x + "  SUSPICIOUS ACTIVITY DETECTED : " + y
        return res

    # main function
    def main(args):
        # global timee
        timee = (tm.get())
        timee = float(timee)
        filename = args["input_file"]
        faceCascade = cv2.CascadeClassifier(
            'models/haarcascade_frontalface_default.xml')
        model = load_model('models/facenet_keras.h5')

        if filename is None:
            isVideo = False
            webcam = cv2.VideoCapture(0)
            webcam.set(3, args['wh'][0])
            webcam.set(4, args['wh'][1])
        else:
            isVideo = True
            webcam = cv2.VideoCapture(filename)
            fps = webcam.get(cv2.webcam_PROP_FPS)
            width = int(webcam.get(cv2.webcam_PROP_FRAME_WIDTH))
            height = int(webcam.get(cv2.webcam_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            name, ext = osp.splitext(filename)
            out = cv2.VideoWriter(
                args["output_file"], fourcc, fps, (width, height))

        # Variable Setting
        hpd = headpose.HeadposeDetection(
            args["landmark_type"], args["landmark_predictor"])  # import headpose
        gaze = GazeTracking()
        yellocard = 0
        redcard = 0
        tempval = 0
        if not timee:
            res = ("Please enter test time(Minute): ")
            message.configure(text=res)
            MsgBox = tk.messagebox.askquestion(
                "Warning", "Please enter time limit of exam in minutes", icon='warning')
            if MsgBox == 'no':
                tk.messagebox.showinfo(
                    'Your need', 'Please go through the readme file properly')
        # Input time for limit test time
        max_time_end = time.time() + (60 * timee)

        

        while(webcam.isOpened()):

            ret, frame = webcam.read()
            gaze.refresh(frame)
            frame = gaze.annotated_frame()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            

            # Get point from pupil
            if gaze.is_blinking():
                yellocard = yellocard - 1
                yellocard = notnegative(yellocard)
            elif gaze.is_right():
                yellocard = yellocard - 1
                yellocard = notnegative(yellocard)
            elif gaze.is_left():
                yellocard = yellocard - 1
                yellocard = notnegative(yellocard)
            elif gaze.is_center():
                yellocard = yellocard - 1
                yellocard = notnegative(yellocard)
            else:
                yellocard = yellocard + 2

            # Get redcard option
            if yellocard > 50:
                yellocard = 0
                tempval = tempval + 1
                redcard = redcard + 1

            if tempval == 1:
                text1 = "WARNING"
                cv2.putText(frame, text1, (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 2)
                tempval = 0

            if redcard == 3:
                warn_img = cv2.imread("Warning/warning.png", cv2.IMREAD_COLOR)
                cv2.imshow('Warning', warn_img)
                cv2.waitKey(1)
                redcard = 3

            if redcard == 6:
                tk.messagebox.showinfo(
                    title="ALERT", message=f"You have been caught cheating more than {redcard} times")
                window.destroy()

            print("<< Level of doubt:", yellocard, " || ",
                  "Number of warnings:", redcard, " >>")

            # Detect head position
            if isVideo:
                frame, angles = hpd.process_image(frame)
                if frame is None:
                    break
                else:
                    out.write(frame)
            else:
                frame, angles = hpd.process_image(frame)

                if angles is None:
                    pass
                else:
                    if angles[0] > 15 or angles[0] < -15 or angles[1] > 15 or angles[1] < -15 or angles[2] > 15 or angles[2] < -15:
                        yellocard = yellocard + 2
                    else:
                        yellocard = yellocard - 1
                        yellocard = notnegative(yellocard)

            yellocard = yellocard + hpd.yello(frame)
            if yellocard < 0:
                yellocard = notnegative(yellocard)

        # Draw a rectangle around the faces and predict the face name
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                crop_frame = frame[y:y+h, x:x+w]
                new_crop = Image.fromarray(crop_frame)
                new_crop = new_crop.resize((160, 160))
                crop_frame = np.asarray(new_crop)
                face_embed = get_embedding(model, crop_frame)
                face_embed = face_embed.reshape(-1, face_embed.shape[0])
                pred = svm.predict(face_embed)
                pred_prob = svm.predict_proba(face_embed)

                # get name
                class_index = pred[0]
                class_probability = pred_prob[0, class_index] * 100
                predict_names = out_encoder.inverse_transform(pred)
                df = pd.read_csv("StudentDetails\StudentDetails.csv")
                recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
                recognizer.read("TrainingImageLabel\Trainner.yml")
                harcascadePath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(harcascadePath)
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                a1 = df.loc[df['Id'] == Id]['Name'].values
                predict_names = a1
                text = '%s (%.3f%%)' % (predict_names, class_probability)

                if (class_probability > 70):
                    cv2.putText(frame, text, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Display the resulting frame
                cv2.imshow('EXAM', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("The test was forcibly terminated by the administrator.")
                client = Client("AC97c237318a9b897aa14ba77982dabc3d",
                                "d37386a62f0205fa472fa20562ee4ae7v")
                # change the "from_" number to your Twilio number and the "to" number
                # to the phone number you signed up for Twilio with, or upgrade your
                # account to send SMS to any phone number
                name = (txt2.get())
                msg = name + " has been caught cheating" + \
                    smsofsus(yellocard, redcard)
                client.messages.create(
                    to="+919167198250", from_="+16196333974", body=msg)
                PrintResult(yellocard, redcard)
                Fail(timee, redcard)
                break
            elif time.time() > max_time_end:
                print(timee, "Minute's test has ended.")
                client = Client("AC97c237318a9b897aa14ba77982dabc3d",
                                "714f4c5df9bb494012b4a3e1b9e0b720")
                # change the "from_" number to your Twilio number and the "to" number
                # to the phone number you signed up for Twilio with, or upgrade your
                # account to send SMS to any phone number
                name = (txt2.get())
                msg = name + " has been caught cheating"
                client.messages.create(
                    to="+919167198250", from_="+16196333974", body=msg)
                PrintResult(yellocard, redcard)
                Fail(timee, redcard)
                break

                # When everything done, release the webcam
                webcam.release()
                if isVideo:
                    out.release()
                    cv2.destroyAllWindows()

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', metavar='FILE', dest='input_file', default=None,
                            help='Input video. If not given, web camera will be used.')
        parser.add_argument(
            '-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
        parser.add_argument('-wh', metavar='N', dest='wh',
                            default=[720, 480], nargs=2, help='Frame size.')
        parser.add_argument('-lt', metavar='N', dest='landmark_type',
                            type=int, default=1, help='Landmark type.')
        parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor',
                            default='gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
        args = vars(parser.parse_args())
        main(args)


def TakeImages():
    Id = (txt.get())
    name = (txt2.get())
    if not Id:
        res = "Please enter Id"
        message.configure(text=res)
        MsgBox = tk.messagebox.askquestion(
            "Warning", "Please enter roll number properly , press yes if you understood", icon='warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo(
                'Your need', 'Please go through the readme file properly')
    elif not name:
        res = "Please enter Name"
        message.configure(text=res)
        MsgBox = tk.messagebox.askquestion(
            "Warning", "Please enter your name properly , press yes if you understood", icon='warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo(
                'Your need', 'Please go through the readme file properly')

    elif(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                sampleNum = sampleNum+1
                cv2.imwrite("TrainingImage\ "+name + "."+Id + '.' +
                            str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)


def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    clear1()
    clear2()
    message.configure(text=res)
    tk.messagebox.showinfo(
        'Completed', 'Your model has been trained successfully!!')


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []

    Ids = []

    for imagePath in imagePaths:

        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

            else:
                Id = 'Unknown'
                tt = str(Id)
            if(conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) +
                            ".jpg", im[y:y+h, x:x+w])
            cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    res = attendance
    message2.configure(text=res)
    res = "Attendance Taken"
    message.configure(text=res)
    tk.messagebox.showinfo(
        'Completed', 'Congratulations ! Your attendance has been marked successfully for the day!!')


def quit_window():
    MsgBox = tk.messagebox.askquestion(
        'Exit Application', 'Are you sure you want to exit the application', icon='warning')
    if MsgBox == 'yes':
        tk.messagebox.showinfo(
            "Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
        window.destroy()


lbl4 = tk.Label(window, text="REGISTER", width=15, fg="white",
                bg="darkblue", height=1, font=('Times New Roman', 20, ' bold '))
lbl4.place(x=1145-x_cord, y=200-y_cord)

takeImg = tk.Button(window, text="CAPTURE IMAGE", command=TakeImages, fg="white", bg="black",
                    width=25, height=2, activebackground="white", font=('Times New Roman', 15, ' bold '))
takeImg.place(x=1110-x_cord, y=250-y_cord)

trainImg = tk.Button(window, text="TRAIN  MODEL", command=TrainImages, fg="white", bg="black",
                     width=25, height=2, activebackground="white", font=('Times New Roman', 15, ' bold '))
trainImg.place(x=1110-x_cord, y=320-y_cord)

lbl5 = tk.Label(window, text="SURVEILLANCE", width=15, fg="white",
                bg="darkblue", height=1, font=('Times New Roman', 20, ' bold '))
lbl5.place(x=1145-x_cord, y=430-y_cord)

trackImg = tk.Button(window, text="MARK ATTENDANCE", command=TrackImages, fg="white", bg="black",
                     width=25, height=2, activebackground="white", font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1110-x_cord, y=480-y_cord)

trackImg = tk.Button(window, text="START EXAM", command=start_exam, fg="white", bg="black",
                     width=25, height=2, activebackground="white", font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1110-x_cord, y=550-y_cord)
trackImg = tk.Button(window, text="START EXAM", command=ExamWindow, fg="white", bg="black",
                     width=25, height=2, activebackground="white", font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1110-x_cord, y=550-y_cord)

quitWindow = tk.Button(window, text="QUIT", command=quit_window, fg="white", bg="red",
                       width=10, height=2, activebackground="pink", font=('Times New Roman', 14, ' bold '))
quitWindow.place(x=1220, y=740-y_cord)

window.mainloop()
