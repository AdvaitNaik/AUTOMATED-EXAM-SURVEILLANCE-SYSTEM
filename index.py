
from tkinter import *


from tkinter.ttk import *
import tkinter.font as font

from tkinter import ttk 
from ttkthemes import ThemedTk
import time



a = 0

def on_press(event):
    global a
    a+=1
    import tkinter
    from tkinter import messagebox
    print('{0} pressed'.format(event.keysym))
    print(a)
    if a>0 and a<5:
        tkinter.messagebox.showwarning(title='Warning', message='Using keyboard is prohibited. You will be removed from the test.')
    if a>4 and a<6:
        tkinter.messagebox.showwarning(title='Final Warning', message='❌❌❌❌❌❌This is your final warning. You will be removed from the test after this warning.❌❌❌❌❌❌')
    if a==6:
        window.destroy()
        
    


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