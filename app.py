from tkinter import *

window=Tk()
window.title("Dispersion Diagram Displayer")

window.geometry("1080x720")
window.iconbitmap("logo.ico")
window.config(background='#33A18B')

label_title=Label(window, text="Propagation modes displayer", font=("Helvetica", 20))
label_title.pack()
window.mainloop()