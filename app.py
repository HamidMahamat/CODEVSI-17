from tkinter import *
window = Tk()
window.title("Dispersion Diagram Displayer")

window.geometry("1080x720")
window.iconbitmap("logo.ico")
window.config(background='#33A18B')

frame = Frame(window)

label_title = Label(window, text="Propagation modes displayer", font=("Palatino Linotype", 20), background='#33A18B',
                    foreground='white')
label_title.pack()

epsilon_t = Entry(frame, foreground='white', background='#33A18B')
epsilon_t.grid(row=0, column=0)

plot_bt = Button(frame, text="Plot")
plot_bt.grid(row=1, column=0)

frame.pack(side=LEFT)

window.mainloop()
