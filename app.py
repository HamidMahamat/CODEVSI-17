import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
 



def create_main_window():
    window=tk.Tk()
    window.title("Dispersion Diagram Displayer")

    #window.geometry("1080x720")
    window.iconbitmap("logo.ico")
    window.config(background='#33A18B')

    label_title=tk.Label(window, text="Propagation modes displayer", font=("Helvetica", 20))

    # layout on the root window
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=5)

    input_frame = create_param_frame(window)
    input_frame.grid(column=0, row=0)

    img = ImageTk.PhotoImage(Image.open('dancing-spider.jpg'))
    panel = tk.Label(window, image = img)
    panel.grid(column=1, row=0,columnspan=1,rowspan=1)

    button_frame = create_button_frame(window)
    button_frame.grid(column=1, row=1)
    
    window.resizable(1, 1)
    window.mainloop()
    
#def image_render():
    
    

def create_view_frame(container):
    frame = ttk.Frame(container)

    img = ImageTk.PhotoImage(Image.open('dancing-spider.jpg'))
    panel = tk.Label(frame, image = img)
    return frame

def create_button_frame(container):
    frame = ttk.Frame(container)
    
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    
    frame2 = ttk.Frame(frame)
    frame2.columnconfigure(0, weight=1)
    frame2.columnconfigure(1, weight=1)
    
    ttk.Label(frame2, text="k_max:").grid(column=0, row=0)
    ttk.Entry(frame2).grid(column=1, row=0, sticky=tk.E)
    frame2.grid(column=1,row=0)
    
    ttk.Button(frame, text='Generate').grid(column=0, row=1)
    
    ttk.Button(frame, text='Save').grid(column=1, row=1)
    
    for widget in frame.winfo_children():
        widget.grid(padx=0, pady=3)

    return frame
    
    

        
def create_param_frame(container):
    frame = ttk.Frame(container)

    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="transversal permittiviy "+u'\u03B5'+u'\u209C'+" :").grid(column=0, row=0)
    ttk.Entry(frame).grid(column=1, row=0, sticky=tk.E)

    ttk.Label(frame, text="longitudinal permittiviy "+u'\u03B5'+u'\u2097'+" :").grid(column=0, row=1)
    ttk.Entry(frame).grid(column=1, row=1, sticky=tk.E)

    ttk.Label(frame, text="diameter d :").grid(column=0, row=2)
    ttk.Entry(frame).grid(column=1, row=2, sticky=tk.E)

    ttk.Label(frame, text="azimuthal index n :").grid(column=0, row=3)
    ttk.Entry(frame).grid(column=1, row=3, sticky=tk.E)


    for widget in frame.winfo_children():
        widget.grid(padx=5, pady=5)
    return frame




if __name__ =="__main__":
	create_main_window()
	
