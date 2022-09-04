import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

view = ""

views= [ "dancing-spider.jpg","dancing-spider-sobel_avec_seuil_270.png","Pitch-Roll-and-Yaw.jpg","im1.jpg","im2.png"]

index = 0
view=+views[index%len(views)]



def create_main_window():
    global view,panel,window,transversal_permittiviy,longitudinal_permittiviy,diameter,azimuthal_index,k_max_var
    window=tk.Tk()
    window.title("Dispersion Diagram Displayer")
    transversal_permittiviy = tk.StringVar()
    longitudinal_permittiviy= tk.StringVar()
    diameter= tk.StringVar()
    azimuthal_index= tk.StringVar()
    k_max_var = tk.StringVar()

    #window.geometry("1080x720")
    #window.iconbitmap("logo.ico")
    window.config(background='#33A18B')

    label_title=tk.Label(window, text="Propagation modes displayer", font=("Helvetica", 20))

    # layout on the root window
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)

    input_frame = create_param_frame(window)
    input_frame.grid(column=0, row=0)

    #img = ImageTk.PhotoImage(file= view)
    #panel = tk.Label(window, image = img)
    #panel.grid(column=1, row=0,columnspan=1,rowspan=1)


    button_frame = create_button_frame(window)
    button_frame.grid(column=1, row=0)
    
    window.resizable(0, 0)
    window.mainloop()
    

    

def create_button_frame(container):
    frame = ttk.Frame(container)
    
    
    
    ttk.Button(frame, text='Generate',command=image_render).grid(column=1, row=0)
    
    ttk.Button(frame, text='Save').grid(column=1, row=1)
    
    for widget in frame.winfo_children():
        widget.grid(padx=0, pady=3)

    return frame
    
    
    
        
def create_param_frame(container):
    frame = ttk.Frame(container)

    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="transversal permittiviy "+u'\u03B5'+u'\u209C'+" :").grid(column=0, row=0)
    ttk.Entry(frame,textvariable=transversal_permittiviy).grid(column=1, row=0, sticky=tk.E)

    ttk.Label(frame, text="longitudinal permittiviy "+u'\u03B5'+u'\u2097'+" :").grid(column=0, row=1)
    ttk.Entry(frame,textvariable=longitudinal_permittiviy).grid(column=1, row=1, sticky=tk.E)

    ttk.Label(frame, text="diameter d :").grid(column=0, row=2)
    ttk.Entry(frame,textvariable=diameter).grid(column=1, row=2, sticky=tk.E)

    ttk.Label(frame, text="azimuthal index n :").grid(column=0, row=3)
    ttk.Entry(frame,textvariable=azimuthal_index).grid(column=1, row=3, sticky=tk.E)
    
    ttk.Label(frame, text="k_max:").grid(column=0, row=4)
    ttk.Entry(frame,textvariable = k_max_var).grid(column=1, row=4, sticky=tk.E)

    for widget in frame.winfo_children():
        widget.grid(padx=5, pady=5)
    return frame

def image_render():
    global view,index,panel,window,transversal_permittiviy,longitudinal_permittiviy,diameter,azimuthal_index,k_max_var
    v1,v2,v3,v4,v5 = None,None,None,None,None
    index =-1
    try : #on s'assure de récuperer les bonnes valeurs
        #print(v1,v2,v3,v4,v5)
        v1= float(transversal_permittiviy.get())
        v2= float(longitudinal_permittiviy.get())
        v3=float(diameter.get())
        v4= float(azimuthal_index.get())
        v5 = float(k_max_var.get())
        index = int(v1+v2+v3+v4)
        print(v1,v2,v3,v4,v5)
        #print("k_max = {}".format(v5))
    except:
        print("les paramètres sont vides ou format incorrect")
        print(v1,v2,v3,v4,v5)
        
    if index ==-1 :
        return
    
    view = views[index%len(views)]
    result = tk.Toplevel(window) 
    img = ImageTk.PhotoImage(file= view)
    graph = tk.Label(result, image = img)
    menubar = tk.Menu(result)  
    file = tk.Menu(menubar, tearoff=0)  
    file.add_command(label="Save")   

    file.add_separator()

    file.add_command(label="Exit", command=result.quit)  
    menubar.add_cascade(label="Fichier", menu=file)  
    
    graph.pack()
    result.config(menu=menubar)  
    result.mainloop()  

if __name__ =="__main__":
	create_main_window()
