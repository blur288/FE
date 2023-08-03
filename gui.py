import tkinter
import tkinter.messagebox
import customtkinter
from customtkinter import CTkFrame
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# pip3 install customtkinter
# pip install tk
# brew install python-tk
# python3 -m pip install --upgrade Pillow

#also have to replace the safwan images with stuff in yourg folders

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Say Cheese!")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Facial Expression Detection", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="By: Safwan, Aden, Ujjawal, Cole, Nickolas, Linus", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.logo_label.grid(row=1, column=0, padx=20, pady=(20, 10)) 

        my_image = customtkinter.CTkImage(Image.open("src/rizz-king.png"), size=(208, 208))
        self.button_image = customtkinter.CTkButton(self.sidebar_frame, image=my_image, text='', border_spacing=0, fg_color="transparent")
        self.button_image.grid(row = 3, column = 0, padx=0, pady=0)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=4, column=0, padx=20, pady=0)
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=5, column=0, padx=20, pady=(0, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 20))

        self.image_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.image_frame.grid(row=1, column=1, padx=(20, 0), pady=(20, 0))
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(4, weight=1)

        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("110%")

    def RenderSliders(self, labels, MaxLabel, progress):
        #percentages here; changing them in the text strings is self-explanatory
        #changing them in the progress bar is in the .set() functions. 0.7 means 70% filled. 0.5 means 50%.
        self.classification = customtkinter.CTkLabel(self.image_frame, text="       You are " + MaxLabel + "       ", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.classification.grid(row=1, column=1, padx=20, pady=5) 

        Emoji = customtkinter.CTkImage(Image.open("src/emoji/" + MaxLabel + ".png"), size=(150, 150))
        self.button_image = customtkinter.CTkButton(self.image_frame, image=Emoji, text='', border_spacing=0, fg_color="transparent")
        self.button_image.grid(row = 1, column = 3, padx=0, pady=0)

        self.classification = customtkinter.CTkLabel(self.image_frame, text=labels[0], font=customtkinter.CTkFont(size=16, weight="bold"))
        self.classification.grid(row=3, column=2, padx=0, pady=5) 

        
        self.progressbar_1 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_1.set(progress[0])
        self.progressbar_1.grid(row=3, column=1, padx=(20, 10), pady=10, sticky="ew")

        self.progressbar_2 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_2.set(progress[1])
        self.progressbar_2.grid(row=3, column=3, padx=(20, 10), pady=10, sticky="ew")

        self.classification = customtkinter.CTkLabel(self.image_frame, text=labels[1], font=customtkinter.CTkFont(size=16, weight="bold"))
        self.classification.grid(row=4, column=2, padx=0, pady=5) 

        self.progressbar_3 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_3.set(progress[2])
        self.progressbar_3.grid(row=4, column=1, padx=(20, 10), pady=10, sticky="ew")

        self.progressbar_4 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_4.set(progress[3])
        self.progressbar_4.grid(row=4, column=3, padx=(20, 10), pady=10, sticky="ew")

        self.classification = customtkinter.CTkLabel(self.image_frame, text=labels[2], font=customtkinter.CTkFont(size=16, weight="bold"))
        self.classification.grid(row=5, column=2, padx=0, pady=5) 

        self.progressbar_5 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_5.set(progress[4])
        self.progressbar_5.grid(row=5, column=1, padx=(20, 10), pady=10, sticky="ew")

        self.progressbar_6 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_6.set(progress[5])
        self.progressbar_6.grid(row=5, column=3, padx=(20, 10), pady=10, sticky="ew")

        self.classification = customtkinter.CTkLabel(self.image_frame, text=labels[3], font=customtkinter.CTkFont(size=16, weight="bold"))
        self.classification.grid(row=6, column=2, padx=0, pady=5) 

        self.progressbar_7 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_7.set(progress[6])
        self.progressbar_7.grid(row=6, column=1, padx=(20, 10), pady=10, sticky="ew")

        self.progressbar_8 = customtkinter.CTkProgressBar(self.image_frame)
        self.progressbar_8.set(progress[7])
        self.progressbar_8.grid(row=6, column=3, padx=(20, 10), pady=10, sticky="ew")

    def RenderImage(self):
        safwan = customtkinter.CTkImage(Image.open("picture/test.jpg"), size=(208, 208))
        self.button_image = customtkinter.CTkButton(self.image_frame, image=safwan, text='', border_spacing=0, fg_color="transparent")
        self.button_image.grid(row = 0, column = 2, padx=0, pady=0)
    
    def ClearWindow(self):
        for widget in self.winfo_children():
            widget.destroy()

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

class Gui:
    def GuiSetup(self):
        self.app = App()
    def GuiRun(self, labels, MaxLabel, progress):
        #self.app.ClearWindow()
        self.app.RenderSliders(labels, MaxLabel, progress)
        self.app.RenderImage()
        self.app.update_idletasks()
        self.app.update()