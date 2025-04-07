import tkinter
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configures window
        self.default_window_width = 1200
        self.default_window_height = 800
        self.authorization_token = ""

        self.title("Image Generator")
        self.geometry(f"{self.default_window_width}x{self.default_window_height}")

        # *** Scrollable Area Start ***

        # Create a canvas that will hold the scrollable frame
        self.canvas = tkinter.Canvas(self, width=self.default_window_width, height=self.default_window_height)
        self.canvas.pack(side="left", fill="both", expand=True)

        # Create a scrollbar and attach it to the canvas
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        # Configure the canvas to work with the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Create a frame inside the canvas where the content will go
        self.scrollable_frame = ctk.CTkFrame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Add the frame to the canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # *** Scrollable Area End ***

        # *** Interface Components Inside Scrollable Frame ***
        
        # Title label at the top of the window
        self.windowlabel = ctk.CTkLabel(
            self.scrollable_frame, 
            text="Nipun's Image Generator", 
            font=ctk.CTkFont(size=30, weight="bold"), 
            padx=50, 
            pady=50, 
            text_color="white"
        )
        self.windowlabel.pack()

        # Label for prompt input field
        self.promptlabel = ctk.CTkLabel(
            self.scrollable_frame, 
            text="Prompt", 
            font=ctk.CTkFont(family="Times New Roman", size=20, weight="bold"), 
            text_color="white"
        )
        self.promptlabel.pack()

        # Entry box for entering prompt
        self.promptentry = ctk.CTkEntry(
            self.scrollable_frame, 
            placeholder_text="Enter your prompt here", 
            width=self.default_window_width-20, 
            height=40
        )
        self.promptentry.pack(padx=20, pady=20)

        # Button for generating image based on the prompt
        self.generatebutton = ctk.CTkButton(
            master=self.scrollable_frame, 
            text="Generate Image", 
            width=self.default_window_width-50, 
            height=40, 
            fg_color="transparent", 
            border_width=2, 
            text_color="white", 
            command=self.generate
        )
        self.generatebutton.pack()

        # Label that will display the generated image
        self.imageview = ctk.CTkLabel(self.scrollable_frame, width=600, height=400)
        self.imageview.pack()

        # *** End of Components Inside Scrollable Frame ***

    def generate(self):
        self.textprompt = self.promptentry.get()

        self.generatebutton.configure(state="disabled")
        
        self.progress = ctk.CTkProgressBar(master=self.scrollable_frame, orientation='horizontal', mode='indeterminate')
        self.progress.pack()
        self.progress.start()

        self.modelid = "CompVis/stable-diffusion-v1-4"
        self.device = torch.device("cuda")
        self.pipe = StableDiffusionPipeline.from_pretrained(self.modelid, variant="fp16", torch_dtype=torch.float16, use_auth_token=self.authorization_token)
        self.pipe.to(self.device)

        with autocast():
            self.image = self.pipe(self.textprompt, guidance_scale=8.5).images[0]
            self.image.save('generatedimage.png')

            # Properly load and display the image using Image.open()
            img_open = Image.open('generatedimage.png')
            self.img = ImageTk.PhotoImage(img_open)

            # Clear existing image and update it with the new one
            self.imageview.configure(image=self.img)  # Update the image in the label
            self.imageview.image = self.img  # Keep a reference to avoid garbage collection

        self.progress.stop()
        self.progress.pack_forget()
        self.generatebutton.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()