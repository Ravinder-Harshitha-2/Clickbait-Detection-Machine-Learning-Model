# Importing necessary libraries
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image 
import joblib

from clickbait_detection_model import detect_clickbait 
from clickbait_detection_model import detect_youtube_clickbait 

# Loading the trained model and CountVectorizer
news_model = joblib.load('clickbait_model.pkl')
news_vectorizer = joblib.load('count_vectorizer.pkl')

# Function to make predictions
def predict_news_clickbait():
    # Taking user input from the Text widget
    input_text = text_entry.get("1.0", tk.END).strip()  
    
    # If the input is empty, it shows an error message
    if not input_text:
        messagebox.showerror("Input Error", "Please enter some text.")
        return
    
    # Useing the detect_clickbait function to get the result
    news_clickbait_prediction = detect_clickbait(input_text)
    
    # Displaying the result in the result label
    prediction_label_1.config(text=f"{news_clickbait_prediction}")

# Function to clear the text widget
def clear_entry():
    text_entry.delete(1.0, 'end')

# Tkinter window
root = Tk()
# name of the window
root.title('Clickbait Detection Machine Learning Model')
# size of the window
root.geometry('900x600')
# background color of the window
root.config(bg='#f50a31')
# code to disable the resizing of output window
root.resizable(0,0)


# Function to switch between tkinter frames
def show_frame(frame):
    frame.tkraise()

# Start frame
info_frame = Frame(root, bg='#f50a31')
info_frame.place(x=10, y=10, height=680, width=880)

# Adding Background Image to start frame
bg_image = Image.open("4.png")
resize_image = bg_image.resize((880,580))
bg_img1 = ImageTk.PhotoImage(resize_image)
bg_image_label1 = Label(info_frame, image=bg_img1)
bg_image_label1.place(x=-2,y=-2)


# Enter button
start_button = Button(info_frame, text="Enter", fg="white", bg="#f50a31", font=("Lucida Sans", 12,"bold"), bd = 0, command=lambda:show_frame(news_frame))
start_button.place(x=350, y=410, height=45, width=150)

# News frame
news_frame = Frame(root, bg='#f50a31')
news_frame.place(x=10, y=10, height=680, width=880)

# Adding Background Image to news frame
bg_image = Image.open("5.png")
resize_image = bg_image.resize((880,580))
bg_img2 = ImageTk.PhotoImage(resize_image)
bg_image_label2 = Label(news_frame, image=bg_img2)
bg_image_label2.place(x=-2,y=-2)

# Instructions
info_text = Label(news_frame, text="This is a News Headline Clickbait Detector that detects if a headline is a clickbait or not based on \n\n machine learning.", fg="white", bg="#020105", font=("Lucida sans", 11))
info_text.place(x=90, y=140)

instruction_text = Label(news_frame, text="Enter a News headline in the input box below and click detect to detect clickbait.", fg="white", bg="#020105", font=("Lucida sans", 11))
instruction_text.place(x=140, y=210)

# Input widget
text_entry = Text(news_frame, font=("Lucida Sans", 12), width=68, height=4) 
text_entry.place(x=100, y=265)

# Detect Button with command to detect clickbait
detect_button_1 = Button(news_frame, text="Detect", fg="white", bg="#4f56e6", bd = 0, font=("Lucida Sans", 11, "bold"), command=predict_news_clickbait)
detect_button_1.place(x=450, y=350, height=40, width=335)

# Clear button to clear text
clear_button = Button(news_frame, text="Clear", fg="white", bg="#f50a31", bd = 0, font=("Lucida Sans", 11, "bold"), command=clear_entry)
clear_button.place(x=100, y=350, height=40, width=335)

# Prediction label with output
prediction_label_1 = Label(news_frame, text=" ", fg="orange", bg="#020105", font=("Lucida Sans", 14, "bold"), width=55, height=3, bd=3,  # Border width
    relief="solid",  
    highlightbackground="white",  
    highlightthickness=2)  
prediction_label_1.place(x=105, y=410)

# Button to switch to youtube frame
switch_button = Button(news_frame, text="Switch to YouTube Clickbait Detection", fg="black", bg="#fff92c", bd = 0, font=("Lucida Sans", 10, "bold"), command=lambda:show_frame(youtube_frame))
switch_button.place(x=420, y=515, height=35, width=300)

# Button to go to main info frame
main_button = Button(news_frame, text="Main", fg="white", bg="#00bf63", bd = 0, font=("Lucida Sans", 10, "bold"), command=lambda:show_frame(info_frame))
main_button.place(x=150, y=515, height=35, width=150)


#-----------------------------------------------------------------------------------------------------------------------

# Loading the trained model and CountVectorizer
youtube_model = joblib.load('youtube_model.pkl')
youtube_vectorizer = joblib.load('youtube_vectorizer.pkl')

# Function to make predictions
def predict_youtube_clickbait():
    # Taking user input from the Text widget
    input_title = title_entry.get("1.0", tk.END).strip()  # Get text from the Text widget (multi-line input)
    
    # If the input is empty, it shows an error message
    if not input_title:
        messagebox.showerror("Input Error", "Please enter some text.")
        return
    
     # Useing the detect_clickbait function to get the result
    youtube_clickbait_prediction = detect_youtube_clickbait(input_title)
    
    # Displaying the result in the result label
    prediction_label_2.config(text=f"{youtube_clickbait_prediction}")

# Function to clear the text widget
def clear_input():
    title_entry.delete(1.0, 'end')

# Youtube frame
youtube_frame = Frame(root, bg='#f50a31')
youtube_frame.place(x=10, y=10, height=680, width=880)

# Adding Background Image to Youtube frame
bg_image = Image.open("6.png")
resize_image = bg_image.resize((880,580))
bg_img3 = ImageTk.PhotoImage(resize_image)
bg_image_label3 = Label(youtube_frame, image=bg_img3)
bg_image_label3.place(x=-2,y=-2)

# Instructions
info_text = Label(youtube_frame, text="This is a YouTube Clickbait Detector that detects if a youtube title is a clickbait or not based on \n\n machine learning.", fg="white", bg="#020105", font=("Lucida sans", 11))
info_text.place(x=85, y=140)

instruction_text = Label(youtube_frame, text="Enter a Youtube title in the input box below and click detect to detect clickbait.", fg="white", bg="#020105", font=("Lucida sans", 11))
instruction_text.place(x=150, y=210)

# Input widget
title_entry = Text(youtube_frame, font=("Lucida Sans", 12), width=68, height=4)  
title_entry.place(x=100, y=265)

# Detect Button with command to detect clickbait
detect_button_2 = Button(youtube_frame, text="Detect", fg="white", bg="#4f56e6", bd = 0, font=("Lucida Sans", 11, "bold"), command=predict_youtube_clickbait)
detect_button_2.place(x=450, y=350, height=40, width=335)

# Clear button to clear text
clear_button = Button(youtube_frame, text="Clear", fg="white", bg="#f50a31", bd = 0, font=("Lucida Sans", 11, "bold"), command=clear_input)
clear_button.place(x=100, y=350, height=40, width=335)

# Prediction label with output
prediction_label_2 = Label(youtube_frame, text=" ", fg="orange", bg="#020105", font=("Lucida Sans", 14, "bold"), width=55, height=3,bd=3,  # Border width
    relief="solid",  
    highlightbackground="white",  
    highlightthickness=2)  
prediction_label_2.place(x=105, y=410)

# Button to switch to youtube frame
switch_button_1 = Button(youtube_frame, text="Switch to News Clickbait Detection", fg="black", bg="#fff92c", bd = 0, font=("Lucida Sans", 10, "bold"), command=lambda:show_frame(news_frame))
switch_button_1.place(x=420, y=515, height=35, width=300)

# Button to go to main info frame
main_button = Button(youtube_frame, text="Main", fg="white", bg="#00bf63", bd = 0, font=("Lucida Sans", 10, "bold"), command=lambda:show_frame(info_frame))
main_button.place(x=150, y=515, height=35, width=150)

# code for info frame to show first
show_frame(info_frame) 

# Closing the loop
root.mainloop()
