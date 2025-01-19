from tkinter import *
import tkinter as tk
from tkinter import messagebox
#from PIL import ImageTk, Image 
# import requests
# import random
# import io
import joblib
from youtube_clickbait_detection_model import detect_youtube_clickbait 
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the trained model and CountVectorizer
model = joblib.load('youtube_model.pkl')
vectorizer = joblib.load('youtube_vectorizer.pkl')

# Function to make predictions
def predict_youtube_clickbait():
    # Get the user input from the Text widget
    input_text = text_entry.get("1.0", tk.END).strip()  # Get text from the Text widget (multi-line input)
    
    # If the input is empty, show an error message
    if not input_text:
        messagebox.showerror("Input Error", "Please enter some text.")
        return
    
     # Use the detect_clickbait function to get the result
    prediction = detect_youtube_clickbait(input_text)
    
    # Display the result in the result label
    prediction_label.config(text=f"Result: {prediction}")


# Output window
root = Tk()
# name of the window
root.title('Clickbait Detection Machine Learning Model')
# size of the output window
root.geometry('900x700')
# background color of the window
root.config(bg='#f50a31')
# code to disable the resizing of output window
root.resizable(0,0)

# Function to switch between frames
def show_frame(frame):
    frame.tkraise()

# Start frame
start_frame = Frame(root)
start_frame.place(x=10, y=10, height=680, width=880)

# Name of the App
Title = Label(start_frame, text="Clickbait Detector", fg="#f50a31", font=("Lucida sans", 18))
Title.place(x=250, y=75)

text_entry = Text(start_frame, font=("Lucida Sans", 10), width=85, height=4)  # height works here for multi-line input
text_entry.place(x=100, y=160)

detect_button = Button(start_frame, text="Detect", fg="white", bg="#4f56e6", bd = 0, font=("Lucida Sans", 10, "bold"), command= predict_youtube_clickbait)
detect_button.place(x=595, y=240, height=35, width=150)

clear_button = Button(start_frame, text="Clear", fg="black", bg="#fff92c", bd = 0, font=("Lucida Sans", 10, "bold"), command=lambda:show_frame(start_frame))
clear_button.place(x=195, y=240, height=35, width=150)


# Enter Amount label and entry
prediction_label = Label(start_frame, text=" ", fg="orange", bg="black", font=("Lucida Sans", 10))
prediction_label.place(x=110, y=350)


root.mainloop()