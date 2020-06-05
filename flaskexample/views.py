from flaskexample import app
from flask import Flask, render_template, redirect, url_for, request
from flask import render_template
from werkzeug.exceptions import abort
import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract

#import re
#import nltk
#import heapq
#import networkx as nx
#from io import StringIO
#from gensim.summarization import summarize




#@app.route('/')
#@app.route('/index')
#def index():
#  return "Hello, World!"

#@app.route('/', methods=['GET', 'POST'])
#def home():
#    return render_template('upload.html')



@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        f.save(f.filename)
        mm = image_to_sum(f.filename)   
        return render_template("success.html", name = mm) #f.filename)  

def image_to_sum(img):
      try:
            img = cv.imread(img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv.filter2D(gray, -1, sharpen_kernel)
            img = Image.fromarray(sharpen)
            raw_text = pytesseract.image_to_string(img)
      except:
            raw_text = 'Wrong format!'
      return raw_text








