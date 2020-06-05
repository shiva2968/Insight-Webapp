from flaskexample import app
from flask import Flask, render_template, redirect, url_for, request
from flask import render_template
from werkzeug.exceptions import abort
import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract

import re
import nltk
import heapq
import networkx as nx
from io import StringIO
import bs4 as bs
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from gensim.summarization import summarize



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


#def image_to_sum(img):
#      try:
#            img = cv.imread(img)
#            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#            sharpen = cv.filter2D(gray, -1, sharpen_kernel)
#            img = Image.fromarray(sharpen)
#            raw_text = pytesseract.image_to_string(img)
#      except:
#            raw_text = 'Wrong format!'
#      return raw_text




def image_to_sum(img):
      img_ext = ['jpg', 'jpeg', 'png', 'gif', 'tiff']
      if (img.split(".")[1].casefold() not in img_ext):
            return 'Wrong format'

#__________PROCESSING THE IMAGE__________
      img = cv.imread(img)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
      sharpen = cv.filter2D(gray, -1, sharpen_kernel)
      img = Image.fromarray(sharpen)
      raw_text = pytesseract.image_to_string(img)

#__________PROCESSING THE TEXT__________
      txt = re.sub(r'\n', ' ', raw_text)
      txt = re.sub(r'\[[0-9]*\]', ' ', txt)
      txt = re.sub(r'\s+', ' ', txt)
      txt = re.sub(r'\s+', ' ', txt)
      
#__________SUMMARY 0__________
      summary_0 = summarize(txt)

#__________SUMMARY 1__________
      sum_len = 3
      formatted_raw_text = re.sub('[^a-zA-Z]', ' ', raw_text)
      sentence_list = nltk.sent_tokenize(raw_text)
      stop_words = stopwords.words('english')

      word_frequencies = {}
      for word in nltk.word_tokenize(formatted_raw_text):
          if word not in stop_words:
              if word not in word_frequencies.keys():
                  word_frequencies[word] = 1
              else:
                  word_frequencies[word] += 1


      maximum_frequncy = max(word_frequencies.values())

      for word in word_frequencies.keys():
          word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


      sentence_scores = {}
      for sent in sentence_list:
          for word in nltk.word_tokenize(sent.lower()):
              if word in word_frequencies.keys():
                  if len(sent.split(' ')) < 30:
                      if sent not in sentence_scores.keys():
                          sentence_scores[sent] = word_frequencies[word]
                      else:
                          sentence_scores[sent] += word_frequencies[word]


      summary_sentences = heapq.nlargest(sum_len, sentence_scores, key = sentence_scores.get)

      summary_1 = ' '.join(summary_sentences)
      return summary_1





