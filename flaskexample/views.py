from flaskexample import app
from flask import Flask, render_template, redirect, url_for, request
from flask import render_template
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

import os
import re
import torch
import nltk
import heapq
import requests 
import urllib
import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from io import StringIO
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from tqdm.notebook import tqdm
from scipy.special import softmax
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
device = torch.device('cpu')




ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

	
@app.route('/')
def upload_form():
	return render_template('index.html')


@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        file = request.files['file'] 
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        txt = image_to_txt(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        smry = smry_gen(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
        if  txt== 'Wrong format':
            return render_template("failure.html")
        if  smry== 'No text is found':
            return render_template("success.html", text='No text is found', summary='', filename=filename) 
        return render_template("success.html", text=txt, summary=smry, filename=filename) 


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
         
def image_to_txt(img):
      img_ext = ['jpg', 'jpeg', 'png', 'gif', 'tiff']
      if (img.split(".")[1].casefold() not in img_ext):
            return 'Wrong format'
      img = cv.imread(img)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      gray = cv.bitwise_not(gray)      
      kernel = np.ones((2, 1), np.uint8)
      img = cv.erode(gray, kernel, iterations=1)
      img = cv.dilate(img, kernel, iterations=1)
      raw_text = pytesseract.image_to_string(img)
      return(raw_text)


def smry_gen(img):
      raw_text = image_to_txt(img)
      if (raw_text==''):
            return 'No text is found'
      if (raw_text=='Wrong format'):
            return 'Wrong format'
      text = re.sub(r'\n', ' ', raw_text)
      text = re.sub(r'\[[0-9]*\]', ' ', text)
      text = re.sub(r'\s+', ' ', text)
      text = re.sub(r'\s+', ' ', text)
      text = re.sub(r'\b-\b', '', text)
      model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
      tokenizer = T5Tokenizer.from_pretrained('t5-small')
      #tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
      optimizer = AdamW(model.parameters(), lr=3e-5)
      directory = 't5-finetuned'
      model.load_state_dict(torch.load(os.path.join(directory,'filename.pth')))
      model.eval()
      text_token = tokenizer.encode(text, return_tensors='pt', max_length=512).to(device)
      #smry = model.generate(text_token, max_length=text_token.shape[1], num_beams=3, no_repeat_ngram_size=3)[0]
      #smry = tokenizer.decode(smry, skip_special_tokens=True)
      summarizer = pipeline("summarization")    
      smry = summarizer(text, max_length=text_token.shape[1])[0]['summary_text']
      return smry


