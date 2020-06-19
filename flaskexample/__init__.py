#from flask import Flask
#app = Flask(__name__)

#from flaskexample import views


from flask import Flask

UPLOAD_FOLDER = 'flaskexample/static/uploads/'

app = Flask(__name__)
from flaskexample import views
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

