from flask import Flask, render_template, request
import base64
import re
from io import BytesIO
from PIL import Image
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/save', methods=['POST'])
def save():
    digit = re.sub('^data:image/.+;base64,', '', request.form['digit'])
    with Image.open(BytesIO(base64.b64decode(digit))) as im:
        im.save("tux.bmp")
        im.close()
    result = subprocess.run(['python', 'predict.py'], stdout=subprocess.PIPE)
    prediction = result.stdout.decode('utf-8')
    # print(prediction)
    return prediction
