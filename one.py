from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pickle
import time


app = Flask(__name__)

# Initialize MediaPipe Hand tracking

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sign', methods=['POST'])
def sign_detection():
    from app import compare_motion
    #compare_motion()
    return jsonify({'result': 'hello world'})
if __name__ == '__main__':
    app.run(debug=True)
