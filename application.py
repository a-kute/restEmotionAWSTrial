from flask import Flask, request, jsonify
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
application = Flask(__name__)

emotion_model = keras.models.load_model('static/full_emotion_model.h5')

@application.route('/')
def hello_world():
    return 'Hello World!'

@application.route('/api/image', methods=['POST'])
def upload_image():
    # check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No posted image. Should be attribute named image.'})
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Empty filename submitted.'})
    if file:
        filename = secure_filename(file.filename)
        file.filename = "static/s1.jpg"
        file.save(file.filename)
        img = keras.preprocessing.image.load_img('static/s1.jpg', target_size=(48, 48), grayscale=True)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        emotion_prediction = emotion_model(img_array)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        response = {'pred': emotion_dict[maxindex]}
        return jsonify(response)
    else:
        return jsonify({'error': 'File has invalid extension'})
if __name__ == '__main__':
    application.run(host='0.0.0.0', debug=True)