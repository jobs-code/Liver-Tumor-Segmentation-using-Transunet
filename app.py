from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from load_data import train  
import cv2
import sqlite3


app = Flask(__name__)
app.secret_key = 'sfdsdcgdfcjdsd98798'

database="test1.db"
conn = sqlite3.connect(database)

cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS register (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT NOT NULL,
        password TEXT NOT NULL
    )
''')

conn.commit()
cursor.close()
conn.close()
def dice_coefficient(y_true, y_pred, smooth=1):
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
        union = tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1)
        return (2. * intersection + smooth) / (union + smooth)

images_dir = 'train_images'
masks_dir = 'train_masks'
#print("programm start")
x_test, y_test = train(images_dir, masks_dir)
#print("Training completed")

@app.route('/')
@app.route('/register', methods=["POST", "GET"])
def register():
        if request.method == "POST":
                name = (request.form["name"])
                email = (request.form["email"])
                password = (request.form["pass"])
                confirm_password = (request.form["pass1"])
                if password==confirm_password :
                        conn = sqlite3.connect(database)
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO register (username, email, password) VALUES (?, ?, ?)", (name, email, password))
                        conn.commit()
                        cursor.close()
                        conn.close()
                else:
                        return "password mismatch"
        return render_template('register.html')


@app.route('/login', methods=["POST", "GET"])
def login():
        if request.method == "POST":
                email = request.form["email"]
                password = request.form["pass"]
                conn = sqlite3.connect(database)
                cursor = conn.cursor()
                cursor.execute("select * from register where email=? and password =?",(email,password))
                data=cursor.fetchone()
                if data:
                        return render_template('index.html')
                else:
                        return "password mismatch"
        return render_template('register.html')

@app.route('/index ')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        num_samples = len(x_test)
        image_index = int(request.form["index"])
        #print(image_index)
        try:
            if 0 <= image_index < num_samples:
                input_image = x_test[image_index]
                Predicted_mask_binary = y_test[image_index]
                loaded_model = load_model('transunet_model.h5', custom_objects={'dice_coefficient': dice_coefficient})

                predicted_mask = loaded_model.predict(np.expand_dims(input_image, axis=0))[0]

                threshold = 0.5
                predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)
                save_dir = 'static'
                os.makedirs(save_dir, exist_ok=True)

                input_image_path = os.path.join(save_dir, 'original_image.png')
                plt.imsave(input_image_path, input_image)

                Predicted_mask_path = os.path.join(save_dir, 'Predicted_mask.png')
                plt.imsave(Predicted_mask_path, Predicted_mask_binary[:, :, 0], cmap='gray')
                area_pixels = np.sum(Predicted_mask_binary[:, :, 0] > 0)
                #print(area_pixels)
                DPI = 300
                cm_per_pixel = 2.54 / DPI
                area_cm2 = area_pixels * cm_per_pixel**2
                side_length_cm = (area_cm2)**0.5 *10

                if side_length_cm==0:
                        stage="No tumor"
                elif  side_length_cm<=1:
                        stage="Stage 1"
                elif side_length_cm<=2:
                        stage="Stage 2"
                else:
                        stage="Stage 3"
                #print("Side length in centimeters:", stage)
                #print(f"Images saved successfully in {save_dir}")
                imag1 = "static/original_image.png"
                imag2 = "static/Predicted_mask.png"
                Img = cv2.imread(imag1)
                Img1 = cv2.imread(imag2)
                img = cv2.addWeighted(Img, 0.5, Img1, 0.7, 0)
                cv2.imwrite('static/final_output.jpg', img)
                return render_template('prediction.html',stage=stage,size=side_length_cm)
            else:
                return (f"Error: Image index {image_index} is out of bounds (0 to {num_samples - 1})")
        except Exception as e:
            return (f"An error occurred: {str(e)}")
            return render_template('index.html')
        else:
            return "Error: Image index is out of bounds."
    else:
        return "Error: Invalid request method." 

if __name__ == '__main__':
    app.run(debug=False, port=500)

