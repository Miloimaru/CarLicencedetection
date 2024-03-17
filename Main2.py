from flask import Flask, render_template, Response, jsonify, request, session, send_file
import cv2
from YoLo.YoLo2 import video_detection
from PIL import Image
import io
import sqlite3
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import configparser

app = Flask(__name__, template_folder='public')

app.config['SECRET_KEY'] = 'muhammadmoin'
app.config['UPLOAD_FOLDER'] = 'static/files'

date_select = ""


def create_plot(data):
    # Create a simple plot using matplotlib
    plt.plot(data)
    plt.xlabel('X-axis label')
    plt.ylabel('Y-axis label')
    plt.title('Your Graph Title')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/show_data', methods=['POST'])
def show_data():
    try:
        # Check if the required parameters are present in the request
        selected_day = request.form['day']
        selected_month = request.form['month']
        selected_year = request.form['year']

        # Construct the selected date
        selected_date = f"{selected_year}-{selected_month.zfill(2)}-{selected_day.zfill(2)}"

        # Fetch data from SQLite
        conn = sqlite3.connect('../pythonProject1/CarDetect/Data_CarDetect.db')
        cursor = conn.cursor()
        query = (f"SELECT car_txt, car_ID, image, plate_image, Date, Province FROM CarDetect WHERE Date = ? ORDER BY "
                 f"car_txt")
        cursor.execute(query, (selected_date,))
        data = cursor.fetchall()
        conn.close()

        decoded_data = []
        for row in data:
            car_txt = row[0]
            car_ID = row[1]
            image = base64.b64encode(row[2]).decode('utf-8') if row[2] else None
            plate_image = base64.b64encode(row[3]).decode('utf-8') if row[3] else None
            Date = row[4]
            Province = row[5]
            decoded_data.append((car_txt, car_ID, image, plate_image, Date, Province))

        return render_template('Show_data.html', data=decoded_data)
    except KeyError as e:
        return f"Bad request: Missing parameter - {e}", 400
    except sqlite3.Error as error:
        print("Error: ", error)
        return "Internal Server Error", 500


@app.route('/show_graph')
def show_graph():
    global date_select
    print(date_select)
    conn = sqlite3.connect('../pythonProject1/CarDetect/Data_CarDetect.db')
    cursor = conn.cursor()

    query = "SELECT Province, COUNT(*) as count FROM CarDetect WHERE Date = ? GROUP BY Province"
    cursor.execute(query, (date_select,))
    results = cursor.fetchall()

    conn.close()

    return render_template('show_graph.html', results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0" , port=5000,debug=True)
