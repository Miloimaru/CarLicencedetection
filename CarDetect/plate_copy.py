import easyocr
import cv2
import numpy as np
import editdistance
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import sqlite3
from datetime import datetime, date
from PIL import Image
import io

# กำหนดขนาดของแผ่นทะเบียน
plate_size = (340, 340)  # x,y


def create_table():
    try:
        with sqlite3.connect("Data_CarDetect.db") as con:
            con.execute('''
                CREATE TABLE IF NOT EXISTS CarDetect (
                    car_txt TEXT ,
                    car_ID TEXT,
                    image BLOB,
                    plate_image BLOB,
                    Date TEXT,
                    Province TEXT,
                    PRIMARY KEY (car_txt, car_ID, Date)
                )
            ''')
    except Exception as e:
        print("Error creating table -> {}".format(e))


def insert_table(car_txt, image, number_res, province_res, photo):
    try:
        create_table()  # Ensure the table exists before inserting data
        photo = convert_photo(photo)
        today = date.today()
        formatted_date = today.strftime('%Y-%m-%d')
        with sqlite3.connect("Data_CarDetect.db") as con:
            con.execute(
                "INSERT INTO CarDetect(car_txt, car_ID, plate_image, image, Date, Province) VALUES(?,?,?,?,?,?)",
                (car_txt, number_res, image, photo, formatted_date, province_res))
    except Exception as e:
        print("Error inserting data -> {}".format(e))


def convert_photo(Pic_name):
    filename = Pic_name
    with open(filename, 'rb') as file:
        photo = file.read()

    return photo


# จัดป้ายทะเบียนเอียงให้ตรง
def get_rec(pts, padding=5):
    # หามุมทั้ง 4 มุมจากจุดต่าง ๆ ของ polygon
    rect = np.zeros((4, 2), dtype="float32")
    # top-left คือจุดที่ x+y มีค่าน้อยที่สุด
    # bottom-right คือจุดที่ x+y มีค่ามากที่สุด
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] - padding
    rect[2] = pts[np.argmax(s)] + padding
    # top-right y-x มีค่าน้อยที่สุด
    # bottom-left y-x มีค่ามากที่สุด
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] + (padding, -padding)
    rect[3] = pts[np.argmax(diff)] + (-padding, padding)
    return rect


# แปลงภาพแผ่นทะเบียน
def transform_plate(image, rect, new_size=(340, 340)):
    dst = np.array([
        [0, 0],
        [new_size[0] - 1, 0],
        [new_size[0] - 1, new_size[1] - 1],
        [0, new_size[1] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (new_size[0], new_size[1]))
    return warp

province = ['นครราชสีมา', 'เชียงใหม่', 'กาญจนบุรี', 'ตาก', 'อุบลราชธานี', 'สุราษฎร์ธานี', 'ชัยภูมิ',
            'แม่ฮ่องสอน', 'เพชรบูรณ์', 'ลำปาง', 'อุดรธานี', 'เชียงราย', 'น่าน', 'เลย', 'ขอนแก่น',
            'พิษณุโลก', 'บุรีรัมย์', 'นครศรีธรรมราช', 'สกลนคร', 'นครสวรรค์', 'ศรีสะเกษ', 'กำแพงเพชร',
            'ร้อยเอ็ด', 'สุรินทร์', 'อุตรดิตถ์', 'สงขลา', 'สระแก้ว', 'กาฬสินธุ์', 'อุทัยธานี', 'สุโขทัย',
            'แพร่', 'ประจวบคีรีขันธ์', 'จันทบุรี', 'พะเยา', 'เพชรบุรี', 'ลพบุรี', 'ชุมพร', 'นครพนม',
            'สุพรรณบุรี', 'ฉะเชิงเทรา', 'มหาสารคาม', 'ราชบุรี', 'ตรัง', 'ปราจีนบุรี', 'กระบี่', 'พิจิตร',
            'ยะลา', 'ลำพูน', 'นราธิวาส', 'ชลบุรี', 'มุกดาหาร', 'บึงกาฬ', 'พังงา', 'ยโสธร',
            'หนองบัวลำภู', 'สระบุรี', 'ระยอง', 'พัทลุง', 'ระนอง', 'อำนาจเจริญ', 'หนองคาย',
            'ตราด', 'พระนครศรีอยุธยา', 'สตูล', 'ชัยนาท', 'นครปฐม', 'นครนายก', 'ปัตตานี',
            'กรุงเทพมหานคร', 'ปทุมธานี', 'สมุทรปราการ', 'อ่างทอง', 'สมุทรสาคร', 'สิงห์บุรี', ]


def main():
    reader_th = easyocr.Reader(['th'])
    plate_model = YOLO("../CarDetect/Car_Counter/yolov8l_segment_new.pt")
    image_folder = "../CarDetect/Car_Counter/mokup"
    for filename in os.listdir(image_folder):
        if filename.endswith(".JPG"):
            try:
                result = plate_model.predict(os.path.join(image_folder, filename))
                res_plotted = result[0].plot()
                plt.figure(figsize=(15, 10))
                plt.imshow(res_plotted[:, :, [2, 1, 0]])
                result[0].boxes.conf
                pts = result[0].masks.xy[0]
                image = result[0].orig_img.copy()

                # รับพิกัดสี่เหลี่ยม
                rect = get_rec(pts)
                image = transform_plate(image, rect, new_size=plate_size)

                plate_img = io.BytesIO()
                Image.fromarray(image).save(plate_img, format='JPEG')
                plate_img = plate_img.getvalue()

                # อ่านข้อความในภาพแผ่นทะเบียน
                result = reader_th.readtext(image)
                # print(result)

                dist = [editdistance.eval(result[-2][1], pp) for pp in province]
                province[np.argmin(dist)]

                # ตัดภาพตาม coordinates ของป้ายทะเบียน
                x1, y1, x2, y2 = 45, 0, 282, 136
                plate_image = image[y1:y2, x1:x2]

                # ตัดภาพตาม coordinates ของจังหวัด
                x1, y1, x2, y2 = 0, 127, 340, 230
                province_image = image[y1:y2, x1:x2]

                # ตัดภาพตาม coordinates ของเลข
                x1, y1, x2, y2 = 35, 226, 316, 340
                number_image = image[y1:y2, x1:x2]

                # Read text in the license plate image
                plate_results = reader_th.readtext(plate_image)
                car_txt = ''
                for (bbox, c_text, prob) in plate_results:
                    # print(f'ป้ายทะเบียน: {text}, Confidence: {prob:.2f}')
                    car_txt = c_text
                    print(f'ป้ายทะเบียน: {car_txt}')

                # Read text in the province image
                province_results = reader_th.readtext(province_image)
                province_res = ''
                for (bbox, text, prob) in province_results:
                    # คำนวณระยะทางแก้ไขและค้นหาค่าใกล้เคียงที่สุด
                    edit_distances = [editdistance.eval(text, ref) for ref in province]
                    closest_match_index = edit_distances.index(min(edit_distances))
                    closest_match = province[closest_match_index]
                    # print(f'จังหวัด: {closest_match}, Edit Distance: {min(edit_distances)}')
                    province_res = closest_match
                    print(f'จังหวัด: {province_res}')

                # Read text in the number image
                number_results = reader_th.readtext(number_image)
                number_res = ''
                for (bbox, num, prob) in number_results:
                    # print(f'เลขทะเบียน: {text}, Confidence: {prob:.2f}')
                    number_res = num
                    print(f'เลขทะเบียน: {number_res}')

                insert_table(car_txt, plate_img, number_res, province_res, os.path.join(image_folder, filename))

            except Exception as e:
                print(f"Error processing image '{filename}': {str(e)}")
                continue  # Skip รูปต่อไป กะน error


if __name__ == "__main__":
    main()
