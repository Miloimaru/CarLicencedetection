import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO
import editdistance

# กำหนดค่าคงที่
size_image = (340, 340)  # กว้าง, ยาว
province_name = ['นครราชสีมา', 'เชียงใหม่', 'กาญจนบุรี', 'ตาก', 'อุบลราชธานี', 'สุราษฎร์ธานี', 'ชัยภูมิ',
                 'แม่ฮ่องสอน', 'เพชรบูรณ์', 'ลำปาง', 'อุดรธานี', 'เชียงราย', 'น่าน', 'เลย', 'ขอนแก่น',
                 'พิษณุโลก', 'บุรีรัมย์', 'นครศรีธรรมราช', 'สกลนคร', 'นครสวรรค์', 'ศรีสะเกษ', 'กำแพงเพชร',
                 'ร้อยเอ็ด', 'สุรินทร์', 'อุตรดิตถ์', 'สงขลา', 'สระแก้ว', 'กาฬสินธุ์', 'อุทัยธานี', 'สุโขทัย',
                 'แพร่', 'ประจวบคีรีขันธ์', 'จันทบุรี', 'พะเยา', 'เพชรบุรี', 'ลพบุรี', 'ชุมพร', 'นครพนม',
                 'สุพรรณบุรี', 'ฉะเชิงเทรา', 'มหาสารคาม', 'ราชบุรี', 'ตรัง', 'ปราจีนบุรี', 'กระบี่', 'พิจิตร',
                 'ยะลา', 'ลำพูน', 'นราธิวาส', 'ชลบุรี', 'มุกดาหาร', 'บึงกาฬ', 'พังงา', 'ยโสธร',
                 'หนองบัวลำภู', 'สระบุรี', 'ระยอง', 'พัทลุง', 'ระนอง', 'อำนาจเจริญ', 'หนองคาย',
                 'ตราด', 'พระนครศรีอยุธยา', 'สตูล', 'ชัยนาท', 'นครปฐม', 'นครนายก', 'ปัตตานี',
                 'กรุงเทพมหานคร', 'ปทุมธานี', 'สมุทรปราการ', 'อ่างทอง', 'สมุทรสาคร', 'สิงห์บุรี',
                 'นนทบุรี', 'ภูเกต', 'สมุทรสงคราม']


def get_rec(pts, padding=5):
    # หามุมทั้ง 4 มุมจากจุดต่าง ๆ ของรูปหลายเหลี่ยม
    rect = np.zeros((4, 2), dtype="float32")
    # มุมบนซ้ายคือจุดที่ x+y มีค่าน้อยที่สุด
    # มุมล่างขวาคือจุดที่ x+y มีค่ามากที่สุด
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] - padding
    rect[2] = pts[np.argmax(s)] + padding
    # มุมบนขวาคือจุดที่ y-x มีค่าน้อยที่สุด
    # มุมล่างซ้ายคือจุดที่ y-x มีค่ามากที่สุด
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


def detect_plate(image_path, ocr_model, plate_size, reader_th):
    # ทำการตรวจจับแผ่นทะเบียนและคืนภาพแผ่นทะเบียนและข้อความ
    result = ocr_model.predict(image_path, save_crop=True, conf=0.7)
    plate_image, plate_text = process_plate_result(result, plate_size, reader_th)
    return plate_image, plate_text


def process_plate_result(result, plate_size, reader_th):
    pts = result[0].masks.xy[0]
    rect = get_rec(pts)
    plate_image = transform_plate(result[0].orig_img.copy(), rect, new_size=plate_size)
    plate_results = reader_th.readtext(plate_image)
    return plate_image, plate_results


def crop_image(image, coordinates):
    # ตัดรูปภาพตามพิกัด
    x1, y1, x2, y2 = coordinates
    return image[y1:y2, x1:x2]


def extract_text(image, reader):
    # สกัดข้อความจากภาพโดยใช้เครื่องมือ OCR
    results = reader.readtext(image)
    return results[0][1] if results else ''


def main():
    # กำหนดโมเดล
    ocr_model = YOLO('CarDetect/Car_Counter/yolov8l_segment_new.pt')
    ocr_model.fuse()
    reader_th = easyocr.Reader(['th'])

    # สร้างไดเร็กทอรีสำหรับผลลัพธ์
    output_dir = "ข้อมูลจังหวัด"
    os.makedirs(output_dir, exist_ok=True)

    # โฟลเดอร์รูปภาพ
    image_folder = "CarDetect/Car_Counter/mokup"

    # ประมวลผลรูปทั้งหมดในโฟลเดอร์
    for filename in os.listdir(image_folder):
        if filename.endswith(".JPG"):
            try:
                # โหลดรูปภาพ
                image_path = os.path.join(image_folder, filename)
                image = cv2.imread(image_path)

                # ตรวจจับแผ่นทะเบียนและสกัดข้อความ
                plate_image, plate_text = detect_plate(image_path, ocr_model, size_image, reader_th)

                # ค้นหาจังหวัดที่ใกล้เคียงที่สุด
                dist = [editdistance.eval(plate_text[-2][1], pp) for pp in province_name]
                closest_match_index = dist.index(min(dist))
                closest_match = province_name[closest_match_index]

                # ตัดรูปภาพ
                province_image = crop_image(plate_image, (0, 127, 340, 230))
                number_image = crop_image(plate_image, (35, 226, 316, 340))

                # สร้างไดเร็กทอรีสำหรับจังหวัดที่ระบุหากยังไม่มี
                province_dir = os.path.join(output_dir, closest_match)
                os.makedirs(province_dir, exist_ok=True)

                # บันทึกรูปภาพและข้อความ
                cv2.imwrite(os.path.join(province_dir, f'{filename}_plate.jpg'), plate_image)
                cv2.imwrite(os.path.join(province_dir, f'{filename}_province.jpg'), province_image)
                cv2.imwrite(os.path.join(province_dir, f'{filename}_number.jpg'), number_image)

                with open(os.path.join(province_dir, f'{filename}.txt'), "w") as text_file:
                    text_file.write(f'ป้ายทะเบียน: {plate_text[0][1]}\n')
                    text_file.write(f'จังหวัด: {closest_match}\n')
                    text_file.write(f'เลขทะเบียน: {extract_text(number_image, reader_th)}\n')

            except Exception as e:
                print(f"Error processing image '{filename}': {str(e)}")
                continue  # Skip รูปต่อไป กะน error


if __name__ == "__main__":
    main()
