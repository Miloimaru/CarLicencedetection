import cv2
import tkinter as tk
from tkinter import Canvas, Button, filedialog, Label, Radiobutton, StringVar
import numpy as np


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MASK")

        self.camera_source = 0
        self.static_image = None
        self.use_webcam = True

        self.camera = cv2.VideoCapture(self.camera_source)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.canvas = Canvas(root, width=640, height=360)
        self.canvas.pack()

        self.mode_var = StringVar()  # ตัวแปรสตริงสำหรับเก็บโหมด (Red หรือ Blue)
        self.mode_var.set("Red")  # กำหนดโหมดเริ่มต้นเป็น "Red"
        self.red_radio = Radiobutton(root, text="Red", font=("Arial", 12), foreground="red", variable=self.mode_var,
                                     value="Red")
        self.red_radio.pack()
        self.blue_radio = Radiobutton(root, text="Blue", font=("Arial", 12), foreground="blue", variable=self.mode_var,
                                      value="Blue")
        self.blue_radio.pack()

        self.save_mask_button = Button(root, text="บันทึก Mask", command=self.save_mask)
        self.save_mask_button.pack()

        self.reset_button = Button(root, text="รีเซ็ต", command=self.reset_app)
        self.reset_button.pack()

        self.polygon_points = []
        self.mask = None
        self.line_start = None
        self.line_end = None

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.update()

        self.label_result = Label(root, text="", font=("Arial", 16))
        self.label_result.pack()

    # วาดรูปหลายเหลี่ยม
    def draw_polygon(self):
        self.canvas.delete("polygon")
        if len(self.polygon_points) >= 3:
            self.canvas.create_polygon(self.polygon_points, outline="red", fill="", tags="polygon")

    # วาดเส้นสีน้ำเงิน
    def draw_line(self):
        self.canvas.delete("line")
        if self.line_start is not None and self.line_end is not None:
            x1, y1 = self.line_start
            x2, y2 = self.line_end
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", tags="line")
            self.label_result.config(text=f"เส้นสีน้ำเงิน: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        if self.mode_var.get() == "Red":
            self.polygon_points.append((x, y))
            self.draw_polygon()
            if len(self.polygon_points) >= 4:
                self.create_mask()
        elif self.mode_var.get() == "Blue":
            if self.line_start is None:
                self.line_start = (x, y)
            else:
                self.line_end = (x, y)
                self.draw_line()

    # สร้างแมสก์
    def create_mask(self):
        if len(self.polygon_points) >= 4:
            mask = np.zeros((360, 640), dtype=np.uint8)
            pts = np.array(self.polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.mask = mask

    # เบันทึกแมสก์เป็นไฟล์รูปภาพ
    def save_mask(self):
        if self.mask is not None:
            cv2.imwrite('Mask.png', self.mask)
            print("บันทึกภาพแมสก์เป็น Mask.png เรียบร้อยแล้ว.")
        else:
            print("ข้อผิดพลาด: ไม่มีแมสก์ที่จะบันทึก.")

    # ปุ่มรีเซ็ต
    def reset_app(self):
        self.polygon_points.clear()
        self.mask = None
        self.line_start = None
        self.line_end = None
        self.canvas.delete("polygon")
        self.canvas.delete("line")
        if self.use_webcam:
            self.camera.release()
            self.camera = cv2.VideoCapture(self.camera_source)
        self.label_result.config(text="")
        print("รีเซ็ตแอปพลิเคชันเรียบร้อยแล้ว.")

    # แสดงผลบนแคนวาส
    def update(self):
        if self.use_webcam:
            ret, frame = self.camera.read()
            if not ret:
                print("ข้อผิดพลาด: ไม่มีการจับเฟรมจากกล้อง.")
                return
        else:
            if self.static_image is not None:
                frame = self.static_image.copy()
            else:
                print("ข้อผิดพลาด: ไม่มีภาพถ่ายที่โหลด.")
                return

        if self.mask is not None and self.mask.dtype == np.uint8:
            resized_mask = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
            masked_frame = cv2.bitwise_and(frame, frame, mask=resized_mask)
            frame = masked_frame
        else:
            print("ทำงานอยู่ ")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(10, self.update)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    app.run()
