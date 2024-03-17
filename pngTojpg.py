from PIL import Image


def convert_png_to_jpg(input_path, output_path, quality=95):
    try:
        image = Image.open(input_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert transparent PNGs to RGB

        image.save(output_path, format='JPEG', quality=quality)
        print(f"Image converted and saved as {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Replace these paths with your file paths
input_png_path = 'static/images/3.png'
output_jpg_path = 'static/images/3.jpg'
convert_png_to_jpg(input_png_path, output_jpg_path)

