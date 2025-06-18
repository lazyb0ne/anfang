
from PIL import Image, ExifTags

def generate_thumbnail(input_path, output_path, size=(300, 300)):
    with Image.open(input_path) as img:
        try:
            # 获取图片方向（如果有 EXIF）
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    img = img.rotate(180, expand=True)
                elif orientation_value == 6:
                    img = img.rotate(270, expand=True)
                elif orientation_value == 8:
                    img = img.rotate(90, expand=True)
        except Exception as e:
            print(f"[缩略图] 无法读取 EXIF 或旋转：{e}")

        img.thumbnail(size)
        img.save(output_path)
