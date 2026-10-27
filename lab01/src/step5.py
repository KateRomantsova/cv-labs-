from PIL import Image, ExifTags

path = "my_photo.jpg"  

exif = Image.open(path).getexif()
data = dict(exif)
data.update(exif.get_ifd(0x8769))   # Exif IFD: тут лежать витримка, ISO, фокусна відстань

wanted = {
    "FocalLength": "Фокусна відстань, мм",
    "FocalLengthIn35mmFilm": "Еквівалентна фокусна (35 мм), мм",
    "ExposureTime": "Витримка, с",
    "ISOSpeedRatings": "ISO",
    "PhotographicSensitivity": "ISO",
    "FNumber": "Діафрагма (f/)",
}

found = False
for tag_id, value in data.items():
    name = ExifTags.TAGS.get(tag_id, tag_id)
    if name in wanted:
        print(f"{wanted[name]}: {value}")
        found = True

if not found:
    print("EXIF відсутній. Потрібен оригінал знімка, не пересланий через месенджер.")