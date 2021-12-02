from fastapi import FastAPI, UploadFile, File
from starlette.responses import RedirectResponse
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract as py

py.pytesseract.tesseract_cmd =  r'E:\New folder\tesseract_ocr\tesseract.exe'

app = FastAPI()

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

#preprocessing the image
def clean(text):
    return re.sub('[^A-Za-z0-9" "]+', '', text)


dataset = "chemicalcarc1.csv"



def carcino(image):

    gray = get_grayscale(image)

    df = py.image_to_data(gray,output_type = 'data.frame')
    df = df[df['conf'] != -1]
    df['text'] = df['text'].apply(lambda x: x.strip())
    df = df[df['text']!=""]
    df['text'] = df['text'].apply(lambda x: x.lower())
    df = df[df.text.str.len() > 3]
    df.text = df.text.replace('\*','', regex = True)
    df.text = df.text.apply(clean)

    #formatting of dataset file
    # d = pd.read_csv('../input/chemicalcsv2/chemicalsimp2.csv')
    # d = d.drop(['Unnamed: 2','ID_v5'], axis = 1)
    # d = d.dropna(axis = 0)
    # d["Chemical Name"] = d["Chemical Name"].apply(clean)
    # d["Chemical Name"] = d["Chemical Name"].apply(lambda x: x.lower())
    # d = d.append({'Chemical Name': 'acesulfame potassium'}, ignore_index = True)
    # d.to_csv('chemicalcarc1', index = False)

    c = pd.read_csv(dataset)
    c = c.append({'Chemical Name':'neotame'}, ignore_index = True)
    chem = c["Chemical Name"].tolist()
    shifted_text_col = list(df['text'].iloc[1:])
    shifted_text_col.append("")
    df['text_2row'] = df['text'] + " " + shifted_text_col
    
    i = 0

    chemical = []
    while i < len(df):
        if df['text_2row'].iloc[i] in chem:
            chemical.append(df.text_2row.iloc[i])
            i += 1
        elif df['text'].iloc[i] in chem:
            chemical.append(df.text.iloc[i])
            i += 1
        else:
            i += 1
    if chemical:
        return chemical
    return "none"


@app.get("/")
def index():
    return RedirectResponse(url = '/docs')
    
@app.post("/image")
async def uploadimage(fileimage: UploadFile = File(...)):
    extension = fileimage.filename.split(".")[-1] in ("jpg", "jpeg", "PNG")
    if not extension:
        return "image must be jpg or png format!"
    bits = (await fileimage.read())
    buff = np.fromstring(bits, np.uint8)
    buff = buff.reshape(1, -1)
    image1 = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    ans = carcino(image1)
    print(ans)
    return {"chemical found": ans}
