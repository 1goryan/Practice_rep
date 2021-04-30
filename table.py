
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from fpdf import FPDF
import math


image = cv2.imread('angl_tabl.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                        cv2.THRESH_BINARY, 15, -2)
horizontal = np.copy(bw)
vertical = np.copy(bw)

cols = horizontal.shape[1]
horizontal_size = cols // 30

rows = vertical.shape[0]
vertical_size = rows // 30

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)



vertical = cv2.bitwise_not(vertical)

horizontal = cv2.bitwise_not(horizontal)

edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                cv2.THRESH_BINARY, 3, -2)

edges_1 = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                cv2.THRESH_BINARY, 3, -2)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours_1, hierarchy_1 = cv2.findContours(edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


x1_list = []
x2_list = []
y1_list = []
y2_list = []

drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)


for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, (255, 255, 255), thickness = 3)
    
for i in range(len(contours_1)):
    cv2.drawContours(drawing, contours_1, i, (255, 255, 255), thickness = 2)  
    
drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

lines = cv2.HoughLinesP(drawing, 1, np.pi / 180, 50, None, 150, 100)

if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        x1_list.append(l[0] // 3.4)
        y1_list.append(l[1] // 3.4)
        x2_list.append(l[2] // 3.4)
        y2_list.append(l[3] // 3.4)


cv2.imshow('1', drawing)
drawing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, (3, 3), iterations = 5)

edges_2 = cv2.Canny(drawing, 0, 255)

contours_3, hierarchy_3 = cv2.findContours(edges_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours_3)):
    if cv2.contourArea(contours_3[i]) > 5000:
        cv2.drawContours(drawing, contours_3, i, (255, 255, 255))
    else:
        cv2.drawContours(drawing, contours_3, i, (0, 0, 0), thickness = 3)

edges_in = cv2.Canny(drawing, 0, 255)
contours_in, hierarchy_in = cv2.findContours(edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


drawing = cv2.erode(drawing, (5, 5), iterations = 2)
pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.add_page()
pdf.set_font('Times', size=18)

mass = [3, 6, 9, 15, 24, 30, 33, 36, 42, 51]

for i in mass:
    pdf.line(x1_list[i] + 10, y1_list[i] +10, x2_list[i] + 10, y2_list[i]+10)

n = 1
for i in range(len(contours_in)):
    if cv2.contourArea(contours_in[i]) < 100000:
        n += 1
        x_list = []
        y_list = []
        x, y, w, h = cv2.boundingRect(contours_in[i])
        y1 = y - 1
        x1 = x - 1
        h1 = h - 4
        w1 = w + 1
        x_list.append(x // 3.2)
        y_list.append(y // 3.2)
        out = image[y1:y1+h, x1:x1+w]
        if n % 2 == 0:
            cv2.imwrite('{}.jpg'.format(i), out)
            src = cv2.imread('{}.jpg'.format(i))
            #cv2.imshow('{}'.format(i), src)
            text = pytesseract.image_to_string(src, lang = 'eng')
            coord = pytesseract.image_to_data(src, output_type = Output.DICT)
            #print(coord)
            #print(text)
            n_box = len(coord['level'])
            w_list = []
            h_list = []
            for i, f in zip(range(1, n_box, 5), range(0, n_box, 4)):
                (x, y, w, h) = (coord['left'][i], coord['top'][i], coord['width'][i], coord['height'][i])
                w_list.append(w // 3.2)
                h_list.append(h // 3.2)            
            for i, f in zip (range(len(w_list)), range(len(x_list))):
                pdf.set_xy(x_list[f] + 15, y_list[f] + 15)
                pdf.cell(w_list[i], h_list[i], txt=text, align = 'C')
                
            
pdf.output('res.pdf')


cv2.waitKey()
cv2.destroyAllWindows






"""
import cv2
import pytesseract
import numpy as np
from fpdf import FPDF

src = cv2.imread('angl_tabl.png')
text = pytesseract.image_to_string(src, lang = 'eng')
print(text)

pdf = FPDF()
pdf.add_page()
pdf.set_font('Times', size=14)
pdf.write(10, text)
pdf.output('res.pdf')
"""

"""
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from fpdf import FPDF



image = cv2.imread('Act.sign.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                        cv2.THRESH_BINARY, 15, -2)
horizontal = np.copy(bw)
vertical = np.copy(bw)

cols = horizontal.shape[1]
horizontal_size = cols // 30

rows = vertical.shape[0]
vertical_size = rows // 30

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)

vertical = cv2.bitwise_not(vertical)

horizontal = cv2.bitwise_not(horizontal)

edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                cv2.THRESH_BINARY, 3, -2)

edges_1 = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                cv2.THRESH_BINARY, 3, -2)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours_1, hierarchy_1 = cv2.findContours(edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, (255, 255, 255), thickness = 3)

for i in range(len(contours_1)):
    cv2.drawContours(drawing, contours_1, i, (255, 255, 255))

cv2.imshow('1', drawing)
drawing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, (3, 3), iterations = 5)

edges_2 = cv2.Canny(drawing, 0, 255)

contours_3, hierarchy_3 = cv2.findContours(edges_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours_3)):
    if cv2.contourArea(contours_3[i]) > 5000:
        cv2.drawContours(drawing, contours_3, i, (255, 255, 255))
    else:
        cv2.drawContours(drawing, contours_3, i, (0, 0, 0), thickness = 3)

edges_in = cv2.Canny(drawing, 0, 255)
contours_in, hierarchy_in = cv2.findContours(edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


drawing = cv2.erode(drawing, (5, 5), iterations = 2)
pdf = FPDF(orientation='L', unit='mm', format='A4')
pdf.add_page()
pdf.add_font('DejaVu', '', '/home/user/.local/lib/python3.8/site-packages/fpdf/DejaVuSansCondensed.ttf', uni=True)
pdf.set_font('DejaVu', size=6)
n = 1
for i in range(len(contours_in)):
    if cv2.contourArea(contours_in[i]) < 100000:
        n += 1
        x_list = []
        y_list = []
        x, y, w, h = cv2.boundingRect(contours_in[i])
        y1 = y - 1
        x1 = x - 1
        h1 = h - 4
        w1 = w + 1
        x_list.append(x // 12)
        y_list.append(y // 12)
        out = image[y1:y1+h, x1:x1+w]
        if n % 2 == 0:
            cv2.imwrite('{}.jpg'.format(i), out)
            src = cv2.imread('{}.jpg'.format(i))
            #cv2.imshow('{}'.format(i), src)
            text = pytesseract.image_to_string(src, lang = 'ukr')
            coord = pytesseract.image_to_data(src, output_type = Output.DICT)
            #print(coord)
            #print(text)
            n_box = len(coord['level'])
            w_list = []
            h_list = []
            for i, f in zip(range(1, n_box, 5), range(0, n_box, 4)):
                (x, y, w, h) = (coord['left'][i], coord['top'][i], coord['width'][i], coord['height'][i])
                w_list.append(w // 8)
                h_list.append(h // 8)            
            for i, f in zip (range(len(w_list)), range(len(x_list))):
                pdf.set_xy(x_list[f],y_list[f])
                pdf.multi_cell(w_list[i], h_list[i], txt=text)
                
            

pdf.output('res.pdf')
cv2.waitKey()
cv2.destroyAllWindows
"""