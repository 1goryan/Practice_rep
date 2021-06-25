"""
import cv2
import numpy as np
import pytesseract

src = cv2.imread('blank.jpeg')
cv2.imshow('1', src)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                            cv2.THRESH_BINARY, 15, -2)


drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)

edges = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                    cv2.THRESH_BINARY, 3, -2)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, (0, 255, 255))


n = 1
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 8000 and cv2.contourArea(contours[i]) < 1000000:
        n += 1
        if n % 2 == 0:
                    
            x, y, w, h = cv2.boundingRect(contours[i])
            y1 = y - 2
            x1 = x - 2
            w += 2
            h += 2
            out = src[y1:y1+h, x1:x1+w]
            cv2.imwrite('blank/{}_blank.jpg'.format(i), out)



drawing = cv2.resize(drawing, (969, 684), interpolation=cv2.INTER_CUBIC)

cv2.imshow('1', drawing)

text = pytesseract.image_to_string(src, lang = 'rus', config = '--psm 6')
print(text)
cv2.waitKey()
cv2.destroyAllWindows
"""

"""
import cv2
import numpy as np
src = cv2.imread('blank/1.jpeg')


def compare(hist_img):

    template = cv2.imread('blank/template.jpg')
    template  = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    h_bins = 100
    v_bins = 100
    histSize = [h_bins, v_bins]

    h_ranges = [0, 180]
    v_ranges = [0, 100]
    ranges = h_ranges + v_ranges 
    channels = [0, 2]

    template_hist = cv2.calcHist([template], channels, None, histSize, ranges, accumulate=False)
    
    compare_method = cv2.HISTCMP_CORREL
       
    compare = cv2.compareHist(template_hist, hist_img, compare_method)
    
    if compare >= 0.9:
        return 1
    elif compare <0.9 and  compare > 0.2:
        return 2
    elif compare <=0.2:
        return 3
    else:
        print("error in comparing") 


def hsv_hist(img):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_bins = 100
    v_bins = 100
    histSize = [h_bins, v_bins]

    h_ranges = [0, 180]
    v_ranges = [0, 100]
    ranges = h_ranges + v_ranges 
    channels = [0, 2]
    hist_img = cv2.calcHist([hsv_img], channels, None, histSize, ranges, accumulate = False)
    
    return hist_img

res_list = []


def main():
    d = 0
    for i in range(4): 
    
        x = 2 + d
        y = 0
        h = 60
        w = 59
        d += 60
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/{}_blank.jpg'.format(i), out)
        h = hsv_hist(out)
        z = compare(h)
        if z == 1:
            res_list.append('empty')
        elif z == 2:
            res_list.append('right answer')
        elif z == 3:
            res_list.append('miss')

main()

print(res_list)


cv2.waitKey()
cv2.destroyAllWindows
"""

"""
import cv2
import numpy as np
import glob
import pytesseract

#src = cv2.imread('blank/123/persp.jpeg')
src = cv2.imread('blank/blank/src_last.jpg')
src = cv2.resize(src, (1241, 1755))
#cv2.imwrite('blank/123/blankkk.jpg', src)
#rc = cv2.imread('blank/123/blankkk.jpg')
src1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY )
#src1 = cv2.bitwise_not(src1)
#src1 = cv2.dilate(src1, (3,3), iterations = 8)
#cv2.imwrite('blank/123/srcbw.jpg', src1)
src1 = cv2.erode(src1, (3,3), iterations = 5)
src1 = cv2.dilate(src1, (3,3), iterations = 5)
height = 1755
width = 1241

hig = src.shape[0]
wid = src.shape[1]

kf_w = wid / width

kf_h = hig / height

kf_s = kf_h * kf_w


edges_in = cv2.Canny(src1, 50, 250)
contours_in, _ = cv2.findContours(edges_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
pointsx = []
pointsy = []

boundRect = [None]*len(contours_in)
contours_poly = [None]*len(contours_in)

drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
z = 0
for i in range(len(contours_in)):
    #if cv2.contourArea(contours_in[i]) > 25000 * kf_s and cv2.contourArea(contours_in[i]) < 300000 * kf_s:
    if cv2.arcLength(contours_in[i], True) > 650 and cv2.arcLength(contours_in[i], True) < 900:  
        
        cv2.drawContours(drawing, contours_in, i, (255, 255, 255), thickness = 1)
        
cv2.imwrite('blank/123/drawing.jpg', drawing)    

edges = cv2.Canny(drawing, 50, 250)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#for i in range(len(contours)):
#    cv2.drawContours(src, contours, i, (50, 0, 255), thickness = 1)

contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)

for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])

for i in range(len(contours)):
    if cv2.arcLength(contours[i], True) > 500 and cv2.contourArea(contours[i]) > 10:
    #if cv2.contourArea(contours[i]) > 1000 * kf_s and cv2.contourArea(contours[i]) < 3000000 * kf_s:
        color = (50, 0, 255)
        cv2.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)


  




cv2.imwrite('blank/123/blankk123.jpg', src)
d = {}
src = cv2.circle(src, (1112, 1460), 2, (0, 255, 255))
cv2.imwrite("blank/srccc.jpg", src)



"""


"""
edges_in = cv2.Canny(src1, 50, 250)
contours_in, _ = cv2.findContours(edges_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pointsx = []
pointsy = []
#c = max(contours_in, key=cv2.contourArea)
boundRect = [None]*len(contours_in)
contours_poly = [None]*len(contours_in)

drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
for i in range(len(contours_in)):
    if cv2.contourArea(contours_in[i]) > 300000 * kf_s and cv2.contourArea(contours_in[i]) < 2000000 * kf_s:
         
        #print(1)
        cv2.drawContours(drawing, contours_in, i, (255, 255, 255), thickness = 1)
        
        #a = contours_in[i]
        #print(a)
        #for i in range(len(a)):
            #pointsx.append(a[i][0][0])
            #pointsy.append(a[i][0][1])
        #points.append(a[0][0][0])
        #points.append(a[0][0][1])
#print(max(pointsx))
#print(max(pointsy))
#cv2.drawContours(src, c, i, (50, 0, 255),hierarhy = hierarhy, maxLevel = 1, thickness = 10)

edges = cv2.Canny(drawing, 50, 250)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#for i in range(len(contours)):
#    cv2.drawContours(src, contours, i, (50, 0, 255), thickness = 1)

contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)

for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])

for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 300000 * kf_s and cv2.contourArea(contours[i]) < 2000000 * kf_s:
        color = (50, 0, 255)
        cv2.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)


print(int(boundRect[0][0]))
print(int(boundRect[0][1]))   
cv2.imwrite("blank/srccc.jpg", src)
c = max(contours_in, key=cv2.contourArea)


#left = tuple(c[c[:, :, 0].argmin()][0])
right = tuple(c[c[:, :, 0].argmax()][0])
#top = tuple(c[c[:, :, 1].argmin()][0])
#bottom = tuple(c[c[:, :, 1].argmax()][0])

#cv2.circle(src, left, 3, (0, 50, 255), -1)
#cv2.circle(src, right, 3, (0, 255, 255), -1)
#cv2.circle(src, top, 3, (255, 50, 0), -1)
#cv2.circle(src, bottom, 3, (255, 255, 0), -1)
#print(right)

#cv2.imwrite('blank/123/blankk123.jpg', src)
d = {}
src = cv2.circle(src, (1112, 1460), 2, (0, 255, 255))
cv2.imwrite("blank/srccc.jpg", src)
"""

"""

def template():

    x = points[8] + 69 * kf_w
    x = int(x)
    y = points[9] + 65 * kf_h
    y = int(y)
    h = 38 * kf_h
    h = int(h)
    w = 34 * kf_w
    w = int(w)
    out = src[y:y+h, x:x+w]
    cv2.imwrite('blank/blank/template1.jpg', out)

template()

def hsv_hist(img):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_bins = 180
    v_bins = 250
    histSize = [h_bins, v_bins]

    h_ranges = [0, 180] #было 0-180
    v_ranges = [0, 250]
    ranges = h_ranges + v_ranges 
    channels = [0, 2]
    hist_img = cv2.calcHist([hsv_img], channels, None, histSize, ranges, accumulate = False)
    
    return hist_img


def compare(hist_img):

    template = cv2.imread('blank/blank/template1.jpg')
    template_hist  = hsv_hist(template)

    compare_method = cv2.HISTCMP_CORREL
       
    compare = cv2.compareHist(template_hist, hist_img, compare_method)
    print(compare)
    if compare >= 0.97:
        return 1
    elif compare < 0.97 and  compare > 0.9:
        return 2
    elif compare <=0.9:
        return 3
    else:
        print("error in comparing") 

def answer_1():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 65 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_1_{}.jpg'.format(i), out)


#src = cv2.line(src, (1054, 1028), (1054, 1553), (255, 0, 255))
#cv2.imwrite("blank/srccc.jpg", src)


def answer_2():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 107 * kf_h
        h = 38 * kf_h
        w = 34 * kf_h
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_2_{}.jpg'.format(i), out)


 
def answer_3():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 150 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_3_{}.jpg'.format(i), out)


def answer_4():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 193 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_4_{}.jpg'.format(i), out)

def answer_5():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 236 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_5_{}.jpg'.format(i), out)


def answer_6():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 279 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_6_{}.jpg'.format(i), out)



def answer_7():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 64 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_1_{}.jpg'.format(i), out)



def answer_8():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 107 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_2_{}.jpg'.format(i), out)

    
def answer_9():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 150 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_3_{}.jpg'.format(i), out)


def answer_10():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 193 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_4_{}.jpg'.format(i), out)


def answer_11():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 236 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_5_{}.jpg'.format(i), out)


def answer_12():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 279 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_6_{}.jpg'.format(i), out)


def answer_13():
    d = 0
    for i in range(5):
        x = points[8] + 706 * kf_w + d
        y = points[9] + 64 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_1_{}.jpg'.format(i), out)


def answer_14():
    d = 0
    for i in range(5):
        x = points[8] + 706 * kf_w + d
        y = points[9] + 107 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_2_{}.jpg'.format(i), out)

  

def answer_15():
    d = 0
    for i in range(5):
        x = points[8] + 706 * kf_w + d
        y = points[9] + 150 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_3_{}.jpg'.format(i), out)


def answer_16():
    d = 0
    for i in range(5):
        x = points[8] + 706 * kf_w + d
        y = points[9] + 193 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_4_{}.jpg'.format(i), out)


def answer_17():
    d = 0
    for i in range(5):
        x = points[8] + 706 * kf_w + d
        y = points[9] + 236 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_5_{}.jpg'.format(i), out)


def answer_18():
    d = 0
    for i in range(5):
        x = points[8] + 706 * kf_w + d
        y = points[9] + 279 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_h
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_6_{}.jpg'.format(i), out)


def answer_19():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 399 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_h
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_1_{}.jpg'.format(i), out)


def answer_20():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 442 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_2_{}.jpg'.format(i), out)


def answer_21():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 485 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_3_{}.jpg'.format(i), out)


def answer_22():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 528 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_4_{}.jpg'.format(i), out)


def answer_23():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w + d
        y = points[9] + 571 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_5_{}.jpg'.format(i), out)

def answer_24():
    d = 0
    for i in range(5):
        x = points[8] + 69 * kf_w+ d
        y = points[9] + 614 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_6_{}.jpg'.format(i), out)


def answer_25():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 399 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_1_{}.jpg'.format(i), out)



def answer_26():
    d = 0
    for i in range(5):
        x = points[8] + 387  * kf_w + d
        y = points[9] + 442 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_2_{}.jpg'.format(i), out)

def answer_27():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 485 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_3_{}.jpg'.format(i), out)


def answer_28():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 528 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_4_{}.jpg'.format(i), out)


def answer_29():
    d = 0
    for i in range(5):
        x = points[8] + 387  * kf_w + d
        y = points[9] + 571 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_5_{}.jpg'.format(i), out)


def answer_30():
    d = 0
    for i in range(5):
        x = points[8] + 387 * kf_w + d
        y = points[9] + 614 * kf_h
        h = 39 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_6_{}.jpg'.format(i), out)

def answer_31():
    d = 0
    for i in range(5):
        x = points[8] + 707 * kf_w + d
        y = points[9] + 399 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_1_{}.jpg'.format(i), out)


def answer_32():
    d = 0
    for i in range(5):
        x = points[8] + 707 * kf_w + d
        y = points[9] + 442 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_2_{}.jpg'.format(i), out)


def answer_33():
    d = 0
    for i in range(5):
        x = points[8] + 707 * kf_w + d
        y = points[9] + 485 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_3_{}.jpg'.format(i), out)

def answer_34():
    d = 0
    for i in range(5):
        x = points[8] + 707 * kf_w + d
        y = points[9] + 528 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_4_{}.jpg'.format(i), out)


def answer_35():
    d = 0
    for i in range(5):
        x = points[8] + 707 * kf_w + d
        y = points[9] + 570 * kf_h
        h = 39 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_5_{}.jpg'.format(i), out)


def answer_36():
    d = 0
    for i in range(5):
        x = points[8] + 707 * kf_w + d
        y = points[9] + 614 * kf_h
        h = 39 * kf_h
        w = 34 * kf_w
        d += 39 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_6_{}.jpg'.format(i), out)


def answer_37():
    d = 0
    for i in range(9):
        x = points[0] + 107 * kf_w + d
        y = points[1] + 73 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 40 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_1_{}.jpg'.format(i), out)


def answer_38():
    d = 0
    for i in range(9):
        x = points[0] + 107 * kf_w + d
        y = points[1] + 202 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 40 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_2_{}.jpg'.format(i), out)

def answer_39():
    d = 0
    for i in range(9):
        x = points[0] + 107 * kf_w + d
        y = points[1] + 331 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 40 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_3_{}.jpg'.format(i), out)


def answer_40():
    d = 0
    for i in range(9):
        x = points[0] + 602 * kf_w + d
        y = points[1] + 73 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 41 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_4_{}.jpg'.format(i), out)

def answer_41():
    d = 0
    for i in range(9):
        x = points[0] + 602 * kf_w + d
        y = points[1] + 202 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 41 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_5_{}.jpg'.format(i), out)

def answer_42():
    d = 0
    for i in range(9):
        x = points[0] + 602 * kf_w + d
        y = points[1] + 330 * kf_h
        h = 38 * kf_h
        w = 34 * kf_w
        d += 41 * kf_w
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_6_{}.jpg'.format(i), out)



"""
"""
answer_1()
answer_2()
answer_3()
answer_4()
answer_5()
answer_6()
answer_7()
answer_8()
answer_9()
answer_10()
answer_11()
answer_12()
answer_13()
answer_14()
answer_15()
answer_16()
answer_17()
answer_18()
answer_19()
answer_20()
answer_21()
answer_22()
answer_23()
answer_24()
answer_25()
answer_26()
answer_27()
answer_28()
answer_29()
answer_30()
answer_31()
answer_32()
answer_33()
answer_34()
answer_35()
answer_36()
answer_37()
answer_38()
answer_39()
answer_40()
answer_41()
answer_42()
"""
"""

def diction():
    namelist = []
    for file in glob.glob('blank/*.jpg'):
        namelist.append(file)
    namelist.sort()
    #print(len(namelist))

    for i in range(len(namelist)):    
        a = cv2.imread(namelist[i])
        a = hsv_hist(a) 
        a = compare(a)
        if a == 1:
            d[i+1]="Answer"
        elif a == 2:
            d[i+1]="Miss"
        elif a == 3:
            d[i+1]="Empty"
        else:
            print('error')

#diction()    
#print(d)
    
def score_miss(dictionary):
    miss = 0   
    for i in range(len(dictionary)):
        if dictionary.get(i) == 'Miss':
            miss += 1        
    print("Колличество исправлений равно: " + str(miss))    

def score_right_answers(dictionary):
    right_score = 0
    wrong_score = 0
    if dictionary[1] == "Answer" and (dictionary[2], dictionary[3], dictionary[4], dictionary[5]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[6] == "Answer" and (dictionary[7], dictionary[8], dictionary[9], dictionary[10]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[11] == "Answer" and (dictionary[12], dictionary[13], dictionary[14], dictionary[15]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[16] == "Answer" and (dictionary[17], dictionary[18], dictionary[19], dictionary[20]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[21] == "Answer" and (dictionary[22], dictionary[23], dictionary[24], dictionary[25]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[26] == "Answer" and (dictionary[27], dictionary[28], dictionary[29], dictionary[30]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[31] == "Answer" and (dictionary[32], dictionary[33], dictionary[34], dictionary[35]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[36] == "Answer" and (dictionary[37], dictionary[38], dictionary[39], dictionary[40]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[41] == "Answer" and (dictionary[42], dictionary[43], dictionary[44], dictionary[45]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[46] == "Answer" and (dictionary[47], dictionary[48], dictionary[49], dictionary[50]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[51] == "Answer" and (dictionary[52], dictionary[53], dictionary[54], dictionary[55]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[56] == "Answer" and (dictionary[57], dictionary[58], dictionary[59], dictionary[60]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[61] == "Answer" and (dictionary[62], dictionary[63], dictionary[64], dictionary[65]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[66] == "Answer" and (dictionary[67], dictionary[68], dictionary[69], dictionary[70]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[71] == "Answer" and (dictionary[72], dictionary[73], dictionary[74], dictionary[75]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[76] == "Answer" and (dictionary[77], dictionary[78], dictionary[79], dictionary[80]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[81] == "Answer" and (dictionary[82], dictionary[83], dictionary[84], dictionary[85]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[86] == "Answer" and (dictionary[87], dictionary[88], dictionary[89], dictionary[90]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[91] == "Answer" and (dictionary[92], dictionary[93], dictionary[94], dictionary[95]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[96] == "Answer" and (dictionary[97], dictionary[98], dictionary[99], dictionary[100]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[101] == "Answer" and (dictionary[102], dictionary[103], dictionary[104], dictionary[105]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[106] == "Answer" and (dictionary[107], dictionary[108], dictionary[109], dictionary[110]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[111] == "Answer" and (dictionary[112], dictionary[113], dictionary[114], dictionary[115]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[116] == "Answer" and (dictionary[117], dictionary[118], dictionary[119], dictionary[120]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[121] == "Answer" and (dictionary[122], dictionary[123], dictionary[124], dictionary[125]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[126] == "Answer" and (dictionary[127], dictionary[128], dictionary[129], dictionary[130]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[131] == "Answer" and (dictionary[132], dictionary[133], dictionary[134], dictionary[135]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[136] == "Answer" and (dictionary[137], dictionary[138], dictionary[139], dictionary[140]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[141] == "Answer" and (dictionary[142], dictionary[143], dictionary[144], dictionary[145]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[146] == "Answer" and (dictionary[147], dictionary[148], dictionary[149], dictionary[150]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[151] == "Answer" and (dictionary[152], dictionary[153], dictionary[154], dictionary[155]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[156] == "Answer" and (dictionary[157], dictionary[158], dictionary[159], dictionary[160]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[161] == "Answer" and (dictionary[162], dictionary[163], dictionary[164], dictionary[165]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[166] == "Answer" and (dictionary[167], dictionary[168], dictionary[169], dictionary[170]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[171] == "Answer" and (dictionary[172], dictionary[173], dictionary[174], dictionary[175]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[176] == "Answer" and (dictionary[177], dictionary[178], dictionary[179], dictionary[180]) != "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[181] == "Answer":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[182] != "Empty" or dictionary[182] != "Miss":
        wrong_score += 1
    
    if dictionary[183] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[184] != "Empty" or dictionary[184] != "Miss":
        wrong_score += 1    

    if dictionary[185] != "Empty" or dictionary[185] != "Miss":
        wrong_score += 1

    if dictionary[186] != "Empty" or dictionary[186] != "Miss":
        wrong_score += 1

    if dictionary[187] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[188] != "Empty" or dictionary[188] != "Miss":
        wrong_score += 1

    if dictionary[189] != "Empty" or dictionary[189] != "Miss":
        wrong_score += 1

    if dictionary[190] != "Empty" or dictionary[190] != "Miss":
        wrong_score += 1
    
    if dictionary[191] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[192] != "Empty" or dictionary[192] != "Miss":
        wrong_score += 1

    if dictionary[193] != "Empty" or dictionary[193] != "Miss":
        wrong_score += 1

    if dictionary[194] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[195] != "Empty" or dictionary[195] != "Miss":
        wrong_score += 1

    if dictionary[196] != "Empty" or dictionary[196] != "Miss":
        wrong_score += 1

    if dictionary[197] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[198] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[199] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[200] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[201] != "Empty" or dictionary[201] != "Miss":
        wrong_score += 1

    if dictionary[202] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[203] != "Empty" or dictionary[203] != "Miss":
        wrong_score += 1
    
    if dictionary[204] != "Empty" or dictionary[204] != "Miss":
        wrong_score += 1

    if dictionary[205] != "Empty" or dictionary[205] != "Miss":
        wrong_score += 1

    if dictionary[206] != "Empty" or dictionary[206] != "Miss":
        wrong_score += 1

    if dictionary[207] != "Empty" or dictionary[207] != "Miss":
        wrong_score += 1

    if dictionary[208] == "Answer":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[209] != "Empty" or dictionary[209] != "Miss":
        wrong_score += 1

    if dictionary[210] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[211] != "Empty" or dictionary[211] != "Miss":
        wrong_score += 1
    
    if dictionary[212] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[213] != "Empty" or dictionary[213] != "Miss":
        wrong_score += 1

    if dictionary[214] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[215] != "Empty" or dictionary[215] != "Miss":
        wrong_score += 1

    if dictionary[216] != "Empty" or dictionary[216] != "Miss":
        wrong_score += 1
    
    if dictionary[217] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[218] != "Empty" or dictionary[218] != "Miss":
        wrong_score += 1

    if dictionary[219] != "Empty" or dictionary[219] != "Miss":
        wrong_score += 1

    if dictionary[220] != "Empty" or dictionary[220] != "Miss":
        wrong_score += 1

    if dictionary[221] != "Empty" or dictionary[221] != "Miss":
        wrong_score += 1

    if dictionary[222] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[223] != "Empty" or dictionary[223] != "Miss":
        wrong_score += 1

    if dictionary[224] != "Empty" or dictionary[224] != "Miss":
        wrong_score += 1

    if dictionary[225] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[226] != "Empty" or dictionary[226] != "Miss":
        wrong_score += 1

    if dictionary[227] != "Empty" or dictionary[227] != "Miss":
        wrong_score += 1

    if dictionary[228] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[229] != "Empty" or dictionary[229] != "Miss":
        wrong_score += 1
    
    if dictionary[230] != "Empty" or dictionary[230] != "Miss":
        wrong_score += 1
    
    if dictionary[231] != "Empty" or dictionary[231] != "Miss":
        wrong_score += 1

    if dictionary[232] != "Empty" or dictionary[232] != "Miss":
        wrong_score += 1

    if dictionary[233] == "Answer":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[234] != "Empty" or dictionary[234] != "Miss":
        wrong_score += 1
    print("Колличество правильных ответов равно: " + str(right_score)) 




#score_right_answers(d)

#score_miss(d)


def blank_name():
    
    x = points[8] + 365 * kf_w
    y = points[9] - 255 * kf_h
    h = 50 * kf_h
    w = 310 * kf_w
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text


def user_name():

    x = points[8] + 100 * kf_w
    y = points[9] - 210 * kf_h
    h = 510 * kf_h
    w = 60 * kf_w
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text

def group():

    x = points[8] + 125 * kf_w
    y = points[9] - 140 * kf_h
    h = 485 * kf_h
    w = 68 * kf_w
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text

print(points[0])

def data():
    
    x = points[0] + 4 * kf_w
    y = points[1] + 394 * kf_h
    h = 73 * kf_h
    w = 480 * kf_w
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text


def sign():
    x = points[0] + 579 *kf_w
    y = points[1] + 394 * kf_h
    h = 73 * kf_h
    w = 485 * kf_w
    out = src[y:y+h, x:x+w]
    img = cv2.imwrite("blank/blank/sign.jpg", out)
    return img



#cv2.imwrite('blank/blank/src_123.jpg', src)



"""
"""
    for i in range(5):
        a = cv2.imread('blank/1_2_{}.jpg'.format(i))
        a = hsv_hist(a) 
        a = compare(a)
        if a == 1:
            d[i+6]="Answer"
        elif a ==2:
            d[i+6]="Miss"
        elif a==3:
            d[i+6]="Empty"
        else:
            print('error')
    
    for i in range(5):
        a = cv2.imread('blank/1_3_{}.jpg'.format(i))
        a = hsv_hist(a) 
        a = compare(a)
        if a == 1:
            d[i+11]="Answer"
        elif a ==2:
            d[i+11]="Miss"
        elif a==3:
            d[i+11]="Empty"
        else:
            print('error')
    
    for i in range(5):
        a = cv2.imread('blank/1_4_{}.jpg'.format(i))
        a = hsv_hist(a) 
        a = compare(a)
        if a == 1:
            d[i+16]="Answer"
        elif a ==2:
            d[i+16]="Miss"
        elif a==3:
            d[i+16]="Empty"
        else:
            print('error')

    for i in range(5):
        a = cv2.imread('blank/1_5_{}.jpg'.format(i))
        a = hsv_hist(a) 
        a = compare(a)
        if a == 1:
            d[i+21]="Answer"
        elif a ==2:
            d[i+21]="Miss"
        elif a==3:
            d[i+21]="Empty"
        else:
            print('error')

    for i in range(5):
        a = cv2.imread('blank/1_6_{}.jpg'.format(i))
        a = hsv_hist(a) 
        a = compare(a)
        if a == 1:
            d[i+26]="Answer"
        elif a ==2:
            d[i+26]="Miss"
        elif a==3:
            d[i+26]="Empty"
        else:
            print('error')
    """
     























"""
import cv2
import numpy as np


img1 = cv2.imread('blank/template.jpg')
cv2.imshow('img1', img1)
img2 = cv2.imread('blank/0_1.jpg')
cv2.imshow('img2', img2)
img3 = cv2.imread('blank/1_1.jpg')
cv2.imshow('img3', img3)
img4 = cv2.imread('blank/2_1.jpg')
cv2.imshow('img4', img4)
img5 = cv2.imread('blank/3_1.jpg')
cv2.imshow('img5', img5)




hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
hsv_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
hsv_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
hsv_img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2HSV)



h_bins = 180
v_bins = 250
histSize = [h_bins, v_bins]

h_ranges = [0, 180]
v_ranges = [0, 250]
ranges = h_ranges + v_ranges 

channels = [0, 1]
compare_method = cv2.HISTCMP_CORREL
hist_img1 = cv2.calcHist([hsv_img1], channels, None, histSize, ranges, accumulate=False)
hist_img2 = cv2.calcHist([hsv_img2], channels, None, histSize, ranges, accumulate=False)
hist_img3 = cv2.calcHist([hsv_img3], channels, None, histSize, ranges, accumulate=False)
hist_img4 = cv2.calcHist([hsv_img4], channels, None, histSize, ranges, accumulate=False)
hist_img5 = cv2.calcHist([hsv_img5], channels, None, histSize, ranges, accumulate=False)

img1_img1 = cv2.compareHist(hist_img1, hist_img1, compare_method)
img1_img2 = cv2.compareHist(hist_img1, hist_img2, compare_method)
img1_img3 = cv2.compareHist(hist_img1, hist_img3, compare_method)
img1_img4 = cv2.compareHist(hist_img1, hist_img4, compare_method)
img1_img5 = cv2.compareHist(hist_img1, hist_img5, compare_method)
print(img1_img1)
print(img1_img2)
print(img1_img3)
print(img1_img4)
print(img1_img5)


cv2.waitKey()
cv2.destroyAllWindows
"""

from fpdf import FPDF
import cv2
import numpy as np
import glob
import pytesseract

src = cv2.imread('blank/123/persp1.jpeg')
#src = cv2.imread('blank/blank/src_last1.jpg')
#src = cv2.resize(src, (1241, 1755))
#cv2.imwrite('blank/123/blankkk.jpg', src)
#rc = cv2.imread('blank/123/blankkk.jpg')
src1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY )
#src1 = cv2.bitwise_not(src1)
#src1 = cv2.dilate(src1, (3,3), iterations = 8)
#cv2.imwrite('blank/123/srcbw.jpg', src1)
#src1 = cv2.erode(src1, (3,3), iterations = 5)
#src1 = cv2.dilate(src1, (3,3), iterations = 5)
height = 1755
width = 1241

hig_src = src.shape[0]
wid_src = src.shape[1]

kf_w = wid_src / width

kf_h = hig_src / height

kf_s = kf_h * kf_w

x = wid_src // 17.73 
y = hig_src // 5 
h = hig_src // 2.44 

w = wid_src - 2*x
x = int(x)
y = int(y)
h = int(h)
w = int(w)

out = src[y:y+h, x:x+w]
cv2.imwrite('blank/out1.jpg', out) 

gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 170, 255, 0)
#cv2.imwrite('blank/out1.jpg', th) 
contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours.sort(key = lambda x: cv2.contourArea(x))
contour_number = len(contours) - 2
#cv2.drawContours(out, contours, contour_number , (50, 0, 255), thickness = 3)
#cv2.imwrite('blank/out1.jpg', out) 

minRect = [None]*1
minRect[0] = cv2.minAreaRect(contours[contour_number])
box = cv2.boxPoints(minRect[0])
box = np.intp(box)
#print(box)
#cv2.drawContours(out, [box], 0, (0, 255, 50), thickness = 3)

cv2.imwrite('blank/out1.jpg', out)        

xlist = []
ylist = []

for i in range(len(box)):
    xlist.append(box[i][0])
    ylist.append(box[i][1])

x_min = min(xlist)
y_min = min(ylist)
x_max = max(xlist)
y_max = max(ylist)

h = y_max - y_min 
w = x_max - x_min 


h_sd = hig_src // 102
w_sd = wid_src // 75.8
h_sd = int(h_sd)
w_sd = int(w_sd)
out = out[y_min+h_sd:y_min+h-h_sd, x_min+w_sd:x_min+w-w_sd]

hig = out.shape[0]
wid = out.shape[1]

h_05 = hig // 2
w_03 = wid // 3

#cv2.line(src, (70 , 1063) , (70, 1065 ), (50, 0, 255), thickness = 3)
#cv2.imwrite('blank/blankk123.jpg', src)

higs = src.shape[0]
wids = src.shape[1]

x = wids // 17.73 
y = higs // 1.65 
h = higs // 4.2 

w = wids - 2 * x 
x = int(x)
y = int(y)
h = int(h)
w = int(w)

outs = src[y:y+h, x:x+w]
cv2.imwrite('blank/outs.jpg', outs) 


gray = cv2.cvtColor(outs, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 170, 255, 0)
#cv2.imwrite('blank/outs.jpg', th) 
contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours.sort(key = lambda x: cv2.contourArea(x))
contour_number = len(contours) - 2
#cv2.drawContours(outs, contours, contour_number , (50, 0, 255), thickness = 3)
#cv2.imwrite('blank/outs.jpg', outs) 

minRect = [None]*1
minRect[0] = cv2.minAreaRect(contours[contour_number])
box = cv2.boxPoints(minRect[0])
box = np.intp(box)
#print(box)
#cv2.drawContours(outs, [box], 0, (0, 255, 50), thickness = 3)
#cv2.imwrite('blank/outs.jpg', outs) 
xlist = []
ylist = []

for i in range(len(box)):
    xlist.append(box[i][0])
    ylist.append(box[i][1])

x_min = min(xlist)
y_min = min(ylist)
x_max = max(xlist)
y_max = max(ylist)

h = y_max - y_min 
w = x_max - x_min 

h_sd = hig // 50
w_sd = wid // 75.8
h_sd = int(h_sd)
w_sd = int(w_sd)
outs = outs[y_min+h_sd:y_min+h-h_sd, x_min+w_sd:x_min+w-w_sd]

cv2.imwrite('blank/outs.jpg', outs) 

higs = outs.shape[0]
wids = outs.shape[1]

hs_03 = higs // 3
ws_05 = wids // 2

def block_1():

    out2 = out[0:h_05, 0:w_03]
    cv2.imwrite('blank/out21.jpg', out2)     
    
    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    cv2.imwrite('blank/out21.jpg', out2) 
    
    pts2 = np.float32([[0, 0], [200, 0], [0, 240], [200, 240]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out21.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    #for i in range(1, len(corners)):
        #print(corners[i])

    img[dst>0.1*dst.max()]=[0,0,255]

    for i in range(1, len(corners)):
        #print(corners[i,0])
        cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)

    #cv2.imshow('image', img)
    #cv2.waitKey(0)

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]

       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h
    
    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (200, 240))

    cv2.imwrite('blank/out211.jpg', result)

    #gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #ret, th = cv2.threshold(gray, 170, 255, 0)
    #contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours.sort(key = lambda x: cv2.contourArea(x))
    #for i in range(len(contours)):
    #    cv2.drawContours(result, contours, i , (50, 50, 255), thickness = 1)
    hh = result.shape[0] // 6
    ww = result.shape[1] // 5
    
    for f in range(5):
        for i in range(6):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8     
            cell = result[y:y+h, x:x+w]
            #gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            ret, th = cv2.threshold(gray, 180, 255, 0)
            #cell = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
            #cell = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            cv2.imwrite('blank/cells/cell0_{}{}.jpg'.format(i,f), cell) 


def block_2():

    out2 = out[0:h_05, w_03:w_03*2]
    cv2.imwrite('blank/out22.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    cv2.imwrite('blank/out22.jpg', out2) 
    
    pts2 = np.float32([[0, 0], [200, 0], [0, 240], [200, 240]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out22.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    img[dst>0.1*dst.max()]=[0,0,255]
 
    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])
    cv2.imwrite('pers/qwerty12.jpeg', img2)
    cv2.imwrite('pers/qwerty1.jpeg', img)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (200, 240))

    cv2.imwrite('blank/out221.jpg', result)

    cv2.waitKey()

    hh = result.shape[0] // 6
    ww = result.shape[1] // 5
    
    for f in range(5):
        for i in range(6):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            cv2.imwrite('blank/cells/cell1_{}{}.jpg'.format(i,f), cell) 

def block_3():

    out2 = out[0:h_05, w_03*2:w_03*3]
    cv2.imwrite('blank/out23.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    cv2.imwrite('blank/out23.jpg', out2) 

    pts2 = np.float32([[0, 0], [200, 0], [0, 240], [200, 240]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out23.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (200, 240))

    cv2.imwrite('blank/out231.jpg', result)

    hh = result.shape[0] // 6
    ww = result.shape[1] // 5
    
    for f in range(5):
        for i in range(6):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            cv2.imwrite('blank/cells/cell2_{}{}.jpg'.format(i,f), cell) 

def block_4():

    out2 = out[h_05:h_05*2, 0:w_03]
    cv2.imwrite('blank/out24.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    cv2.imwrite('blank/out24.jpg', out2) 
    
    pts2 = np.float32([[0, 0], [200, 0], [0, 240], [200, 240]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out24.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        
    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
   
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (200, 240))

    cv2.imwrite('blank/out241.jpg', result)  
    
    hh = result.shape[0] // 6
    ww = result.shape[1] // 5
    
    for f in range(5):
        for i in range(6):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            cv2.imwrite('blank/cells/cell3_{}{}.jpg'.format(i,f), cell)

def block_5():

    out2 = out[h_05:h_05*2, w_03:w_03*2]
    cv2.imwrite('blank/out25.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    cv2.imwrite('blank/out25.jpg', out2) 
    
    pts2 = np.float32([[0, 0], [200, 0], [0, 240], [200, 240]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out25.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (200, 240))

    cv2.imwrite('blank/out251.jpg', result)

    hh = result.shape[0] // 6
    ww = result.shape[1] // 5
    
    for f in range(5):
        for i in range(6):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            cv2.imwrite('blank/cells/cell4_{}{}.jpg'.format(i,f), cell)

def block_6():

    out2 = out[h_05:h_05*2, w_03*2:w_03*3]
    cv2.imwrite('blank/out26.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    cv2.imwrite('blank/out26.jpg', out2) 
    
    pts2 = np.float32([[0, 0], [200, 0], [0, 240], [200, 240]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out26.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        
    img[dst>0.1*dst.max()]=[0,0,255]
    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
 
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (200, 240))

    cv2.imwrite('blank/out261.jpg', result)

    hh = result.shape[0] // 6
    ww = result.shape[1] // 5
    
    for f in range(5):
        for i in range(6):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            #gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            #ret, th = cv2.threshold(gray, 170, 255, 0)
            cv2.imwrite('blank/cells/cell5_{}{}.jpg'.format(i,f), cell)





def block_7():

    out2 = outs[0:hs_03, ws_05*0:ws_05*1]
    cv2.imwrite('blank/out31.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    #cv2.imwrite('blank/out31.jpg', drawing) 

    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out31.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (360, 80))

    cv2.imwrite('blank/out311.jpg', result)

    hh = result.shape[0] // 2
    ww = result.shape[1] // 9
    
    for f in range(9):
        
        for i in range(2):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            if i > 0:
                cv2.imwrite('blank/cells/cell61_{}{}.jpg'.format(i,f), cell) 

def block_8():

    out2 = outs[0:hs_03, ws_05*1:ws_05*2]
    cv2.imwrite('blank/out32.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    #cv2.imwrite('blank/out31.jpg', drawing) 

    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out32.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (360, 80))

    cv2.imwrite('blank/out322.jpg', result)

    hh = result.shape[0] // 2
    ww = result.shape[1] // 9
    
    for f in range(9):
        
        for i in range(2):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            if i > 0:
                cv2.imwrite('blank/cells/cell62_{}{}.jpg'.format(i,f), cell) 
            

def block_9():

    out2 = outs[hs_03:hs_03*2, ws_05*0:ws_05*1]
    cv2.imwrite('blank/out33.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    #cv2.imwrite('blank/out31.jpg', drawing) 

    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out33.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (360, 80))

    cv2.imwrite('blank/out333.jpg', result)

    hh = result.shape[0] // 2
    ww = result.shape[1] // 9
    
    for f in range(9):
        
        for i in range(2):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            if i > 0:
                cv2.imwrite('blank/cells/cell63_{}{}.jpg'.format(i,f), cell) 

def block_10():

    out2 = outs[hs_03:hs_03*2, ws_05*1:ws_05*2]
    cv2.imwrite('blank/out34.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    #cv2.imwrite('blank/out31.jpg', drawing) 

    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out34.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (360, 80))

    cv2.imwrite('blank/out344.jpg', result)

    hh = result.shape[0] // 2
    ww = result.shape[1] // 9
    
    for f in range(9):
        
        for i in range(2):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            if i > 0:
                cv2.imwrite('blank/cells/cell64_{}{}.jpg'.format(i,f), cell) 


def block_11():

    out2 = outs[hs_03*2:hs_03*3, ws_05*0:ws_05*1]
    cv2.imwrite('blank/out35.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)
    
    #cv2.imwrite('blank/draw.jpg', drawing) 

    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out35.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    #for i in range(1, len(corners)):
        #print(corners[i])

    img[dst>0.1*dst.max()]=[0,0,255]

    #for i in range(1, len(corners)):
        #print(corners[i,0])
        #cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)

    #cv2.imshow('image', img)
    #cv2.waitKey(0)

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (360, 80))

    cv2.imwrite('blank/out355.jpg', result)

    hh = result.shape[0] // 2
    ww = result.shape[1] // 9
    
    for f in range(9):
        
        for i in range(2):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            if i > 0:
                cv2.imwrite('blank/cells/cell65_{}{}.jpg'.format(i,f), cell) 


def block_12():

    out2 = outs[hs_03*2:hs_03*3, ws_05*1:ws_05*2]
    cv2.imwrite('blank/out36.jpg', out2)

    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((out2.shape[0], out2.shape[1], 3), dtype=np.uint8)
    contours.sort(key = lambda x: cv2.contourArea(x))
    contour_number = len(contours) - 2
    cv2.drawContours(drawing, contours, contour_number , (255, 255, 255), thickness = 3)

    #cv2.imwrite('blank/out31.jpg', drawing) 

    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])
    img = drawing
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,50,29,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    img2 = cv2.imread('blank/out36.jpg')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    img[dst>0.1*dst.max()]=[0,0,255]

    k_h = out2.shape[0] / img.shape[0]
    k_w = out2.shape[1] / img.shape[1]
       
    a10 = corners[1][0] * k_w
    a11 = corners[1][1] * k_h
    a20 = corners[2][0] * k_w
    a21 = corners[2][1] * k_h
    a30 = corners[3][0] * k_w
    a31 = corners[3][1] * k_h
    a40 = corners[4][0] * k_w
    a41 = corners[4][1] * k_h

    a1 = a10 + a11
    a2 = a20 + a21
    a3 = a30 + a31
    a4 = a40 + a41 

    if a1 < a2 and a3 < a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a30, a31], [a40, a41]])
    elif a1 > a2 and a3 < a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a30, a31], [a40, a41]])
    elif a1 < a2 and a3 > a4:
        pts1 = np.float32([[a10, a11], [a20, a21], [a40, a41], [a30, a31]])
    elif a1 > a2 and a3 > a4:
        pts1 = np.float32([[a20, a21], [a10, a11], [a40, a41], [a30, a31]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (360, 80))

    cv2.imwrite('blank/out366.jpg', result)

    hh = result.shape[0] // 2
    ww = result.shape[1] // 9
    
    for f in range(9):
        
        for i in range(2):
            x = ww * f+4  
            y = hh * i+4  
            h = hh -7      
            w = ww -8      
            cell = result[y:y+h, x:x+w]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            if i > 0:
                cv2.imwrite('blank/cells/cell66_{}{}.jpg'.format(i,f), cell) 

block_1()
block_2()
block_3()
block_4()
block_5()
block_6()
block_7()
block_8()
block_9()
block_10()
block_11()
block_12()



def data():
    
    x = wid_src // 12.41 
    y = hig_src // 1.186 
    h = hig_src // 25 
    w = wid_src // 2.48
    
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)

    out2 = src[y:y+h, x:x+w]
    cv2.imwrite('blank/data.jpg', out2) 
    text = pytesseract.image_to_string(out2, config = '--psm 6 --oem 3 ')
    print(text)
    return text

b = data()


def sign():
    
    x = wid_src // 1.91 
    y = hig_src // 1.19 
    h = hig_src // 8.775 
    w = wid_src // 2.48
    
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)

    out2 = src[y:y+h, x:x+w]
    cv2.imwrite('blank/sign.jpg', out2) 
    
#cv2.line(src, (800 , 50) , (800, 3500), (50, 0, 255), thickness = 3)
#cv2.imwrite('blank/blankk123.jpg', src)

def name():
    
    x = wid_src // 10.342 
    y = hig_src // 11.7 
    h = hig_src // 8.775 
    w = wid_src // 1.55
    
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)

    out2 = src[y:y+h, x:x+w]

    cv2.imwrite('blank/name.jpg', out2)
    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 170, 255, 0) 
    cv2.imwrite('blank/name.jpg', th)
    text = pytesseract.image_to_string(th, lang = 'rus' , config = '--psm 6')
    #print(text)
    return text

name()


def template():

    img = cv2.imread('blank/template/cell1_00.jpg')
    return img

def hsv_hist(img):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    h_ranges = [0, 180] #было 0-180
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]
    
    hist_img = cv2.calcHist([hsv_img], channels, None, histSize, ranges, accumulate = False)
    #cv2.normalize(hist_img, hist_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return hist_img

def compare(hist_img):

    temp = template()
    template_hist  = hsv_hist(temp)
    
    compare_method = cv2.HISTCMP_INTERSECT
       
    compare = cv2.compareHist(template_hist, hist_img, compare_method)
    #print(compare)
    if compare >=376:
        return 1
    elif compare <376:
        return 2
    
    #if compare >= 0.98:
    #    return 1
    #elif compare < 0.98 and  compare > 0.94:
    #    return 2
    #elif compare <=94:
    #    return 3
    else:
        print("error in comparing") 

d = {}
def diction():
    namelist = []
    for file in glob.glob('blank/cells/*.jpg'):
        namelist.append(file)
    namelist.sort()
    #print(namelist)
    #print(len(namelist))

    for i in range(len(namelist)):    
        a = cv2.imread(namelist[i])
        a = hsv_hist(a)
        #print(i+1) 
        a = compare(a)
        if a == 1:
            d[i+1]="Filled"
        #elif a == 3:
        #    d[i+1]="Miss"
        elif a == 2:
            d[i+1]="Empty"
        else:
            print('error')

diction()    
#print(d)

 
def score_right_answers(dictionary):
    right_score = 0
    wrong_score = 0
    if dictionary[1] == "Filled" and dictionary[2] != "Filled" and dictionary[3] != "Filled" and dictionary[4] != "Filled" and dictionary[5] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[8] == "Filled" and dictionary[6] != "Filled" and dictionary[7] != "Filled" and dictionary[9] != "Filled" and dictionary[10] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[15] == "Filled" and dictionary[11] != "Filled" and dictionary[12] != "Filled" and dictionary[13] != "Filled" and dictionary[14] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[17] == "Filled" and dictionary[16] != "Filled" and dictionary[18] != "Filled" and dictionary[19] != "Filled" and dictionary[20] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[25] == "Filled" and dictionary[21] != "Filled" and dictionary[22] != "Filled" and dictionary[23] != "Filled" and dictionary[24] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[26] == "Filled" and dictionary[27] != "Filled" and dictionary[28] != "Filled" and dictionary[29] != "Filled" and dictionary[30] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    

    if dictionary[32] == "Filled" and dictionary[31] != "Filled" and dictionary[33] != "Filled" and dictionary[34] != "Filled" and dictionary[35] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[38] == "Filled" and dictionary[36] != "Filled" and dictionary[37] != "Filled" and dictionary[39] != "Filled" and dictionary[40] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[44] == "Filled" and dictionary[41] != "Filled" and dictionary[42] != "Filled" and dictionary[43] != "Filled" and dictionary[45] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[50] == "Filled" and dictionary[46] != "Filled" and dictionary[47] != "Filled" and dictionary[48] != "Filled" and dictionary[49] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[54] == "Filled" and dictionary[51] != "Filled" and dictionary[52] != "Filled" and dictionary[53] != "Filled" and dictionary[55] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[58] == "Filled" and dictionary[56] != "Filled" and dictionary[57] != "Filled" and dictionary[59] != "Filled" and dictionary[60] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    

    if dictionary[61] == "Filled" and dictionary[62] != "Filled" and dictionary[63] != "Filled" and dictionary[64] != "Filled" and dictionary[65] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[67] == "Filled" and dictionary[66] != "Filled" and dictionary[68] != "Filled" and dictionary[69] != "Filled" and dictionary[70] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[71] == "Filled" and dictionary[72] != "Filled" and dictionary[73] != "Filled" and dictionary[74] != "Filled" and dictionary[75] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[77] == "Filled" and dictionary[76] != "Filled" and dictionary[78] != "Filled" and dictionary[79] != "Filled" and dictionary[80] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[81] == "Filled" and dictionary[82] != "Filled" and dictionary[83] != "Filled" and dictionary[84] != "Filled" and dictionary[85] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[86] == "Filled" and dictionary[87] != "Filled" and dictionary[88] != "Filled" and dictionary[89] != "Filled" and dictionary[90] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

   

    if dictionary[92] == "Filled" and dictionary[91] != "Filled" and dictionary[93] != "Filled" and dictionary[94] != "Filled" and dictionary[95] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[100] == "Filled" and dictionary[96] != "Filled" and dictionary[97] != "Filled" and dictionary[98] != "Filled" and dictionary[99] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[103] == "Filled" and dictionary[101] != "Filled" and dictionary[102] != "Filled" and dictionary[104] != "Filled" and dictionary[105] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[106] == "Filled" and dictionary[107] != "Filled" and dictionary[108] != "Filled" and dictionary[109] != "Filled" and dictionary[110] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[115] == "Filled" and dictionary[111] != "Filled" and dictionary[112] != "Filled" and dictionary[113] != "Filled" and dictionary[114] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[117] == "Filled" and dictionary[116] != "Filled" and dictionary[118] != "Filled" and dictionary[119] != "Filled" and dictionary[120] != "Filled":
        right_score += 1
    else:
        wrong_score += 1


    

    if dictionary[121] == "Filled" and dictionary[122] != "Filled" and dictionary[123] != "Filled" and dictionary[124] != "Filled" and dictionary[125] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[129] == "Filled" and dictionary[126] != "Filled" and dictionary[127] != "Filled" and dictionary[128] != "Filled" and dictionary[130] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[133] == "Filled" and dictionary[131] != "Filled" and dictionary[132] != "Filled" and dictionary[134] != "Filled" and dictionary[135] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[140] == "Filled" and dictionary[136] != "Filled" and dictionary[137] != "Filled" and dictionary[138] != "Filled" and dictionary[139] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[141] == "Filled" and dictionary[142] != "Filled" and dictionary[143] != "Filled" and dictionary[144] != "Filled" and dictionary[145] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[150] == "Filled" and dictionary[146] != "Filled" and dictionary[147] != "Filled" and dictionary[148] != "Filled" and dictionary[149] != "Filled":
        right_score += 1
    else:
        wrong_score += 1




    if dictionary[151] == "Filled" and dictionary[152] != "Filled" and dictionary[153] != "Filled" and dictionary[154] != "Filled" and dictionary[155] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[158] == "Filled" and dictionary[156] != "Filled" and dictionary[157] != "Filled" and dictionary[159] != "Filled" and dictionary[160] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[165] == "Filled" and dictionary[161] != "Filled" and dictionary[162] != "Filled" and dictionary[163] != "Filled" and dictionary[164] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[168] == "Filled" and dictionary[166] != "Filled" and dictionary[167] != "Filled" and dictionary[169] != "Filled" and dictionary[170] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[175] == "Filled" and dictionary[171] != "Filled" and dictionary[172] != "Filled" and dictionary[173] != "Filled" and dictionary[174] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[179] == "Filled" and dictionary[176] != "Filled" and dictionary[177] != "Filled" and dictionary[178] != "Filled" and dictionary[180] != "Filled":
        right_score += 1
    else:
        wrong_score += 1





    if dictionary[181] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[182] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[183] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[184] == "Filled":
        right_score += 1
    else:
        wrong_score += 1   

    if dictionary[185] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[186] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[187] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[188] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[189] != "Filled":
        right_score += 1
    else:
        wrong_score += 1



    if dictionary[190] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[191] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[192] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[193] == "Filled":
        right_score += 1
    else:
        wrong_score += 1   

    if dictionary[194] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[195] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[196] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[197] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[198] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    

    
    if dictionary[199] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[200] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[201] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[202] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[203] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[204] == "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[205] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[206] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[207] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[208] == "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[209] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[210] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[211] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[212] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[213] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[214] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[215] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[216] != "Filled":
        right_score += 1
    else:
        wrong_score += 1



    if dictionary[217] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[218] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[219] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[220] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[221] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[222] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[223] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[224] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[225] == "Filled":
        right_score += 1
    else:
        wrong_score += 1




    if dictionary[226] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[227] != "Filled":
        right_score += 1
    else:
        wrong_score += 1
    
    if dictionary[228] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[229] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[230] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[231] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[232] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[233] == "Filled":
        right_score += 1
    else:
        wrong_score += 1

    if dictionary[234] != "Filled":
        right_score += 1
    else:
        wrong_score += 1

    #print("Колличество правильных ответов равно: " + str(right_score)) 
    #print("Колличество ошибочных ответов равно: " + str(wrong_score)) 
    return right_score, wrong_score




import re
a = name()
score = score_right_answers(d)








def pdf2(text, dat):
    dat = re.sub("[^а-яі.'*єЇ()0-9]+",'', dat, flags=re.IGNORECASE)
    text = text.split('\n')
    pdf=FPDF("P", "mm", "A4")
    pdf.add_page()
    pdf.add_font('DejaVu', '', '/home/user/.local/lib/python3.8/site-packages/fpdf/DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', size=14)
    pdf.set_xy(100, 25)
    pdf.cell(10, 10, txt = text[0], border = 0, align = 'C')
    pdf.set_xy(30, 38)
    pdf.cell(10, 10, txt = text[1], border = 0, align = 'L')
    pdf.line(43, 45, 120, 45)
    pdf.set_xy(30, 48)
    pdf.cell(10, 10, txt = text[2], border = 0, align = 'L')
    pdf.line(49, 55, 120, 55)
    
    pdf.rect(25, 63, 160, 120)
    pdf.rect(40, 73, 35, 42)
    pdf.rect(40, 131, 35, 42)
    pdf.rect(90, 73, 35, 42)
    pdf.rect(90, 131, 35, 42)
    pdf.rect(140, 73, 35, 42)
    pdf.rect(140, 131, 35, 42)

    pdf.line(47, 73, 47, 115)
    pdf.line(54, 73, 54, 115)
    pdf.line(61, 73, 61, 115)
    pdf.line(68, 73, 68, 115)
    
    pdf.line(40, 80, 75, 80)
    pdf.line(40, 87, 75, 87)
    pdf.line(40, 94, 75, 94)
    pdf.line(40, 101, 75, 101)
    pdf.line(40, 108, 75, 108)

    

    pdf.line(97, 73, 97, 115)
    pdf.line(104, 73, 104, 115)
    pdf.line(111, 73, 111, 115)
    pdf.line(118, 73, 118, 115)
    
    pdf.line(90, 80, 125, 80)
    pdf.line(90, 87, 125, 87)
    pdf.line(90, 94, 125, 94)
    pdf.line(90, 101, 125, 101)
    pdf.line(90, 108, 125, 108)


    pdf.line(147, 73, 147, 115)
    pdf.line(154, 73, 154, 115)
    pdf.line(161, 73, 161, 115)
    pdf.line(168, 73, 168, 115)
    
    pdf.line(140, 80, 175, 80)
    pdf.line(140, 87, 175, 87)
    pdf.line(140, 94, 175, 94)
    pdf.line(140, 101, 175, 101)
    pdf.line(140, 108, 175, 108)



    pdf.line(47, 131, 47, 173)
    pdf.line(54, 131, 54, 173)
    pdf.line(61, 131, 61, 173)
    pdf.line(68, 131, 68, 173)
    
    pdf.line(40, 138, 75, 138)
    pdf.line(40, 145, 75, 145)
    pdf.line(40, 152, 75, 152)
    pdf.line(40, 159, 75, 159)
    pdf.line(40, 166, 75, 166)

    

    pdf.line(97, 131, 97, 173)
    pdf.line(104, 131, 104, 173)
    pdf.line(111, 131, 111, 173)
    pdf.line(118, 131, 118, 173)
    
    pdf.line(90, 138, 125, 138)
    pdf.line(90, 145, 125, 145)
    pdf.line(90, 152, 125, 152)
    pdf.line(90, 159, 125, 159)
    pdf.line(90, 166, 125, 166)


    pdf.line(147, 131, 147, 173)
    pdf.line(154, 131, 154, 173)
    pdf.line(161, 131, 161, 173)
    pdf.line(168, 131, 168, 173)
    
    pdf.line(140, 138, 175, 138)
    pdf.line(140, 145, 175, 145)
    pdf.line(140, 152, 175, 152)
    pdf.line(140, 159, 175, 159)
    pdf.line(140, 166, 175, 166)


    pdf.rect(18, 188 , 174, 60)
    pdf.rect(40, 190 , 63, 14)
    pdf.rect(40, 211 , 63, 14)
    pdf.rect(40, 232 , 63, 14)

    pdf.rect(120, 190 , 63, 14)
    pdf.rect(120, 211 , 63, 14)
    pdf.rect(120, 232 , 63, 14)


    pdf.line(40, 197, 103, 197)
    pdf.line(40, 218, 103, 218)
    pdf.line(40, 239, 103, 239)

    pdf.line(120, 197, 183, 197)
    pdf.line(120, 218, 183, 218)
    pdf.line(120, 239, 183, 239)


    k = 0
    for i in range(9):
        pdf.line(47+k, 190, 47+k, 204)
        k += 7
    k = 0
    for i in range(9):
        pdf.line(127+k, 190, 127 +k, 204)
        k += 7


    k = 0
    for i in range(9):
        pdf.line(47+k, 211, 47+k, 225)
        k += 7
    k = 0
    for i in range(9):
        pdf.line(127+k, 211, 127 +k, 225)
        k += 7
    

    k = 0
    for i in range(9):
        pdf.line(47+k, 232, 47+k, 246)
        k += 7
    k = 0
    for i in range(9):
        pdf.line(127+k, 232, 127 +k, 246)
        k += 7


    pdf.set_xy(25, 260)
    pdf.set_font('DejaVu', size=12)
    pdf.cell(10, 10, txt = dat, border = 0, align = 'L')
    pdf.line(25, 268, 105, 268)
    pdf.set_xy(95, 268)
    pdf.cell(5, 5, txt = "Дата", border = 0, align = 'L')
    pdf.line(120, 268, 195, 268)
    pdf.set_xy(177, 268)
    pdf.cell(5, 5, txt = "Подпись", border = 0, align = 'L')
    



    points = [[40, 73], [90, 73],[140, 73], [40, 131], [90, 131],[140, 131]]

    i_mass = [0,30, 60, 90, 120, 150]
    for i, i1 in zip (i_mass, range(len(i_mass))):
        f = 0
        z_mass = [0,1,2,3,4,5]
        for z in z_mass:
            for f in range(5):
                if d[i + (f+1)+(5*z)] == "Filled":
                    pdf.line(points[i1][0]+7*f, points[i1][1]+7*z, points[i1][0]+7+7*f, points[i1][1]+7+7*z)
                    pdf.line(points[i1][0]+7*f, points[i1][1]+7+7*z, points[i1][0]+7+7*f, points[i1][1]+7*z)
                

    
    points = [[40, 197], [120, 197],[40, 218], [120, 218], [40, 239],[120, 239]]
    
    mass_i = [180, 189, 198, 207, 216, 225]
    for i in range(len(mass_i)):
        for f in range(9):
            if d[mass_i[i]+f+1] == "Filled":
                pdf.line(points[i][0]+f*7, points[i][1], points[i][0]+7+f*7, points[i][1]+7)
                pdf.line(points[i][0]+f*7, points[i][1]+7, points[i][0]+7+f*7, points[i][1])
            
    
    
    pdf.set_font('DejaVu', size=12, )
    pdf.set_xy(41, 67)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(48, 67)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(55, 67)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(62, 67)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(69, 67)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')


    pdf.set_xy(91, 67)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(98, 67)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(105, 67)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(112, 67)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(119, 67)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')


    pdf.set_xy(141, 67)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(148, 67)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(155, 67)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(162, 67)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(169, 67)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')


    pdf.set_xy(41, 125)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(48, 125)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(55, 125)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(62, 125)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(69, 125)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')


    pdf.set_xy(91, 125)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(98, 125)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(105, 125)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(112, 125)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(119, 125)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')


    pdf.set_xy(141, 125)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(148, 125)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(155, 125)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(162, 125)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(169, 125)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')



    pdf.set_xy(41, 191)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(48, 191)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(55, 191)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(62, 191)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(69, 191)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')
    pdf.set_xy(76, 191)
    pdf.cell(5, 5, txt = "F", border = 0, align = 'C')
    pdf.set_xy(83, 191)
    pdf.cell(5, 5, txt = "G", border = 0, align = 'C')
    pdf.set_xy(90, 191)
    pdf.cell(5, 5, txt = "H", border = 0, align = 'C')
    pdf.set_xy(97, 191)
    pdf.cell(5, 5, txt = "I", border = 0, align = 'C')

    pdf.set_xy(121, 191)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(128, 191)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(135, 191)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(142, 191)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(149, 191)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')
    pdf.set_xy(156, 191)
    pdf.cell(5, 5, txt = "F", border = 0, align = 'C')
    pdf.set_xy(163, 191)
    pdf.cell(5, 5, txt = "G", border = 0, align = 'C')
    pdf.set_xy(170, 191)
    pdf.cell(5, 5, txt = "H", border = 0, align = 'C')
    pdf.set_xy(177, 191)
    pdf.cell(5, 5, txt = "I", border = 0, align = 'C')



    pdf.set_xy(41, 212)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(48, 212)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(55, 212)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(62, 212)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(69, 212)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')
    pdf.set_xy(76, 212)
    pdf.cell(5, 5, txt = "F", border = 0, align = 'C')
    pdf.set_xy(83, 212)
    pdf.cell(5, 5, txt = "G", border = 0, align = 'C')
    pdf.set_xy(90, 212)
    pdf.cell(5, 5, txt = "H", border = 0, align = 'C')
    pdf.set_xy(97, 212)
    pdf.cell(5, 5, txt = "I", border = 0, align = 'C')

    pdf.set_xy(121, 212)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(128, 212)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(135, 212)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(142, 212)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(149, 212)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')
    pdf.set_xy(156, 212)
    pdf.cell(5, 5, txt = "F", border = 0, align = 'C')
    pdf.set_xy(163, 212)
    pdf.cell(5, 5, txt = "G", border = 0, align = 'C')
    pdf.set_xy(170, 212)
    pdf.cell(5, 5, txt = "H", border = 0, align = 'C')
    pdf.set_xy(177, 212)
    pdf.cell(5, 5, txt = "I", border = 0, align = 'C')



    pdf.set_xy(41, 233)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(48, 233)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(55, 233)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(62, 233)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(69, 233)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')
    pdf.set_xy(76, 233)
    pdf.cell(5, 5, txt = "F", border = 0, align = 'C')
    pdf.set_xy(83, 233)
    pdf.cell(5, 5, txt = "G", border = 0, align = 'C')
    pdf.set_xy(90, 233)
    pdf.cell(5, 5, txt = "H", border = 0, align = 'C')
    pdf.set_xy(97, 233)
    pdf.cell(5, 5, txt = "I", border = 0, align = 'C')

    pdf.set_xy(121, 233)
    pdf.cell(5, 5, txt = "A", border = 0, align = 'C')
    pdf.set_xy(128, 233)
    pdf.cell(5, 5, txt = "B", border = 0, align = 'C')
    pdf.set_xy(135, 233)
    pdf.cell(5, 5, txt = "C", border = 0, align = 'C')
    pdf.set_xy(142, 233)
    pdf.cell(5, 5, txt = "D", border = 0, align = 'C')
    pdf.set_xy(149, 233)
    pdf.cell(5, 5, txt = "E", border = 0, align = 'C')
    pdf.set_xy(156, 233)
    pdf.cell(5, 5, txt = "F", border = 0, align = 'C')
    pdf.set_xy(163, 233)
    pdf.cell(5, 5, txt = "G", border = 0, align = 'C')
    pdf.set_xy(170, 233)
    pdf.cell(5, 5, txt = "H", border = 0, align = 'C')
    pdf.set_xy(177, 233)
    pdf.cell(5, 5, txt = "I", border = 0, align = 'C')

    
    pdf.set_xy(35, 74)
    pdf.cell(5, 5, txt = "1", border = 0, align = 'C')
    pdf.set_xy(35, 81)
    pdf.cell(5, 5, txt = "2", border = 0, align = 'C')
    pdf.set_xy(35, 88)
    pdf.cell(5, 5, txt = "3", border = 0, align = 'C')
    pdf.set_xy(35, 95)
    pdf.cell(5, 5, txt = "4", border = 0, align = 'C')
    pdf.set_xy(35, 102)
    pdf.cell(5, 5, txt = "5", border = 0, align = 'C')
    pdf.set_xy(35, 109)
    pdf.cell(5, 5, txt = "6", border = 0, align = 'C')

    pdf.set_xy(84, 74)
    pdf.cell(5, 5, txt = "7", border = 0, align = 'C')
    pdf.set_xy(84, 81)
    pdf.cell(5, 5, txt = "8", border = 0, align = 'C')
    pdf.set_xy(84, 88)
    pdf.cell(5, 5, txt = "9", border = 0, align = 'C')
    pdf.set_xy(84, 95)
    pdf.cell(5, 5, txt = "10", border = 0, align = 'C')
    pdf.set_xy(84, 102)
    pdf.cell(5, 5, txt = "11", border = 0, align = 'C')
    pdf.set_xy(84, 109)
    pdf.cell(5, 5, txt = "12", border = 0, align = 'C')

    pdf.set_xy(134, 74)
    pdf.cell(5, 5, txt = "13", border = 0, align = 'C')
    pdf.set_xy(134, 81)
    pdf.cell(5, 5, txt = "14", border = 0, align = 'C')
    pdf.set_xy(134, 88)
    pdf.cell(5, 5, txt = "15", border = 0, align = 'C')
    pdf.set_xy(134, 95)
    pdf.cell(5, 5, txt = "16", border = 0, align = 'C')
    pdf.set_xy(134, 102)
    pdf.cell(5, 5, txt = "17", border = 0, align = 'C')
    pdf.set_xy(134, 109)
    pdf.cell(5, 5, txt = "18", border = 0, align = 'C')


    pdf.set_xy(34, 132)
    pdf.cell(5, 5, txt = "19", border = 0, align = 'C')
    pdf.set_xy(34, 139)
    pdf.cell(5, 5, txt = "20", border = 0, align = 'C')
    pdf.set_xy(34, 146)
    pdf.cell(5, 5, txt = "21", border = 0, align = 'C')
    pdf.set_xy(34, 153)
    pdf.cell(5, 5, txt = "22", border = 0, align = 'C')
    pdf.set_xy(34, 160)
    pdf.cell(5, 5, txt = "23", border = 0, align = 'C')
    pdf.set_xy(34, 167)
    pdf.cell(5, 5, txt = "24", border = 0, align = 'C')

    pdf.set_xy(84, 132)
    pdf.cell(5, 5, txt = "25", border = 0, align = 'C')
    pdf.set_xy(84, 139)
    pdf.cell(5, 5, txt = "26", border = 0, align = 'C')
    pdf.set_xy(84, 146)
    pdf.cell(5, 5, txt = "27", border = 0, align = 'C')
    pdf.set_xy(84, 153)
    pdf.cell(5, 5, txt = "28", border = 0, align = 'C')
    pdf.set_xy(84, 160)
    pdf.cell(5, 5, txt = "29", border = 0, align = 'C')
    pdf.set_xy(84, 167)
    pdf.cell(5, 5, txt = "30", border = 0, align = 'C')

    pdf.set_xy(134, 132)
    pdf.cell(5, 5, txt = "31", border = 0, align = 'C')
    pdf.set_xy(134, 139)
    pdf.cell(5, 5, txt = "32", border = 0, align = 'C')
    pdf.set_xy(134, 146)
    pdf.cell(5, 5, txt = "33", border = 0, align = 'C')
    pdf.set_xy(134, 153)
    pdf.cell(5, 5, txt = "34", border = 0, align = 'C')
    pdf.set_xy(134, 160)
    pdf.cell(5, 5, txt = "35", border = 0, align = 'C')
    pdf.set_xy(134, 167)
    pdf.cell(5, 5, txt = "36", border = 0, align = 'C')
    

    pdf.set_xy(34, 198)
    pdf.cell(5, 5, txt = "37", border = 0, align = 'C')
    pdf.set_xy(34, 219)
    pdf.cell(5, 5, txt = "38", border = 0, align = 'C')
    pdf.set_xy(34, 240)
    pdf.cell(5, 5, txt = "39", border = 0, align = 'C')

    pdf.set_xy(114, 198)
    pdf.cell(5, 5, txt = "40", border = 0, align = 'C')
    pdf.set_xy(114, 219)
    pdf.cell(5, 5, txt = "41", border = 0, align = 'C')
    pdf.set_xy(114, 240)
    pdf.cell(5, 5, txt = "42", border = 0, align = 'C')




    pdf.add_page()
    pdf.set_xy(10, 10)
    pdf.set_font('DejaVu', size=17)
    pdf.cell(10, 10, txt = text[0], border = 0, align = 'L')
    pdf.set_xy(10, 10)
    pdf.cell(10, 30, txt = text[1], border = 0, align = 'L')
    pdf.set_xy(10, 10)
    pdf.cell(10, 50, txt = text[2], border = 0, align = 'L')
    pdf.set_xy(10, 10)
    pdf.cell(10, 70, txt = "Дата выполнения: " + dat, border = 0, align = 'L')
    pdf.set_xy(10, 10)
    pdf.cell(10, 90, txt = "Колличество верных ответов: " + str(score[0]), border = 0, align = 'L')
    pdf.set_xy(10, 10)
    pdf.cell(10, 110, txt = "Колличество ошибок: " + str(score[1]), border = 0, align = 'L')
    pdf.set_xy(10, 10)
    res = 200 / 90 * score[0] 
    res = round(res, 1)
    pdf.cell(10, 130, txt = "Итоговый балл: " + str(res), border = 0, align = 'L')
    
    

    



    pdf.output('blank/res.pdf')
    

pdf2(a, b)























"""
def block_1():

    #cv2.resize(out, (1101, 720))
    #gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    #ret, th = cv2.threshold(gray, 120, 255, 0)
    #img = cv2.erode(th, (7,7), iterations = 5)
    #img = cv2.dilate(img, (7,7), iterations = 5)
    #cv2.imwrite('blank/7_5_{}.jpg'.format(1), img)
    #edges = cv2.Canny(img, 50, 250)
    #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    #for i in range(len(contours)):
        #if cv2.contourArea(contours[i]) > 1000 * kf_s and cv2.contourArea(contours[i]) < 2000000 * kf_s:
        #cv2.drawContours(out, contours, i, (50, 0, 255), thickness = 3)
    #cv2.imwrite('blank/7_5_{}.jpg'.format(2), out)   
      
    x = wid // 2
    y = out_hig // 7.778
    h = out_hig // 2.613

    w = out_wid // 5.128
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    out2 = out[y:y+h, x:x+w]
    cv2.imwrite('blank/out21.jpg', out2) 

    
    #cv2.resize(out2, (1101, 720))
    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 120, 255, 0)
    #cv2.imwrite('blank/7_5_{}.jpg'.format(4), th) 
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    contours.sort(key = lambda x: cv2.contourArea(x))
    #print(len(contours))
    
    contour_number = len(contours) - 2
    cv2.drawContours(out2, contours, contour_number , (50, 0, 255), thickness = 3)
    print(contour_number)
    #cv2.circle(out2,(24, 34), 10, (0, 255, 255))
    cv2.imwrite('blank/out21.jpg', out2)




def block_2():
      
    x = out_wid // 2.395
    y = out_hig // 7.778
    h = out_hig // 2.613

    w = out_wid // 5.128
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    out2 = out[y:y+h, x:x+w]
    cv2.imwrite('blank/out22.jpg', out2) 

    cv2.resize(out2, (1101, 720))
    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 120, 255, 0)
    #cv2.imwrite('blank/7_5_{}.jpg'.format(4), th) 
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #c = sorted(contours, key = cv2.contourArea)

    #print(len(contours))
    x = []
    y = []
    for i in range(len(contours[1])):
        x.append(contours[1][i][0][0])
        y.append(contours[1][i][0][1])        
    cv2.drawContours(out2, contours, 1, (50, 0, 255), thickness = 3)
    
    a=(min(x))
    b=(min(y))

    cv2.circle(out2,(a, b), 10, (0, 255, 255))
    cv2.imwrite('blank/out22.jpg', out2)
     


def block_3():

    out_hig = out.shape[0]
    out_wid = out.shape[1]
    
    x = out_wid // 1.44
    y = out_hig // 7.778
    h = out_hig // 2.613

    w = out_wid // 5.128
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    out2 = out[y:y+h, x:x+w]
    cv2.imwrite('blank/out23.jpg', out2) 

    #cv2.line(out, (900, 2150), (9000, 2150), (50, 255, 0), thickness = 4)
    #cv2.imwrite('blank/out12354.jpg', out) 


def block_4():
      
    x = out_wid // 6.837
    y = out_hig // 1.76
    h = out_hig // 2.613

    w = out_wid // 5.128
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    out2 = out[y:y+h, x:x+w]
    cv2.imwrite('blank/out24.jpg', out2) 

    cv2.resize(out2, (1101, 720))
    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 120, 255, 0)
    #cv2.imwrite('blank/7_5_{}.jpg'.format(4), th) 
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #c = sorted(contours, key = cv2.contourArea)

    #print(len(contours))
    x = []
    y = []
    for i in range(len(contours[1])):
        x.append(contours[1][i][0][0])
        y.append(contours[1][i][0][1])        
    cv2.drawContours(out2, contours, 1, (50, 0, 255), thickness = 3)
    
   #print(min(x))
    #print(min(y))

    #cv2.circle(out2,(24, 34), 10, (0, 255, 255))
    #cv2.imwrite('blank/out21.jpg', out2)
     

def block_5():
      
    x = out_wid // 2.395
    y = out_hig // 1.76
    h = out_hig // 2.613

    w = out_wid // 5.128
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    out2 = out[y:y+h, x:x+w]
    cv2.imwrite('blank/out25.jpg', out2) 

    cv2.resize(out2, (1101, 720))
    gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 120, 255, 0)
    cv2.imwrite('blank/7_5_{}.jpg'.format(4), th) 
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    x = []
    y = []
    for i in range(len(contours[1])):
        x.append(contours[1][i][0][0])
        y.append(contours[1][i][0][1])        
    cv2.drawContours(out2, contours, 1, (50, 0, 255), thickness = 3)
    
    #print(min(x))
    #print(min(y))

    #cv2.circle(out3,(24, 34), 10, (0, 255, 255))
    #cv2.imwrite('blank/out21.jpg', out3)


def block_6():
   
    x = out_wid // 1.44
    y = out_hig // 1.76
    h = out_hig // 2.613

    w = out_wid // 5.128
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    out2 = out[y:y+h, x:x+w]
    cv2.imwrite('blank/out26.jpg', out2) 

    #cv2.line(out, (4270, 500), (42070, 500), (50, 255, 0), thickness = 4)
    #cv2.imwrite('blank/out23.jpg', out) 

#block_1()
#block_2()
#block_3()
#block_4()
#block_5()
#block_6()
"""