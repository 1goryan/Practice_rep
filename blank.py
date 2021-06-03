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

import cv2
import numpy as np
import glob
import pytesseract

src = cv2.imread('blank/blank/src_last1.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY )


d = {}

def template():

    x = 214
    y = 430
    h = 38
    w = 34
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
        x = 214 + d
        y = 429
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_1_{}.jpg'.format(i), out)

#src = cv2.rectangle(src, (685, 1480), (1170, 1553), (255, 0, 255))
#cv2.imwrite("blank/srccc.jpg", src)


def answer_2():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 472
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_2_{}.jpg'.format(i), out)

 
def answer_3():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 515
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_3_{}.jpg'.format(i), out)


def answer_4():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 558
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_4_{}.jpg'.format(i), out)

def answer_5():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 601
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_5_{}.jpg'.format(i), out)


def answer_6():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 644
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/1_6_{}.jpg'.format(i), out)


def answer_7():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 429
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_1_{}.jpg'.format(i), out)


def answer_8():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 472
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_2_{}.jpg'.format(i), out)

    
def answer_9():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 515
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_3_{}.jpg'.format(i), out)


def answer_10():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 558
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_4_{}.jpg'.format(i), out)


def answer_11():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 601
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_5_{}.jpg'.format(i), out)


def answer_12():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 644
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/2_6_{}.jpg'.format(i), out)

def answer_13():
    d = 0
    for i in range(5):
        x = 851 + d
        y = 429
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_1_{}.jpg'.format(i), out)

def answer_14():
    d = 0
    for i in range(5):
        x = 851 + d
        y = 472
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_2_{}.jpg'.format(i), out)
   

def answer_15():
    d = 0
    for i in range(5):
        x = 851 + d
        y = 515
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_3_{}.jpg'.format(i), out)

def answer_16():
    d = 0
    for i in range(5):
        x = 851 + d
        y = 558
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_4_{}.jpg'.format(i), out)


def answer_17():
    d = 0
    for i in range(5):
        x = 851 + d
        y = 601
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_5_{}.jpg'.format(i), out)

def answer_18():
    d = 0
    for i in range(5):
        x = 851 + d
        y = 644
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/3_6_{}.jpg'.format(i), out)

def answer_19():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 764
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_1_{}.jpg'.format(i), out)


def answer_20():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 807
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_2_{}.jpg'.format(i), out)


def answer_21():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 850
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_3_{}.jpg'.format(i), out)


def answer_22():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 893
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_4_{}.jpg'.format(i), out)


def answer_23():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 936
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_5_{}.jpg'.format(i), out)


def answer_24():
    d = 0
    for i in range(5):
        x = 214 + d
        y = 979
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/4_6_{}.jpg'.format(i), out)


def answer_25():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 764
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_1_{}.jpg'.format(i), out)

def answer_26():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 807
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_2_{}.jpg'.format(i), out)

def answer_27():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 850
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_3_{}.jpg'.format(i), out)


def answer_28():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 893
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_4_{}.jpg'.format(i), out)


def answer_29():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 936
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_5_{}.jpg'.format(i), out)


def answer_30():
    d = 0
    for i in range(5):
        x = 532 + d
        y = 979
        h = 39
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/5_6_{}.jpg'.format(i), out)


def answer_31():
    d = 0
    for i in range(5):
        x = 852 + d
        y = 764
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_1_{}.jpg'.format(i), out)


def answer_32():
    d = 0
    for i in range(5):
        x = 852 + d
        y = 807
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_2_{}.jpg'.format(i), out)

def answer_33():
    d = 0
    for i in range(5):
        x = 852 + d
        y = 850
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_3_{}.jpg'.format(i), out)


def answer_34():
    d = 0
    for i in range(5):
        x = 852 + d
        y = 893
        h = 38
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_4_{}.jpg'.format(i), out)

def answer_35():
    d = 0
    for i in range(5):
        x = 852 + d
        y = 935
        h = 39
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_5_{}.jpg'.format(i), out)


def answer_36():
    d = 0
    for i in range(5):
        x = 852 + d
        y = 979
        h = 39
        w = 34
        d += 39
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/6_6_{}.jpg'.format(i), out)


def answer_37():
    d = 0
    for i in range(9):
        x = 213 + d
        y = 1159
        h = 38
        w = 34
        d += 40
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_1_{}.jpg'.format(i), out)


def answer_38():
    d = 0
    for i in range(9):
        x = 213 + d
        y = 1288
        h = 38
        w = 34
        d += 40
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_2_{}.jpg'.format(i), out)


def answer_39():
    d = 0
    for i in range(9):
        x = 213 + d
        y = 1417
        h = 38
        w = 34
        d += 40
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_3_{}.jpg'.format(i), out)

def answer_40():
    d = 0
    for i in range(9):
        x = 708 + d
        y = 1159
        h = 38
        w = 34
        d += 41
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_4_{}.jpg'.format(i), out)

def answer_41():
    d = 0
    for i in range(9):
        x = 708 + d
        y = 1288
        h = 38
        w = 34
        d += 41
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_5_{}.jpg'.format(i), out)

def answer_42():
    d = 0
    for i in range(9):
        x = 708 + d
        y = 1417
        h = 38
        w = 34
        d += 41
        out = src[y:y+h, x:x+w]
        cv2.imwrite('blank/7_6_{}.jpg'.format(i), out)





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
    
    x = 510
    y = 110
    h = 50
    w = 310
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text

def user_name():

    x = 245
    y = 155
    h = 510
    w = 60
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text

def group():

    x = 270
    y = 225
    h = 485
    w = 68
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text

def data():
    
    x = 110
    y = 1480
    h = 73
    w = 480
    out = src[y:y+h, x:x+w]
    text = pytesseract.image_to_string(out, lang = 'rus', config = '--psm 6')
    return text

def sign():
    x = 685
    y = 1480
    h = 73
    w = 485
    out = src[y:y+h, x:x+w]
    img = cv2.imwrite("blank/blank/sign.jpg", out)
    return img


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