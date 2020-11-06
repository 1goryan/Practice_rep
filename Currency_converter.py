import requests
from bs4 import BeautifulSoup

url = 'https://minfin.com.ua/currency/nbu/'

source = requests.get(url)
main_text = source.text
soup = BeautifulSoup(main_text,'html.parser')

table = soup.find('table', {'class' : 'table-auto'} ) # если find без All то получаем только первое совпадение
tr = table.findAll('td', {'class' : 'responsive-hide'})
mn = []
for f in tr:
    h = f.text[1:7]
    mn.append(h)
      
for i in range(len(mn)):
    mn[i] = float(mn[i])

def converter (uah, currency):
    if currency == "USD":
        course = uah / mn[0]
    elif currency == "EUR":
        course = uah / mn[1]
    elif currency == "RUB":
        course = uah / mn[2]
    elif currency == "PLN":
        course = uah / mn[3]
    elif currency == "CHF":
        course = uah / mn[4]
    elif currency == "GBR":
        course = uah / mn[5]
    else:
        print("please select the correct currency") 

    print(str(uah) + " UAH = " + str(course) + " " + currency)

converter(float(input("Write your sum of UAH ")), str(input("Available currency: USD, EUR, RUB, PLN, CHF, GBR ")))