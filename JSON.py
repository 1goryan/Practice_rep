"""#Работа с JSON
import json #импорт модуля json

nums = [4, 76, 2, 67, 23, 7, 78, 98] #создан список для примера
#записываем в файл
file_name = "E:\Pyth0n\python\modules\\nums.json" # в переменную положим файл с именем nums.json
with open(file_name, 'w') as f:
    json.dump(nums, f) #используется для сохранения данных в ранее созданый файл


#открываем файл
file = "E:\Pyth0n\python\modules\\nums.json"
with open(file) as f1:
    nums_new = json.load(f1)

print(nums_new)"""

import json

def get_user_name():
    #получаем имя пользователя, если оно есть
    file_name = "E:\Pyth0n\python\modules\\user.json"
    try:
        with open(file_name) as f1 :
            user_name = json.load(f1)
    except FileNotFoundError:
        return None
    else:
        return user_name
       
def user_greeting():
    #приветствие пользователя
    user_name = get_user_name()
    if user_name:
        print("Wellcome " + user_name)
    else:
        user_name = input("Введите ваше имя")
        file_name = "E:\Pyth0n\python\modules\\user.json"    
        with open(file_name, "w", encoding="utf-8") as f: #encoding="utf-8" нужен для чтения кирилицы
            json.dump(user_name, f, ensure_ascii=False) #ensure_ascii=False отключаем кодировку ascii
            print("We saved your name as " + user_name)
    
user_greeting()