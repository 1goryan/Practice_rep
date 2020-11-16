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

file_name = "E:\Pyth0n\python\modules\\user.json"
try:
    with open(file_name) as f1 :
        user = json.load(f1)
except:
    user_name = input("Enter you name")
    with open(file_name, "w", encoding="utf-8") as f: #encoding="utf-8" нужен для чтения кирилицы
        json.dump(user_name, f, ensure_ascii=False) #ensure_ascii=False отключаем кодировку ascii
        print("We saved your name as " + user_name)
else:
    print("Wellcome " + user)