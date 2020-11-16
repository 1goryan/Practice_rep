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
# программа приветствует пользователя беря его имя с json файла, если имя не задано, то
# потребуется его ввести и имя будет закодировано в json файл, если имя не совпадает
# то программа потребует его ввести
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

def get_new_username():
    user_name = input("Enter you name")
    file_name = "E:\Pyth0n\python\modules\\user.json"    
    with open(file_name, "w", encoding="utf-8") as f: 
        json.dump(user_name, f, ensure_ascii=False) 
        print("We saved your name as " + user_name)
            
def user_greeting(user):
    #приветствие пользователя
    user_name = get_user_name()
    if user_name == user:
        print("Wellcome " + user_name)
    else:
        print("You aren't " + user_name + " you must login")
        user_name = get_new_username()
    
user_greeting(input())

"""# cоздана программа, которая принимает число, кодирует его в json, а потом декодирует и выдает
# его, если число не біло задано вернет вас к функции задания числа
import json

def favourite_number ():
    num = input("Enter your favourite number")
    file_name = "E:\Pyth0n\python\modules\\number.json"
    with open(file_name, 'w') as f:
        json.dump(num, f)

def show_favourite_number():
    file_name = "E:\Pyth0n\python\modules\\number.json"
    try:
        with open(file_name) as f1:
            num = json.load(f1)
    except:
        num = favourite_number()
    else:
        print("You favourite number is " + num)   

show_favourite_number()"""