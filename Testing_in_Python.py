#Тестирование
"""def full_name(first, second, middle=" " ): #делаем третий параметр необязательным
    if middle == " ":
        full = first + ' ' + second
    else:
        full = first + ' ' + second + ' ' + middle
    return full


from Testing_in_Python import full_name

print("for stop the test enter symbol 'Q'")
while True:
    first = input("\n enter your name ")
    if first == 'Q':
        break
    last = input("\n enter your second name ")
    if last == 'Q':
        break

    format_name = full_name(first, last)
    print("\n format name " + format_name)

import unittest 
#from Testing_in_Python import full_name

class Name_test_case(unittest.TestCase): #Наследуем от класса unittest метод TestCase
    #тест для функции full_name 

    def test_first_last_name(self): #любой метод, который начинается test_ при запуске программы они будут сразу отрабатывать
        #проверяем имена вида 'Ihor' 'Poltavets' работает нормально?
        format_name = full_name('Ihor', 'Poltavets') # вызываем тестируемую функцию и кладем ее в переменную
        self.assertEqual(format_name, 'Ihor Poltavets') # этот метод сравниваем данные с первого аргумента со вторым,если они совпадают-тест пройден

    def test_first_last_middle(self):
        #проверяем имена вида 'Ihor' 'Poltavets' 'Alexandrovich' работает нормально?
        format_name = full_name('Ihor', 'Poltavets', 'Alexandrovich')
        self.assertEqual(format_name, 'Ihor Poltavets Alexandrovich' )
#if __name__ == "__main__": #сравниваем присвоенное значение __name__
#    unittest.main()
unittest.main()"""

"""import unittest

def city_country(city, country, population =" "):
    if population == " ":
        return city + " " +  country
    else:
        return city + " " +  country + " " + population

class City_test(unittest.TestCase):
   #тест для функции city_country
    
    def test_city_county(self):
        city_1 = city_country('Kharkiv', 'Ukraine')
        self.assertEqual(city_1, 'Kharkiv Ukraine')

    def test_city_county_population(self):
        city_2 = city_country('Kharkiv', 'Ukraine', '1,419')
        self.assertEqual(city_2, 'Kharkiv Ukraine 1,419')

if __name__ == "__main__":
    unittest.main()"""

import unittest

def calculator(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '/':
        return num1 / num2
    elif operation == '*':
        return num1 * num2
    else:
        print("Choose correct operation") 

class Calculator_test(unittest.TestCase):


    def test_calculator(self):
        operation = input()
        if operation == '+':
            calc1 = calculator(6, 3, '+')
            self.assertEqual(calc1, 9)
        elif operation == '-':
            calc1 = calculator(6, 3, '-')
            self.assertEqual(calc1, 3)
        elif operation == '/':
            calc1 = calculator(6, 3, '/')
            self.assertEqual(calc1, 2)
        elif operation == '*':
            calc1 = calculator(6, 3, '*')
            self.assertEqual(calc1, 18)
        else :
            print("Error operation")
        
if __name__ == "__main__":
    unittest.main()