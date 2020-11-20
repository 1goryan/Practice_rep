#Тестирование
def full_name(first, second, middle=" " ): #делаем третий параметр необязательным
    if middle == " ":
        full = first + ' ' + second
    else:
        full = first + ' ' + second + ' ' + middle
    return full


"""from Testing_in_Python import full_name

print("for stop the test enter symbol 'Q'")
while True:
    first = input("\n enter your name ")
    if first == 'Q':
        break
    last = input("\n enter your second name ")
    if last == 'Q':
        break

    format_name = full_name(first, last)
    print("\n format name " + format_name)"""

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
#if __name__ == "__name__": #сравниваем присвоенное значение __name__
#    unittest.main()
unittest.main()



