                                #Классы
"""class Dog():        #простая модель собаки

    def __init__(self, name, age): #инициализируем атрибуты
        self.name = name
        self.age = age
        #print("dog created")

    def sit(self): #собака будет садиться по команде
        print(self.name.title() + " sat down") #title первая буква заглавная

    def jump(self): #собака будет прыгать по команде
        print(self.name.title() + " jumped")

    def bark(self): #собака будет гафкать
        print(self.name + ": Gaf-Gaf")

my_dog = Dog("Topik", 4)
my_dog2 = Dog("Nik", 7)
poppy = Dog("Poppy", 2)
print(my_dog.age)
print(my_dog.name)

my_dog.sit()
my_dog2.jump()
poppy.bark()"""

"""Cоздан класс с именем Restaurant Метод __init__() класса Restaurant содержит два атрибута: restaurant_name и cuisine_type 
Создан метод describe_restaurant(), который выводит два атрибута, и метод open_restaurant(), который выводит
сообщение о том, что ресторан открыт
Создан экземпляр с именем restaurant1 Выведено два атрибута по отдельности, затем вызваны оба метода
Три ресторана: 
Созданы три разных экземпляра, для каждого экземпляра вызван метод describe_restaurant()"""

"""class Restaurant():

    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type


    def describe_restaurant(self):
        print("The restaurant is named " + self.restaurant_name)
        print("Cuisine type of restaurant is " + self.cuisine_type)

    def open_restaurant(self):
        print("The restaurant " + self.restaurant_name + " is open" )

restaurant1 = Restaurant("Tasty", "fastfood")
restaurant2 = Restaurant("Delishios", "ukrainian food")
restaurant3 = Restaurant("Dainty", "chinese")

print(restaurant1.restaurant_name)
print(restaurant1.cuisine_type)
restaurant1.describe_restaurant()
restaurant1.open_restaurant()
restaurant2.describe_restaurant()
restaurant3.describe_restaurant() """


"""Создан класс с именем User Создано два атрибута first_name и last_name, а затем еще несколько атрибутов, которые обычно хранятся в профиле пользователя 
Написан метод describe_user(), который выводит сводку с информацией о пользователе 
Создан еще один метод greet_user() для вывода персонального приветствия для пользователя
Создано несколько экземпляров, представляющих разных пользователей
Вызованы оба метода"""

class User():

    def __init__(self, first_name, second_name, age, sex):
        self.first_name = first_name
        self.second_name = second_name
        self.age = age
        self.sex = sex

    def describe_user(self):
        print("first name " + self.first_name)
        print("second name " + self.second_name)
        print("age " + str(self.age))
        print("sex " + self.sex)

    def greet_user(self):
        print("Hello " + self.first_name + " " + self.second_name) 

user1 = User("Ihor", "Poltavets", 21, "Male")
user2 = User("Anna", "Ivanova", 19, "Female") 
user3 = User("Ivan", "Pavlov", 32, "Male")
user4 = User("Katya", "Vasilina", 24, "Female")

user1.describe_user()
user2.greet_user()