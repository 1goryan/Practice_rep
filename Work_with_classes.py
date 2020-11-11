                    #Работа с классами и экземплярами
"""class Car():
    #класс по созданию автомобиля
    def __init__(self, make, model, year):
        #инициализация атрибутов авто
        self.make = make
        self.model = model 
        self.year = year
        self.odometer_reading = 0 #созадние атрибута, который задаем в экземларе

    def decribshion_car(self):
        #возвращаем описание авто
        desc = str(self.year) + " " + self.make + " " + self.model
        return desc.title()

    def read_odometer(self):
        #выводим пробег авто
        print("Mileage of this car is " + str(self.odometer_reading) + " km")

    def update_odometer(self, km):
        #устанавливаем значение на одометре и защишаем от уменьшения пробега
        if km >= self.odometer_reading :
            self.odometer_reading = km
        else :
            print("It's impossible")

    def increase_odometer(self, km):
        #увеличиваем значение на одометре на заданный пробег, с защитой от отрицательного пробега#
        if km <= 0 :
            km = int(input("Enter a value greater than 0 "))
            self.odometer_reading += km
        else:  
            self.odometer_reading += km

my_car1 = Car("Audi", "A4", 2017)
#print(my_car1.decribshion_car())
#my_car1.odometer_reading = 30 # изменяем атрибут через обращение к нему
my_car1.update_odometer(40) #изменяем атрибут через метод
my_car1.increase_odometer(7) #увеличиваем через метод
my_car1.read_odometer() """

"""Создан класс пользователь для сайта)
Добавлен метод залогиниться
Создан метод разлогиниться
Создан метод, который выводит количество заходов этим пользователем"""
class User():
    
    def __init__(self, login, password,):
        
        self.login = login 
        self.password = password 
        self.number_of_inputs = 0
              

    def login_site(self, lg, pw):
        if self.login == lg and self.password == pw:
            print(self.login + " entered the site")
            self.number_of_inputs += 1
        elif self.login == lg and self.password != pw:
            print("incorrect password")
        elif self.login != lg and self.password == pw:
            print("incorrect login")
        else:
            print("incorrect login and password")
        
    def log_off(self):
        print(first_name + " " + second_name + " left the site")

    def number_of_visits(self):
        print(self.number_of_inputs)   
    
user1 = User("Ihor", "qwerty")
user2 = User("Ivan", "123456")
user1.login_site("Ihor", "qwerty")
user2.login_site("Ivan", "123456")
user2.number_of_visits()
user1.number_of_visits()
