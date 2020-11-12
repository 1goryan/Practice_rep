                            #Наследование классов
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

#В класс Батарей добавлен метод разряда батареи, который принимает параметр пробега
#и внутри умножает пробег на расход батареии и потом отнимает данные с атрибута батареи
#Добавлен метод заряда батарей к 100% значению

class Battery(): #сохдаем самостоятельный класс для батареи
    
    def __init__(self, battery = 100): #если при создании атрибута не укажем battery, он будет принят за 100
        self.battery = battery
        self.charge = battery

    def decribshion_battery(self):
        #выводит инфо о мощности батареи
        print("This car has a battery with a power " + str(self.battery) + " kw-hr")

    def battery_discharge(self, km):
        self.battery -= km * 0.16
        return self.battery

    def battery_charge(self):
        self.battery = self.charge
        return self.battery

class Electriacal_car(Car): #в скобках указывает от какого класса мы наследываем атирибуты
    #аспекты для электромобиля
    def __init__(self, make, model, year):
    #инициализация атрибутов класса родителя
        super().__init__(make, model, year) #вызвать метод инит с родительского класса
        #self.battery = 100
        self.battery = Battery() #атрибут батарея берем с класса

    #def decribshion_battery(self): # этот метод работает только для этого класса
        #выводит инфо о мощности батареи
        #print("This car has a battery with a power " + str(self.battery))

    def decribshion_car(self): #переопредиление родительского метода
        desc = str(self.year) + " " + self.model
        return desc.title()



tesla = Electriacal_car("Tesla", "S", 2017)
tesla.battery.decribshion_battery() #обращение к методам класса батарея
print(tesla.battery.battery_discharge(350))
print(tesla.battery.battery_charge())
#print(tesla.decribshion_car())
#tesla.decribshion_battery()"""

#Создан основной класс ресторан
#Созданы атрибуты еды, напитков
#Добавлены методы посадочных  мест, время работы, количество сотрудников
#Создан дочерний класс кальян
#добавлены атрибуты сорты табака при помощи словаря
#добавлен метод заказать кальян

class Restaurant():

    def __init__(self, name, number_of_seats, open_time, close_time, workers):
        self.name = name
        self.number_of_seats = number_of_seats
        self.open_time = open_time
        self.close_time = close_time
        self.workers = workers
        self.menu_1 = "meet" 
        self.menu_2 = "potatoes"
        self.menu_3 = "alcohol"
        self.menu_4 = "water"

    def seats(self, number):
        if number > self.number_of_seats:
            print("Restaurant hasn't enought seats")
        else:
            print("Wellcome to " + self.name + " restaurant")
    
    def working_hours(self):
        print("The restaurant open from " + self.open_time  + " to " + self.close_time)

    def number_of_workers(self):
        number = int(input("Enter the required number of workers "))
        if number > self.workers:
            number -= self.workers
            print("We're missing " + str(number) + " workers")
        elif number == self.workers:
            print("We have the required number of worker")
        else:
            number -= self.workers
            print("We have " + str(abs(number)) + " extra workers")   

class Hookah(Restaurant):

    def __init__(self, name, number_of_seats, open_time, close_time, workers):
        super().__init__(name, number_of_seats, open_time, close_time, workers)
        self.tobacco_grade = {"Nakhla" : 200 , "Adalia" : 150, "DarkSide" : 250}
        
    def order_hookah(self):
        for k,v in self.tobacco_grade.items():
            print(k + " costs " + str(v) + " UAH")                   
        
rest = Restaurant("First", 30, "9:00", "22:30", 25)
rest.working_hours()
rest_2 = Hookah("Second", 25, "8:30", "3:30", 5)
rest_2.order_hookah()