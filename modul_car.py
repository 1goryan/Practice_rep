class Car():
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

class Electrical_car(Car): #в скобках указывает от какого класса мы наследываем атирибуты
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
