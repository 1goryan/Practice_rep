"""#from modul_car import Car #импорт класса Car с модуля modul_car, другие классы не будут работать
from modul_car import Car, Electrical_car #вызов двух классов с модуля
a4 = Car("Audi", "a4", 2016)
print(a4.decribshion_car())
tesla = Electrical_car("Tesla", "S", 2017)
tesla.battery.decribshion_battery() #без импорта нужного класса с модуля не будет работать

import modul_car #импорт всего модуля
a4 = modul_car.Car("Audi", "a4", 2016 ) #имя модуля.имя класса
tesla = modul_car.Electrical_car("Tesla", "S", 2017) #имя модуля.имя класса
print(a4.decribshion_car())
tesla.battery.decribshion_battery() #без импорта нужного класса с модуля не будет работать"""


from modul_car import * #импортируем все с модуля
a4 = Car("Audi", "a4", 2016 ) #имя модуля.имя класса
tesla = Electrical_car("Tesla", "S", 2017) #имя модуля.имя класса
print(a4.decribshion_car())
tesla.battery.decribshion_battery() #без импорта нужного класса с модуля не будет работать"""