                                    #Тестированние классов
#assertEqual(a, b) проверяет, что a == b
#assertNotEqual(a, b) проверяет, что a != b
#assertTrue(x) проверяет, что x истинно
#assertFalse(x) проверяет, что x ложно
#assertIn(элемент, список) проверяет, что элемент есть в списке
#assertNotIn(элемент, список) проверяет, что элемента нет в списке


class AnonymousSurvey():
    #класс для анонимных ответов

    def __init__ (self, question):
        #сохраняем вопрос
        self.question = question
        self.responses = []

    def show_question(self):
        #вывод вопроса
        print(self.question)

    def save_response(self, new_response):
        #сохраняем ответ
        self.responses.append(new_response)

    def show_results(self):
        #вывод полученных ответов
        print("Answers: " )
        for response in self.responses:
            print("- " + response)


"""question = "Which programming language are you most interested in?"
my_survey = AnonymousSurvey(question)

my_survey.show_question()
print("Press Q to exit poll")

while True:
    response = input("Language: ")
    if response == 'Q':
        break
    my_survey.save_response(response)

print("\n Thanks for answers")

my_survey.show_results()"""

#from Testing_class import AnonymousSurvey
import unittest


class TestAnonymousSurvey(unittest.TestCase):
    #Тесты для класса AnonymousSurvey
    
    def setUp(self): 
        #создание опроса и набора ответов для всех методов
        question = "Which programming language are you most interested in?"
        self.my_survey = AnonymousSurvey(question)
        self.responses = ['Java', 'Python', 'C#', 'Go', 'JavaScript']

    def test_save_single_response(self):
        #проверка, что ответ был сохранен в нужном месте в списке
        self.my_survey.save_response(self.responses[1]) 
        self.assertIn("Python", self.my_survey.responses)

    def  test_save_list_responses(self):
        #проверяем, что в сохраненном списке есть правильный ответ
        for response in self.responses: 
            self.my_survey.save_response(response) 

        for response in self.responses:
            self.assertIn("Python", self.my_survey.responses)      


if __name__ == "__main__":
    unittest.main()


