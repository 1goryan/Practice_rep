from collections import Counter

def count_words(path):
    
    txt = open(path, 'r')
    #txt = open('E:\Pyth0n\python\modules\soome.txt', 'r')
    text = txt.read()
    
    word_list = []
    for word in text.split(): #в переменную ворд записать каждое слово с txt
        clear_word = "" #сюда будут записаны только буквы
        for letter in word: #каждое слово разбиваем на символы
            if letter.isalpha(): #если символ с алфавита, записываем его в letter
                clear_word += letter.lower() #записываем буквы в clear_word
        word_list.append(clear_word) #записываем наши "чистые" слова в массив
    txt.close
    print(Counter(word_list))
    

count_words(input("Введите путь файла"))