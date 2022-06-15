import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json', encoding='UTF8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # разбиение слов на массив
    sentence_words = nltk.word_tokenize(sentence)

    # вычленение каждого слова - приведение к базовой форме
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# возвращает набор слов массива: 0 или 1 для слов, которые существуют в предложении
def bag_of_words(sentence, words, show_details=True):

    # токенизация паттернов
    sentence_words = clean_up_sentence(sentence)

    # словарь слов - словарная матрица
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # присвоить 1, если текущее слово находится в словарной позиции
                bag[i] = 1
                if show_details:
                    print ("Найдено в словаре: %s" % word)
    return(np.array(bag))

def predict_class(sentence):

    # фильтрация прогнозов
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    # вероятность сортировки
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    if (result == ""):
        result = 'Извини, я тебя не понимаю...'
    return result


# tkinter GUI
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Вы: " + msg + '\n\n')
        ChatBox.config(foreground="black", font=("Corbel", 13))
    
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        
        ChatBox.insert(END, "Бот: " + res + '\n\n')
            
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
 

root = Tk()
root.title("Чат-бот")
root.geometry("500x500")
root.resizable(width=TRUE, height=FALSE)

# Создание окна чат-бота
ChatBox = Text(root, bd=0, bg="old lace", height="8", width="50", font="Corbel",)
ChatBox.config(state=DISABLED)

# Скроллбар
scrollbar = Scrollbar(root, command=ChatBox.yview)
ChatBox['yscrollcommand'] = scrollbar.set

# Поле для ввода сообщения
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Corbel")

# Кнопка отправки сообщения
SendButton = Button(root, font=("Corbel", 13, 'bold'), text="Отправить", width="12", height="5",
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command=send)

# Расположение элементов
scrollbar.place(x=476, y=6, height=386)
ChatBox.place(x=6, y=6, height=400, width=470)
EntryBox.place(x=6, y=420, height=60, width=365)
SendButton.place(x=380, y=420, height=60, width=110)

root.mainloop()
