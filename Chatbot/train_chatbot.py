import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('intents.json', encoding='UTF8').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # токенизация каждого слова
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # добавление док-ов
        documents.append((word, intent['tag']))

        # добавление в классы
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

# лемматизация опустить каждое слово и удалить дубликаты
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

# сортировка классов
classes = sorted(list(set(classes)))

# док-ы = комбинация между patterns и intents
print (len(documents), "documents")

# классы - intents
print (len(classes), "classes", classes)

# слова - словарь
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# создание тренировочных данных
training = []

# создание пустого массива для вывода
output_empty = [0] * len(classes)

# тренировочный набор, набор слов для каждого предложения
for doc in documents:
    # создание массива слов
    bag = []

    # список токенизированных слов для шаблона
    pattern_words = doc[0]

    # лемматизировать каждое слово - создать базовое слово, пытаясь представить связанные слова
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # создание массива слов с 1, если совпадение слов найдено в текущем шаблоне
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # вывод — «0» для каждого тега и «1» для текущего тега (для каждого шаблона)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# перетасовка функций и превращение в np.array
random.shuffle(training)
training = np.array(training)

# создание списков обучений и тестов. X - паттерны, Y - намерения
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Данные для обучения созданы")

# Создание модели - 3 слоя.
# Первый слой 128 нейронов, второй слой 64 нейрона и 3-й выходной слой содержит количество нейронов.
# равно количеству намерений, чтобы предсказать намерение вывода с помощью softmax
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Компиляция модели
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# тренировка и сохранение модели
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=32, verbose=1)
model.save('chatbot_model.h5', hist)

print("Модель сохранена")

