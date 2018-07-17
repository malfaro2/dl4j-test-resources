from keras.models import Sequential
from keras.layers import Dense

optim = 'rmsprop'

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=optim, metrics=['accuracy'])

model.save(optim + '.h5')