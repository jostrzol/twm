def generate_model():
  model = Sequential()                                 # Linear stacking of layers

  # Convolution Layer 1
  model.add(Conv2D(64, (3, 3), padding="same", input_shape=(32,32,3)))
  model.add(Activation('relu') )
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Dropout(0.2))

  # Convolution Layer 2
  model.add(Conv2D(128, (3, 3), padding="same"))
  model.add(Activation('relu'))
  model.add(Conv2D(128, (3, 3), padding="same"))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  # Convolution Layer 3
  model.add(Conv2D(256, (3, 3), padding="same"))
  model.add(Activation('relu'))
  model.add(Conv2D(256, (3, 3), padding="same"))
  model.add(Activation('relu'))
  model.add(Conv2D(256, (3, 3), padding="same"))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())                                 # Flatten final output matrix into a vector


  model.add(Dropout(0.5))

  # Fully Connected Layer
  model.add(Dense(1024))
  model.add(Activation('relu'))

  # Fully Connected Layer
  model.add(Dense(1024))
  model.add(Activation('relu'))

  # Fully Connected Layer
  model.add(Dense(10))                                 # final 10 FC nodes
  model.add(Activation('softmax'))                     # softmax activation

  model.summary()

  adam = tf.optimizers.Adam(learning_rate=0.001)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

  return model