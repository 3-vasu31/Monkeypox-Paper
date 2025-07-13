from tensorflow.keras.applications import EfficientNetB3, DenseNet121, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adamax

def create_model(base_model, input_shape, class_count):
    base_model = base_model(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
    base_model.trainable = True
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(class_count, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_efficientnet(input_shape, class_count):
    return create_model(EfficientNetB3, input_shape, class_count)

def create_densenet(input_shape, class_count):
    return create_model(DenseNet121, input_shape, class_count)

def create_inceptionresnet(input_shape, class_count):
    return create_model(InceptionResNetV2, input_shape, class_count)