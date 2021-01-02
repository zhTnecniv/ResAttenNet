from components import getAttentionModule, getResidualUnit, getResNextUnit
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Conv2D, Input, Flatten, Dense, ReLU
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2

def residualAttention56():
    model_input = Input(shape=(32,32,3))
    x = Conv2D(filters=32, kernel_size=(5,5), padding="same")(model_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x = getResidualUnit(x,128)
    x = getAttentionModule(x,1,2,1,3, getResidualUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,2,getResidualUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = AveragePooling2D(pool_size=(4,4),strides=(1,1))(x)
    x = Flatten()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

def residualAttentionNeXt56():
    model_input = Input(shape=(32,32,3))
    x = Conv2D(filters=32, kernel_size=(5,5), padding="same")(model_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x = getResidualUnit(x,128)
    x = getAttentionModule(x,1,2,1,3, getResNextUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,2,getResNextUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,1,getResNextUnit)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = AveragePooling2D(pool_size=(4,4),strides=(1,1))(x)
    x = Flatten()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

def residualAttention92():
    model_input = Input(shape=(32,32,3))
    x = Conv2D(filters=32, kernel_size=(5,5), padding="same")(model_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x = getResidualUnit(x,128)
    x = getAttentionModule(x,1,2,1,3, getResidualUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,2,getResidualUnit)
    x = getAttentionModule(x,1,2,1,2,getResidualUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = AveragePooling2D(pool_size=(4,4),strides=(1,1))(x)
    x = Flatten()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

def residualAttentionNeXt92():
    model_input = Input(shape=(32,32,3))
    x = Conv2D(filters=32, kernel_size=(5,5), padding="same")(model_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x = getResidualUnit(x,128)
    x = getAttentionModule(x,1,2,1,3, getResNextUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,2,getResNextUnit)
    x = getAttentionModule(x,1,2,1,2,getResNextUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,1,getResNextUnit)
    x = getAttentionModule(x,1,2,1,1,getResNextUnit)
    x = getAttentionModule(x,1,2,1,1,getResNextUnit)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = AveragePooling2D(pool_size=(4,4),strides=(1,1))(x)
    x = Flatten()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

def residualAttention128():
    model_input = Input(shape=(32,32,3))
    x = Conv2D(filters=32, kernel_size=(5,5), padding="same")(model_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x = getResidualUnit(x,128)
    x = getAttentionModule(x,1,2,1,3,getResidualUnit)
    x = getAttentionModule(x,1,2,1,3,getResidualUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,2,getResidualUnit)
    x = getAttentionModule(x,1,2,1,2,getResidualUnit)
    x = getAttentionModule(x,1,2,1,2,getResidualUnit)
    x = getResidualUnit(x,256)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getAttentionModule(x,1,2,1,1,getResidualUnit)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = getResidualUnit(x,512)
    x = AveragePooling2D(pool_size=(4,4),strides=(1,1))(x)
    x = Flatten()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

def resnet50V2Model():
    baseModel = ResNet50V2(
        weights=None,
        include_top=False,
        input_shape=(32,32,3),
    )
    model_input = Input(shape=(32,32,3))
    x = baseModel(model_input)
    x = GlobalAveragePooling2D()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

def resnet101V2Model():
    baseModel = ResNet101V2(
        weights=None,
        include_top=False,
        input_shape=(32,32,3),
    )
    model_input = Input(shape=(32,32,3))
    x = baseModel(model_input)
    x = GlobalAveragePooling2D()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

def resnet152V2Model():
    baseModel = ResNet152V2(
        weights=None,
        include_top=False,
        input_shape=(32,32,3),
    )
    model_input = Input(shape=(32,32,3))
    x = baseModel(model_input)
    x = GlobalAveragePooling2D()(x)
    model_output = Dense(10, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model