from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, BatchNormalization, Conv2D, Multiply, Add,  ReLU, Concatenate
import math


def getResidualUnit(initial_x, output_filters):
    ## main branch
    input_shape = initial_x.shape[1:]
    x = BatchNormalization()(initial_x)
    x = ReLU()(x)
    x = Conv2D(filters=input_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=input_shape[-1], kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=output_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)

    ## highway
    x_highway = initial_x
    if input_shape[-1] != output_filters:
        x_highway = Conv2D(filters=output_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(x_highway)

    ## combine main branch and highway
    output_x = Add()([x, x_highway])

    return output_x


def getResNextUnit(initial_x, output_filters):
    input_shape = initial_x.shape[1:]
    cardinality = 32

    middle_layer_n_filters = math.ceil(initial_x.shape[-1] / 10)

    ## highway
    output_x = initial_x
    if input_shape[-1] != output_filters:
        output_x = Conv2D(filters=output_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(output_x)

    ## main branch
    x_store = []
    for _ in range(cardinality):
        x = BatchNormalization()(initial_x)
        x = ReLU()(x)
        x = Conv2D(filters=middle_layer_n_filters, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=middle_layer_n_filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x_store.append(x)
    x = Concatenate(axis=-1)(x_store)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=output_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    output_x = Add()([x, output_x])

    return output_x


def getAttentionModule(initial_x, p, t, r, depth, unitFunc):
    ## pre-processing residual units
    x = initial_x
    for _ in range(p):
        x = unitFunc(x, x.shape[-1])

    ## trunk branch
    x_trunk = x
    for _ in range(t):
        x_trunk = unitFunc(x_trunk, x_trunk.shape[-1])

    ## softmask branch
    x_softmask = x
    x_softmask = MaxPooling2D(pool_size=(2, 2))(x_softmask)
    for _ in range(r):
        x_softmask = unitFunc(x_softmask, x_softmask.shape[-1])

    ## softmask branch: down sampling
    x_softmask_skip_outputs = []
    for _ in range(depth - 1):

        ## softmask branch: downsampling, main subbranch
        x_softmask_main = x_softmask
        x_softmask_main = MaxPooling2D(pool_size=(2, 2))(x_softmask_main)
        for _ in range(r):
            x_softmask_main = unitFunc(x_softmask_main, x_softmask_main.shape[-1])

        ## softmask branch: downsampling, skip subbranch
        x_softmask_skip = x_softmask
        x_softmask_skip = unitFunc(x_softmask_skip, x_softmask_skip.shape[-1])
        x_softmask_skip_outputs.append(x_softmask_skip)

        x_softmask = x_softmask_main

    ## softmask branch: up sampling
    for _ in range(depth - 1):

        ## softmask branch: upsampling, main subbranch
        for _ in range(r):
            x_softmask = unitFunc(x_softmask, x_softmask.shape[-1])
        x_softmask = UpSampling2D(size=(2, 2), interpolation="bilinear")(x_softmask)

        ## softmask branch: upsampling, combine main subbranch and skip branch
        x_softmask = Add()([x_softmask, x_softmask_skip_outputs.pop()])

    ## softmask branch
    for _ in range(r):
        x_softmask = unitFunc(x_softmask, x_softmask.shape[-1])
    x_softmask = UpSampling2D(size=(2, 2), interpolation="bilinear")(x_softmask)

    ## softmask branch
    x_softmask = Conv2D(filters=initial_x.shape[-1], kernel_size=(1, 1), strides=(1, 1))(x_softmask)
    x_softmask = Conv2D(filters=initial_x.shape[-1], kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(
        x_softmask)

    # combine trunk branch and softmask branch
    x = Multiply()([x_softmask, x_trunk])
    x = Add()([x, x_trunk])

    # post-processing residual units
    for _ in range(p):
        x = unitFunc(x, x.shape[-1])

    return x
