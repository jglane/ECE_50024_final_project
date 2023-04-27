import tensorflow as tf
import tensorflow_addons as tfa

class Residual(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding='same', activation='relu'):
        super(Residual, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=activation)
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        out = tf.keras.layers.add([x, inputs])
        return tf.keras.layers.ReLU()(out)
    
def build_generator(IMG_DIM=256):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((IMG_DIM, IMG_DIM, 3)))

    # c7s1-64
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # d128
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # d256
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())
    
    # 9 x R256
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))

    # u128
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # u64
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # c7s1-3
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same', activation='tanh'))

    return model

def build_discriminator(IMG_DIM=256):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((IMG_DIM, IMG_DIM, 3)))
    
    # C64
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2))
    model.add(tf.keras.layers.LeakyReLU(0.2))

    # C128
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    # C256
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    # C512
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))

    # Convolution to produce a 1D output
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=4, activation='sigmoid'))

    return model

class CycleGAN(tf.keras.Model):
    def __init__(self, G, F, D_X, D_Y, *args, **kwargs):
        super(CycleGAN, self).__init__()
        self.G = G
        self.F = F
        self.D_X = D_X
        self.D_Y = D_Y
    
    def compile(self, G_opt, F_opt, D_X_opt, D_Y_opt, generator_loss, discriminator_loss, cycle_loss, identity_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.G_opt = G_opt
        self.F_opt = F_opt
        self.D_X_opt = D_X_opt
        self.D_Y_opt = D_Y_opt
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.cycle_loss = cycle_loss
        self.identity_loss = identity_loss

    def train_step(self, data):
        real_X, real_Y = data
        
        with tf.GradientTape(persistent=True) as tape:
            # Generate images
            fake_Y = self.G(real_X, training=True)
            cycled_X = self.F(fake_Y, training=True)
            fake_X = self.F(real_Y, training=True)
            cycled_Y = self.G(fake_X, training=True)
            same_X = self.F(real_X, training=True)
            same_Y = self.G(real_Y, training=True)

            # Discriminator output
            D_X_real = self.D_X(real_X, training=True)
            D_X_fake = self.D_X(fake_X, training=True)
            D_Y_real = self.D_Y(real_Y, training=True)
            D_Y_fake = self.D_Y(fake_Y, training=True)

            # Calculate the loss
            total_cycle_loss = self.cycle_loss(real_X, cycled_X) + self.cycle_loss(real_Y, cycled_Y)
            G_loss = total_cycle_loss + self.generator_loss(D_Y_fake, tf.ones_like(D_Y_fake)) + self.identity_loss(real_Y, same_Y)
            F_loss = total_cycle_loss + self.generator_loss(D_X_fake, tf.ones_like(D_X_fake)) + self.identity_loss(real_X, same_X)

            D_X_loss = self.discriminator_loss(D_X_real, D_X_fake)
            D_Y_loss = self.discriminator_loss(D_Y_real, D_Y_fake)

        # Calculate the gradients for generator and discriminator
        G_grad = tape.gradient(G_loss, self.G.trainable_variables)
        F_grad = tape.gradient(F_loss, self.F.trainable_variables)

        D_X_grad = tape.gradient(D_X_loss, self.D_X.trainable_variables)
        D_Y_grad = tape.gradient(D_Y_loss, self.D_Y.trainable_variables)

        # Apply the gradients to the optimizer
        self.G_opt.apply_gradients(zip(G_grad, self.G.trainable_variables))
        self.F_opt.apply_gradients(zip(F_grad, self.F.trainable_variables))

        self.D_X_opt.apply_gradients(zip(D_X_grad, self.D_X.trainable_variables))
        self.D_Y_opt.apply_gradients(zip(D_Y_grad, self.D_Y.trainable_variables))

        return {'G_loss': G_loss, 'F_loss': F_loss, 'D_X_loss': D_X_loss, 'D_Y_loss': D_Y_loss}