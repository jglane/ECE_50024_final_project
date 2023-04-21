import tensorflow as tf

class GeneratorLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def call(self, fake, ones_like_fake):
        return self.loss(ones_like_fake, fake)

class DiscriminatorLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def call(self, real, fake):
        real_loss = self.loss(tf.ones_like(real), real)
        fake_loss = self.loss(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) / 2

class CycleLoss(tf.keras.losses.Loss):
    def __init__(self, lambd=10):
        super().__init__()
        self.lambd = lambd
    
    def call(self, real, cycled):
        return self.lambd * tf.reduce_mean(tf.abs(real - cycled))

class IdentityLoss(tf.keras.losses.Loss):
    def __init__(self, lambd=10):
        super().__init__()
        self.lambd = lambd
    
    def call(self, real, same):
        return self.lambd * tf.reduce_mean(tf.abs(real - same)) / 2