@tf.function
def gradient_ascent(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        pred = model(img)
    grads = tape.gradient(pred, img)
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return img

def deprocess(img):
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    img += 0.5
    img = np.clip(img, 0, 1)

    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def visualize():
    iterations = 100
    learning_rate = 2
    
    img = tf.random.uniform((1, 160, 160, 3))
    img = (img-0.5)*0.25

    for iteration in range(iterations):
        img = gradient_ascent(img, learning_rate)

    img = deprocess(img[0].numpy())
    return img
