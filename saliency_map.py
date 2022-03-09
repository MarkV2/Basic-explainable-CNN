def do_salience(image, model, label, loss_fn):
    image = tf.expand_dims(image, [0])
    label = tf.expand_dims(label, [0])

    inputs = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        prediction = model(inputs)
        loss = loss_fn(label, prediction)

    gradients = tape.gradient(loss, inputs)
    grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)

    normalized_tensor = tf.cast(255 * (grayscale_tensor - tf.reduce_min(grayscale_tensor)) / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)), tf.uint8)

    normalized_tensor = tf.squeeze(normalized_tensor)

    # To superimpose
    image = np.reshape(image, image[0].shape)
    gradient_color = cv2.applyColorMap(normalized_tensor.numpy(), cv2.COLORMAP_HOT)
    gradient_color = gradient_color / 255.0
    
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    super_imposed = cv2.addWeighted(image.astype('float64'), 0.5, gradient_color, 0.5, 0.0)

    plt.figure(figsize = (10, 10))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Saliency heatmap')
    plt.imshow(normalized_tensor, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('Superimposed heatmap')
    plt.imshow(super_imposed)
    plt.show()
