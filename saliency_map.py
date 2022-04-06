def saliency_map(image, label, model, loss_fn, cmap = plt.cm.afmhot):
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

    #
    # Visualisations
    #

    titles = ['Original image', 'Saliency heatmap', 'Overlay']
    images = [[tf.squeeze(image)], [normalized_tensor], (normalized_tensor, tf.squeeze(image))]
    cmaps = [[None], [cmap], (cmap, None)]
    alphas = [[1], [1], (1, 0.4)]

    plt.figure(figsize = (8, 8))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.axis('off')
        plt.title(titles[i])
        for j in range(len(images[i])):
            plt.imshow(images[i][j], cmap = cmaps[i][j], alpha = alphas[i][j])
    plt.tight_layout()
    plt.show()
