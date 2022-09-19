def integrated_gradients(image, label, baseline, model, loss_fn, cmap = plt.cm.inferno, kernel_size = 9, sigma = 3, m_steps = 50, batch_size = 32):
    # NOTE: kernel_size and sigma was chosen randomly
    # Both kernel_size and sigma are necessary for baseline = 'blur'
    if baseline == 'black':
        baseline = tf.zeros_like(image)
    elif baseline == 'white':
        baseline = tf.ones_like(image)
    elif baseline == 'random':
        baseline = tf.random.uniform(shape=image.shape, minval=0.0, maxval=1.0)
    elif baseline == 'blur':
        def gausian_kernel(size, sigma):
            x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
            y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

            xs, ys = tf.meshgrid(x_range, y_range)

            num = (tf.exp(tf.cast(-(xs**2 + ys**2), dtype=tf.float32) / (tf.constant(2.0) * sigma**2)))
            den = (2.0 * tf.math.acos(-1.0) * sigma**2)
            kernel = num / den

            return tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)

        kernel = gausian_kernel(kernel_size, sigma)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

        r, g, b = tf.split(image, [1, 1, 1], axis=-1)
        r_blur = tf.nn.conv2d(tf.expand_dims(r, 0), kernel, [1, 1, 1, 1], padding='SAME')
        g_blur = tf.nn.conv2d(tf.expand_dims(g, 0), kernel, [1, 1, 1, 1], padding='SAME')
        b_blur = tf.nn.conv2d(tf.expand_dims(b, 0), kernel, [1, 1, 1, 1], padding='SAME')

        blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
        baseline = tf.squeeze(blur_image, axis=0)

    def compute_gradients(images, labels, loss_fn, model):
        with tf.GradientTape() as tape:
            tape.watch(images)
            pred = model(images)
            loss = loss_fn(labels, pred)
        return tape.gradient(loss, images)

    def interpolate_images(baseline, image, alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    def integral_approximation(gradients):
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    @tf.function
    def one_batch(baseline, image, alpha_batch, labels, loss_fn, model):
        interpolated_path_input_batch = interpolate_images(baseline, image, alpha_batch)
        gradient_batch = compute_gradients(interpolated_path_input_batch, labels, loss_fn, model)
        return gradient_batch


    alphas = tf.linspace(0.0, 1.0, m_steps+1)
    gradient_batches = []
    for alpha in tf.range(0, len(alphas), batch_size):
        _from = alpha
        to = tf.minimum(_from + batch_size, len(alphas))
        alpha_batch = alphas[_from:to]

        labels = np.array([label] * alpha_batch.shape[0])

        gradient_batch = one_batch(baseline, image, alpha_batch, labels, loss_fn, model)
        gradient_batches.append(gradient_batch)

    total_gradients = tf.concat(gradient_batches, axis=0)
    avg_gradients = integral_approximation(total_gradients)

    integrated_gradients = (image - baseline) * avg_gradients

    #
    # Visualisations
    # 

    mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1)

    titles = ['Baseline', 'Original Image', 'Attribution mask', 'Overlay']
    images = [[baseline], [image], [mask], (mask, image)]
    cmaps = [[None], [None], [cmap], (cmap, None)]
    alphas = [[1], [1], [1], (1, 0.4)]
    
    plt.figure(figsize=(6, 6))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.axis('off')
        plt.title(titles[i])
        for j in range(len(images[i])):
            plt.imshow(images[i][j], cmap = cmaps[i][j], alpha = alphas[i][j])

    plt.tight_layout()
    plt.show()
