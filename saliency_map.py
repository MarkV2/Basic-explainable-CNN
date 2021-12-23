def do_salience(image, model, label, loss_fn):
  label = np.asarray(label).reshape((1, label.shape[0]))
  
  with tf.GradientTape() as tape: 
    inputs = tf.convert_to_tensor(image)
    tape.watch(inputs) 
    prediction = model(inputs) 
    loss = loss_fn(
        label, prediction
    )

  gradients = tape.gradient(loss, inputs)
  grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1) 

  normalized_tensor = tf.cast(
    255 * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
    / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)), 
    tf.uint8)

  normalized_tensor = tf.squeeze(normalized_tensor)
  
  #superimpose
  image = np.reshape(image, image[0].shape)
  gradient_color = cv2.applyColorMap(normalized_tensor.numpy(), cv2.COLORMAP_HOT)
  gradient_color = gradient_color / 255.0
  super_imposed = cv2.addWeighted(image.astype('float64'), 0.5, gradient_color, 0.5, 0.0)


  fig = plt.figure(figsize=(10, 10))
  fig.add_subplot(1, 3, 1)
  plt.axis('off')
  plt.title("Original image")
  plt.imshow(image)

  fig.add_subplot(1, 3, 2)
  plt.axis('off')
  plt.title("Saliency heatmap")
  plt.imshow(normalized_tensor, cmap='gray')

  fig.add_subplot(1, 3, 3)
  plt.axis('off')
  plt.title("Superimposed heatmap")
  plt.imshow(super_imposed)
  plt.show() 
