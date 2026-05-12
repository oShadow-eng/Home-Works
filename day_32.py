kernel_tf = kernel.reshape(3, 3, 1, 1)

feature_maps = tf.nn.conv2d(
    images,
    kernel_tf,
    strides=strides,
    padding=padding
)

if use_relu:
    feature_maps = tf.nn.relu(feature_maps)

avg_pooled = tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=strides,
    padding="valid"
)(feature_maps)

global_pooled = tf.keras.layers.GlobalAveragePooling2D()(feature_maps)

print("feature_maps shape:", feature_maps.shape)
print("avg_pooled shape:", avg_pooled.shape)
print("global_pooled shape:", global_pooled.shape)

show_image(feature_maps[0, :, :, 0].numpy(), "Feature map: Image A")
show_image(feature_maps[1, :, :, 0].numpy(), "Feature map: Image B")
