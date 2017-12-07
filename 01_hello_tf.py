import tensorflow as tf

hello = tf.constant('Hello TensorFlow!')
sess = tf.Session()

print(sess.run(hello))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a + b

print(sess.run(adder, feed_dict={a:3, b:4.5}))
print(sess.run(adder, feed_dict={a:[1,3], b:[2,4]}))


