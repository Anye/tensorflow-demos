# -*- coding: utf-8 -*-
from gen_model import create_layer
import tensorflow as tf
import config
import numpy as np
from gen_image import convert2gray, gen_random_captcha_image, array_to_text

if __name__ == '__main__':
    test_count = 100
    right_count = 0
    x_input = tf.placeholder(tf.float32, [None, config.IMAGE_HEIGHT * config.IMAGE_WIDTH])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    output = create_layer(x_input, keep_prob)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, config.MAX_CAPTCHA, config.CHAR_SET_LEN]), 2)

        for x in range(0, test_count):
            label_text, captcha_image = gen_random_captcha_image()
            captcha_image = convert2gray(captcha_image)
            captcha_image = captcha_image.flatten() / 255
            text_list = sess.run(predict, feed_dict={x_input: [captcha_image], keep_prob: 1})

            text = text_list[0].tolist()
            vector = np.zeros(config.MAX_CAPTCHA * config.CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * config.CHAR_SET_LEN + n] = 1
                i += 1
            predict_text = array_to_text(vector)
            if predict_text == label_text:
                right_count = right_count + 1
                result = 'y'
            else:
                result = 'n'
            print("label is : {} <----> predict is : {} result: {}".format(label_text, predict_text, result))

        print("all result %s/%s , accuracy: %s" % (right_count, test_count, right_count/test_count))
