import tensorflow as tf
from tensorflow.contrib import ffmpeg
#for tensorflow1.3
#from tensorflow.contrib import keras as tf_keras
from tensorflow import keras as tf_keras

import os


voc_size = 20000
embedding_size = 256
video = tf.layers.Input(shape = (None, 360, 640, 3),
                        name = 'video')

cnn = tf.keras.applications.InceptionV3(weights='imagenet',
                                        include_top=False,
                                        pooling='avg')
cnn.trainable = False

frame_features = tf.keras.layers.TimeDistributed(cnn)(video)

video_vector = tf.keras.layers.LSTM(256)(frame_features)

#########################################################################
# Each question will be at most 100 word long,
#question = keras.Input(shape=(100, ), dtype='int32', name='question')

tanmu = tf.layers.Input(shape=(None, ), dtype='int32', name='tanmu')

tanmu_embedded = tf.keras.layers.Embedding(voc_size,embedding_size)(tanmu)
tanmu_vector = tf.keras.layers.LSTM(256)(tanmu_embedded)

########################################################################
twist_vector = tf.keras.layers.concatenate([video_vector,tanmu_vector])
x = tf.keras.layers.Dense(256,activation=tf.nn.relu)(twist_vector)
pred = tf.keras.layers.Dense(voc_size)(x)
########################################################################

model = tf.keras.models.Model([video,tanmu],pred)

model.compile(optimizer = tf.train.AdamOptimizer(),
              #loss = tf.nn.softmax_cross_entropy_with_logits(logits=frame_features,labels=))
              loss ='categorical_crossentropy')

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('/home/xucl/app/tensorboard_log/keras_training')

    file_path = '/home/xucl/app/data/bilibili/video/jpg/'
    files = os.listdir(file_path)
    file_name_list = [os.path.join(file_path,file_name) for file_name in files]

    movie = []
    for file_path in file_name_list:
      img_bin = tf.read_file(file_path)
      img = tf.image.decode_jpeg(img_bin)
      img_ev = img.eval()
      movie.append(img_ev)
      print len(img_ev)
      print len(img_ev[0])
      print len(img_ev[0][0])

      ##################################################
      img_slim = tf.expand_dims(img, 0)
      img_show = tf.summary.image(file_path,img_slim)
      img_summary_op = sess.run(img_show)
      summary_writer.add_summary(img_summary_op)
      ###################################################
    #feed_dict = {video:movie}
    #excrete_list = cnn

    #sess.run(excrete_list,feed_dict)
    model.fit([movie[:-1]],[movie[1:]])
