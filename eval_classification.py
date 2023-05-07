import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import adda


def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('weights')
@click.option('--gpu', default='0')
def main(dataset, split, model, weights, gpu):
    is_adda_coarse = ("adda" in weights and "coarse" in weights)
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    dataset_name = dataset
    split_name = split
    dataset = adda.data.get_dataset(dataset, shuffle=False)
    split = getattr(dataset, split)
    model_fn = adda.models.get_model_fn(model)
    im, label = split.tf_ops(capacity=2)
    im = adda.models.preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], batch_size=1)

    net, layers = model_fn(im_batch, is_training=False)
    net = tf.argmax(net, -1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    var_dict = adda.util.collect_vars(model)
    restorer = tf.train.Saver(var_list=var_dict)
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Evaluating {}'.format(weights))
    restorer.restore(sess, weights)

    class_correct = np.zeros(dataset.num_classes, dtype=np.int32)
    class_counts = np.zeros(dataset.num_classes, dtype=np.int32)

    save = np.array([])

    # USE FOR REGULAR EVALUATIONS
    # for i in tqdm(range(len(split))):
    #     predictions, gt = sess.run([net, label_batch])
    #     class_counts[gt[0]] += 1
    #     save = np.append(save, predictions)
    #     if predictions[0] == gt[0]:
    #         class_correct[gt[0]] += 1

    # USE FOR SUBCLASS (FINE) EVALUATIONS
    for i in tqdm(range(len(split))):
        predictions, gt = sess.run([net, label_batch])
        class_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
        class_counts[class_index[gt[0]]] += 1
        save = np.append(save, predictions)
        if predictions[0] == gt[0]:
            class_correct[class_index[gt[0]]] += 1

    if is_adda_coarse:
        if dataset_name == "mnist":
            if split_name == "test":
                np.save("data/mnist/test_predictions_2loop.npy", save)
            else:
                np.save("data/mnist/train_predictions_2loop.npy", save)
        if dataset_name == "usps":
            if split_name == "test":
                np.save("data/usps/test_predictions_2loop.npy", save)
            else:
                np.save("data/usps/train_predictions_2loop.npy", save)

    logging.info('Class accuracies:')
    logging.info('    ' + format_array(class_correct / class_counts))
    logging.info('Overall accuracy:')
    logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))

    coord.request_stop()
    coord.join(threads)
    sess.close()
    

if __name__ == '__main__':
    main()
