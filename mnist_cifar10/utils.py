import tensorflow as tf


# -------------------------------------------------------------
#    Helpers
# -------------------------------------------------------------

def load_mnist(batch_size, data_dir, augmentation=False, stddev=0.0, adv_subset=1000, workers=4):
    from data_loader import get_mnist

    trainloader, _, classes = get_mnist(batch_size=batch_size,
                                        train=True,
                                        path=data_dir,
                                        augmentation=augmentation,
                                        std=stddev,
                                        shuffle=True,
                                        workers=workers
                                        )

    testloader, _, _ = get_mnist(batch_size=batch_size,
                                 train=False,
                                 path=data_dir,
                                 shuffle=False,
                                 workers=workers
                                 )

    adv_testloader, _, _ = get_mnist(batch_size=batch_size,
                                     train=False,
                                     path=data_dir,
                                     shuffle=False,
                                     adversarial=True,
                                     subset=adv_subset,
                                     workers=workers
                                     )

    input_shape = (None, 28, 28, 1)

    return trainloader, testloader, adv_testloader, input_shape, len(classes)


def load_cifar10(batch_size, data_dir, augmentation=False, stddev=0.0, adv_subset=1000, workers=4):
    from data_loader import get_cifar10

    trainloader, _, classes = get_cifar10(batch_size=batch_size,
                                          train=True,
                                          path=data_dir,
                                          augmentation=augmentation,
                                          std=stddev,
                                          shuffle=True,
                                          workers=workers
                                          )

    testloader, _, _ = get_cifar10(batch_size=batch_size,
                                   train=False,
                                   path=data_dir,
                                   shuffle=False,
                                   workers=workers
                                   )

    adv_testloader, _, _ = get_cifar10(batch_size=batch_size,
                                       train=False,
                                       path=data_dir,
                                       shuffle=False,
                                       adversarial=True,
                                       subset=adv_subset,
                                       workers=workers
                                       )

    input_shape = (None, 32, 32, 3)

    return trainloader, testloader, adv_testloader, input_shape, len(classes)


def variable_summaries(var, name=None, collections=['training'], histo=True):
    with tf.device('/gpu:0'):
        if name is None:
            name = var.op.name

        var_shape = var.get_shape().as_list()
        var_dim = 1.0
        for dim in var_shape[1:]:
            var_dim *= dim

        with tf.name_scope('Compute-Mean'):
            mean = tf.reduce_mean(var)

        with tf.name_scope('Compute-Stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        with tf.name_scope('Compute-Max'):
            max = tf.reduce_max(var)

        with tf.name_scope('Compute-Min'):
            min = tf.reduce_min(var)

        # Write summaries
        tf.summary.scalar(name + '/mean', mean, collections=collections)
        tf.summary.scalar(name + '/stddev', stddev, collections=collections)
        tf.summary.scalar(name + '/max', max, collections=collections)
        tf.summary.scalar(name + '/min', min, collections=collections)
        if histo:
            tf.summary.histogram(name, tf.identity(var), collections=collections)
