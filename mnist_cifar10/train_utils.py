import time
import numpy as np
import tensorflow as tf
from cleverhans.attacks import ProjectedGradientDescent


# -------------------------------------------------------------
#    Training loop
# -------------------------------------------------------------

def model_train(sess, x, y, x_pair1, x_pair2, is_training, dataloader, train_step, evaluate=None, adv_evaluate=None,
                args=None, writer_train=None, verbose=False):
    with sess.as_default():

        # Evaluate once before the training starts
        if evaluate is not None: evaluate(0)
        if adv_evaluate is not None: adv_evaluate(0)

        iterations = 0  # count total number of iterations

        if verbose: print('\nStart training loop...')
        begin = time.time()

        # Get training summaries
        training_summaries = tf.get_collection('training')

        # Training loop
        for epoch in range(args['epochs']):
            epoch_iterations = 0  # count number of iterations per epoch
            prev = time.time()

            # Perform single epoch
            n_samples = 0
            for i, data in enumerate(dataloader, start=0):
                # Get next batch from the dataloader object
                X_train, Y_train = data
                X_train, Y_train = X_train.numpy(), Y_train.numpy()

                # Change format for the input from NCHW to NHWC
                X_train = X_train.transpose([0, 2, 3, 1])

                # Reshape and apply one hot encoding to labels
                Y_train = np.reshape(Y_train, newshape=-1)
                Y_train = np.eye(args['n_classes'])[Y_train]

                # Collect total number of samples
                n_samples += X_train.shape[0]

                if args['clp']:
                    # Split batch in two halves for clean logit pairing
                    X_train1 = X_train[:X_train.shape[0] // 2]
                    X_train2 = X_train[X_train.shape[0] // 2:]

                    feed_dict = {x: X_train, y: Y_train, x_pair1: X_train1, x_pair2: X_train2, is_training: True}

                else:
                    feed_dict = {x: X_train, y: Y_train, is_training: True}

                if iterations % 100 == 0:
                    # Perform single SGD step and write summaries
                    _, training_summaries_val = sess.run([train_step, training_summaries], feed_dict=feed_dict)

                    # Write tensorboard summaries
                    for summary in training_summaries_val:
                        writer_train.add_summary(summary, iterations)
                else:
                    # Perform single SGD step
                    _ = sess.run([train_step], feed_dict=feed_dict)

                epoch_iterations += 1
                iterations += 1

            cur = time.time()

            if verbose:
                print(
                    '- Finished epoch {:d}. Performed {:d} iterations in {:.4f} seconds'.format(epoch, epoch_iterations,
                                                                                                (cur - prev)))

            # Evaluate at the end of the epoch
            if (epoch % args['eval_step'] == 0) or (epoch + 1 == args['epochs']):
                if evaluate is not None:
                    evaluate(epoch + 1)

            if (epoch % args['adv_eval_step'] == 0) or (epoch + 1 == args['epochs']):
                if adv_evaluate is not None:
                    adv_evaluate(epoch + 1)

    end = time.time()

    if verbose:
        print("Completed model training in {:.4f} seconds.".format((end - begin)))

    return True


# -------------------------------------------------------------
#    Evaluation during training
# -------------------------------------------------------------

def clean_eval(sess, x, y, is_training, dataloader, n_classes, logits, predictions):
    # Define test accuracy symbolically
    correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1), tf.argmax(predictions, axis=tf.rank(predictions) - 1))

    # Define test loss symbolically
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_sum(loss)

    test_accuracy, test_loss = 0.0, 0.0

    n_samples = 0
    for i, data in enumerate(dataloader, start=0):
        # Get next batch from the pytorch dataloader object
        X_test, Y_test = data
        X_test, Y_test = X_test.numpy(), Y_test.numpy()

        # Change format for the input from NCHW to NHWC
        X_test = X_test.transpose([0, 2, 3, 1])

        # Reshape and apply one hot encoding to labels
        Y_test = np.reshape(Y_test, newshape=-1)
        Y_test = np.eye(n_classes)[Y_test]

        # Collect total number of samples
        n_samples += X_test.shape[0]

        feed_dict = {x: X_test, y: Y_test, is_training: False}

        with sess.as_default():
            cur_corr_preds, cur_loss = sess.run([correct_preds, loss], feed_dict=feed_dict)

        # Accumulate batch accuracy and loss
        test_accuracy += cur_corr_preds[:].sum()
        test_loss += cur_loss

    # Divide by number of examples to get final value
    test_accuracy /= n_samples
    avg_test_loss = test_loss / n_samples

    return test_accuracy, test_loss, avg_test_loss


def adversarial_eval(sess, x, y, is_training, dataloader, n_classes, predictions, adv_preds, eval_all=True):
    label = tf.argmax(y, axis=tf.rank(y) - 1)
    predicted_label = tf.argmax(predictions, axis=tf.rank(y) - 1)

    loss, accuracy = 0.0, 0.0

    # Define accuracy and loss on adversarial examples
    adv_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=adv_preds, labels=y))
    adv_correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1), tf.argmax(adv_preds, axis=tf.rank(adv_preds) - 1))

    n_samples = 0
    for i, data in enumerate(dataloader, start=0):
        # Get next batch from the pytorch dataloader object
        X_test, Y_test = data
        X_test, Y_test = X_test.numpy(), Y_test.numpy()

        # Change format for the input from NCHW to NHWC
        X_test = X_test.transpose([0, 2, 3, 1])

        # Reshape and apply one hot encoding to labels
        Y_test = np.reshape(Y_test, newshape=-1)
        Y_test = np.eye(n_classes)[Y_test]

        feed_dict = {x: X_test, y: Y_test, is_training: False}

        # Evaluate only on correctly classified samples
        if not eval_all:
            with sess.as_default():
                # Get labels
                label_val, predicted_label_val = sess.run([label, predicted_label], feed_dict=feed_dict)

                # Find misclassifications
                delete = np.where(label_val != predicted_label_val)[0]

                # Remove misclassified data from the batch
                X_test = np.delete(X_test, delete, axis=0)
                Y_test = np.delete(Y_test, delete, axis=0)

                # Check if batch is empty now
                if len(X_test) == 0:
                    print('All samples in the batch were misclassified')
                    continue

                # Update feed-dict
                feed_dict = {x: X_test, y: Y_test, is_training: False}

        # Collect total number of samples
        n_samples += X_test.shape[0]

        # Evaluate
        with sess.as_default():
            adv_loss_val, adv_correct_preds_val = sess.run([adv_loss, adv_correct_preds], feed_dict=feed_dict)

        # Accumulate
        accuracy += (adv_correct_preds_val)[:].sum()
        loss += adv_loss_val

    # Divide by number of examples to get final value
    accuracy /= n_samples
    loss /= n_samples

    return accuracy, loss


# -------------------------------------------------------------
#    Adversarial training (AT) and adversarial logit pairing (ALP)
# -------------------------------------------------------------

def get_at_loss(sess, x, y, model, eps, eps_iter, iterations):
    # Set up PGD attack graph using Cleverhans library

    pgd_params = {
        'ord': np.inf,
        'y': y,
        'eps': eps / 255,
        'eps_iter': eps_iter / 255,
        'nb_iter': iterations,
        'rand_init': True,
        'rand_minmax': eps / 255,
        'clip_min': 0.,
        'clip_max': 1.,
        'sanity_checks': True
    }

    pgd = ProjectedGradientDescent(model, sess=sess)
    adv_x = pgd.generate(x, **pgd_params)
    adv_logits = model.get_logits(adv_x)

    # Add summary for adversarial training images
    with tf.device('/gpu:0'):
        with tf.name_scope('Adversarial-Image-Summaries'):
            tf.summary.image('adv-input', adv_x, max_outputs=2, family='Adversarial-Training',
                             collections=['training'])

    adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=adv_logits, labels=y)
    adv_loss = tf.reduce_mean(adv_loss)

    return adv_loss, adv_logits


def get_alp_loss(sess, x, y, logits, adv_logits, model, eps, eps_iter, iterations):
    if adv_logits is None:
        pgd_params = {
            'ord': np.inf,
            'y': y,
            'eps': eps / 255,
            'eps_iter': eps_iter / 255,
            'nb_iter': iterations,
            'rand_init': True,
            'rand_minmax': eps / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True
        }

        pgd = ProjectedGradientDescent(model, sess=sess)
        adv_x = pgd.generate(x, **pgd_params)
        adv_logits = model.get_logits(adv_x)

    adv_pairing_loss = tf.losses.mean_squared_error(logits, adv_logits)

    return adv_pairing_loss


# -------------------------------------------------------------
#    Logit squeezing (LSQ) and clean logit pairing (CLP)
# -------------------------------------------------------------

def get_lsq_loss(x, model):
    logits = model.get_logits(x)
    logits = tf.layers.flatten(logits)

    squeezing_loss = tf.reduce_mean(tf.norm(logits, ord='euclidean', axis=1))

    return squeezing_loss


def get_clp_loss(x_pair1, x_pair2, model):
    logits1 = model.get_logits(x_pair1)
    logits2 = model.get_logits(x_pair2)

    clean_pairing_loss = tf.losses.mean_squared_error(logits1, logits2, loss_collection=None)

    return clean_pairing_loss
