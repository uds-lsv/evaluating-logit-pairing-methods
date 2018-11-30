import argparse
import os
from cleverhans.attacks import ProjectedGradientDescent, SPSA
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf


def eval_robustness(ARGS, verbose=True):
    #############################################
    # Load pre-trained model
    #############################################

    if verbose:
        print('\n- Loading pre-trained model...')

    # Build evaluation graph
    eval_graph = tf.Graph()
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=eval_graph, config=config)

    # Define input TF placeholder
    with eval_graph.as_default():
        with tf.device('/gpu:0'):
            # Define placeholders
            with tf.name_scope('Placeholders'):
                x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='inputs')
                y = tf.placeholder(dtype=tf.float32, shape=(None, n_classes), name='labels')
                is_training = tf.placeholder_with_default(False, shape=(), name='is-training')

            # Define model
            with tf.name_scope('Model'):
                model = Model(nb_classes=n_classes, input_shape=input_shape, is_training=is_training)

            # Define forward-pass
            with tf.name_scope('Logits'):
                logits = model.get_logits(x)
            with tf.name_scope('Probs'):
                preds = tf.nn.softmax(logits)

            # Restore the pre-trained model
            with sess.as_default():
                saver = tf.train.Saver()
                saver.restore(sess, ARGS.restore_path + '/model.ckpt')

            # Define accuracy ops
            with tf.name_scope('Accuracy'):
                ground_truth = tf.argmax(y, axis=1)
                predicted_label = tf.argmax(preds, axis=1)
                correct_prediction = tf.equal(predicted_label, ground_truth)
                clean_acc = tf.reduce_mean(tf.to_float(correct_prediction), name='accuracy')

            # Define PGD adversary
            if ARGS.attack == 'PGD':
                if verbose:
                    print('\n- Building {:s} attack graph...'.format(ARGS.attack))

                with tf.name_scope('PGD-Attacker'):
                    pgd_params = {
                        'ord': np.inf,
                        'y': y,
                        'eps': ARGS.eps / 255,
                        'eps_iter': ARGS.eps_iter / 255,
                        'nb_iter': ARGS.nb_iter,
                        'rand_init': ARGS.rand_init,
                        'rand_minmax': ARGS.eps / 255,
                        'clip_min': 0.,
                        'clip_max': 1.,
                        'sanity_checks': True
                    }

                    pgd = ProjectedGradientDescent(model, sess=None)
                    adv_x = pgd.generate(x, **pgd_params)

            # Define SPSA adversary
            elif ARGS.attack == 'SPSA':
                if verbose:
                    print('\n- Building {:s} attack graph...'.format(ARGS.attack))

                with tf.name_scope('PGD-Attacker'):
                    spsa_params = {
                        'y': y,
                        'eps': ARGS.eps / 255,
                        'nb_iter': ARGS.nb_iter,
                        'spsa_samples': ARGS.spsa_samples,
                        'spsa_iters': ARGS.spsa_iters,
                        'clip_min': 0.,
                        'clip_max': 1.,
                        'learning_rate': ARGS.spsa_lr,
                        'delta': ARGS.spsa_delta
                    }

                    spsa = SPSA(model, sess=sess)
                    adv_x = spsa.generate(x, **spsa_params)
            else:
                raise NotImplementedError

            with tf.name_scope('Logits'):
                adv_logits = model.get_logits(adv_x)
            with tf.name_scope('Probs'):
                adv_preds = tf.nn.softmax(adv_logits)

            adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=adv_logits, labels=y)
            adv_predicted_label = tf.argmax(adv_preds, axis=1)
            correct_prediction = tf.equal(adv_predicted_label, ground_truth)
            adv_accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name='adv-accuracy')
            is_adv_example = tf.not_equal(ground_truth, adv_predicted_label)

    #############################################
    # Run evaluation
    #############################################

    if verbose:
        print('\n- Running robustness evaluation against {:s} attacker...\n'.format(ARGS.attack))

    if ARGS.attack == 'PGD':
        clean, adv_mean, adv_worstcase = run_pgd_eval(x, y, is_training, sess, adv_testloader, clean_acc, adv_accuracy,
                                                      adv_loss, is_adv_example, ARGS,
                                                      save_loss_dist=False, verbose=verbose)

    elif ARGS.attack == 'SPSA':
        clean, adv_mean = run_spsa_eval(x, y, is_training, sess, adv_testloader, clean_acc, adv_accuracy, adv_loss,
                                        is_adv_example, ARGS,
                                        save_loss_dist=False, verbose=verbose)
        adv_worstcase = adv_mean
    else:
        raise NotImplementedError

    return clean, adv_mean, adv_worstcase


def run_pgd_eval(x, y, is_training, sess, adv_testloader, clean_acc, adv_accuracy, adv_loss, is_adv_example, ARGS,
                 save_loss_dist=False, verbose=True):
    restarts_mask = np.zeros(shape=(1, ARGS.samples))
    restarts_clean_acc = []
    restarts_adv_acc = []
    restarts_loss_dist = []

    for _ in tqdm(range(ARGS.restarts)):
        n_samples = 0

        restart_adv_examples = np.zeros(shape=(1, ARGS.samples))
        restart_loss_dist = np.zeros(shape=(1, ARGS.samples))
        restart_clean_acc = []
        restart_adv_acc = []

        for i, data in enumerate(adv_testloader, start=0):
            # Get next batch from the pytorch dataloader object
            X_test, Y_test = data
            X_test, Y_test = X_test.numpy(), Y_test.numpy()

            # Change format for the input from NCHW to NHWC
            X_test = X_test.transpose([0, 2, 3, 1])

            # Reshape and apply one hot encoding to labels
            Y_test = np.reshape(Y_test, newshape=-1)
            Y_test = np.eye(n_classes)[Y_test]

            feed_dict = {x: X_test, y: Y_test, is_training: False}

            # Collect total number of samples
            n_samples += X_test.shape[0]

            # Evaluate on current batch
            with sess.as_default():
                clean_acc_val, adv_accuracy_val, adv_loss_val, is_adv_example_val = sess.run(
                    [clean_acc, adv_accuracy, adv_loss, is_adv_example],
                    feed_dict=feed_dict)

                # Collect adversarial examples and accuracies for each batch
                restart_adv_examples[0, i * ARGS.batch_size:(i + 1) * ARGS.batch_size] = is_adv_example_val
                restart_clean_acc.append(clean_acc_val)
                restart_adv_acc.append(adv_accuracy_val)
                restart_loss_dist[0, i * ARGS.batch_size:(i + 1) * ARGS.batch_size] = adv_loss_val

                # Collect adversarial examples
                restarts_mask = np.concatenate([restarts_mask, restart_adv_examples])

        restarts_clean_acc.append(np.mean(restart_clean_acc))
        restarts_adv_acc.append(np.mean(restart_adv_acc))
        restarts_loss_dist.append(restart_loss_dist)

    restarts_mask = restarts_mask[1:, :]  # remove first row
    aggregated_mask = np.max(restarts_mask, axis=0)  # select only the most harmful restarts for every sample

    # Make numpy arrays and reshape
    restarts_clean_acc = np.asarray(restarts_clean_acc)
    restarts_adv_acc = np.asarray(restarts_adv_acc)
    restarts_loss_dist = np.asarray(restarts_loss_dist)
    restarts_loss_dist = np.reshape(restarts_loss_dist, newshape=(ARGS.restarts, ARGS.samples))

    if save_loss_dist:
        # Persist distribution data
        np.save(ARGS.log_dir + '/loss-distribution.npy', restarts_loss_dist)

    if verbose:
        print('\n------------------------- RESULTS -------------------------')
        print('model: {:s}'.format(ARGS.restore_path))
        print('PGD: eps: {:.2f}, eps_iter: {:.2f}, nb_iter: {:d}, restarts: {:d}'.format(ARGS.eps, ARGS.eps_iter,
                                                                                         ARGS.nb_iter,
                                                                                         ARGS.restarts))
        print('samples: {:d}'.format(ARGS.samples))
        print('clean accuracy (mean): {:.4f}'.format(np.mean(restarts_clean_acc)))
        print('adversarial accuracy (mean): {:.4f}'.format(np.mean(restarts_adv_acc)))
        print('adversarial accuracy (worstcase): {:.4f}'.format(1.0 - np.mean(aggregated_mask)))
        print('--------------------------- END ---------------------------\n')

    return np.mean(restarts_clean_acc), np.mean(restarts_adv_acc), 1.0 - np.mean(aggregated_mask)


def run_spsa_eval(x, y, is_training, sess, adv_testloader, clean_acc, adv_accuracy, adv_loss, is_adv_example, ARGS,
                  save_loss_dist=False, verbose=False):
    n_samples = 0
    clean_accuracies = []
    adv_accuracies = []
    adv_losses = []
    adv_examples_mask = []

    for i, data in tqdm(list(enumerate(adv_testloader, start=0))):
        # Get next batch (containing a single sample) from the pytorch dataloader object
        X_test, Y_test = data
        X_test, Y_test = X_test.numpy(), Y_test.numpy()

        # Change format for the input from NCHW to NHWC
        X_test = X_test.transpose([0, 2, 3, 1])

        # Reshape and apply one hot encoding to labels
        Y_test = np.reshape(Y_test, newshape=-1)
        Y_test = np.eye(n_classes)[Y_test]

        feed_dict = {x: X_test, y: Y_test, is_training: False}

        # Collect total number of samples
        n_samples += X_test.shape[0]

        # Evaluate on current batch
        with sess.as_default():
            clean_acc_val, adv_accuracy_val, adv_loss_val, is_adv_example_val = sess.run(
                [clean_acc, adv_accuracy, adv_loss, is_adv_example],
                feed_dict=feed_dict)

            # Collect adversarial examples and accuracies for each batch
            clean_accuracies.append(clean_acc_val)
            adv_accuracies.append(adv_accuracy_val)
            adv_losses.append(adv_loss_val)
            adv_examples_mask.append(is_adv_example_val)

    if save_loss_dist:
        # Persist distribution data
        np.save(ARGS.log_dir + '/loss-distribution.npy', adv_losses)

    if verbose:
        print('\n------------------------- RESULTS -------------------------')
        print('model: {:s}'.format(ARGS.restore_path))
        print(
            'SPSA: eps: {:.2f}, spsa_samples: {:d}, spsa_iters: {:d}, nb_iter: {:d}, spsa_lr: {:.4f}, spsa_delta: {:.4f}'.format(
                ARGS.eps, ARGS.spsa_samples,
                ARGS.spsa_iters,
                ARGS.nb_iter, ARGS.spsa_lr, ARGS.spsa_delta))
        print('samples: {:d}'.format(ARGS.samples))
        print('clean accuracy (mean): {:.4f}'.format(np.mean(clean_accuracies)))
        print('adversarial accuracy (mean): {:.4f}'.format(np.mean(adv_accuracies)))
        print('--------------------------- END ---------------------------\n')

    return np.mean(clean_accuracies), np.mean(adv_accuracies)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument('--dataset', type=str, default='mnist', help='Data set used for training')

    # Dirs
    parser.add_argument('--data_dir', type=str, default='/path/to/mnist', help='Directory containing the datasets')
    parser.add_argument('--restore_path', type=str, default='/path/to/pretrained/model/checkpoints',
                        help='Directory containing the model weights')
    parser.add_argument('--log_dir', type=str, default='/path/to/log/dir', help='Directory for storing log files')

    # Attacks
    parser.add_argument('--attack', type=str, default='PGD', help='Adversary')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of (random) samples from the test set to evaluate on')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size used during evaluation')

    # PGD
    parser.add_argument('--eps', type=float, default=76.5, help='Adversarial perturbation')
    parser.add_argument('--eps_iter', type=float, default=2.55, help='Step size of PGD attack')
    parser.add_argument('--nb_iter', type=int, default=40, help='Iterations of PGD attack')
    parser.add_argument('--rand_init', type=bool, default=True, help='Peform initial random step')
    parser.add_argument('--restarts', type=int, default=1, help='Restarts of PGD attack')

    # SPSA
    parser.add_argument('--spsa_samples', type=int, default=8192,
                        help='Number of inputs to evaluate at a single time. The true batch size (the number of evaluated inputs for each update) is spsa_samples * spsa_iters')
    parser.add_argument('--spsa_iters', type=int, default=1,
                        help='Number of model evaluations before performing an update, where each evaluation is on spsa_samples different inputs.')
    parser.add_argument('--spsa_lr', type=float, default=0.01, help='')
    parser.add_argument('--spsa_delta', type=float, default=0.01, help='')

    ARGS, unparsed = parser.parse_known_args()

    # Load dataset and specify model
    if ARGS.dataset == 'mnist':
        from utils import load_mnist

        _, _, adv_testloader, input_shape, n_classes = load_mnist(ARGS.batch_size, ARGS.data_dir,
                                                                  augmentation=False, stddev=0.0,
                                                                  adv_subset=ARGS.samples, workers=0)

        from models import LeNet as Model

    elif ARGS.dataset == 'cifar10':
        from utils import load_cifar10

        _, _, adv_testloader, input_shape, n_classes = load_cifar10(ARGS.batch_size, ARGS.data_dir,
                                                                    augmentation=False, stddev=0.0,
                                                                    adv_subset=ARGS.samples, workers=0)

        from models import ResNet20_v2 as Model

    else:
        raise NotImplementedError('Only MNIST and CIFAR10 are supported.')

    # Start evaluation
    eval_robustness(ARGS)
