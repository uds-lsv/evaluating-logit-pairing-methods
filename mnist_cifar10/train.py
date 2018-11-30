import argparse
import os
import numpy as np
from utils import variable_summaries
from train_utils import model_train, get_at_loss, get_alp_loss, get_lsq_loss, get_clp_loss
from cleverhans.attacks import ProjectedGradientDescent

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf


def train(ARGS):
    # Define helper function for evaluating on test data during training
    def eval(epoch):
        from train_utils import clean_eval
        test_accuracy, test_loss, _ = clean_eval(sess, x, y, is_training, testloader, n_classes, logits,
                                                 preds)
        # Write tensorboard summary
        acc_summary = tf.Summary()
        acc_summary.value.add(tag='Evaluation/accuracy/test', simple_value=test_accuracy)
        writer_test.add_summary(acc_summary, epoch)

        # Write tensorboard summary
        err_summary = tf.Summary()
        err_summary.value.add(tag='Evaluation/error/test', simple_value=1.0 - test_accuracy)
        writer_test.add_summary(err_summary, epoch)

        # Write tensorboard summary
        loss_summary = tf.Summary()
        loss_summary.value.add(tag='Evaluation/loss/test', simple_value=test_loss)
        writer_test.add_summary(loss_summary, epoch)

    # Define helper function for evaluating on adversarial test data during training
    def adv_eval(epoch):
        from train_utils import adversarial_eval
        adv_accuracy, adv_loss = adversarial_eval(sess, x, y, is_training, adv_testloader, n_classes, preds, adv_preds,
                                                  eval_all=True)

        # Write tensorboard summary
        acc_summary = tf.Summary()
        acc_summary.value.add(tag='Evaluation/adversarial-accuracy/test', simple_value=adv_accuracy)
        writer_test.add_summary(acc_summary, epoch)

        # Write tensorboard summary
        err_summary = tf.Summary()
        err_summary.value.add(tag='Evaluation/adversarial-error/test', simple_value=1.0 - adv_accuracy)
        writer_test.add_summary(err_summary, epoch)

        # Write tensorboard summary
        loss_summary = tf.Summary()
        loss_summary.value.add(tag='Evaluation/adversarial-loss/test', simple_value=adv_loss)
        writer_test.add_summary(loss_summary, epoch)

    # Define computational graph
    with tf.Graph().as_default() as g:
        # Define placeholders
        with tf.device('/gpu:0'):
            with tf.name_scope('Placeholders'):
                x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='inputs')
                x_pair1 = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x-pair1')
                x_pair2 = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x-pair2')
                y = tf.placeholder(dtype=tf.float32, shape=(None, n_classes), name='labels')
                is_training = tf.placeholder_with_default(True, shape=(), name='is-training')

        # Define TF session
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=g, config=config)

        # Define model
        with tf.name_scope('Model'):
            with tf.device('/gpu:0'):
                model = Model(nb_classes=n_classes, input_shape=input_shape, is_training=is_training)

                # Define forward-pass
                with tf.name_scope('Logits'):
                    logits = model.get_logits(x)
                with tf.name_scope('Probs'):
                    preds = tf.nn.softmax(logits)

                with tf.name_scope('Accuracy'):
                    ground_truth = tf.argmax(y, axis=1)
                    predicted_label = tf.argmax(preds, axis=1)
                    correct_prediction = tf.equal(predicted_label, ground_truth)
                    acc = tf.reduce_mean(tf.to_float(correct_prediction), name='accuracy')
                    tf.add_to_collection('accuracies', acc)

                    err = tf.identity(1.0 - acc, name='error')
                    tf.add_to_collection('accuracies', err)

                # Define losses
                with tf.name_scope('Losses'):
                    ce_loss, wd_loss, clp_loss, lsq_loss, at_loss, alp_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    adv_logits = None

                    if ARGS.ct:
                        with tf.name_scope('Cross-Entropy-Loss'):
                            ce_loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y),
                                name='cross-entropy-loss')

                            tf.add_to_collection('losses', ce_loss)

                    if ARGS.at:
                        with tf.name_scope('Adversarial-Cross-Entropy-Loss'):
                            at_loss, adv_logits = get_at_loss(sess, x, y, model, ARGS.eps, ARGS.eps_iter,
                                                              ARGS.nb_iter)
                            at_loss = tf.identity(at_loss, name='at-loss')
                            tf.add_to_collection('losses', at_loss)

                    with tf.name_scope('Regularizers'):
                        if ARGS.wd:
                            with tf.name_scope('Weight-Decay'):
                                for var in tf.trainable_variables():
                                    if 'beta' in var.op.name:
                                        # Do not regularize bias of batch normalization
                                        continue
                                    # print('regularizing: ', var.op.name)
                                    wd_loss += tf.nn.l2_loss(var)

                                reg_loss = tf.identity(wd_loss, name='wd-loss')
                                tf.add_to_collection('losses', reg_loss)

                        if ARGS.alp:
                            with tf.name_scope('Adversarial-Logit-Pairing'):
                                alp_loss = get_alp_loss(sess, x, y, logits, adv_logits, model, ARGS.eps,
                                                        ARGS.eps_iter,
                                                        ARGS.nb_iter)

                                alp_loss = tf.identity(alp_loss, name='alp-loss')
                                tf.add_to_collection('losses', alp_loss)

                        if ARGS.clp:
                            with tf.name_scope('Clean-Logit-Pairing'):
                                clp_loss = get_clp_loss(x_pair1, x_pair2, model)
                                clp_loss = tf.identity(clp_loss, name='clp-loss')
                                tf.add_to_collection('losses', clp_loss)

                        if ARGS.lsq:
                            with tf.name_scope('Logit-Squeezing'):
                                lsq_loss = get_lsq_loss(x, model)
                                lsq_loss = tf.identity(lsq_loss, name='lsq-loss')
                                tf.add_to_collection('losses', lsq_loss)

                    with tf.name_scope('Total-Loss'):
                        # Define objective function
                        total_loss = (ARGS.ct_lambda * ce_loss) + (ARGS.at_lambda * at_loss) + (
                                ARGS.wd_lambda * wd_loss) + (
                                             ARGS.clp_lambda * clp_loss) + (ARGS.lsq_lambda * lsq_loss) + (
                                             ARGS.alp_lambda * alp_loss)

                        total_loss = tf.identity(total_loss, name='total-loss')
                        tf.add_to_collection('losses', total_loss)

                # Define PGD adversary
                with tf.name_scope('PGD-Attacker'):
                    pgd_params = {'ord': np.inf,
                                  'y': y,
                                  'eps': ARGS.eps / 255,
                                  'eps_iter': ARGS.eps_iter / 255,
                                  'nb_iter': ARGS.nb_iter,
                                  'rand_init': True,
                                  'rand_minmax': ARGS.eps / 255,
                                  'clip_min': 0.,
                                  'clip_max': 1.,
                                  'sanity_checks': True
                                  }

                    pgd = ProjectedGradientDescent(model, sess=sess)
                    adv_x = pgd.generate(x, **pgd_params)

                    with tf.name_scope('Logits'):
                        adv_logits = model.get_logits(adv_x)
                    with tf.name_scope('Probs'):
                        adv_preds = tf.nn.softmax(adv_logits)

        # Define optimizer
        with tf.device('/gpu:0'):
            with tf.name_scope('Optimizer'):
                # Define global step variable
                global_step = tf.get_variable(
                    name='global_step',
                    shape=[],  # scalar
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                    trainable=False
                )

                optimizer = tf.train.AdamOptimizer(learning_rate=ARGS.lr, beta1=0.9, beta2=0.999, epsilon=1e-6,
                                                   use_locking=False, name='Adam')
                trainable_vars = tf.trainable_variables()

                update_bn_ops = tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS)  # this collection stores the moving_mean and moving_variance ops
                #  for batch normalization
                with tf.control_dependencies(update_bn_ops):
                    grads_and_vars = optimizer.compute_gradients(total_loss, trainable_vars)
                    train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Add Tensorboard summaries
        with tf.device('/gpu:0'):
            # Create file writers
            writer_train = tf.summary.FileWriter(ARGS.log_dir + '/train', graph=g)
            writer_test = tf.summary.FileWriter(ARGS.log_dir + '/test')

            # Add summary for input images
            with tf.name_scope('Image-Summaries'):
                # Create image summary ops
                tf.summary.image('input', x, max_outputs=2, collections=['training'])

            # Add summaries for the training losses
            losses = tf.get_collection('losses')
            for entry in losses:
                tf.summary.scalar(entry.name, entry, collections=['training'])

            # Add summaries for the training accuracies
            accs = tf.get_collection('accuracies')
            for entry in accs:
                tf.summary.scalar(entry.name, entry, collections=['training'])

            # Add summaries for all trainable vars
            for var in trainable_vars:
                tf.summary.histogram(var.op.name, var, collections=['training'])
                var_norm = tf.norm(var, ord='euclidean')
                tf.summary.scalar(var.op.name + '/l2norm', var_norm, collections=['training'])

            # Add summaries for variable gradients
            for grad, var in grads_and_vars:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad, collections=['training'])
                    grad_norm = tf.norm(grad, ord='euclidean')
                    tf.summary.scalar(var.op.name + '/gradients/l2norm', grad_norm, collections=['training'])

            # Add summaries for the logits and model predictions
            with tf.name_scope('Logits-Summaries'):
                variable_summaries(tf.identity(logits, name='logits'), name='logits',
                                   collections=['training', 'test'],
                                   histo=True)
            with tf.name_scope('Predictions-Summaries'):
                variable_summaries(tf.identity(preds, name='predictions'), name='predictions',
                                   collections=['training', 'test'], histo=True)

        # Initialize all variables
        with sess.as_default():
            tf.global_variables_initializer().run()

        # Collect training params
        train_params = {
            'epochs': ARGS.epochs,
            'eval_step': ARGS.eval_step,
            'adv_eval_step': ARGS.adv_eval_step,
            'n_classes': n_classes,
            'clp': ARGS.clp
        }

        # Start training loop
        model_train(sess, x, y, x_pair1, x_pair2, is_training, trainloader, train_step, args=train_params,
                    evaluate=eval, adv_evaluate=adv_eval, writer_train=writer_train)

        # Save the trained model
        if ARGS.save:
            save_path = os.path.join(ARGS.save_dir, ARGS.filename)
            saver = tf.train.Saver(var_list=tf.global_variables())
            saver.save(sess, save_path)
            print("Saved model at {:s}".format(str(ARGS.save_dir)))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Dirs
    parser.add_argument('--data_dir', type=str, default='/path/to/mnist',
                        help='Directory containing the datasets')
    parser.add_argument('--log_dir', type=str,
                        default='/path/to/log/dir',
                        help='Directory for storing log files')
    parser.add_argument('--save', type=bool, default=True, help='Save the model at the end')
    parser.add_argument('--save_dir', type=str,
                        default='/path/to/model/checkpoints/',
                        help='Directory for storing model')
    parser.add_argument('--filename', type=str, default='model.ckpt', help='Filename of trained model')

    # Inputs
    parser.add_argument('--dataset', type=str, default='mnist', help='Data set used for training')
    parser.add_argument('--noise', type=bool, default=False, help='Add Gaussian noise during training')
    parser.add_argument('--stddev', type=float, default=0.5, help='Standard deviation used for Gaussian noise')
    parser.add_argument('--adv_subset', type=int, default=1000,
                        help='Size of a subset of the test data that is used for crafting adversarial examples during training')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size used during training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--eval_step', type=int, default=1, help='When to evaluate on clean test examples')
    parser.add_argument('--adv_eval_step', type=int, default=1, help='When to evaluate on adversarial test examples')
    parser.add_argument('--ct', type=bool, default=True, help='Train on clean examples')
    parser.add_argument('--ct_lambda', type=float, default=1.0, help='Weight of loss on clean examples')
    parser.add_argument('--at', type=bool, default=False, help='Train on adversarial examples')
    parser.add_argument('--at_lambda', type=float, default=0.0, help='Weight of loss on adversarial examples')
    parser.add_argument('--wd', type=bool, default=True, help='Add weight decay')
    parser.add_argument('--wd_lambda', type=float, default=0.0, help='Weight of l2 regularizer')
    parser.add_argument('--alp', type=bool, default=False, help='Add ALP regularizer')
    parser.add_argument('--alp_lambda', type=float, default=0.0, help='Weight of ALP loss')
    parser.add_argument('--clp', type=bool, default=True, help='Add CLP regularizer')
    parser.add_argument('--clp_lambda', type=float, default=0.0, help='Weight of CLP loss')
    parser.add_argument('--lsq', type=bool, default=True, help='Add LSQ regularizer')
    parser.add_argument('--lsq_lambda', type=float, default=0.5, help='Weight of LSQ loss')

    # PGD
    parser.add_argument('--eps', type=float, default=76.5, help='Adversarial perturbation')
    parser.add_argument('--eps_iter', type=float, default=2.55, help='Step size of PGD attack')
    parser.add_argument('--nb_iter', type=float, default=40, help='Iterations of PGD attack')

    ARGS, unparsed = parser.parse_known_args()

    # Load dataset and specify model
    if ARGS.dataset == 'mnist':
        from utils import load_mnist

        trainloader, testloader, adv_testloader, input_shape, n_classes = load_mnist(ARGS.batch_size, ARGS.data_dir,
                                                                                     ARGS.noise, ARGS.stddev,
                                                                                     ARGS.adv_subset)

        from models import LeNet as Model

    elif ARGS.dataset == 'cifar10':
        from utils import load_cifar10

        trainloader, testloader, adv_testloader, input_shape, n_classes = load_cifar10(ARGS.batch_size, ARGS.data_dir,
                                                                                       ARGS.noise, ARGS.stddev,
                                                                                       ARGS.adv_subset)

        from models import ResNet20_v2 as Model

    else:
        raise NotImplementedError('Only MNIST and CIFAR10 are supported.')

    # Start training
    train(ARGS)
