import tensorflow
import tensorflow.contrib.slim

import microscopeimagequality.miq


class MiqTest(tensorflow.test.TestCase):
    def test_add_loss_training_runs(self):
        with self.test_session():
            targets = tensorflow.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

            inputs = tensorflow.constant([[0.7, 0.3, 0.0], [0.9, 0.1, 0.0], [0.6, 0.4, 0.0], [0.0, 0.4, 0.6]])

            predictions = tensorflow.contrib.layers.fully_connected(inputs, 3)

            microscopeimagequality.miq.add_loss(targets, predictions, use_rank_loss=True)

            total_loss = tensorflow.losses.get_total_loss()

            tensorflow.summary.scalar("Total Loss", total_loss)

            optimizer = tensorflow.train.AdamOptimizer(0.000001)

            # Set up training.
            train_op = tensorflow.contrib.slim.learning.create_train_op(total_loss, optimizer)

            # Run training.
            tensorflow.contrib.slim.learning.train(train_op, None, number_of_steps=5, log_every_n_steps=5)
