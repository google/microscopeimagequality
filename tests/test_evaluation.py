import os
import tempfile

import PIL.Image
import numpy
import tensorflow
import tensorflow.contrib.slim

import microscopeimagequality.evaluation


class Evaluation(tensorflow.test.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.test_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.test_dir = tempfile.mkdtemp()
        self.patch_width = 28
        self.image_shape = (int(numpy.sqrt(self.batch_size) * self.patch_width),
                            int(numpy.sqrt(self.batch_size) * self.patch_width))

    def testAnnotatePatch(self):
        image_width = 28
        image = numpy.expand_dims(
            numpy.expand_dims(
                numpy.zeros((image_width, image_width)), axis=0), axis=3)
        annotated_image = microscopeimagequality.evaluation.annotate_patch(image, prediction=0, label=0)
        expected_image_width = (
            image_width * microscopeimagequality.evaluation._IMAGE_ANNOTATION_MAGNIFICATION_PERCENT / 100)
        self.assertEquals(annotated_image.shape,
                          (1, expected_image_width, expected_image_width, 1))

        def check_image_matches_golden(prediction, label):
            annotated_image = microscopeimagequality.evaluation.annotate_patch(
                image, prediction=prediction, label=label)
            test_image = numpy.squeeze(annotated_image).astype(numpy.uint8)
            golden = numpy.array(
                PIL.Image.open(
                    os.path.join(self.test_data_directory,
                                 'annotated_image_predicted_{}_label_{}.png'.format(
                                     prediction, label))))
            self.assertEquals(golden.shape, test_image.shape)
            self.assertEquals(golden.dtype, test_image.dtype)

            numpy.testing.assert_array_equal(golden, test_image)

        check_image_matches_golden(0, 0)
        check_image_matches_golden(0, 1)
        check_image_matches_golden(1, 0)
        check_image_matches_golden(1, 1)

    def testAnnotateClassificationErrorsRuns(self):
        num_classes = 5
        images = tensorflow.zeros([self.batch_size, self.patch_width, self.patch_width, 1])
        predictions = tensorflow.zeros([self.batch_size, ])
        labels = tensorflow.zeros([self.batch_size, ])
        probabilities = tensorflow.zeros([self.batch_size, num_classes])
        microscopeimagequality.evaluation.annotate_classification_errors(images, predictions, labels,
                                                          probabilities, self.image_shape[0],
                                                          self.image_shape[1])

    def testAnnotateClassificationErrorsRunsInTensorFlow(self):
        g = tensorflow.Graph()
        with g.as_default():
            num_classes = 5
            images = tensorflow.zeros(
                [self.batch_size, self.patch_width, self.patch_width, 1])
            predictions = tensorflow.zeros([self.batch_size, ])
            labels = tensorflow.zeros([self.batch_size, ])
            probabilities = tensorflow.zeros([self.batch_size, num_classes])
            image, summary = microscopeimagequality.evaluation.annotate_classification_errors(
                images, predictions, labels, probabilities, self.image_shape[0],
                self.image_shape[1])

            sv = tensorflow.train.Supervisor()
            with sv.managed_session() as sess:
                [_, image_np] = sess.run([summary, image])
            self.assertEqual((1, self.patch_width * numpy.sqrt(self.batch_size),
                              self.patch_width * numpy.sqrt(self.batch_size), 3),
                             image_np.shape)

    def testGetConfusionMatrix(self):
        predicted_probabilities = numpy.array([[0.4, 0.6], [0, 1]])
        confusion_matrix = microscopeimagequality.evaluation.get_confusion_matrix(
            predicted_probabilities, [0, 1],
            os.path.join(self.test_dir, 'confusion_matrix.png'),
            'Test confusion matrix',
            use_predictions_instead_of_probabilities=True)
        confusion_matrix_expected = numpy.array([[0.0, 1.0], [0.0, 1.0]])
        self.assertAllEqual(confusion_matrix_expected, confusion_matrix)

    def testGetConfusionMatrixWithProbabilities(self):
        predicted_probabilities = numpy.array([[0.4, 0.6], [0, 1]])
        confusion_matrix = microscopeimagequality.evaluation.get_confusion_matrix(
            predicted_probabilities, [0, 1],
            os.path.join(self.test_dir, 'confusion_matrix_probabilities.png'),
            'Test confusion matrix with probabilities',
            use_predictions_instead_of_probabilities=False)
        confusion_matrix_expected = numpy.array([[0.4, 0.6], [0, 1]], dtype=numpy.float32)
        self.assertAllEqual(confusion_matrix_expected, confusion_matrix)

    def testGetAggregatedPredictionTruePositive1(self):
        with self.test_session() as sess:
            probabilities = tensorflow.constant(
                [[0.0, 1.0], [0.2, 0.8], [0.5, 0.5], [0.9, 0.1]])
            labels = tensorflow.constant([1, 1, 1, 1])
            prediction, label = microscopeimagequality.evaluation.get_aggregated_prediction(probabilities,
                                                                             labels,
                                                                             self.batch_size)

            self.assertEquals(sess.run(label), 1)
            self.assertEquals(sess.run(prediction), 1)

    def testGetAggregatedPredictionTruePositive2(self):
        with self.test_session() as sess:
            probabilities = tensorflow.constant(
                [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.0, 1.0]])
            labels = tensorflow.constant([1, 1, 1, 1])
            prediction, label = microscopeimagequality.evaluation.get_aggregated_prediction(probabilities,
                                                                             labels,
                                                                             self.batch_size)
            self.assertEquals(sess.run(label), 1)
            self.assertEquals(sess.run(prediction), 1)

    def testGetAggregatedPredictionTrueNegative1(self):
        with self.test_session() as sess:
            probabilities = tensorflow.constant(
                [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [1.0, 0.0]])
            labels = tensorflow.constant([0, 0, 0, 0])
            prediction, label = microscopeimagequality.evaluation.get_aggregated_prediction(probabilities,
                                                                             labels,
                                                                             self.batch_size)
            self.assertEquals(sess.run(label), 0)
            self.assertEquals(sess.run(prediction), 0)

    def testGetAggregatedPredictionTrueNegative2(self):
        with self.test_session() as sess:
            probabilities = tensorflow.constant(
                [[1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
            labels = tensorflow.constant([0, 0, 0, 0])
            prediction, label = microscopeimagequality.evaluation.get_aggregated_prediction(probabilities,
                                                                             labels,
                                                                             self.batch_size)
            self.assertEquals(sess.run(label), 0)
            self.assertEquals(sess.run(prediction), 0)

    def testGetAggregatedPredictionRequiresIdenticalLabels(self):
        with self.test_session() as sess:
            probabilities = tensorflow.constant(
                [[0.0, 1.0], [0.0, 0.1], [0.0, 0.1], [0.0, 1.0]])
            labels = tensorflow.constant([0, 0, 0, 1])
            _, label = microscopeimagequality.evaluation.get_aggregated_prediction(probabilities, labels,
                                                                    self.batch_size)
            with self.assertRaises(tensorflow.errors.InvalidArgumentError):
                self.assertEquals(sess.run(label), 0)

    def testAggregatedPredictionWithAccuracy(self):
        with self.test_session() as sess:
            probabilities = tensorflow.constant(
                [[1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
            labels = tensorflow.constant([0, 0, 0, 0], dtype=tensorflow.int64)
            prediction, label = microscopeimagequality.evaluation.get_aggregated_prediction(probabilities,
                                                                             labels,
                                                                             self.batch_size)
            self.assertEquals(sess.run(label), 0)
            self.assertEquals(sess.run(prediction), 0)

            # Check that 'prediction' and 'label' are valid inputs to this function.
            tensorflow.contrib.slim.metrics.aggregate_metric_map({
                'Accuracy': tensorflow.contrib.metrics.streaming_accuracy(prediction, label),
            })

    def testVisualizeImagePredictionsRuns(self):
        num_patches = 4
        patches = numpy.ones((num_patches, 28, 28, 1))
        probabilities = numpy.ones((num_patches, 2)) / 2.0
        labels = numpy.ones(num_patches, dtype=numpy.int32)
        microscopeimagequality.evaluation.visualize_image_predictions(
            patches,
            probabilities,
            labels,
            self.image_shape[0],
            self.image_shape[1],
            show_plot=True)

    def testGetClassRgbIsHsv(self):
        class_rgb = microscopeimagequality.evaluation._get_class_rgb(11, 0)
        self.assertEquals(3, len(class_rgb))
        # This is the first HSV color.
        self.assertEquals((1.0, 0, 0), class_rgb)

    def testGetCertainty(self):
        certainty = microscopeimagequality.evaluation.get_certainty(numpy.array([0.5, 0.5]))
        self.assertEquals(0.0, certainty)
        certainty = microscopeimagequality.evaluation.get_certainty(numpy.array([1.0, 0.0]))
        self.assertEquals(1.0, certainty)

    def testGetRgbImageRuns(self):
        num_rows = 4
        patch_width = 28
        num_patches = num_rows ** 2
        num_classes = 2
        image_width = patch_width * num_rows
        image = numpy.ones((image_width, image_width, 1))
        patches = numpy.ones((num_patches, patch_width, patch_width, 1))
        probabilities = numpy.ones(
            (num_patches, num_classes), dtype=numpy.float32) / num_classes
        labels = numpy.ones(num_patches, dtype=numpy.int32)

        rgb_image = microscopeimagequality.evaluation.get_rgb_image(
            numpy.max(image), patches, probabilities, labels,
            (image_width, image_width))
        self.assertEquals(rgb_image.shape, (image_width, image_width, 3))

        # Check function runs with all probabilities set to 0, without
        # divide-by-zero issues.

        probabilities = numpy.zeros(
            (num_patches, num_classes), dtype=numpy.float32) / num_classes

        rgb_image = microscopeimagequality.evaluation.get_rgb_image(
            numpy.max(image), patches, probabilities, labels,
            (image_width, image_width))
        self.assertEquals(rgb_image.shape, (image_width, image_width, 3))

    def testAggregatePredictionFromProbabilities(self):
        probabilities = numpy.array([[1.0 / 3, 1.0 / 3, 1.0 / 3], [0.0, 0.2, 0.8]])

        (predicted_class, certainties, probabilities_averaged
         ) = microscopeimagequality.evaluation.aggregate_prediction_from_probabilities(probabilities)
        self.assertEquals(2, predicted_class)
        numpy.testing.assert_allclose(
            numpy.array([0.0, 0.2, 0.8]), probabilities_averaged, atol=1e-3)
        expected_certainties = {
            'mean': numpy.float64(0.272),
            'max': numpy.float64(0.545),
            'aggregate': numpy.float64(0.545),
            'weighted': numpy.float64(0.545)
        }
        self.assertDictEqual(expected_certainties, certainties)

    def testAggregatePredictionFromProbabilitiesLeastCertain(self):
        probabilities = numpy.ones((2, 3)) / 3.0
        (predicted_class, certainties, probabilities_averaged
         ) = microscopeimagequality.evaluation.aggregate_prediction_from_probabilities(probabilities)
        self.assertEquals(0, predicted_class)
        numpy.testing.assert_allclose(
            numpy.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
            probabilities_averaged,
            atol=1e-3)
        expected_certainties = {
            'mean': numpy.float64(0.0),
            'max': numpy.float64(0.0),
            'aggregate': numpy.float64(0.0),
            'weighted': numpy.float64(0.0)
        }
        self.assertDictEqual(expected_certainties, certainties)

    def testAggregatePredictionFromProbabilitiesMostCertain(self):
        probabilities = numpy.array([[1.0 / 3, 1.0 / 3, 1.0 / 3], [0.0, 0.0, 1.0]])
        (predicted_class, certainties, probabilities_averaged
         ) = microscopeimagequality.evaluation.aggregate_prediction_from_probabilities(probabilities)
        self.assertEquals(2, predicted_class)
        numpy.testing.assert_allclose(
            numpy.array([0, 0, 1.0]), probabilities_averaged, atol=1e-3)
        expected_certainties = {
            'mean': numpy.float64(0.5),
            'max': numpy.float64(1.0),
            'aggregate': numpy.float64(1.0),
            'weighted': numpy.float64(1.0)
        }
        self.assertDictEqual(expected_certainties, certainties)

    def testAggregatePredictionFromProbabilitiesWithProduct(self):
        probabilities = numpy.array([[0.25, 0.25, 0.5], [0.1, 0.2, 0.7]])
        probabilities_aggregated = microscopeimagequality.evaluation.aggregate_prediction_from_probabilities(
            probabilities, microscopeimagequality.evaluation.METHOD_PRODUCT).probabilities
        expected = probabilities[0, :] * probabilities[1, :]
        expected /= expected.sum()
        numpy.testing.assert_allclose(
            numpy.array(expected), probabilities_aggregated, atol=1e-3)

    def testAggregatePredictionFromProbabilitiesWithProduct2(self):
        probabilities = numpy.array(
            [[0.25, 0.25, 0.5], [0.1, 0.2, 0.7], [0.4, 0.3, 0.3]])
        probabilities_aggregated = microscopeimagequality.evaluation.aggregate_prediction_from_probabilities(
            probabilities, microscopeimagequality.evaluation.METHOD_PRODUCT).probabilities
        numpy.testing.assert_allclose(
            numpy.array([0.077, 0.115, 0.807]), probabilities_aggregated, atol=1e-3)

    def testAddRgbAnnotation(self):
        image = numpy.zeros((20, 20, 3))
        predicted_rgb = (1, 0, 0)
        actual_rgb = (0, 1, 0)
        max_value = 1
        image_rgb = microscopeimagequality.evaluation._add_rgb_annotation(image, predicted_rgb, actual_rgb,
                                                           max_value)
        image_expected = numpy.zeros((20, 20, 3))
        image_expected[0:microscopeimagequality.evaluation.BORDER_SIZE, :, 1] = 1
        image_expected[-1 * microscopeimagequality.evaluation.BORDER_SIZE:, :, 0] = 1
        numpy.testing.assert_array_equal(image_rgb, image_expected)

    def testPatchesToImageNonSquare(self):
        num_rows = 2
        num_cols = 3
        num_patches = num_rows * num_cols
        patch_width = 28
        image_shape = patch_width * num_rows, patch_width * num_cols
        patches = numpy.ones((num_patches, patch_width, patch_width, 1))
        image = microscopeimagequality.evaluation._patches_to_image(patches, image_shape)
        self.assertEquals(image.shape, (image_shape[0], image_shape[1], 1))

        with self.assertRaises(ValueError):
            image_shape_invalid = (20, 20)
            microscopeimagequality.evaluation._patches_to_image(patches, image_shape_invalid)

    def testSetBorderPixels(self):
        image = numpy.zeros((5, 5, 1))
        image_expected = numpy.ones((5, 5, 1))
        image_expected[2, 2, :] = 0
        image_with_border = microscopeimagequality.evaluation._set_border_pixels(image, value=1)
        numpy.testing.assert_array_equal(image_with_border, image_expected)

    def testApplyImageGamma(self):
        image = numpy.array([1.0, 2.0])
        image_original = numpy.copy(image)
        image_with_gamma = microscopeimagequality.evaluation.apply_image_gamma(image, gamma=0.5)

        # Check original image is unmodified.
        numpy.testing.assert_array_equal(image, image_original)

        # Check gamma has been applied
        image_expected = numpy.array([0.5, 2.0])
        numpy.testing.assert_array_equal(image_with_gamma, image_expected)

    def testGetModelAndMetricsWithoutTrueLabels(self):
        g = tensorflow.Graph()
        with g.as_default():
            images = tensorflow.zeros(
                [self.batch_size, self.patch_width, self.patch_width, 1])
            num_classes = 11
            one_hot_labels = tensorflow.zeros([self.batch_size, num_classes])
            labels = microscopeimagequality.evaluation.get_model_and_metrics(images, num_classes,
                                                              one_hot_labels, False).labels

            sv = tensorflow.train.Supervisor()
            with sv.managed_session() as sess:
                [labels_np] = sess.run([labels])

            self.assertTrue(numpy.all(-1 == labels_np))

    def testGetModelAndMetricsWithTrueLabels(self):
        g = tensorflow.Graph()
        with g.as_default():
            batch_size = 2
            images = tensorflow.zeros([batch_size, self.patch_width, self.patch_width, 1])
            one_hot_labels = tensorflow.constant([0, 1, 0, 1], dtype=tensorflow.float32, shape=(2, 2))

            num_classes = 11
            labels = microscopeimagequality.evaluation.get_model_and_metrics(images, num_classes,
                                                              one_hot_labels, False).labels

            sv = tensorflow.train.Supervisor()
            with sv.managed_session() as sess:
                [labels_np] = sess.run([labels])

            self.assertTrue(numpy.all(1 == labels_np))

    def testSaveInferenceResultsRuns(self):
        num_classes = 3
        aggregate_probabilities = numpy.ones((self.batch_size, num_classes))
        aggregate_labels = range(self.batch_size)
        certainties = {
            'mean': [0.5] * self.batch_size,
            'max': [0.8] * self.batch_size,
            'aggregate': [0.9] * self.batch_size,
            'weighted': [1.0] * self.batch_size
        }
        orig_names = ['orig_name'] * self.batch_size
        aggregate_predictions = range(self.batch_size)
        output_path = os.path.join(self.test_dir, 'results.csv')
        microscopeimagequality.evaluation.save_inference_results(aggregate_probabilities, aggregate_labels,
                                                  certainties, orig_names,
                                                  aggregate_predictions, output_path)

    def testSaveAndLoadResults(self):
        num_classes = 3
        aggregate_probabilities = numpy.ones((self.batch_size, num_classes))
        aggregate_probabilities[0, 2] = 3
        aggregate_labels = list(range(self.batch_size))
        certainties = {
            'mean': [numpy.float64(1.0 / 3)] * self.batch_size,
            'max': [0.0] * self.batch_size,
            'aggregate': [0.9] * self.batch_size,
            'weighted': [1.0] * self.batch_size
        }
        orig_names = ['orig_name'] * self.batch_size
        aggregate_predictions = list(range(self.batch_size))
        test_directory = os.path.join(self.test_dir, 'save_load_test')
        os.makedirs(test_directory)
        output_path = os.path.join(test_directory, 'results.csv')
        microscopeimagequality.evaluation.save_inference_results(aggregate_probabilities, aggregate_labels,
                                                  certainties, orig_names,
                                                  aggregate_predictions, output_path)

        (aggregate_probabilities_2, aggregate_labels_2, certainties_2, orig_names_2,
         aggregate_predictions_2) = microscopeimagequality.evaluation.load_inference_results(test_directory)
        numpy.testing.assert_array_equal(aggregate_probabilities,
                                         aggregate_probabilities_2)
        self.assertEquals(aggregate_labels, aggregate_labels_2)
        self.assertEquals(certainties['mean'], certainties_2['mean'])
        self.assertEquals(certainties['max'], certainties_2['max'])
        self.assertEquals(certainties['aggregate'], certainties_2['aggregate'])
        self.assertEquals(certainties['weighted'], certainties_2['weighted'])
        self.assertEquals(orig_names, orig_names_2)
        self.assertEquals(aggregate_predictions, aggregate_predictions_2)

    def testSaveResultPlotsRuns(self):
        num_classes = 4
        aggregate_probabilities = numpy.ones((self.batch_size, num_classes))
        aggregate_labels = range(self.batch_size)
        microscopeimagequality.evaluation.save_result_plots(
            aggregate_probabilities,
            aggregate_labels,
            save_confusion=True,
            output_directory=self.test_dir)
        microscopeimagequality.evaluation.save_result_plots(
            aggregate_probabilities,
            aggregate_labels,
            save_confusion=False,
            output_directory=self.test_dir)

    def testSavePredictionHistogramRuns(self):
        probabilities = numpy.array(((0.0, 1.0), (0.5, 0.5), (0.2, 0.8)))
        predictions = numpy.array([0, 1, 0])
        assert probabilities.shape[0] == len(predictions)
        microscopeimagequality.evaluation.save_prediction_histogram(
            predictions,
            os.path.join(self.test_dir, 'histogram.png'),
            probabilities.shape[1])
