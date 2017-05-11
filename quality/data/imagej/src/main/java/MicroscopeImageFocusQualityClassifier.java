import io.scif.img.ImgOpener;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

/**
 * Command to apply the Microscopy image focus quality classifier model on an input (16-bit,
 * greyscale image).
 *
 * <p>This command will show both the input image and an annotated image marking regions of the
 * image with their focus quality.
 *
 * <p>This is a first draft, some TODOs:
 *
 * <ul>
 *   <li>Generate the annotated image from the model's output quality for each tensor (and then set
 *       {@code annotatedImage} to the annotated image). For now, the patch qualities are just
 *       dumped into the console log.
 *   <li>Avoid loading the model from disk on every invocation of the command, as that slows down
 *       the classifer (loading ~100MB of data into memory on every invocation)
 *   <li>Perhaps package the classification model with the plugin instead of asking the user to
 *       locate the model on their local disk.
 * </ul>
 */
@Plugin(type = Command.class, menuPath = "Microscopy>Focus Quality")
public class MicroscopeImageFocusQualityClassifier implements Command {

  @Parameter private LogService logService;

  @Parameter(label = "Microscope Image")
  private File imageFile;

  @Parameter(label = "Focus Quality Model", style = "directory")
  private File modelDir;

  @Parameter(type = ItemIO.OUTPUT)
  private Img<UnsignedShortType> originalImage;

  @Parameter(type = ItemIO.OUTPUT)
  private Dataset annotatedImage;

  // Same as the tag used in export_saved_model in the Python code.
  private static final String MODEL_TAG = "inference";
  // Same as tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  // in Python. Perhaps this should be an exported constant in TensorFlow's Java API.
  private static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default";

  /*
   * The run() method is where we do the actual 'work' of the command.
   *
   * TODO(ashankar): The current implementation is extremely sub-optimal as the model
   * is being loaded on every call to run(). The model is pretty big (~100MB) and the
   * cost of loaded should be ammortized. Perhaps the model should be loaded once statically,
   * or implemented as a service plugin?
   */
  @Override
  public void run() {
    final long loadModelStart = System.nanoTime();
    try (SavedModelBundle model = SavedModelBundle.load(modelDir.getAbsolutePath(), MODEL_TAG)) {
      final long loadModelEnd = System.nanoTime();
      logService.info(
          String.format(
              "Loaded microscope focus image quality model in %dms",
              (loadModelEnd - loadModelStart) / 1000000));

      // Extract names from the model signature.
      // The strings "input", "probabilities" and "patches" are meant to be in sync with
      // the model exporter (export_saved_model()) in Python.
      final SignatureDef sig =
          MetaGraphDef.parseFrom(model.metaGraphDef())
              .getSignatureDefOrThrow(DEFAULT_SERVING_SIGNATURE_DEF_KEY);
      originalImage =
          new ImgOpener()
              .openImg(
                  imageFile.getAbsolutePath(),
                  new ArrayImgFactory<UnsignedShortType>(),
                  new UnsignedShortType());
      validateFormat(originalImage);
      try (Tensor inputTensor = inputImageTensor(originalImage)) {
        final long runModelStart = System.nanoTime();
        final List<Tensor> fetches =
            model
                .session()
                .runner()
                .feed(opName(sig.getInputsOrThrow("input")), inputTensor)
                .fetch(opName(sig.getOutputsOrThrow("probabilities")))
                .fetch(opName(sig.getOutputsOrThrow("patches")))
                .run();
        final long runModelEnd = System.nanoTime();
        try (Tensor probabilities = fetches.get(0);
            Tensor patches = fetches.get(1)) {
          logService.info(
              String.format(
                  "Ran image through model in %dms", (runModelEnd - runModelStart) / 1000000));
          logService.info("Probabilities shape: " + Arrays.toString(probabilities.shape()));
          logService.info("Patches shape: " + Arrays.toString(patches.shape()));

          float[][] floatProbs =
              new float[(int) probabilities.shape()[0]][(int) probabilities.shape()[1]];
          probabilities.copyTo(floatProbs);
          for (int i = 0; i < probabilities.shape()[0]; ++i) {
            logService.info(
                String.format("Patch %02d probabilities: %s", i, Arrays.toString(floatProbs[i])));
          }

          final int npatches = (int) patches.shape()[0];
          final int patchSide = (int) patches.shape()[1];
          assert patchSide == (int) patches.shape()[2]; // Square patches
          assert patches.shape()[3] == 1;

          // Log an error to force the console log to display
          // (otherwise the user will have to know to display the console window).
          // Of course, this will go away once the annotate image is generated.
          logService.error(
              "TODO: Display annotated image. Till then, see the beautiful log messages above");
        }
      }

    } catch (final Exception exc) {
      // Use the LogService to report the error.
      logService.error(exc);
    }
  }

  private void validateFormat(Img<UnsignedShortType> image) throws IOException {
    int ndims = image.numDimensions();
    if (ndims != 2) {
      long[] dims = new long[ndims];
      image.dimensions(dims);
      throw new IOException(
          "Can only process greyscale images, not an image with "
              + ndims
              + " dimensions ("
              + Arrays.toString(dims)
              + ")");
    }
  }

  /**
   * Convert an Img object into a Tensor suitable for input to the focus quality classification
   * model.
   */
  private Tensor inputImageTensor(Img<UnsignedShortType> image) throws IOException {
    final int width = (int) image.dimension(0);
    final int height = (int) image.dimension(1);
    logService.info("Width = " + width + ", height = " + height);

    final RandomAccess<UnsignedShortType> r = image.randomAccess();
    float[][] pixels = new float[height][width];
    final int pos[] = new int[2];
    for (int x = 0; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        pos[0] = x;
        pos[1] = y;
        r.setPosition(pos);
        pixels[y][x] = (float) r.get().get() / 65535;
      }
    }
    // An opportunity for optimization here: Instead of filling in a 2D pixels array,
    // create a flattened array and use:
    // Tensor.create(new long[]{height, width}, FloatBuffer.wrap(pixels));
    // That will save some reflection cost if the Tensor.create() call here is too expensive.
    final long start = System.nanoTime();
    Tensor t = Tensor.create(pixels);
    final long end = System.nanoTime();
    logService.info(
        String.format("Created Tensor from %dx%d image in %dns", height, width, (end - start)));
    return t;
  }

  // The SignatureDef inputs and outputs contain names of the form <operation_name>:<output_index>,
  // where for this model, <output_index> is always 0. This function trims the ":0" suffix to
  // get the operation name.
  private static String opName(TensorInfo t) {
    final String n = t.getName();
    if (n.endsWith(":0")) {
      return n.substring(0, n.lastIndexOf(":0"));
    }
    return n;
  }

  public static void main(String[] args) {
    final ImageJ ij = new ImageJ();
    ij.launch(args);
    ij.command().run(MicroscopeImageFocusQualityClassifier.class, true);
  }
}
