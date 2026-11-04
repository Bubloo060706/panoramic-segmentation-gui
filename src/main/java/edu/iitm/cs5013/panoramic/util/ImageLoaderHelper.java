package edu.iitm.cs5013.panoramic.util;

import javafx.scene.image.Image;

import java.nio.file.Path;

/**
 * Universal image loading and dimension query helper.
 * Uses high-performance native Java {@link ImagePreprocessor} to ensure
 * any panoramic image of any resolution (from small test crops to 8K+ panoramas)
 * and any JPEG encoding (e.g. progressive, SOF type 0xce, custom camera EXIF markers)
 * is accepted, decoded, sanitized, and displayed without external Python dependencies.
 */
public class ImageLoaderHelper {

    /**
     * Determines dimensions [width, height] of any image file.
     * Uses fast binary marker inspection, ImageReader headers, and Prism decoding fallback.
     */
    public static int[] getImageDimensions(Path path) {
        return ImagePreprocessor.getDimensions(path);
    }

    /**
     * Loads an Image into JavaFX cleanly.
     * If the native JavaFX loader trips on proprietary camera markers,
     * automatically sanitizes the image via ImagePreprocessor into target/cache/clean_<filename>
     * and loads the clean baseline image at full original resolution.
     */
    public static Image loadJavaFxImage(Path path) {
        return ImagePreprocessor.loadJavaFxImage(path);
    }

    /**
     * Sanitizes non-standard JPEGs into clean baseline JPEGs in target/cache.
     */
    public static Path sanitizeImage(Path rawPath) {
        return ImagePreprocessor.sanitizeImage(rawPath, null);
    }

    /**
     * Queries dimensions via ImagePreprocessor engine.
     */
    public static int[] queryDimensionsViaPreprocessor(Path path) {
        return ImagePreprocessor.getDimensions(path);
    }

    /**
     * Compatibility alias for existing callers.
     */
    public static int[] queryDimensionsViaPython(Path path) {
        return ImagePreprocessor.getDimensions(path);
    }
}
