package edu.iitm.cs5013.panoramic.visualization;

import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;

import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

/**
 * Panorama-aware image and mask overlay renderer.
 * Recognizes the equirectangular 360° geometry and supports horizontal wrap-around rotation
 * so that objects spanning the 0°/360° boundary can be inspected without discontinuity.
 */
public class PanoramaImageRenderer {

    /**
     * Creates a colored mask image from a canonical SegmentationResult.
     */
    public WritableImage renderMaskImage(SegmentationResult result) {
        if (result == null || result.getMaskData() == null) {
            return null;
        }

        MaskData mask = result.getMaskData();
        int width = mask.getWidth();
        int height = mask.getHeight();

        Map<Integer, Color> palette = buildColorPalette(result);
        WritableImage output = new WritableImage(width, height);
        PixelWriter writer = output.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = mask.getLabel(x, y);
                Color color = palette.getOrDefault(label, Color.rgb(128, 128, 128));
                writer.setColor(x, y, color);
            }
        }

        return output;
    }

    /**
     * Creates an overlay blending the base panoramic image and the segmentation mask.
     *
     * @param baseImage The original equirectangular panoramic image.
     * @param result The segmentation result.
     * @param alpha The mask opacity (0.0 to 1.0).
     * @return Blended WritableImage.
     */
    public WritableImage renderOverlay(Image baseImage, SegmentationResult result, double alpha) {
        if (baseImage == null) {
            return renderMaskImage(result);
        }
        if (result == null || result.getMaskData() == null) {
            return copyImage(baseImage);
        }

        int width = (int) baseImage.getWidth();
        int height = (int) baseImage.getHeight();
        MaskData mask = result.getMaskData();

        double clampedAlpha = Math.clamp(alpha, 0.0, 1.0);
        double baseWeight = 1.0 - clampedAlpha;

        Map<Integer, Color> palette = buildColorPalette(result);
        WritableImage output = new WritableImage(width, height);
        PixelWriter writer = output.getPixelWriter();
        PixelReader baseReader = baseImage.getPixelReader();

        // Scale factors if mask dimensions differ from base image
        double scaleX = (double) mask.getWidth() / width;
        double scaleY = (double) mask.getHeight() / height;

        for (int y = 0; y < height; y++) {
            int maskY = Math.min((int) (y * scaleY), mask.getHeight() - 1);
            for (int x = 0; x < width; x++) {
                int maskX = Math.min((int) (x * scaleX), mask.getWidth() - 1);
                int label = mask.getLabel(maskX, maskY);
                Color maskColor = palette.getOrDefault(label, Color.rgb(128, 128, 128));
                Color baseColor = baseReader.getColor(x, y);

                double r = (baseColor.getRed() * baseWeight) + (maskColor.getRed() * clampedAlpha);
                double g = (baseColor.getGreen() * baseWeight) + (maskColor.getGreen() * clampedAlpha);
                double b = (baseColor.getBlue() * baseWeight) + (maskColor.getBlue() * clampedAlpha);

                writer.setColor(x, y, new Color(r, g, b, 1.0));
            }
        }

        return output;
    }

    /**
     * Performs horizontal wrap-around rotation on an equirectangular image by a specified yaw angle in degrees.
     * In a 360° equirectangular projection, horizontal translation wraps cyclically from the 360° right edge
     * to the 0° left edge, allowing continuous inspection of wrap-around entities.
     *
     * @param source The equirectangular image to rotate.
     * @param yawDegrees The horizontal shift angle in degrees [0, 360).
     * @return A shifted, cyclically wrapped WritableImage.
     */
    public WritableImage renderWithHorizontalOffset(Image source, double yawDegrees) {
        if (source == null) return null;

        int width = (int) source.getWidth();
        int height = (int) source.getHeight();

        // Normalize yaw to [0, 360)
        double normalizedYaw = ((yawDegrees % 360.0) + 360.0) % 360.0;
        int shiftPixels = (int) Math.round((normalizedYaw / 360.0) * width);

        if (shiftPixels == 0) {
            return copyImage(source);
        }

        WritableImage wrapped = new WritableImage(width, height);
        PixelWriter writer = wrapped.getPixelWriter();
        PixelReader reader = source.getPixelReader();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Cylindrical wrap around modulo arithmetic
                int srcX = (x + shiftPixels) % width;
                Color c = reader.getColor(srcX, y);
                writer.setColor(x, y, c);
            }
        }

        return wrapped;
    }

    private Map<Integer, Color> buildColorPalette(SegmentationResult result) {
        Map<Integer, Color> palette = new HashMap<>();
        if (result.getDetectedClasses() != null) {
            for (SemanticClass sc : result.getDetectedClasses()) {
                palette.put(sc.getId(), Color.rgb(sc.getRed(), sc.getGreen(), sc.getBlue()));
            }
        }
        return palette;
    }

    private WritableImage copyImage(Image src) {
        int w = (int) src.getWidth();
        int h = (int) src.getHeight();
        WritableImage copy = new WritableImage(w, h);
        PixelWriter writer = copy.getPixelWriter();
        PixelReader reader = src.getPixelReader();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                writer.setColor(x, y, reader.getColor(x, y));
            }
        }
        return copy;
    }
}
