package edu.iitm.cs5013.panoramic.validation;

import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Validates inputs, files, configurations, and result compatibility across the application.
 */
public class ValidationService {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of("jpg", "jpeg", "png");

    public ValidationResult validatePanoramicImage(Path path) {
        if (path == null) {
            return ValidationResult.failure("Panoramic image path cannot be null");
        }
        if (!Files.exists(path)) {
            return ValidationResult.failure("Image file does not exist: " + path);
        }
        if (!Files.isRegularFile(path)) {
            return ValidationResult.failure("Path is not a regular file: " + path);
        }
        if (!Files.isReadable(path)) {
            return ValidationResult.failure("Image file is not readable: " + path);
        }

        String fileName = path.getFileName().toString().toLowerCase();
        int dotIdx = fileName.lastIndexOf('.');
        if (dotIdx == -1) {
            return ValidationResult.failure("Image file has no extension: " + fileName);
        }
        String ext = fileName.substring(dotIdx + 1);
        if (!SUPPORTED_EXTENSIONS.contains(ext)) {
            return ValidationResult.failure("Unsupported image format '." + ext + "'. Supported: " + SUPPORTED_EXTENSIONS);
        }

        try {
            long sizeBytes = Files.size(path);
            if (sizeBytes == 0) {
                return ValidationResult.failure("Image file is empty (0 bytes)");
            }

            BufferedImage img = ImageIO.read(path.toFile());
            if (img == null) {
                return ValidationResult.failure("Could not decode image (file may be corrupted): " + fileName);
            }

            List<String> warnings = new ArrayList<>();
            int w = img.getWidth();
            int h = img.getHeight();

            // Typical 360 equirectangular panoramas have a 2:1 aspect ratio
            double ratio = (double) w / h;
            if (Math.abs(ratio - 2.0) > 0.05) {
                warnings.add(String.format(
                        "Image aspect ratio is %.2f:1 (%dx%d). Standard 360° equirectangular panoramas typically have a 2:1 ratio.",
                        ratio, w, h));
            }

            return ValidationResult.successWithWarnings(warnings);

        } catch (IOException e) {
            return ValidationResult.failure("Failed to read image attributes: " + e.getMessage());
        }
    }

    public ValidationResult validateToolsSelection(List<SegmentationTool> tools) {
        if (tools == null || tools.isEmpty()) {
            return ValidationResult.failure("At least one segmentation tool must be selected");
        }
        List<String> warnings = new ArrayList<>();
        for (SegmentationTool t : tools) {
            if (!t.isAvailable()) {
                warnings.add("Tool '" + t.getMetadata().getName() + "' currently reports unavailable status");
            }
        }
        return ValidationResult.successWithWarnings(warnings);
    }

    public ValidationResult validateResultForEvaluation(SegmentationResult prediction, SegmentationResult groundTruth) {
        if (prediction == null) {
            return ValidationResult.failure("Prediction result is null");
        }
        if (groundTruth == null) {
            return ValidationResult.failure("Ground truth result is null");
        }
        if (!prediction.isSuccess()) {
            return ValidationResult.failure("Prediction tool execution failed: " + prediction.getErrorMessage());
        }
        if (!groundTruth.isSuccess()) {
            return ValidationResult.failure("Ground truth result is invalid: " + groundTruth.getErrorMessage());
        }
        if (prediction.getMaskData() == null) {
            return ValidationResult.failure("Prediction contains no mask data");
        }
        if (groundTruth.getMaskData() == null) {
            return ValidationResult.failure("Ground truth contains no mask data");
        }

        if (prediction.getWidth() != groundTruth.getWidth() || prediction.getHeight() != groundTruth.getHeight()) {
            return ValidationResult.failure(String.format(
                    "Dimensions mismatch: prediction is %dx%d, ground truth is %dx%d",
                    prediction.getWidth(), prediction.getHeight(), groundTruth.getWidth(), groundTruth.getHeight()
            ));
        }

        return ValidationResult.success();
    }
}
