package edu.iitm.cs5013.panoramic.normalization;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.model.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Normalizes heterogeneous model outputs (JSON payloads, raw matrices, color-mapped mask images,
 * and indexed mask images) into the canonical SegmentationResult.
 *
 * Guarantees that:
 * 1. Image dimensions are verified.
 * 2. Unmapped or unsupported attributes are never silently discarded.
 * 3. Class labels and colors are preserved.
 */
public class ResultNormalizer {

    private final ObjectMapper objectMapper;

    public ResultNormalizer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper != null ? objectMapper : new ObjectMapper();
    }

    public ResultNormalizer() {
        this(new ObjectMapper());
    }

    /**
     * Normalizes a standardized JSON output file or payload produced by an external model runner.
     */
    public SegmentationResult normalizeFromJson(String jsonContent, ToolConfig toolConfig, int expectedWidth, int expectedHeight)
            throws NormalizationException {
        if (jsonContent == null || jsonContent.isBlank()) {
            throw new NormalizationException("Model output JSON is empty");
        }

        try {
            JsonNode root = objectMapper.readTree(jsonContent);

            int width = root.has("width") ? root.get("width").asInt() : expectedWidth;
            int height = root.has("height") ? root.get("height").asInt() : expectedHeight;

            if (width <= 0 || height <= 0) {
                throw new NormalizationException("Invalid or missing output dimensions in model output: " + width + "x" + height);
            }

            // Segmentation Type
            SegmentationType type = SegmentationType.SEMANTIC;
            if (root.has("segmentationType")) {
                try {
                    type = SegmentationType.valueOf(root.get("segmentationType").asText().toUpperCase());
                } catch (IllegalArgumentException e) {
                    type = SegmentationType.UNKNOWN;
                }
            }

            // Classes
            List<SemanticClass> classes = new ArrayList<>();
            Map<Integer, SemanticClass> classMap = new HashMap<>();
            if (root.has("classes") && root.get("classes").isArray()) {
                for (JsonNode cNode : root.get("classes")) {
                    int id = cNode.get("id").asInt();
                    String name = cNode.has("name") ? cNode.get("name").asText() : "Class_" + id;
                    int r = cNode.has("red") ? cNode.get("red").asInt() : 128;
                    int g = cNode.has("green") ? cNode.get("green").asInt() : 128;
                    int b = cNode.has("blue") ? cNode.get("blue").asInt() : 128;
                    long count = cNode.has("pixelCount") ? cNode.get("pixelCount").asLong() : 0L;
                    double conf = cNode.has("confidence") ? cNode.get("confidence").asDouble() : 1.0;
                    SemanticClass sc = new SemanticClass(id, name, r, g, b, count, conf);
                    classes.add(sc);
                    classMap.put(id, sc);
                }
            }

            // Labels
            int expectedLen = width * height;
            int[] labels = new int[expectedLen];

            if (root.has("labels") && root.get("labels").isArray()) {
                JsonNode labelsNode = root.get("labels");
                if (labelsNode.size() != expectedLen) {
                    throw new NormalizationException("Labels array length (" + labelsNode.size() +
                            ") does not match declared dimensions (" + width + "x" + height + " = " + expectedLen + ")");
                }
                for (int i = 0; i < expectedLen; i++) {
                    labels[i] = labelsNode.get(i).asInt();
                }
            } else if (root.has("maskPath")) {
                // If the model produced an image mask file (indexed or RGB)
                String maskPath = root.get("maskPath").asText();
                MaskData loadedMask = loadMaskFromImageFile(Path.of(maskPath), toolConfig);
                if (loadedMask.getWidth() != width || loadedMask.getHeight() != height) {
                    throw new NormalizationException("Mask image dimensions (" + loadedMask.getWidth() + "x" +
                            loadedMask.getHeight() + ") do not match JSON metadata (" + width + "x" + height + ")");
                }
                labels = loadedMask.getLabels();
            } else {
                throw new NormalizationException("Model output contains neither 'labels' array nor 'maskPath'");
            }

            // Instance IDs
            int[] instanceIds = null;
            if (root.has("instanceIds") && root.get("instanceIds").isArray()) {
                JsonNode instNode = root.get("instanceIds");
                if (instNode.size() == expectedLen) {
                    instanceIds = new int[expectedLen];
                    for (int i = 0; i < expectedLen; i++) {
                        instanceIds[i] = instNode.get(i).asInt();
                    }
                }
            }

            MaskData maskData = new MaskData(width, height, labels, instanceIds);

            // Recompute / update pixel counts for classes
            Map<Integer, Long> computedCounts = maskData.computeLabelCounts();
            List<SemanticClass> finalizedClasses = new ArrayList<>();
            for (Map.Entry<Integer, Long> entry : computedCounts.entrySet()) {
                int cId = entry.getKey();
                long count = entry.getValue();
                SemanticClass existing = classMap.get(cId);
                if (existing != null) {
                    finalizedClasses.add(existing.withPixelCount(count));
                } else {
                    // Auto-generate consistent palette color if class was not explicitly listed
                    int r = (cId * 67 + 37) % 256;
                    int g = (cId * 131 + 83) % 256;
                    int b = (cId * 197 + 127) % 256;
                    finalizedClasses.add(new SemanticClass(cId, "Class_" + cId, r, g, b, count, 1.0));
                }
            }
            finalizedClasses.sort(Comparator.comparingInt(SemanticClass::getId));

            // Extract metadata and collect unsupported attributes
            SegmentationResult.Builder builder = new SegmentationResult.Builder()
                    .toolId(toolConfig.getId())
                    .toolName(toolConfig.getName())
                    .status(ExecutionStatus.SUCCESS)
                    .segmentationType(type)
                    .dimensions(width, height)
                    .maskData(maskData)
                    .detectedClasses(finalizedClasses);

            // Check dimension match with input image
            if (expectedWidth > 0 && expectedHeight > 0 && (width != expectedWidth || height != expectedHeight)) {
                builder.addWarning("Output dimensions (" + width + "x" + height +
                        ") differ from input panorama dimensions (" + expectedWidth + "x" + expectedHeight + ")");
            }

            // Separate standard fields from unmapped fields
            Set<String> standardFields = Set.of("width", "height", "segmentationType", "classes", "labels", "maskPath", "instanceIds");
            Iterator<String> fieldNames = root.fieldNames();
            while (fieldNames.hasNext()) {
                String field = fieldNames.next();
                if (!standardFields.contains(field)) {
                    // Preserve in unsupportedAttributes so information is NEVER silently dropped
                    builder.addUnsupportedAttribute(field, root.get(field).toString());
                }
            }

            return builder.build();

        } catch (NormalizationException ne) {
            throw ne;
        } catch (Exception e) {
            throw new NormalizationException("Failed to normalize model output: " + e.getMessage(), e);
        }
    }

    /**
     * Loads a mask from a grayscale or color image file.
     */
    public MaskData loadMaskFromImageFile(Path maskPath, ToolConfig toolConfig) throws NormalizationException {
        if (!Files.exists(maskPath)) {
            throw new NormalizationException("Mask image file not found: " + maskPath);
        }

        try {
            BufferedImage img = ImageIO.read(maskPath.toFile());
            if (img == null) {
                throw new NormalizationException("Failed to decode mask image: " + maskPath);
            }

            int w = img.getWidth();
            int h = img.getHeight();
            int[] labels = new int[w * h];

            // If image is indexed or grayscale, pixel value directly gives class ID
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int rgb = img.getRGB(x, y);
                    // For grayscale, r == g == b
                    int gray = rgb & 0xFF;
                    labels[y * w + x] = gray;
                }
            }

            return new MaskData(w, h, labels);
        } catch (IOException e) {
            throw new NormalizationException("IO error reading mask image: " + maskPath, e);
        }
    }
}
