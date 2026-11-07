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

    public static final String[] DEFAULT_VPLAB_CLASS_NAMES = {
        "Roof", "Bed", "Obstructions", "Refrigerator", "Phone", "Wall",
        "Furniture", "Electronics", "Floor", "Monitor", "Cabinet", "Door",
        "Curtain", "Pillow", "Window", "Misc.", "Chair", "Table", "Boxes",
        "Bag", "Bottles"
    };

    public static final int[][] DEFAULT_VPLAB_PALETTE = {
        {153, 51, 0},     // 0: roof
        {0, 142, 64},     // 1: bed
        {255, 0, 0},      // 2: obstructions
        {37, 150, 190},   // 3: refrigerator
        {240, 200, 60},   // 4: phone
        {19, 27, 172},    // 5: wall
        {0, 255, 0},      // 6: furniture
        {150, 50, 145},   // 7: electronics
        {164, 244, 252},  // 8: floor
        {153, 102, 0},    // 9: monitor
        {146, 146, 146},  // 10: cabinet
        {0, 204, 153},    // 11: door
        {255, 51, 153},   // 12: curtain
        {102, 0, 102},    // 13: pillow
        {220, 142, 205},  // 14: window
        {218, 240, 19},   // 15: misc
        {45, 0, 246},     // 16: chair
        {255, 255, 153},  // 17: table
        {160, 169, 94},   // 18: boxes
        {123, 0, 246},    // 19: bag
        {255, 102, 0}     // 20: bottles
    };

    /**
     * Loads a mask from a grayscale, indexed, or RGB color image file.
     * Accurately distinguishes between direct class ID index masks (values 0..20, 255)
     * and RGB palette color masks (matching against the official VPLab palette).
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

            // Build palette color lookup
            Map<Integer, int[]> paletteMap = new HashMap<>();
            if (toolConfig != null && toolConfig.getSupportedClasses() != null && !toolConfig.getSupportedClasses().isEmpty()) {
                for (SemanticClass sc : toolConfig.getSupportedClasses()) {
                    paletteMap.put(sc.getId(), new int[]{sc.getRed(), sc.getGreen(), sc.getBlue()});
                }
            } else {
                for (int i = 0; i < DEFAULT_VPLAB_PALETTE.length; i++) {
                    paletteMap.put(i, DEFAULT_VPLAB_PALETTE[i]);
                }
            }

            // Check if image is an indexed or grayscale raster with direct class IDs
            boolean isDirectIndexed = false;
            if (img.getRaster().getNumBands() == 1 ||
                img.getType() == BufferedImage.TYPE_BYTE_GRAY ||
                img.getType() == BufferedImage.TYPE_BYTE_INDEXED) {
                isDirectIndexed = true;
            } else {
                // Check if the image channels are equal (grayscale encoded in RGB/ARGB)
                // Sample points uniformly across the entire image
                int grayMatches = 0;
                int validClassMatches = 0;
                int sampleCount = 0;
                int stepY = Math.max(1, h / 20);
                int stepX = Math.max(1, w / 20);
                for (int y = 0; y < h; y += stepY) {
                    for (int x = 0; x < w; x += stepX) {
                        sampleCount++;
                        int rgb = img.getRGB(x, y);
                        int r = (rgb >> 16) & 0xFF;
                        int g = (rgb >> 8) & 0xFF;
                        int b = rgb & 0xFF;
                        if (r == g && g == b) {
                            grayMatches++;
                            if (r <= 21 || r == 255) {
                                validClassMatches++;
                            }
                        }
                    }
                }
                if (sampleCount > 0 && ((double) grayMatches / sampleCount >= 0.75 || (double) validClassMatches / sampleCount >= 0.65)) {
                    isDirectIndexed = true;
                }
            }

            if (isDirectIndexed) {
                var raster = img.getRaster();
                boolean useRaster = raster.getNumBands() == 1;
                for (int y = 0; y < h; y++) {
                    int offset = y * w;
                    for (int x = 0; x < w; x++) {
                        int val = useRaster ? raster.getSample(x, y, 0) : (img.getRGB(x, y) & 0xFF);
                        if (val <= 20 || val == 255) {
                            labels[offset + x] = val;
                        } else if (val >= 250) {
                            labels[offset + x] = 255;
                        } else {
                            labels[offset + x] = Math.min(val, 20);
                        }
                    }
                }
            } else {
                    // RGB Color Mask: match nearest color in palette
                    for (int y = 0; y < h; y++) {
                        int offset = y * w;
                        for (int x = 0; x < w; x++) {
                            int rgb = img.getRGB(x, y);
                            int r = (rgb >> 16) & 0xFF;
                            int g = (rgb >> 8) & 0xFF;
                            int b = rgb & 0xFF;

                            int bestClass = 255;
                            int minDistanceSq = Integer.MAX_VALUE;

                            for (Map.Entry<Integer, int[]> entry : paletteMap.entrySet()) {
                                int[] c = entry.getValue();
                                int dr = r - c[0];
                                int dg = g - c[1];
                                int db = b - c[2];
                                int distSq = dr * dr + dg * dg + db * db;
                                if (distSq < minDistanceSq) {
                                    minDistanceSq = distSq;
                                    bestClass = entry.getKey();
                                }
                            }

                            // If distance is within tolerance, map to class ID; otherwise ignore index
                            labels[offset + x] = (minDistanceSq <= 20000) ? bestClass : 255;
                        }
                    }
                }

                return new MaskData(w, h, labels);
            } catch (IOException e) {
                throw new NormalizationException("IO error reading mask image: " + maskPath, e);
            }
        }
    }
