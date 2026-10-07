package edu.iitm.cs5013.panoramic.adapter;

import edu.iitm.cs5013.panoramic.config.ExecutionType;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.model.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;

/**
 * Deterministic Test/Mock Adapter for testing the Java framework independently
 * of external model runtimes and hardware dependencies.
 *
 * Distinct from real model adapters, this adapter generates mathematically predictable
 * panoramic semantic masks including boundary wrap-around segments.
 */
public class DeterministicTestAdapter implements SegmentationAdapter<MaskData> {

    private final ToolMetadata metadata;
    private final ToolConfig config;
    private final boolean simulateFailure;
    private int cachedWidth = 800;
    private int cachedHeight = 400;

    public DeterministicTestAdapter(String toolId, String toolName, boolean simulateFailure) {
        this.metadata = new ToolMetadata(
                toolId,
                toolName,
                "Deterministic mock adapter for testing and regression",
                "1.0.0-TEST",
                Set.of(SegmentationType.SEMANTIC),
                Set.of("jpg", "jpeg", "png")
        );
        this.config = new ToolConfig(
                toolId,
                toolName,
                "Mock adapter",
                ExecutionType.MOCK,
                List.of(),
                "",
                "",
                10L,
                true,
                Map.of("mock", true),
                List.of(
                        new SemanticClass(0, "Ceiling", 120, 180, 240),
                        new SemanticClass(1, "Wall", 220, 190, 140),
                        new SemanticClass(2, "Floor", 100, 100, 100),
                        new SemanticClass(3, "Window", 60, 220, 180),
                        new SemanticClass(4, "WrapAroundObject", 240, 50, 50)
                )
        );
        this.simulateFailure = simulateFailure;
    }

    public DeterministicTestAdapter() {
        this("deterministic-mock", "Deterministic Mock Segmenter", false);
    }

    @Override
    public ToolMetadata getMetadata() {
        return metadata;
    }

    @Override
    public ToolConfig getConfiguration() {
        return config;
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public void preprocess(SegmentationRequest request) throws SegmentationException {
        request.validate();
        if (simulateFailure) {
            throw new SegmentationException(metadata.getId(), "Simulated preprocessing failure for testing");
        }
        try {
            BufferedImage img = ImageIO.read(request.getImageNioPath().toFile());
            if (img != null) {
                this.cachedWidth = img.getWidth();
                this.cachedHeight = img.getHeight();
            }
        } catch (IOException e) {
            throw new SegmentationException(metadata.getId(), "Failed to read image dimensions", e);
        }
    }

    @Override
    public MaskData executeRaw(SegmentationRequest request) throws SegmentationException {
        if (simulateFailure) {
            throw new SegmentationException(metadata.getId(), "Simulated model execution failure for testing");
        }

        int w = cachedWidth;
        int h = cachedHeight;
        int[] labels = new int[w * h];

        // 1. Equirectangular geometric layers:
        // Top 25%: Ceiling (Class 0)
        // Middle 50%: Wall (Class 1)
        // Bottom 25%: Floor (Class 2)
        int ceilingLimit = h / 4;
        int floorLimit = (3 * h) / 4;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                if (y < ceilingLimit) {
                    labels[idx] = 0; // Ceiling
                } else if (y >= floorLimit) {
                    labels[idx] = 2; // Floor
                } else {
                    labels[idx] = 1; // Wall
                }
            }
        }

        // 2. Interior window in the center
        int winXStart = (3 * w) / 8;
        int winXEnd = (5 * w) / 8;
        int winYStart = (3 * h) / 8;
        int winYEnd = (5 * h) / 8;
        for (int y = winYStart; y < winYEnd; y++) {
            for (int x = winXStart; x < winXEnd; x++) {
                labels[y * w + x] = 3; // Window
            }
        }

        // 3. Wrap-around object crossing the 0° / 360° boundary (x = 0 and x = w-1)
        // Object spans from [w - w/16, w - 1] to [0, w/16] in the middle wall band
        int wrapSpan = Math.max(2, w / 16);
        int objYStart = h / 3;
        int objYEnd = (2 * h) / 3;
        for (int y = objYStart; y < objYEnd; y++) {
            // Left boundary portion
            for (int x = 0; x < wrapSpan; x++) {
                labels[y * w + x] = 4; // WrapAroundObject
            }
            // Right boundary portion
            for (int x = w - wrapSpan; x < w; x++) {
                labels[y * w + x] = 4; // WrapAroundObject
            }
        }

        return new MaskData(w, h, labels);
    }

    @Override
    public SegmentationResult postprocess(MaskData rawOutput, SegmentationRequest request) {
        List<SemanticClass> classes = config.getSupportedClasses();
        Map<Integer, Long> counts = rawOutput.computeLabelCounts();

        List<SemanticClass> updatedClasses = new ArrayList<>();
        for (SemanticClass sc : classes) {
            long count = counts.getOrDefault(sc.getId(), 0L);
            updatedClasses.add(sc.withPixelCount(count));
        }

        return new SegmentationResult.Builder()
                .toolId(metadata.getId())
                .toolName(metadata.getName())
                .status(ExecutionStatus.SUCCESS)
                .dimensions(rawOutput.getWidth(), rawOutput.getHeight())
                .maskData(rawOutput)
                .detectedClasses(updatedClasses)
                .addMetadata("isMock", true)
                .addMetadata("hasBoundaryWrapAround", true)
                .addUnsupportedAttribute("mockSpecialFeature", "ignored_mock_attribute_for_testing")
                .executionDurationMs(25)
                .build();
    }

    @Override
    public SegmentationException translateError(Throwable t, SegmentationRequest request) {
        if (t instanceof SegmentationException se) {
            return se;
        }
        return new SegmentationException(metadata.getId(), "DeterministicTestAdapter error: " + t.getMessage(), t);
    }
}
