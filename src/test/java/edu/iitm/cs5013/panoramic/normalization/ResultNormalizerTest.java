package edu.iitm.cs5013.panoramic.normalization;

import edu.iitm.cs5013.panoramic.config.ExecutionType;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ResultNormalizerTest {

    private ResultNormalizer normalizer;
    private ToolConfig sampleConfig;

    @BeforeEach
    void setUp() {
        normalizer = new ResultNormalizer();
        sampleConfig = new ToolConfig(
                "test-tool", "Test Tool", "Desc", ExecutionType.PROCESS,
                List.of("python"), ".", "", 30L, true, null, null
        );
    }

    @Test
    void testNormalizeValidJson() throws NormalizationException {
        String json = """
                {
                  "width": 2,
                  "height": 2,
                  "segmentationType": "SEMANTIC",
                  "labels": [0, 1, 1, 0],
                  "classes": [
                    { "id": 0, "name": "Sky", "red": 50, "green": 100, "blue": 200 },
                    { "id": 1, "name": "Ground", "red": 100, "green": 150, "blue": 50 }
                  ],
                  "customCameraRig": "Insta360-Pro2",
                  "equirectangularHorizonPitch": 0.05
                }
                """;

        SegmentationResult result = normalizer.normalizeFromJson(json, sampleConfig, 2, 2);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals(2, result.getWidth());
        assertEquals(2, result.getHeight());
        assertEquals(2, result.getDetectedClasses().size());

        // Verify custom attributes were preserved in unsupportedAttributes
        assertEquals(2, result.getUnsupportedAttributes().size());
        assertTrue(result.getUnsupportedAttributes().containsKey("customCameraRig"));
        assertTrue(result.getUnsupportedAttributes().containsKey("equirectangularHorizonPitch"));
    }

    @Test
    void testMismatchedDimensionsGeneratesWarning() throws NormalizationException {
        String json = """
                {
                  "width": 4,
                  "height": 2,
                  "labels": [0, 1, 0, 1, 1, 0, 1, 0]
                }
                """;

        // Expected dimensions were 8x4, output is 4x2 (e.g. downsampled inference)
        SegmentationResult result = normalizer.normalizeFromJson(json, sampleConfig, 8, 4);
        assertNotNull(result);
        assertFalse(result.getWarnings().isEmpty());
        assertTrue(result.getWarnings().get(0).contains("differ from input panorama"));
    }

    @Test
    void testMalformedLabelsLengthThrows() {
        String json = """
                {
                  "width": 2,
                  "height": 2,
                  "labels": [0, 1, 0]
                }
                """;
        // 2x2 = 4, but labels has 3
        assertThrows(NormalizationException.class, () ->
                normalizer.normalizeFromJson(json, sampleConfig, 2, 2));
    }
}
