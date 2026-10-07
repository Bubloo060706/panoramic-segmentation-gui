package edu.iitm.cs5013.panoramic.adapter;

import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.config.ToolConfigurationLoader;
import edu.iitm.cs5013.panoramic.model.*;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AdapterTest {

    private final Path testImgPath = Path.of("src/test/resources/test_panorama.png");

    @Test
    void testDeterministicTestAdapterExecution() throws SegmentationException {
        DeterministicTestAdapter adapter = new DeterministicTestAdapter();
        SegmentationRequest request = new SegmentationRequest(testImgPath.toString(), "deterministic-mock");

        SegmentationResult result = adapter.segment(request);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals("deterministic-mock", result.getToolId());
        assertEquals(800, result.getWidth());
        assertEquals(400, result.getHeight());

        // Verify detected classes include ceiling, wall, floor, window, and wrap-around object
        assertFalse(result.getDetectedClasses().isEmpty());
        assertTrue(result.findClassById(4).isPresent(), "WrapAroundObject should be detected");

        // Verify wrap-around object continuity at left and right boundaries
        MaskData mask = result.getMaskData();
        int midY = 400 / 2;
        // x=0 and x=799 must both be Class 4 (WrapAroundObject)
        assertEquals(4, mask.getLabel(0, midY), "Left boundary pixel should belong to wrap-around object");
        assertEquals(4, mask.getLabel(799, midY), "Right boundary pixel should belong to wrap-around object");

        // Check unsupported mock attribute was preserved
        assertTrue(result.getUnsupportedAttributes().containsKey("mockSpecialFeature"));
    }

    @Test
    void testDeterministicTestAdapterFailureHandling() {
        DeterministicTestAdapter failAdapter = new DeterministicTestAdapter("fail-mock", "Failing Mock", true);
        SegmentationRequest request = new SegmentationRequest(testImgPath.toString(), "fail-mock");

        SegmentationException ex = assertThrows(SegmentationException.class, () -> failAdapter.segment(request));
        assertTrue(ex.getMessage().contains("Simulated"));
    }

    @Test
    void testRealProcessModelAdapterExecution() throws Exception {
        ToolConfigurationLoader loader = new ToolConfigurationLoader();
        List<ToolConfig> configs = loader.loadFromFile(Path.of("config/tools.json"));
        ToolConfig pythonToolConfig = configs.stream()
                .filter(c -> c.getId().equals("pano-seg-py"))
                .findFirst()
                .orElseThrow();

        ProcessModelAdapter adapter = new ProcessModelAdapter(pythonToolConfig);
        assertTrue(adapter.isAvailable(), "Real Python script runner should be available");

        SegmentationRequest request = new SegmentationRequest(testImgPath.toString(), "pano-seg-py");
        SegmentationResult result = adapter.segment(request);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals("pano-seg-py", result.getToolId());
        assertEquals(800, result.getWidth());
        assertEquals(400, result.getHeight());

        // Verify detected classes from real Python segmentation
        List<SemanticClass> classes = result.getDetectedClasses();
        assertFalse(classes.isEmpty());
        assertTrue(classes.stream().anyMatch(c -> c.getName().equals("Ceiling")));
        assertTrue(classes.stream().anyMatch(c -> c.getName().equals("Wall")));
        assertTrue(classes.stream().anyMatch(c -> c.getName().equals("Floor")));

        // Verify unmapped metadata from Python output was preserved
        assertTrue(result.getUnsupportedAttributes().containsKey("model") ||
                result.getUnsupportedAttributes().containsKey("projection"));
    }
}
