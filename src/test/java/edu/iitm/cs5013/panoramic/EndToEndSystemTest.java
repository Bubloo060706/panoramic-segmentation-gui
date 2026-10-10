package edu.iitm.cs5013.panoramic;

import edu.iitm.cs5013.panoramic.adapter.DeterministicTestAdapter;
import edu.iitm.cs5013.panoramic.adapter.ProcessModelAdapter;
import edu.iitm.cs5013.panoramic.comparison.EvaluationResult;
import edu.iitm.cs5013.panoramic.comparison.MetricsCalculator;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.config.ToolConfigurationLoader;
import edu.iitm.cs5013.panoramic.config.ToolRegistry;
import edu.iitm.cs5013.panoramic.controller.AsyncSegmentationCoordinator;
import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;
import edu.iitm.cs5013.panoramic.validation.ValidationResult;
import edu.iitm.cs5013.panoramic.validation.ValidationService;
import edu.iitm.cs5013.panoramic.visualization.PanoramaImageRenderer;
import javafx.application.Platform;
import javafx.scene.image.WritableImage;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Complete End-to-End integration test exercising the full project workflow:
 * 1. Validate panoramic image input.
 * 2. Discover and register heterogeneous segmentation tools.
 * 3. Asynchronously execute multiple tools (including real Python external model).
 * 4. Verify canonical result normalization and unmapped attribute preservation.
 * 5. Verify panorama horizontal wrap-around rendering.
 * 6. Compute quantitative comparison metrics against ground truth reference.
 */
class EndToEndSystemTest {

    private static AsyncSegmentationCoordinator coordinator;

    @BeforeAll
    static void setUp() {
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException ignored) {}
        coordinator = new AsyncSegmentationCoordinator(2);
    }

    @AfterAll
    static void tearDown() {
        if (coordinator != null) {
            coordinator.shutdown();
        }
    }

    @Test
    void testCompleteMultiToolWorkflow() throws Exception {
        Path inputPanorama = Path.of("src/test/resources/test_panorama.png");

        // 1. Validation
        ValidationService validator = new ValidationService();
        ValidationResult valRes = validator.validatePanoramicImage(inputPanorama);
        assertTrue(valRes.isValid(), "Panorama validation should succeed");

        // 2. Discover and configure tools
        ToolConfigurationLoader loader = new ToolConfigurationLoader();
        List<ToolConfig> configs = loader.loadFromFile(Path.of("config/tools.json"));
        ToolConfig pythonCfg = configs.stream().filter(c -> c.getId().equals("panofusion")).findFirst().orElseThrow();

        SegmentationTool realPythonTool = new ProcessModelAdapter(pythonCfg);
        SegmentationTool mockTool = new DeterministicTestAdapter("gt-tool", "Ground Truth Generator", false);

        ToolRegistry registry = new ToolRegistry();
        registry.registerTool(realPythonTool);
        registry.registerTool(mockTool);

        assertEquals(2, registry.size());

        // 3. Multi-Tool Asynchronous Execution
        CompletableFuture<List<SegmentationResult>> execFuture = coordinator.executeAllAsync(
                inputPanorama,
                List.of(realPythonTool, mockTool),
                null
        );

        List<SegmentationResult> results = execFuture.get(15, TimeUnit.SECONDS);
        assertEquals(2, results.size());

        SegmentationResult pythonResult = results.stream().filter(r -> r.getToolId().equals("panofusion")).findFirst().orElseThrow();
        SegmentationResult mockResult = results.stream().filter(r -> r.getToolId().equals("gt-tool")).findFirst().orElseThrow();

        assertTrue(pythonResult.isSuccess(), "Python tool execution should succeed");
        assertTrue(mockResult.isSuccess(), "Mock tool execution should succeed");

        // 4. Verify canonical representation and boundary preservation
        assertEquals(800, pythonResult.getWidth());
        assertEquals(400, pythonResult.getHeight());
        assertEquals(800, mockResult.getWidth());
        assertEquals(400, mockResult.getHeight());

        // Python tool produced equirectangular metadata that must be preserved
        assertFalse(pythonResult.getUnsupportedAttributes().isEmpty(), "Unmapped attributes should be preserved");

        // 5. Panorama wrap-around visualization verification
        PanoramaImageRenderer renderer = new PanoramaImageRenderer();
        WritableImage mockMask = renderer.renderMaskImage(mockResult);
        assertNotNull(mockMask);

        // 360-degree rotation must produce matching dimensions
        WritableImage rotated = renderer.renderWithHorizontalOffset(mockMask, 180.0);
        assertEquals(800, (int) rotated.getWidth());
        assertEquals(400, (int) rotated.getHeight());

        // 6. Quantitative comparison metrics
        MetricsCalculator metricsCalc = new MetricsCalculator();
        EvaluationResult evalResult = metricsCalc.evaluate(pythonResult, mockResult);

        assertTrue(evalResult.isQuantitativeValid(), "Quantitative evaluation should be valid for matching dimensions");
        assertTrue(evalResult.getPixelAccuracy() > 0.0, "Pixel accuracy should be greater than 0");
        assertTrue(evalResult.getMeanIoU() > 0.0, "mIoU should be greater than 0");
        assertTrue(evalResult.getMeanDice() > 0.0, "Mean Dice should be greater than 0");

        System.out.println("End-to-End Evaluation Output:\n" + evalResult.getFormattedSummary());
    }
}
