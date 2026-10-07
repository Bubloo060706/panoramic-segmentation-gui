package edu.iitm.cs5013.panoramic.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SegmentationModelTest {

    @Test
    void testMaskDataDimensionsAndCoordinates() {
        int w = 3;
        int h = 2;
        int[] labels = {
                1, 2, 1,
                3, 0, 2
        };
        MaskData mask = new MaskData(w, h, labels);

        assertEquals(3, mask.getWidth());
        assertEquals(2, mask.getHeight());
        assertEquals(1, mask.getLabel(0, 0));
        assertEquals(2, mask.getLabel(1, 0));
        assertEquals(1, mask.getLabel(2, 0));
        assertEquals(3, mask.getLabel(0, 1));
        assertEquals(0, mask.getLabel(1, 1));
        assertEquals(2, mask.getLabel(2, 1));

        assertThrows(IndexOutOfBoundsException.class, () -> mask.getLabel(3, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> mask.getLabel(0, 2));

        Map<Integer, Long> counts = mask.computeLabelCounts();
        assertEquals(2L, counts.get(1));
        assertEquals(2L, counts.get(2));
        assertEquals(1L, counts.get(3));
        assertEquals(1L, counts.get(0));
    }

    @Test
    void testMaskDataInvalidLengthThrows() {
        assertThrows(IllegalArgumentException.class, () -> new MaskData(2, 2, new int[]{1, 2, 3}));
        assertThrows(IllegalArgumentException.class, () -> new MaskData(-1, 2, new int[]{1, 2}));
    }

    @Test
    void testSegmentationResultPreservesUnsupportedAttributes() {
        SemanticClass sc = new SemanticClass(1, "wall", 255, 0, 0, 100, 0.95);
        MaskData mask = new MaskData(2, 1, new int[]{1, 0});

        SegmentationResult result = new SegmentationResult.Builder()
                .toolId("hoho-net")
                .toolName("HoHoNet Panoramic Segmenter")
                .dimensions(2, 1)
                .maskData(mask)
                .addClass(sc)
                .addMetadata("inferenceTimeSec", 0.42)
                .addUnsupportedAttribute("depthMapProjection", "equirectangular_cube_6")
                .addUnsupportedAttribute("pointCloudPCD", "points.pcd")
                .addWarning("Depth map was produced but is not mapped into 2D semantic classes")
                .build();

        assertTrue(result.isSuccess());
        assertEquals("hoho-net", result.getToolId());
        assertEquals(1, result.getDetectedClasses().size());
        assertEquals("wall", result.getDetectedClasses().get(0).getName());

        // Verify unmapped attributes are explicitly preserved rather than dropped
        assertEquals(2, result.getUnsupportedAttributes().size());
        assertTrue(result.getUnsupportedAttributes().containsKey("depthMapProjection"));
        assertTrue(result.getUnsupportedAttributes().containsKey("pointCloudPCD"));
        assertEquals(1, result.getWarnings().size());
    }

    @Test
    void testSegmentationRequestValidation(@TempDir Path tempDir) throws IOException {
        Path validImg = tempDir.resolve("pano.png");
        Files.write(validImg, new byte[]{1, 2, 3});

        SegmentationRequest validReq = new SegmentationRequest(validImg.toString(), "tool-1");
        assertDoesNotThrow(validReq::validate);

        Path invalidExt = tempDir.resolve("pano.txt");
        Files.write(invalidExt, new byte[]{1, 2, 3});
        SegmentationRequest invalidExtReq = new SegmentationRequest(invalidExt.toString(), "tool-1");
        assertThrows(IllegalArgumentException.class, invalidExtReq::validate);

        SegmentationRequest nonExistentReq = new SegmentationRequest(tempDir.resolve("missing.png").toString(), "tool-1");
        assertThrows(IllegalArgumentException.class, nonExistentReq::validate);
    }
}
