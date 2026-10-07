package edu.iitm.cs5013.panoramic.visualization;

import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;
import javafx.application.Platform;
import javafx.scene.image.PixelReader;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PanoramaVisualizationTest {

    private final PanoramaImageRenderer renderer = new PanoramaImageRenderer();

    @BeforeAll
    static void initJavaFX() {
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException ignored) {
            // Toolkit already initialized
        }
    }

    @Test
    void testHorizontalWrapAroundRotation() {
        // Create 4x1 image with distinct colors [Red, Green, Blue, Yellow]
        int width = 4;
        int height = 1;
        WritableImage testImage = new WritableImage(width, height);
        PixelWriter writer = testImage.getPixelWriter();
        writer.setColor(0, 0, Color.RED);
        writer.setColor(1, 0, Color.GREEN);
        writer.setColor(2, 0, Color.BLUE);
        writer.setColor(3, 0, Color.YELLOW);

        // Shift 90 degrees = 1/4 of 360 = 1 pixel shift
        WritableImage shifted90 = renderer.renderWithHorizontalOffset(testImage, 90.0);
        PixelReader reader90 = shifted90.getPixelReader();
        // Index 0 in shifted should be Green (index 1 of source)
        assertEquals(Color.GREEN, reader90.getColor(0, 0));
        // Index 1 should be Blue
        assertEquals(Color.BLUE, reader90.getColor(1, 0));
        // Index 2 should be Yellow
        assertEquals(Color.YELLOW, reader90.getColor(2, 0));
        // Index 3 should be Red (wrapped around from index 0)
        assertEquals(Color.RED, reader90.getColor(3, 0));

        // Shift 360 degrees = full rotation, should equal original
        WritableImage shifted360 = renderer.renderWithHorizontalOffset(testImage, 360.0);
        PixelReader reader360 = shifted360.getPixelReader();
        assertEquals(Color.RED, reader360.getColor(0, 0));
        assertEquals(Color.GREEN, reader360.getColor(1, 0));
        assertEquals(Color.BLUE, reader360.getColor(2, 0));
        assertEquals(Color.YELLOW, reader360.getColor(3, 0));
    }

    @Test
    void testMaskRenderAndOverlay() {
        MaskData mask = new MaskData(2, 2, new int[]{0, 1, 1, 0});
        SegmentationResult res = new SegmentationResult.Builder()
                .dimensions(2, 2)
                .maskData(mask)
                .addClass(new SemanticClass(0, "Sky", 0, 0, 255))
                .addClass(new SemanticClass(1, "Ground", 0, 255, 0))
                .build();

        WritableImage maskImage = renderer.renderMaskImage(res);
        assertNotNull(maskImage);
        assertEquals(2, (int) maskImage.getWidth());
        assertEquals(2, (int) maskImage.getHeight());

        PixelReader reader = maskImage.getPixelReader();
        // (0,0) is Sky (Blue)
        assertEquals(Color.BLUE, reader.getColor(0, 0));
        // (1,0) is Ground (Green)
        assertEquals(Color.rgb(0, 255, 0), reader.getColor(1, 0));

        // Blend with base image
        WritableImage base = new WritableImage(2, 2);
        WritableImage overlay = renderer.renderOverlay(base, res, 0.5);
        assertNotNull(overlay);
        assertEquals(2, (int) overlay.getWidth());
        assertEquals(2, (int) overlay.getHeight());
    }
}
