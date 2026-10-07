package edu.iitm.cs5013.panoramic.gui;

import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import javafx.application.Platform;
import javafx.stage.Stage;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class GuiWorkflowTest {

    @BeforeAll
    static void initJavaFX() {
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException ignored) {
            // Already started
        }
    }

    @Test
    void testGuiControllerInitializationAndImageLoad() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<MainViewController> controllerRef = new AtomicReference<>();

        Platform.runLater(() -> {
            try {
                MainViewController controller = new MainViewController(new Stage());
                controllerRef.set(controller);

                Path testImg = Path.of("src/test/resources/test_panorama.png");
                controller.loadImage(testImg);

                assertNotNull(controller.getRoot());
                assertFalse(controller.getToolRegistry().getAllTools().isEmpty());
            } finally {
                latch.countDown();
            }
        });

        assertTrue(latch.await(5, TimeUnit.SECONDS), "JavaFX UI initialization timed out");
        assertNotNull(controllerRef.get());
    }
}
