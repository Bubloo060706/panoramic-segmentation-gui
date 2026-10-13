package edu.iitm.cs5013.panoramic.util;

import javafx.scene.image.Image;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ImageLoaderHelperTest {

    @Test
    void testGetDimensionsStandardImage() {
        Path path = Path.of("src/test/resources/test_panorama.png");
        int[] dims = ImageLoaderHelper.getImageDimensions(path);
        assertNotNull(dims);
        assertEquals(800, dims[0]);
        assertEquals(400, dims[1]);
    }

    @Test
    void testGetDimensionsNonExistent() {
        assertNull(ImageLoaderHelper.getImageDimensions(null));
        assertNull(ImageLoaderHelper.getImageDimensions(Path.of("non_existent_file.jpg")));
    }

    @Test
    void testGetDimensionsNonStandardHighResJpeg() {
        Path hPath = Path.of("H.jpg");
        if (Files.exists(hPath)) {
            int[] dims = ImageLoaderHelper.getImageDimensions(hPath);
            assertNotNull(dims);
            assertEquals(7808, dims[0]);
            assertEquals(3744, dims[1]);
        }
    }
}
