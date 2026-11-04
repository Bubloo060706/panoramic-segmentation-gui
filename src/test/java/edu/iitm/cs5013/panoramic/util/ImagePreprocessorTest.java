package edu.iitm.cs5013.panoramic.util;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ImagePreprocessorTest {

    @Test
    void testProcessStandardImage() {
        Path input = Path.of("src/test/resources/test_panorama.png");
        Path output = Path.of("target/test_cache/clean_standard.png");

        ImagePreprocessor.PreprocessResult res = ImagePreprocessor.process(input, output);
        assertTrue(res.success());
        assertEquals(800, res.width());
        assertEquals(400, res.height());
        assertNotNull(res.cleanPath());
        assertTrue(Files.exists(Path.of(res.cleanPath())));
    }

    @Test
    void testGetDimensionsStandardImage() {
        Path input = Path.of("src/test/resources/test_panorama.png");
        int[] dims = ImagePreprocessor.getDimensions(input);
        assertNotNull(dims);
        assertEquals(800, dims[0]);
        assertEquals(400, dims[1]);
    }

    @Test
    void testProcessNonExistentFile() {
        ImagePreprocessor.PreprocessResult res = ImagePreprocessor.process(Path.of("invalid_file.xyz"), null);
        assertFalse(res.success());
        assertNotNull(res.error());
        assertTrue(res.error().contains("File not found"));
    }

    @Test
    void testGetDimensionsHighResCameraJpeg() {
        Path hPath = Path.of("H.jpg");
        if (Files.exists(hPath)) {
            int[] dims = ImagePreprocessor.getDimensions(hPath);
            assertNotNull(dims);
            assertEquals(7808, dims[0]);
            assertEquals(3744, dims[1]);
        }
    }

    @Test
    void testSanitizeImageCaching() {
        Path input = Path.of("src/test/resources/test_panorama.png");
        Path cleanPath = ImagePreprocessor.sanitizeImage(input, null);
        assertNotNull(cleanPath);
        assertTrue(Files.exists(cleanPath));

        // Calling again should return cached path immediately
        Path cached = ImagePreprocessor.sanitizeImage(input, null);
        assertEquals(cleanPath, cached);
    }
}
