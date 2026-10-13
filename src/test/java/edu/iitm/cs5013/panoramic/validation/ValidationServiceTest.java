package edu.iitm.cs5013.panoramic.validation;

import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ValidationServiceTest {

    private ValidationService service;

    @BeforeEach
    void setUp() {
        service = new ValidationService();
    }

    @Test
    void testValidatePanoramicImageMissingPath() {
        ValidationResult res = service.validatePanoramicImage(null);
        assertFalse(res.isValid());
        assertTrue(res.getFormattedErrors().contains("null"));

        res = service.validatePanoramicImage(Path.of("non_existent_panorama_file.jpg"));
        assertFalse(res.isValid());
        assertTrue(res.getFormattedErrors().contains("does not exist"));
    }

    @Test
    void testValidatePanoramicImageInvalidExtension(@TempDir Path tempDir) throws IOException {
        Path txt = tempDir.resolve("pano.txt");
        Files.writeString(txt, "test content");

        ValidationResult res = service.validatePanoramicImage(txt);
        assertFalse(res.isValid());
        assertTrue(res.getFormattedErrors().contains("Unsupported image format"));
    }

    @Test
    void testValidatePanoramicImageEmptyFile(@TempDir Path tempDir) throws IOException {
        Path empty = tempDir.resolve("empty.jpg");
        Files.createFile(empty);

        ValidationResult res = service.validatePanoramicImage(empty);
        assertFalse(res.isValid());
        assertTrue(res.getFormattedErrors().contains("empty"));
    }

    @Test
    void testValidatePanoramicImageValidFile() {
        Path valid = Path.of("src/test/resources/test_panorama.png");
        ValidationResult res = service.validatePanoramicImage(valid);
        assertTrue(res.isValid());
    }

    @Test
    void testValidatePanoramicImageNonStandardJpeg() {
        Path hPath = Path.of("H.jpg");
        if (Files.exists(hPath)) {
            ValidationResult res = service.validatePanoramicImage(hPath);
            assertTrue(res.isValid(), "H.jpg should be valid despite SOF 0xce / non-standard camera markers: " + res.getFormattedErrors());
        }
    }

    @Test
    void testValidateToolsSelectionEmpty() {
        ValidationResult res = service.validateToolsSelection(List.of());
        assertFalse(res.isValid());
        assertTrue(res.getFormattedErrors().contains("At least one"));
    }

    @Test
    void testValidateEvaluationMismatchedDimensions() {
        SegmentationResult r1 = new SegmentationResult.Builder()
                .dimensions(100, 50)
                .maskData(new MaskData(100, 50, new int[5000]))
                .build();
        SegmentationResult r2 = new SegmentationResult.Builder()
                .dimensions(200, 100)
                .maskData(new MaskData(200, 100, new int[20000]))
                .build();

        ValidationResult res = service.validateResultForEvaluation(r1, r2);
        assertFalse(res.isValid());
        assertTrue(res.getFormattedErrors().contains("mismatch"));
    }
}
