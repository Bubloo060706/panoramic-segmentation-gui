package edu.iitm.cs5013.panoramic.util;

import javafx.scene.image.Image;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;

/**
 * Universal image loading and dimension query helper.
 * Transparently bridges Java/JavaFX with standard OpenCV/PIL preprocessing to ensure
 * any panoramic image of any resolution (from small test crops to 8K+ panoramas)
 * and any JPEG encoding (e.g. progressive, SOF type 0xce, custom camera EXIF markers)
 * is accepted and displayed without restrictions.
 */
public class ImageLoaderHelper {

    /**
     * Determines dimensions [width, height] of any image file.
     * Uses fast header-only inspection first, falls back to OpenCV preprocessor
     * if the image contains non-standard camera markers (e.g. JPEG SOF 0xce).
     */
    public static int[] getImageDimensions(Path path) {
        if (path == null || !Files.exists(path)) {
            return null;
        }

        // 1. Try fast header inspection via ImageReader (no full bitmap raster decode)
        try (ImageInputStream iis = ImageIO.createImageInputStream(path.toFile())) {
            if (iis != null) {
                Iterator<ImageReader> readers = ImageIO.getImageReaders(iis);
                if (readers.hasNext()) {
                    ImageReader reader = readers.next();
                    try {
                        reader.setInput(iis);
                        int w = reader.getWidth(0);
                        int h = reader.getHeight(0);
                        if (w > 0 && h > 0) {
                            return new int[]{w, h};
                        }
                    } finally {
                        reader.dispose();
                    }
                }
            }
        } catch (Exception ignored) {
            // Header reader failed on proprietary camera markers (e.g. SOF 0xce)
        }

        // 2. Fallback to Python preprocessor (OpenCV / PIL)
        return queryDimensionsViaPython(path);
    }

    /**
     * Loads an Image into JavaFX cleanly.
     * If the native JavaFX loader trips on camera markers (e.g. marker 0xa0),
     * automatically sanitizes the image via Python/OpenCV into target/cache/clean_<filename>
     * and loads the clean image at full original resolution.
     */
    public static Image loadJavaFxImage(Path path) {
        if (path == null || !Files.exists(path)) {
            return null;
        }

        // 1. Try direct loading
        try {
            Image candidate = new Image(new FileInputStream(path.toFile()));
            if (!candidate.isError() && candidate.getWidth() > 0 && candidate.getHeight() > 0) {
                return candidate;
            }
        } catch (Exception ignored) {
        }

        // 2. Sanitize and load
        Path cleanPath = sanitizeImage(path);
        if (cleanPath != null && Files.exists(cleanPath)) {
            try {
                Image cleanImage = new Image(new FileInputStream(cleanPath.toFile()));
                if (!cleanImage.isError() && cleanImage.getWidth() > 0) {
                    return cleanImage;
                }
            } catch (Exception ignored) {
            }
        }

        return null;
    }

    /**
     * Sanitizes non-standard JPEGs into clean baseline JPEGs in target/cache.
     */
    public static Path sanitizeImage(Path rawPath) {
        try {
            Path cacheDir = Path.of("target", "cache");
            Files.createDirectories(cacheDir);
            String cleanFileName = "clean_" + rawPath.getFileName().toString();
            Path cleanPath = cacheDir.resolve(cleanFileName);

            // If already cached and valid, reuse
            if (Files.exists(cleanPath) && Files.size(cleanPath) > 0) {
                return cleanPath;
            }

            ProcessBuilder pb = new ProcessBuilder(
                    "python", "scripts/image_preprocessor.py",
                    "--input", rawPath.toAbsolutePath().toString(),
                    "--output", cleanPath.toAbsolutePath().toString()
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();
            boolean finished = process.waitFor(20, TimeUnit.SECONDS);
            if (finished && Files.exists(cleanPath) && Files.size(cleanPath) > 0) {
                return cleanPath;
            }
        } catch (Exception ignored) {
        }
        return rawPath;
    }

    public static int[] queryDimensionsViaPython(Path path) {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "python", "scripts/image_preprocessor.py",
                    "--input", path.toAbsolutePath().toString()
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("\"width\"") && line.contains("\"height\"")) {
                        int wIdx = line.indexOf("\"width\":");
                        int hIdx = line.indexOf("\"height\":");
                        if (wIdx != -1 && hIdx != -1) {
                            String wSub = line.substring(wIdx + 8).replaceAll("[^0-9].*", "").trim();
                            String hSub = line.substring(hIdx + 9).replaceAll("[^0-9].*", "").trim();
                            int w = Integer.parseInt(wSub);
                            int h = Integer.parseInt(hSub);
                            process.waitFor();
                            return new int[]{w, h};
                        }
                    }
                }
            }
            process.waitFor();
        } catch (Exception ignored) {
        }
        return null;
    }
}
