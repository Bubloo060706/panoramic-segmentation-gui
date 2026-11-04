package edu.iitm.cs5013.panoramic.util;

import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;

/**
 * Universal Image Preprocessor & Normalizer for Panoramic Framework.
 * Replaces scripts/image_preprocessor.py with a high-performance, native Java solution.
 * Handles standard images as well as non-standard camera JPEGs (e.g. SOF 0xce, SOS markers,
 * proprietary camera EXIF markers like 0xa0).
 *
 * Can be used:
 * 1. As an in-process Java API: ImagePreprocessor.process(inputPath, outputPath), getDimensions(path), etc.
 * 2. As a standalone CLI: java edu.iitm.cs5013.panoramic.util.ImagePreprocessor --input <file> [--output <file>]
 */
public class ImagePreprocessor {

    /**
     * DTO representing the result of image preprocessing, matching the JSON schema
     * of the original Python image_preprocessor.py.
     */
    public record PreprocessResult(boolean success, int width, int height, String cleanPath, String error) {
        public String toJson() {
            if (success) {
                return String.format("{\"success\":true,\"width\":%d,\"height\":%d,\"cleanPath\":\"%s\"}",
                        width, height, cleanPath != null ? cleanPath.replace("\\", "/") : "");
            } else {
                String safeErr = error != null ? error.replace("\"", "\\\"").replace("\n", " ") : "Unknown error";
                return String.format("{\"success\":false,\"error\":\"%s\"}", safeErr);
            }
        }
    }

    /**
     * Fast header-only dimensions query using direct JPEG/PNG binary marker parsing,
     * ImageIO ImageReader, and JavaFX decoding fallback.
     */
    public static int[] getDimensions(Path path) {
        if (path == null || !Files.exists(path)) {
            return null;
        }

        // 1. Direct binary header inspection (fastest, 0ms, no memory allocation)
        int[] fastDims = readBinaryHeaderDimensions(path);
        if (fastDims != null && fastDims[0] > 0 && fastDims[1] > 0) {
            return fastDims;
        }

        // 2. ImageIO ImageReader header inspection
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
        }

        // 3. Fallback: JavaFX native prism loader (handles extended camera markers / SOF 0xce)
        try {
            Image fxImg = new Image(new FileInputStream(path.toFile()));
            if (!fxImg.isError() && fxImg.getWidth() > 0 && fxImg.getHeight() > 0) {
                return new int[]{(int) Math.round(fxImg.getWidth()), (int) Math.round(fxImg.getHeight())};
            }
        } catch (Exception ignored) {
        }

        // 4. Fallback: ImageIO full read
        try {
            BufferedImage bimg = ImageIO.read(path.toFile());
            if (bimg != null && bimg.getWidth() > 0 && bimg.getHeight() > 0) {
                return new int[]{bimg.getWidth(), bimg.getHeight()};
            }
        } catch (Exception ignored) {
        }

        return null;
    }

    /**
     * Loads a BufferedImage from any supported image file.
     * Uses ImageIO first; if non-standard JPEG markers fail standard ImageIO,
     * transparently decodes via JavaFX Prism and converts to BufferedImage.
     */
    public static BufferedImage loadBufferedImage(Path path) {
        if (path == null || !Files.exists(path)) {
            return null;
        }

        // 1. Try ImageIO direct read
        try {
            BufferedImage bimg = ImageIO.read(path.toFile());
            if (bimg != null) {
                return bimg;
            }
        } catch (Exception ignored) {
        }

        // 2. Try JavaFX loader and convert via SwingFXUtils
        try {
            Image fxImg = new Image(new FileInputStream(path.toFile()));
            if (!fxImg.isError() && fxImg.getWidth() > 0 && fxImg.getHeight() > 0) {
                return SwingFXUtils.fromFXImage(fxImg, null);
            }
        } catch (Exception ignored) {
        }

        return null;
    }

    /**
     * Loads an Image for JavaFX cleanly at full resolution.
     * If the raw image fails direct loading, sanitizes it to a clean baseline JPEG
     * in target/cache and loads the clean image.
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

        // 2. Sanitize and load from clean baseline copy
        Path cleanPath = sanitizeImage(path, null);
        if (cleanPath != null && Files.exists(cleanPath) && !cleanPath.equals(path)) {
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
     * Sanitizes non-standard JPEGs / images into a clean baseline JPEG.
     * If outputPath is null, writes into target/cache/clean_<filename>.
     *
     * @param rawPath   Source image path
     * @param cleanPath Target sanitized path (optional)
     * @return Path to the sanitized clean image
     */
    public static Path sanitizeImage(Path rawPath, Path cleanPath) {
        if (rawPath == null || !Files.exists(rawPath)) {
            return null;
        }

        try {
            if (cleanPath == null) {
                Path cacheDir = Path.of("target", "cache");
                Files.createDirectories(cacheDir);
                String cleanFileName = "clean_" + rawPath.getFileName().toString();
                cleanPath = cacheDir.resolve(cleanFileName);
            } else if (cleanPath.getParent() != null) {
                Files.createDirectories(cleanPath.getParent());
            }

            // If already cached and valid, reuse
            if (Files.exists(cleanPath) && Files.size(cleanPath) > 0) {
                return cleanPath;
            }

            // Decode image
            BufferedImage bimg = loadBufferedImage(rawPath);
            if (bimg != null) {
                // Ensure RGB without alpha for JPEG writing if needed
                BufferedImage rgbImg = bimg;
                if (bimg.getType() == BufferedImage.TYPE_CUSTOM || bimg.getColorModel().hasAlpha()) {
                    rgbImg = new BufferedImage(bimg.getWidth(), bimg.getHeight(), BufferedImage.TYPE_INT_RGB);
                    java.awt.Graphics2D g = rgbImg.createGraphics();
                    g.drawImage(bimg, 0, 0, null);
                    g.dispose();
                }

                String format = cleanPath.getFileName().toString().toLowerCase().endsWith(".png") ? "png" : "jpg";
                boolean written = ImageIO.write(rgbImg, format, cleanPath.toFile());
                if (written && Files.exists(cleanPath) && Files.size(cleanPath) > 0) {
                    return cleanPath;
                }
            }
        } catch (Exception ignored) {
        }

        return rawPath;
    }

    /**
     * Preprocesses an image matching the behavior and contract of image_preprocessor.py:
     * - Verifies input file exists
     * - Decodes image and retrieves width/height
     * - If outputPath is specified, writes sanitized image to disk
     * - Returns PreprocessResult with width, height, cleanPath
     */
    public static PreprocessResult process(Path inputPath, Path outputPath) {
        if (inputPath == null || !Files.exists(inputPath)) {
            return new PreprocessResult(false, 0, 0, null, "File not found: " + inputPath);
        }

        int[] dims = getDimensions(inputPath);
        if (dims == null) {
            return new PreprocessResult(false, 0, 0, null, "Could not decode image: " + inputPath);
        }

        Path cleanPath = inputPath;
        if (outputPath != null) {
            cleanPath = sanitizeImage(inputPath, outputPath);
            if (cleanPath == null || !Files.exists(cleanPath)) {
                return new PreprocessResult(false, 0, 0, null, "Failed to sanitize image to: " + outputPath);
            }
        }

        return new PreprocessResult(true, dims[0], dims[1],
                cleanPath.toAbsolutePath().normalize().toString().replace("\\", "/"), null);
    }

    public static PreprocessResult process(String inputStr, String outputStr) {
        Path inPath = inputStr != null ? Path.of(inputStr) : null;
        Path outPath = outputStr != null ? Path.of(outputStr) : null;
        return process(inPath, outPath);
    }

    /**
     * Fast binary header parser for JPEG (SOF markers) and PNG (IHDR chunk).
     * Extracts dimensions in sub-millisecond time without allocating bitmaps.
     */
    private static int[] readBinaryHeaderDimensions(Path path) {
        try (InputStream is = Files.newInputStream(path)) {
            byte[] header = is.readNBytes(8);
            if (header.length < 8) {
                return null;
            }

            // Check PNG (signature 89 50 4E 47 0D 0A 1A 0A)
            if ((header[0] & 0xFF) == 0x89 && header[1] == 'P' && header[2] == 'N' && header[3] == 'G') {
                byte[] ihdr = is.readNBytes(16); // 4 bytes chunk len, 4 bytes chunk type 'IHDR', 4 bytes W, 4 bytes H
                if (ihdr.length == 16 && ihdr[4] == 'I' && ihdr[5] == 'H' && ihdr[6] == 'D' && ihdr[7] == 'R') {
                    int w = ((ihdr[8] & 0xFF) << 24) | ((ihdr[9] & 0xFF) << 16) | ((ihdr[10] & 0xFF) << 8) | (ihdr[11] & 0xFF);
                    int h = ((ihdr[12] & 0xFF) << 24) | ((ihdr[13] & 0xFF) << 16) | ((ihdr[14] & 0xFF) << 8) | (ihdr[15] & 0xFF);
                    if (w > 0 && h > 0) {
                        return new int[]{w, h};
                    }
                }
            }

            // Check JPEG (0xFF 0xD8)
            if ((header[0] & 0xFF) == 0xFF && (header[1] & 0xFF) == 0xD8) {
                // Rewind by reopening stream
                try (InputStream jpegIs = Files.newInputStream(path)) {
                    jpegIs.skip(2); // Skip SOI
                    while (true) {
                        int b = jpegIs.read();
                        if (b == -1) break;
                        if (b != 0xFF) continue;
                        while (b == 0xFF) {
                            b = jpegIs.read();
                        }
                        if (b == -1) break;
                        int marker = b & 0xFF;
                        if (marker == 0xD9 || marker == 0xDA) break; // EOI or SOS
                        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue; // TEM or RST

                        int len1 = jpegIs.read();
                        int len2 = jpegIs.read();
                        if (len1 == -1 || len2 == -1) break;
                        int length = (len1 << 8) | len2;

                        // SOF0 (baseline), SOF1 (extended seq), SOF2 (progressive)
                        if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) {
                            int precision = jpegIs.read();
                            int h1 = jpegIs.read();
                            int h2 = jpegIs.read();
                            int w1 = jpegIs.read();
                            int w2 = jpegIs.read();
                            if (precision != -1 && h1 != -1 && h2 != -1 && w1 != -1 && w2 != -1) {
                                int h = (h1 << 8) | h2;
                                int w = (w1 << 8) | w2;
                                if (w > 0 && h > 0) {
                                    return new int[]{w, h};
                                }
                            }
                            break;
                        } else {
                            long toSkip = length - 2;
                            while (toSkip > 0) {
                                long skipped = jpegIs.skip(toSkip);
                                if (skipped <= 0) break;
                                toSkip -= skipped;
                            }
                        }
                    }
                }
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    /**
     * CLI entry point matching scripts/image_preprocessor.py interface.
     * Arguments:
     *   --input <path>     (required)
     *   --output <path>    (optional)
     */
    public static void main(String[] args) {
        String inputPath = null;
        String outputPath = null;

        for (int i = 0; i < args.length; i++) {
            if ("--input".equals(args[i]) && i + 1 < args.length) {
                inputPath = args[++i];
            } else if ("--output".equals(args[i]) && i + 1 < args.length) {
                outputPath = args[++i];
            }
        }

        if (inputPath == null || inputPath.isBlank()) {
            System.out.println("{\"success\":false,\"error\":\"Missing required argument --input\"}");
            System.exit(1);
        }

        File inFile = new File(inputPath);
        if (!inFile.exists()) {
            System.out.println("{\"success\":false,\"error\":\"File not found: " + inputPath + "\"}");
            System.exit(1);
        }

        PreprocessResult result = process(Path.of(inputPath), outputPath != null ? Path.of(outputPath) : null);
        System.out.println(result.toJson());

        if (!result.success()) {
            System.exit(2);
        } else {
            System.exit(0);
        }
    }
}
