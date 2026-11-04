/*
 * Universal Image Preprocessor & Normalizer for Panoramic Framework (Single-File Java Edition).
 * Can be executed directly via:
 *   java scripts/ImagePreprocessor.java --input <path> [--output <path>]
 * Or used within the JavaFX application.
 */

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;

public class ImagePreprocessor {

    public static int[] getDimensions(Path path) {
        if (path == null || !Files.exists(path)) {
            return null;
        }

        // 1. Direct binary header inspection (fastest, 0ms)
        int[] fastDims = readBinaryHeaderDimensions(path);
        if (fastDims != null && fastDims[0] > 0 && fastDims[1] > 0) {
            return fastDims;
        }

        // 2. ImageIO ImageReader header-only inspection
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
                            return new int[] { w, h };
                        }
                    } finally {
                        reader.dispose();
                    }
                }
            }
        } catch (Exception ignored) {
        }

        // 3. ImageIO full read fallback
        try {
            BufferedImage bimg = ImageIO.read(path.toFile());
            if (bimg != null) {
                return new int[] { bimg.getWidth(), bimg.getHeight() };
            }
        } catch (Exception ignored) {
        }

        return null;
    }

    public static Path sanitize(Path inputPath, Path outputPath) {
        if (inputPath == null || !Files.exists(inputPath)) {
            return null;
        }
        try {
            if (outputPath != null && outputPath.getParent() != null) {
                Files.createDirectories(outputPath.getParent());
            }

            BufferedImage bimg = ImageIO.read(inputPath.toFile());
            if (bimg != null && outputPath != null) {
                BufferedImage rgb = bimg;
                if (bimg.getType() == BufferedImage.TYPE_CUSTOM || bimg.getColorModel().hasAlpha()) {
                    rgb = new BufferedImage(bimg.getWidth(), bimg.getHeight(), BufferedImage.TYPE_INT_RGB);
                    java.awt.Graphics2D g = rgb.createGraphics();
                    g.drawImage(bimg, 0, 0, null);
                    g.dispose();
                }
                String format = outputPath.getFileName().toString().toLowerCase().endsWith(".png") ? "png" : "jpg";
                ImageIO.write(rgb, format, outputPath.toFile());
                return outputPath;
            }
        } catch (Exception ignored) {
        }
        return inputPath;
    }

    private static int[] readBinaryHeaderDimensions(Path path) {
        try (InputStream is = Files.newInputStream(path)) {
            byte[] header = is.readNBytes(8);
            if (header.length < 8)
                return null;

            // PNG
            if ((header[0] & 0xFF) == 0x89 && header[1] == 'P' && header[2] == 'N' && header[3] == 'G') {
                byte[] ihdr = is.readNBytes(16);
                if (ihdr.length == 16 && ihdr[4] == 'I' && ihdr[5] == 'H' && ihdr[6] == 'D' && ihdr[7] == 'R') {
                    int w = ((ihdr[8] & 0xFF) << 24) | ((ihdr[9] & 0xFF) << 16) | ((ihdr[10] & 0xFF) << 8)
                            | (ihdr[11] & 0xFF);
                    int h = ((ihdr[12] & 0xFF) << 24) | ((ihdr[13] & 0xFF) << 16) | ((ihdr[14] & 0xFF) << 8)
                            | (ihdr[15] & 0xFF);
                    if (w > 0 && h > 0)
                        return new int[] { w, h };
                }
            }

            // JPEG
            if ((header[0] & 0xFF) == 0xFF && (header[1] & 0xFF) == 0xD8) {
                try (InputStream jpegIs = Files.newInputStream(path)) {
                    jpegIs.skip(2);
                    while (true) {
                        int b = jpegIs.read();
                        if (b == -1)
                            break;
                        if (b != 0xFF)
                            continue;
                        while (b == 0xFF) {
                            b = jpegIs.read();
                        }
                        if (b == -1)
                            break;
                        int marker = b & 0xFF;
                        if (marker == 0xD9 || marker == 0xDA)
                            break;
                        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7))
                            continue;

                        int len1 = jpegIs.read();
                        int len2 = jpegIs.read();
                        if (len1 == -1 || len2 == -1)
                            break;
                        int length = (len1 << 8) | len2;

                        if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) {
                            int precision = jpegIs.read();
                            int h1 = jpegIs.read();
                            int h2 = jpegIs.read();
                            int w1 = jpegIs.read();
                            int w2 = jpegIs.read();
                            if (precision != -1 && h1 != -1 && h2 != -1 && w1 != -1 && w2 != -1) {
                                int h = (h1 << 8) | h2;
                                int w = (w1 << 8) | w2;
                                if (w > 0 && h > 0)
                                    return new int[] { w, h };
                            }
                            break;
                        } else {
                            long toSkip = length - 2;
                            while (toSkip > 0) {
                                long skipped = jpegIs.skip(toSkip);
                                if (skipped <= 0)
                                    break;
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

        Path inPath = Path.of(inputPath);
        if (!Files.exists(inPath)) {
            System.out.println("{\"success\":false,\"error\":\"File not found: " + inputPath + "\"}");
            System.exit(1);
        }

        int[] dims = getDimensions(inPath);
        if (dims == null) {
            System.out.println("{\"success\":false,\"error\":\"Could not decode image: " + inputPath + "\"}");
            System.exit(2);
        }

        Path cleanPath = inPath;
        if (outputPath != null) {
            cleanPath = sanitize(inPath, Path.of(outputPath));
            if (cleanPath == null || !Files.exists(cleanPath)) {
                System.out.println(
                        "{\"success\":false,\"error\":\"Failed to write clean image to: " + outputPath + "\"}");
                System.exit(2);
            }
        }

        String absPath = cleanPath.toAbsolutePath().normalize().toString().replace("\\", "/");
        System.out.printf("{\"success\":true,\"width\":%d,\"height\":%d,\"cleanPath\":\"%s\"}%n",
                dims[0], dims[1], absPath);
        System.exit(0);
    }
}
