package edu.iitm.cs5013.panoramic;

import edu.iitm.cs5013.panoramic.adapter.DeterministicTestAdapter;
import edu.iitm.cs5013.panoramic.adapter.ProcessModelAdapter;
import edu.iitm.cs5013.panoramic.comparison.EvaluationResult;
import edu.iitm.cs5013.panoramic.comparison.MetricsCalculator;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.config.ToolConfigurationLoader;
import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationRequest;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;
import edu.iitm.cs5013.panoramic.visualization.PanoramaImageRenderer;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;
import javafx.scene.image.WritableImage;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Automates the complete panoramic segmentation demo workflow:
 * 1. Loads 360 equirectangular panorama.
 * 2. Executes both the real external Python model and the deterministic test model.
 * 3. Renders and exports mask images, overlays, and 360 wrap-around views.
 * 4. Computes quantitative metrics and exports an evaluation report.
 */
public class BatchDemoRunner {

    public static void main(String[] args) throws Exception {
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException ignored) {}

        Path inputPath = Path.of("data/sample_panorama.jpg");
        Path outputDir = Path.of("data/output");
        Files.createDirectories(outputDir);

        System.out.println("====================================================");
        System.out.println("  STARTING AUTOMATED PANORAMIC SEGMENTATION PIPELINE");
        System.out.println("====================================================");
        System.out.println("Input Panorama: " + inputPath.toAbsolutePath());

        // 1. Load tools from configuration
        ToolConfigurationLoader loader = new ToolConfigurationLoader();
        List<ToolConfig> configs = loader.loadFromFile(Path.of("config/tools.json"));
        ToolConfig pyCfg = configs.stream().filter(c -> c.getId().equals("pano-seg-py")).findFirst().orElseThrow();

        ProcessModelAdapter pythonTool = new ProcessModelAdapter(pyCfg);
        DeterministicTestAdapter mockTool = new DeterministicTestAdapter("gt-reference", "Ground Truth Reference", false);

        // 2. Run Python external model
        System.out.println("\n[1/4] Running External Python Panoramic Model (pano-seg-py)...");
        SegmentationRequest pyReq = new SegmentationRequest(inputPath.toString(), "pano-seg-py");
        SegmentationResult pyRes = pythonTool.segment(pyReq);
        System.out.println("  ✓ Python Model Output: " + pyRes.getWidth() + "x" + pyRes.getHeight() +
                ", detected " + pyRes.getDetectedClasses().size() + " categories in " + pyRes.getExecutionDurationMs() + "ms");

        // 3. Run Ground Truth reference model
        System.out.println("\n[2/4] Running Reference Model (gt-reference)...");
        SegmentationRequest mockReq = new SegmentationRequest(inputPath.toString(), "gt-reference");
        SegmentationResult mockRes = mockTool.segment(mockReq);
        System.out.println("  ✓ Reference Model Output: " + mockRes.getWidth() + "x" + mockRes.getHeight() +
                ", detected " + mockRes.getDetectedClasses().size() + " categories");

        // 4. Render and export visual assets
        System.out.println("\n[3/4] Rendering & Exporting Visual Overlays and 360 Wrap-Around Views...");
        PanoramaImageRenderer renderer = new PanoramaImageRenderer();

        BufferedImage baseAwt = ImageIO.read(inputPath.toFile());
        Image baseJfx = SwingFXUtils.toFXImage(baseAwt, null);

        // A. Python Model Mask & Overlay
        WritableImage pyMaskJfx = renderer.renderMaskImage(pyRes);
        BufferedImage pyMaskAwt = SwingFXUtils.fromFXImage(pyMaskJfx, null);
        File pyMaskFile = outputDir.resolve("01_python_segmentation_mask.png").toFile();
        ImageIO.write(pyMaskAwt, "png", pyMaskFile);
        System.out.println("  ✓ Exported: " + pyMaskFile.getName());

        WritableImage pyOverlayJfx = renderer.renderOverlay(baseJfx, pyRes, 0.60);
        BufferedImage pyOverlayAwt = SwingFXUtils.fromFXImage(pyOverlayJfx, null);
        File pyOverlayFile = outputDir.resolve("02_python_blended_overlay.png").toFile();
        ImageIO.write(pyOverlayAwt, "png", pyOverlayFile);
        System.out.println("  ✓ Exported: " + pyOverlayFile.getName());

        // B. 360 Horizontal Wrap-Around Rotations (demonstrating boundary continuity)
        WritableImage rot90Jfx = renderer.renderWithHorizontalOffset(pyOverlayJfx, 90.0);
        File rot90File = outputDir.resolve("03_panoramic_wrap_around_yaw_90.png").toFile();
        ImageIO.write(SwingFXUtils.fromFXImage(rot90Jfx, null), "png", rot90File);
        System.out.println("  ✓ Exported: " + rot90File.getName() + " (90° horizontal yaw shift)");

        WritableImage rot180Jfx = renderer.renderWithHorizontalOffset(pyOverlayJfx, 180.0);
        File rot180File = outputDir.resolve("04_panoramic_wrap_around_yaw_180.png").toFile();
        ImageIO.write(SwingFXUtils.fromFXImage(rot180Jfx, null), "png", rot180File);
        System.out.println("  ✓ Exported: " + rot180File.getName() + " (180° horizontal yaw shift)");

        // C. Side-by-Side Comparison Composite
        int w = baseAwt.getWidth();
        int h = baseAwt.getHeight();
        BufferedImage compImage = new BufferedImage(w * 2, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = compImage.createGraphics();
        g.drawImage(baseAwt, 0, 0, null);
        g.drawImage(pyOverlayAwt, w, 0, null);
        g.setColor(java.awt.Color.WHITE);
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.drawString("Original 360° Panorama", 20, 30);
        g.drawString("Python Segmentation Overlay", w + 20, 30);
        g.dispose();
        File compFile = outputDir.resolve("05_side_by_side_comparison.png").toFile();
        ImageIO.write(compImage, "png", compFile);
        System.out.println("  ✓ Exported: " + compFile.getName());

        // 5. Compute Quantitative Metrics
        System.out.println("\n[4/4] Calculating Quantitative Comparison Metrics...");
        MetricsCalculator calculator = new MetricsCalculator();
        EvaluationResult eval = calculator.evaluate(pyRes, mockRes);

        System.out.println("----------------------------------------------------");
        System.out.println(eval.getFormattedSummary());
        System.out.println("----------------------------------------------------");

        // Write metrics report text file
        File reportFile = outputDir.resolve("evaluation_report.txt").toFile();
        Files.writeString(reportFile.toPath(), eval.getFormattedSummary());
        System.out.println("  ✓ Report saved to: " + reportFile.getName());

        System.out.println("\nALL PIPELINE STAGES COMPLETED SUCCESSFULLY!");
        System.exit(0);
    }
}
