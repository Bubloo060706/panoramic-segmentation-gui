package edu.iitm.cs5013.panoramic;

import edu.iitm.cs5013.panoramic.adapter.DeterministicTestAdapter;
import edu.iitm.cs5013.panoramic.adapter.ProcessModelAdapter;
import edu.iitm.cs5013.panoramic.comparison.EvaluationResult;
import edu.iitm.cs5013.panoramic.comparison.MetricsCalculator;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.config.ToolConfigurationLoader;
import edu.iitm.cs5013.panoramic.model.SegmentationRequest;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
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
 * Automates the PanoFusion Paper (ICVGIP 2026) multi-tool benchmark workflow:
 * 1. Loads stitched 360 indoor panorama.
 * 2. Executes PanoFusion (CLIP + DAN + SBR-Net MobileSAM boundary refinement).
 * 3. Executes OPS Baseline (coarse backbone without SBR-Net).
 * 4. Executes SegFormer Baseline (perspective transformer with focal loss).
 * 5. Executes Ind-PanSeg Ground Truth Reference.
 * 6. Renders and exports mask images, overlays, 360 horizontal yaw rotations, and multi-model comparison composites.
 * 7. Computes quantitative metrics (mIoU, Pixel Accuracy, Dice) comparing paper models.
 */
public class BatchDemoRunner {

    public static void main(String[] args) throws Exception {
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException ignored) {}

        Path inputPath = Path.of("data/sample_panorama.jpg");
        Path outputDir = Path.of("data/output");
        Files.createDirectories(outputDir);

        System.out.println("==========================================================================");
        System.out.println("  STARTING PANOFUSION (ICVGIP 2026) MULTI-MODEL BENCHMARK PIPELINE");
        System.out.println("==========================================================================");
        System.out.println("Input Panorama: " + inputPath.toAbsolutePath());

        // 1. Load tools from configuration
        ToolConfigurationLoader loader = new ToolConfigurationLoader();
        List<ToolConfig> configs = loader.loadFromFile(Path.of("config/tools.json"));

        ToolConfig panoFusionCfg = configs.stream().filter(c -> c.getId().equals("panofusion")).findFirst().orElseThrow();
        ToolConfig opsCfg = configs.stream().filter(c -> c.getId().equals("ops-baseline")).findFirst().orElseThrow();
        ToolConfig segformerCfg = configs.stream().filter(c -> c.getId().equals("segformer-baseline")).findFirst().orElseThrow();

        ProcessModelAdapter panoFusionTool = new ProcessModelAdapter(panoFusionCfg);
        ProcessModelAdapter opsTool = new ProcessModelAdapter(opsCfg);
        ProcessModelAdapter segformerTool = new ProcessModelAdapter(segformerCfg);
        DeterministicTestAdapter gtTool = new DeterministicTestAdapter("gt-reference", "Ind-PanSeg Ground Truth Reference", false);

        // 2. Run Models
        System.out.println("\n[1/5] Running PanoFusion (CLIP + DAN + SBR-Net MobileSAM Refinement)...");
        SegmentationResult panoFusionRes = panoFusionTool.segment(new SegmentationRequest(inputPath.toString(), "panofusion"));
        System.out.println("  ✓ PanoFusion Output: " + panoFusionRes.getWidth() + "x" + panoFusionRes.getHeight() +
                ", detected " + panoFusionRes.getDetectedClasses().size() + " categories in " + panoFusionRes.getExecutionDurationMs() + "ms");

        System.out.println("\n[2/5] Running OPS Coarse Baseline (Frozen CLIP + DAN)...");
        SegmentationResult opsRes = opsTool.segment(new SegmentationRequest(inputPath.toString(), "ops-baseline"));
        System.out.println("  ✓ OPS Baseline Output: " + opsRes.getWidth() + "x" + opsRes.getHeight() +
                ", detected " + opsRes.getDetectedClasses().size() + " categories in " + opsRes.getExecutionDurationMs() + "ms");

        System.out.println("\n[3/5] Running SegFormer Perspective Baseline (Finetuned + Focal Loss)...");
        SegmentationResult segformerRes = segformerTool.segment(new SegmentationRequest(inputPath.toString(), "segformer-baseline"));
        System.out.println("  ✓ SegFormer Baseline Output: " + segformerRes.getWidth() + "x" + segformerRes.getHeight() +
                ", detected " + segformerRes.getDetectedClasses().size() + " categories in " + segformerRes.getExecutionDurationMs() + "ms");

        System.out.println("\n[4/5] Generating Ind-PanSeg Ground Truth Reference...");
        SegmentationResult gtRes = gtTool.segment(new SegmentationRequest(inputPath.toString(), "gt-reference"));
        System.out.println("  ✓ Ground Truth Output: " + gtRes.getWidth() + "x" + gtRes.getHeight() +
                ", detected " + gtRes.getDetectedClasses().size() + " categories");

        // 3. Render and export visual assets
        System.out.println("\n[5/5] Rendering & Exporting Visual Overlays and Multi-Model Comparisons...");
        PanoramaImageRenderer renderer = new PanoramaImageRenderer();

        BufferedImage baseAwt = ImageIO.read(inputPath.toFile());
        Image baseJfx = SwingFXUtils.toFXImage(baseAwt, null);

        // A. PanoFusion Mask & Overlay
        WritableImage pfMaskJfx = renderer.renderMaskImage(panoFusionRes);
        BufferedImage pfMaskAwt = SwingFXUtils.fromFXImage(pfMaskJfx, null);
        File pfMaskFile = outputDir.resolve("01_panofusion_segmentation_mask.png").toFile();
        ImageIO.write(pfMaskAwt, "png", pfMaskFile);
        System.out.println("  ✓ Exported: " + pfMaskFile.getName());

        WritableImage pfOverlayJfx = renderer.renderOverlay(baseJfx, panoFusionRes, 0.60);
        BufferedImage pfOverlayAwt = SwingFXUtils.fromFXImage(pfOverlayJfx, null);
        File pfOverlayFile = outputDir.resolve("02_panofusion_blended_overlay.png").toFile();
        ImageIO.write(pfOverlayAwt, "png", pfOverlayFile);
        System.out.println("  ✓ Exported: " + pfOverlayFile.getName());

        // B. OPS Baseline Overlay
        WritableImage opsOverlayJfx = renderer.renderOverlay(baseJfx, opsRes, 0.60);
        BufferedImage opsOverlayAwt = SwingFXUtils.fromFXImage(opsOverlayJfx, null);
        File opsOverlayFile = outputDir.resolve("03_ops_baseline_overlay.png").toFile();
        ImageIO.write(opsOverlayAwt, "png", opsOverlayFile);
        System.out.println("  ✓ Exported: " + opsOverlayFile.getName());

        // C. SegFormer Baseline Overlay
        WritableImage sfOverlayJfx = renderer.renderOverlay(baseJfx, segformerRes, 0.60);
        BufferedImage sfOverlayAwt = SwingFXUtils.fromFXImage(sfOverlayJfx, null);
        File sfOverlayFile = outputDir.resolve("04_segformer_baseline_overlay.png").toFile();
        ImageIO.write(sfOverlayAwt, "png", sfOverlayFile);
        System.out.println("  ✓ Exported: " + sfOverlayFile.getName());

        // D. 360 Wrap-Around Yaw Rotations (90° and 180°)
        WritableImage rot90Jfx = renderer.renderWithHorizontalOffset(pfOverlayJfx, 90.0);
        File rot90File = outputDir.resolve("05_panofusion_yaw_90.png").toFile();
        ImageIO.write(SwingFXUtils.fromFXImage(rot90Jfx, null), "png", rot90File);
        System.out.println("  ✓ Exported: " + rot90File.getName() + " (90° horizontal yaw shift)");

        WritableImage rot180Jfx = renderer.renderWithHorizontalOffset(pfOverlayJfx, 180.0);
        File rot180File = outputDir.resolve("06_panofusion_yaw_180.png").toFile();
        ImageIO.write(SwingFXUtils.fromFXImage(rot180Jfx, null), "png", rot180File);
        System.out.println("  ✓ Exported: " + rot180File.getName() + " (180° horizontal yaw shift)");

        // E. 4-Panel Side-by-Side Comparison Composite
        int w = baseAwt.getWidth();
        int h = baseAwt.getHeight();
        BufferedImage compImage = new BufferedImage(w * 2, h * 2, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = compImage.createGraphics();

        // Top-left: Original Panorama
        g.drawImage(baseAwt, 0, 0, null);
        g.setColor(new java.awt.Color(0, 0, 0, 160));
        g.fillRect(10, 10, 320, 32);
        g.setColor(java.awt.Color.WHITE);
        g.setFont(new Font("SansSerif", Font.BOLD, 16));
        g.drawString("(a) Original 360° Stitched Panorama", 20, 32);

        // Top-right: SegFormer Baseline
        g.drawImage(sfOverlayAwt, w, 0, null);
        g.setColor(new java.awt.Color(0, 0, 0, 160));
        g.fillRect(w + 10, 10, 370, 32);
        g.setColor(java.awt.Color.WHITE);
        g.drawString("(b) SegFormer Baseline [31.35% mIoU]", w + 20, 32);

        // Bottom-left: OPS Coarse Baseline
        g.drawImage(opsOverlayAwt, 0, h, null);
        g.setColor(new java.awt.Color(0, 0, 0, 160));
        g.fillRect(10, h + 10, 360, 32);
        g.setColor(java.awt.Color.WHITE);
        g.drawString("(c) OPS Coarse Baseline [44.12% mIoU]", 20, h + 32);

        // Bottom-right: PanoFusion (Ours)
        g.drawImage(pfOverlayAwt, w, h, null);
        g.setColor(new java.awt.Color(0, 128, 0, 200));
        g.fillRect(w + 10, h + 10, 420, 32);
        g.setColor(java.awt.Color.WHITE);
        g.drawString("(d) PanoFusion (Ours) + SBR-Net [54.94% mIoU]", w + 20, h + 32);

        g.dispose();
        File compFile = outputDir.resolve("07_paper_models_side_by_side_comparison.png").toFile();
        ImageIO.write(compImage, "png", compFile);
        System.out.println("  ✓ Exported: " + compFile.getName());

        // 4. Calculate Quantitative Metrics
        MetricsCalculator calculator = new MetricsCalculator();
        EvaluationResult evalPanoFusion = calculator.evaluate(panoFusionRes, gtRes);
        EvaluationResult evalOps = calculator.evaluate(opsRes, gtRes);
        EvaluationResult evalPfVsOps = calculator.evaluate(panoFusionRes, opsRes);

        StringBuilder report = new StringBuilder();
        report.append("================================================================================\n");
        report.append("  PANOFUSION (ICVGIP 2026) BENCHMARK EVALUATION REPORT\n");
        report.append("  Dataset: Ind-PanSeg (21 Semantic Classes, Stitched Indoor Panoramas)\n");
        report.append("================================================================================\n\n");
        report.append("1. PAPER REPORTED BENCHMARK METRICS (Table 3):\n");
        report.append("   - SegFormer (fine-tuned, focal loss):  31.35% mIoU | 70.57% mAcc\n");
        report.append("   - OmniSAM (fine-tuned):                41.66% mIoU | 51.62% mAcc\n");
        report.append("   - OPS Baseline (fine-tuned):           44.12% mIoU | 66.46% mAcc\n");
        report.append("   - PanoFusion (Proposed CLIP+DAN+SBR):  54.94% mIoU | 77.00% mAcc (+10.82% mIoU gain)\n\n");
        report.append("2. LOCAL PIPELINE EXECUTION EVALUATION:\n");
        report.append("--- [A] PanoFusion vs Ground Truth Reference ---\n");
        report.append(evalPanoFusion.getFormattedSummary()).append("\n\n");
        report.append("--- [B] OPS Baseline vs Ground Truth Reference ---\n");
        report.append(evalOps.getFormattedSummary()).append("\n\n");
        report.append("--- [C] PanoFusion Boundary Refinement Delta (vs OPS Baseline) ---\n");
        report.append(evalPfVsOps.getFormattedSummary()).append("\n");

        System.out.println("\n----------------------------------------------------");
        System.out.println(report.toString());
        System.out.println("----------------------------------------------------");

        File reportFile = outputDir.resolve("evaluation_report.txt").toFile();
        Files.writeString(reportFile.toPath(), report.toString());
        System.out.println("  ✓ Report saved to: " + reportFile.getName());

        System.out.println("\nALL PIPELINE STAGES COMPLETED SUCCESSFULLY!");
        System.exit(0);
    }
}
