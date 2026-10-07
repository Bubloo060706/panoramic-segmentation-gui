package edu.iitm.cs5013.panoramic.gui;

import edu.iitm.cs5013.panoramic.comparison.EvaluationResult;
import edu.iitm.cs5013.panoramic.comparison.MetricsCalculator;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;
import edu.iitm.cs5013.panoramic.visualization.PanoramaImageRenderer;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;

import java.util.*;

/**
 * Multi-tool visual comparison and quantitative metrics pane.
 * Synchronizes horizontal panoramic inspection across tools and renders evaluation metrics
 * when compatible ground truth is available.
 */
public class ComparisonViewPane extends VBox {

    private final PanoramaImageRenderer renderer = new PanoramaImageRenderer();
    private final MetricsCalculator metricsCalculator = new MetricsCalculator();

    private final Slider syncYawSlider = new Slider(0.0, 360.0, 0.0);
    private final HBox comparisonGrid = new HBox(12);
    private final VBox metricsPanel = new VBox(8);
    private final Label metricsStatusLabel = new Label("No comparison performed yet.");

    private Image baseImage;
    private final Map<String, SegmentationResult> toolResults = new LinkedHashMap<>();
    private SegmentationResult groundTruthResult;

    public ComparisonViewPane() {
        setSpacing(12);
        setPadding(new Insets(12));
        VBox.setVgrow(this, Priority.ALWAYS);

        // Header and Yaw sync bar
        HBox topBar = new HBox(16);
        topBar.setAlignment(Pos.CENTER_LEFT);
        Label title = new Label("Cross-Model Comparison");
        title.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        Label yawLabel = new Label("Synchronized 360° Yaw:");
        syncYawSlider.setPrefWidth(160);
        syncYawSlider.valueProperty().addListener((obs, o, n) -> refreshViews());

        topBar.getChildren().addAll(title, new Label("|"), yawLabel, syncYawSlider);

        // Comparison images container
        ScrollPane scrollPane = new ScrollPane(comparisonGrid);
        scrollPane.setFitToHeight(true);
        scrollPane.setFitToWidth(false);
        scrollPane.setPannable(true);
        scrollPane.setMinHeight(260);
        scrollPane.setStyle("-fx-background-color: transparent; -fx-border-color: #cbd5e1; -fx-border-radius: 6px;");

        // Metrics Panel
        metricsPanel.getStyleClass().add("card");
        Label metricsHeading = new Label("Quantitative Evaluation Metrics");
        metricsHeading.getStyleClass().add("section-header");
        metricsPanel.getChildren().addAll(metricsHeading, metricsStatusLabel);

        getChildren().addAll(topBar, scrollPane, metricsPanel);
    }

    public void updateData(Image baseImage, Map<String, SegmentationResult> results, SegmentationResult groundTruth) {
        this.baseImage = baseImage;
        this.toolResults.clear();
        if (results != null) {
            this.toolResults.putAll(results);
        }
        this.groundTruthResult = groundTruth;

        refreshViews();
        recalculateMetrics();
    }

    private void refreshViews() {
        comparisonGrid.getChildren().clear();
        double yaw = syncYawSlider.getValue();

        // 1. Original Image Tile
        if (baseImage != null) {
            Image shiftedBase = renderer.renderWithHorizontalOffset(baseImage, yaw);
            VBox baseTile = createViewerTile("Original 360° Panorama", shiftedBase, "Aspect: 2:1 Equirectangular");
            comparisonGrid.getChildren().add(baseTile);
        }

        // 2. Ground Truth Tile if loaded
        if (groundTruthResult != null && groundTruthResult.isSuccess()) {
            Image gtOverlay = renderer.renderOverlay(baseImage, groundTruthResult, 0.55);
            Image shiftedGt = renderer.renderWithHorizontalOffset(gtOverlay, yaw);
            VBox gtTile = createViewerTile("Ground Truth Reference", shiftedGt, "Verified Annotations");
            comparisonGrid.getChildren().add(gtTile);
        }

        // 3. Model Tiles
        for (Map.Entry<String, SegmentationResult> entry : toolResults.entrySet()) {
            SegmentationResult res = entry.getValue();
            if (res.isSuccess()) {
                Image overlay = renderer.renderOverlay(baseImage, res, 0.55);
                Image shifted = renderer.renderWithHorizontalOffset(overlay, yaw);
                String sub = String.format("Classes: %d | Time: %d ms",
                        res.getDetectedClasses().size(), res.getExecutionDurationMs());
                VBox tile = createViewerTile(res.getToolName(), shifted, sub);
                comparisonGrid.getChildren().add(tile);
            } else {
                VBox failTile = createFailedTile(res.getToolName(), res.getErrorMessage());
                comparisonGrid.getChildren().add(failTile);
            }
        }
    }

    private VBox createViewerTile(String title, Image image, String subtitle) {
        VBox box = new VBox(6);
        box.setPrefWidth(380);
        box.setAlignment(Pos.TOP_CENTER);
        box.getStyleClass().add("card");

        Label titleLabel = new Label(title);
        titleLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #1e293b;");

        ImageView iv = new ImageView(image);
        iv.setFitWidth(360);
        iv.setPreserveRatio(true);
        iv.setSmooth(true);

        Label subLabel = new Label(subtitle);
        subLabel.setStyle("-fx-text-fill: #64748b; -fx-font-size: 11px;");

        box.getChildren().addAll(titleLabel, iv, subLabel);
        return box;
    }

    private VBox createFailedTile(String title, String error) {
        VBox box = new VBox(6);
        box.setPrefWidth(380);
        box.setAlignment(Pos.CENTER);
        box.getStyleClass().addAll("card", "error-banner");

        Label titleLabel = new Label(title + " (Failed)");
        titleLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #991b1b;");

        Label errLabel = new Label(error != null ? error : "Execution failed");
        errLabel.setWrapText(true);
        errLabel.setStyle("-fx-text-fill: #991b1b;");

        box.getChildren().addAll(titleLabel, errLabel);
        return box;
    }

    private void recalculateMetrics() {
        metricsPanel.getChildren().clear();
        Label heading = new Label("Quantitative Evaluation Metrics");
        heading.getStyleClass().add("section-header");
        metricsPanel.getChildren().add(heading);

        if (groundTruthResult == null || !groundTruthResult.isSuccess()) {
            Label note = new Label("Notice: Quantitative evaluation (mIoU, Pixel Accuracy, Dice/F1) is unavailable because " +
                    "no Ground Truth reference mask is currently loaded. Side-by-side visual comparison is active above. " +
                    "Use 'Load Ground Truth' to compute numerical metrics.");
            note.setStyle("-fx-text-fill: #64748b; -fx-font-style: italic;");
            note.setWrapText(true);
            metricsPanel.getChildren().add(note);
            return;
        }

        // Table for quantitative comparisons
        GridPane table = new GridPane();
        table.setHgap(16);
        table.setVgap(8);
        table.setPadding(new Insets(8));

        // Table Header
        table.add(styledHeader("Tool Name"), 0, 0);
        table.add(styledHeader("Pixel Accuracy"), 1, 0);
        table.add(styledHeader("Mean IoU (mIoU)"), 2, 0);
        table.add(styledHeader("Mean Dice (F1)"), 3, 0);
        table.add(styledHeader("Status"), 4, 0);

        int row = 1;
        for (SegmentationResult pred : toolResults.values()) {
            EvaluationResult eval = metricsCalculator.evaluate(pred, groundTruthResult);

            table.add(new Label(pred.getToolName()), 0, row);

            if (eval.isQuantitativeValid()) {
                table.add(new Label(String.format("%.2f%%", eval.getPixelAccuracy() * 100)), 1, row);
                table.add(new Label(String.format("%.4f", eval.getMeanIoU())), 2, row);
                table.add(new Label(String.format("%.4f", eval.getMeanDice())), 3, row);

                Label okBadge = new Label("Calculated");
                okBadge.getStyleClass().add("status-badge-success");
                table.add(okBadge, 4, row);
            } else {
                table.add(new Label("N/A"), 1, row);
                table.add(new Label("N/A"), 2, row);
                table.add(new Label("N/A"), 3, row);

                Label warnBadge = new Label(eval.getInvalidationReason());
                warnBadge.setStyle("-fx-text-fill: #b45309; -fx-font-size: 11px;");
                warnBadge.setWrapText(true);
                table.add(warnBadge, 4, row);
            }
            row++;
        }

        metricsPanel.getChildren().add(table);
    }

    private Label styledHeader(String text) {
        Label l = new Label(text);
        l.setStyle("-fx-font-weight: bold; -fx-text-fill: #334155;");
        return l;
    }
}
