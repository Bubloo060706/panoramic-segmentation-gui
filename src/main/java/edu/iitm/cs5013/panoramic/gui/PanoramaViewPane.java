package edu.iitm.cs5013.panoramic.gui;

import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;
import edu.iitm.cs5013.panoramic.visualization.PanoramaImageRenderer;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Slider;
import javafx.scene.control.ToggleGroup;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;

import java.util.List;

/**
 * Interactive panoramic visualization pane.
 * Renders the 360° equirectangular image, mask overlays, interactive 360° yaw wrap-around rotation,
 * and mouse pixel inspection.
 */
public class PanoramaViewPane extends VBox {

    public enum DisplayMode {
        OVERLAY,
        MASK_ONLY,
        ORIGINAL_ONLY
    }

    private final PanoramaImageRenderer renderer = new PanoramaImageRenderer();
    private final ImageView imageView = new ImageView();
    private final Slider opacitySlider = new Slider(0.0, 1.0, 0.55);
    private final Slider yawSlider = new Slider(0.0, 360.0, 0.0);
    private final Label pixelInfoLabel = new Label("Coordinates: (--, --) | Class: --");
    private final FlowPane legendPane = new FlowPane();

    private Image baseImage;
    private SegmentationResult currentResult;
    private WritableImage renderedComposite;
    private DisplayMode currentMode = DisplayMode.OVERLAY;

    public PanoramaViewPane() {
        setSpacing(10);
        setPadding(new Insets(10));
        VBox.setVgrow(this, Priority.ALWAYS);

        // Controls bar (Mode toggles, Opacity, Yaw rotation)
        HBox controlBar = createControlBar();

        // Image container with scroll and aspect ratio handling
        imageView.setPreserveRatio(true);
        imageView.setSmooth(true);
        imageView.fitWidthProperty().bind(widthProperty().subtract(30));

        StackPane imageHolder = new StackPane(imageView);
        imageHolder.setAlignment(Pos.CENTER);
        imageHolder.getStyleClass().add("viewport-container");
        imageHolder.setMinHeight(320);
        VBox.setVgrow(imageHolder, Priority.ALWAYS);

        // Pixel inspector on mouse move
        setupPixelInspection();

        // Legend pane
        legendPane.setHgap(12);
        legendPane.setVgap(6);
        legendPane.setPadding(new Insets(4, 0, 4, 0));

        getChildren().addAll(controlBar, imageHolder, pixelInfoLabel, legendPane);
    }

    private HBox createControlBar() {
        HBox bar = new HBox(16);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.setPadding(new Insets(4, 0, 4, 0));

        // Display Mode Radios
        ToggleGroup group = new ToggleGroup();
        RadioButton rbOverlay = new RadioButton("Blended Overlay");
        rbOverlay.setToggleGroup(group);
        rbOverlay.setSelected(true);
        rbOverlay.setOnAction(e -> {
            currentMode = DisplayMode.OVERLAY;
            updateRendering();
        });

        RadioButton rbMask = new RadioButton("Mask Only");
        rbMask.setToggleGroup(group);
        rbMask.setOnAction(e -> {
            currentMode = DisplayMode.MASK_ONLY;
            updateRendering();
        });

        RadioButton rbOrig = new RadioButton("Original Panorama");
        rbOrig.setToggleGroup(group);
        rbOrig.setOnAction(e -> {
            currentMode = DisplayMode.ORIGINAL_ONLY;
            updateRendering();
        });

        // Opacity
        Label opLabel = new Label("Overlay Opacity:");
        opacitySlider.setPrefWidth(100);
        opacitySlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (currentMode == DisplayMode.OVERLAY) {
                updateRendering();
            }
        });

        // 360 Yaw Rotation (Wrap-Around)
        Label yawLabel = new Label("360° Horizontal Yaw:");
        yawSlider.setPrefWidth(120);
        yawSlider.valueProperty().addListener((obs, oldVal, newVal) -> updateRendering());

        bar.getChildren().addAll(
                rbOverlay, rbMask, rbOrig,
                new Label("|"),
                opLabel, opacitySlider,
                new Label("|"),
                yawLabel, yawSlider
        );
        return bar;
    }

    public void setContent(Image baseImage, SegmentationResult result) {
        this.baseImage = baseImage;
        this.currentResult = result;
        updateLegend();
        updateRendering();
    }

    private void updateRendering() {
        if (baseImage == null && (currentResult == null || currentResult.getMaskData() == null)) {
            imageView.setImage(null);
            return;
        }

        Image toRender = null;
        double alpha = opacitySlider.getValue();

        switch (currentMode) {
            case ORIGINAL_ONLY -> toRender = baseImage;
            case MASK_ONLY -> {
                if (currentResult != null && currentResult.getMaskData() != null) {
                    toRender = renderer.renderMaskImage(currentResult);
                } else {
                    toRender = baseImage;
                }
            }
            case OVERLAY -> {
                if (currentResult != null && currentResult.getMaskData() != null) {
                    toRender = renderer.renderOverlay(baseImage, currentResult, alpha);
                } else {
                    toRender = baseImage;
                }
            }
        }

        if (toRender != null) {
            double yaw = yawSlider.getValue();
            renderedComposite = renderer.renderWithHorizontalOffset(toRender, yaw);
            imageView.setImage(renderedComposite);
        }
    }

    private void updateLegend() {
        legendPane.getChildren().clear();
        if (currentResult == null || currentResult.getDetectedClasses() == null) {
            return;
        }

        List<SemanticClass> classes = currentResult.getDetectedClasses();
        long totalPixels = currentResult.getMaskData() != null ?
                (long) currentResult.getMaskData().getWidth() * currentResult.getMaskData().getHeight() : 1L;

        for (SemanticClass sc : classes) {
            HBox item = new HBox(6);
            item.setAlignment(Pos.CENTER_LEFT);

            Rectangle swatch = new Rectangle(14, 14);
            swatch.setFill(Color.rgb(sc.getRed(), sc.getGreen(), sc.getBlue()));
            swatch.setStroke(Color.web("#94a3b8"));
            swatch.setArcWidth(4);
            swatch.setArcHeight(4);

            double pct = totalPixels > 0 ? (sc.getPixelCount() * 100.0 / totalPixels) : 0.0;
            Label label = new Label(String.format("%s (%.1f%%)", sc.getName(), pct));
            label.setStyle("-fx-font-size: 11px;");

            item.getChildren().addAll(swatch, label);
            legendPane.getChildren().add(item);
        }
    }

    private void setupPixelInspection() {
        imageView.setOnMouseMoved(e -> {
            if (renderedComposite == null || currentResult == null || currentResult.getMaskData() == null) {
                return;
            }

            double ivW = imageView.getBoundsInLocal().getWidth();
            double ivH = imageView.getBoundsInLocal().getHeight();
            if (ivW <= 0 || ivH <= 0) return;

            double origW = renderedComposite.getWidth();
            double origH = renderedComposite.getHeight();

            int imgX = (int) ((e.getX() / ivW) * origW);
            int imgY = (int) ((e.getY() / ivH) * origH);

            if (imgX >= 0 && imgX < origW && imgY >= 0 && imgY < origH) {
                // Reverse yaw calculation to map back to original mask coordinates
                double yaw = yawSlider.getValue();
                double normalizedYaw = ((yaw % 360.0) + 360.0) % 360.0;
                int shiftPixels = (int) Math.round((normalizedYaw / 360.0) * origW);
                int maskX = ((imgX + shiftPixels) % (int) origW);

                MaskData mask = currentResult.getMaskData();
                int mX = (int) ((double) maskX / origW * mask.getWidth());
                int mY = (int) ((double) imgY / origH * mask.getHeight());
                mX = Math.clamp(mX, 0, mask.getWidth() - 1);
                mY = Math.clamp(mY, 0, mask.getHeight() - 1);

                int classId = mask.getLabel(mX, mY);
                String className = currentResult.findClassById(classId)
                        .map(SemanticClass::getName)
                        .orElse("Class " + classId);

                pixelInfoLabel.setText(String.format(
                        "Panorama Position: (%d, %d) | Equirectangular Yaw: %.1f° | Detected Segment: %s (ID: %d)",
                        mX, mY, (mX * 360.0 / mask.getWidth()), className, classId
                ));
            }
        });
    }

    public ImageView getImageView() {
        return imageView;
    }

    public Slider getYawSlider() {
        return yawSlider;
    }
}
