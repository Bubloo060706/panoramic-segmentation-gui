package edu.iitm.cs5013.panoramic.gui;

import edu.iitm.cs5013.panoramic.adapter.DeterministicTestAdapter;
import edu.iitm.cs5013.panoramic.adapter.HttpModelAdapter;
import edu.iitm.cs5013.panoramic.adapter.ProcessModelAdapter;
import edu.iitm.cs5013.panoramic.config.ExecutionType;
import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.config.ToolRegistry;
import edu.iitm.cs5013.panoramic.controller.AsyncSegmentationCoordinator;
import edu.iitm.cs5013.panoramic.controller.ToolExecutionListener;
import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.normalization.ResultNormalizer;
import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;
import edu.iitm.cs5013.panoramic.validation.ValidationResult;
import edu.iitm.cs5013.panoramic.validation.ValidationService;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Parent;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import edu.iitm.cs5013.panoramic.util.ImageLoaderHelper;

/**
 * Main application coordinator and JavaFX presentation layer.
 * Coordinates input loading, tool registry discovery, asynchronous inference execution,
 * visualization updates, and multi-tool comparison.
 */
public class MainViewController {

    private final Stage stage;
    private final ToolRegistry toolRegistry;
    private final ValidationService validationService;
    private final AsyncSegmentationCoordinator coordinator;
    private final ResultNormalizer normalizer;

    // UI Controls
    private BorderPane rootLayout;
    private Label inputImageLabel;
    private Label groundTruthLabel;
    private VBox toolCheckboxesBox = new VBox(6);
    private Button runButton;
    private ProgressBar progressBar;
    private Label statusMessageLabel;
    private VBox errorAlertBox;
    private Label errorAlertLabel;

    private TabPane tabPane;
    private PanoramaViewPane singleViewPane;
    private ComparisonViewPane comparisonViewPane;
    private TextArea diagnosticsArea;
    private ComboBox<String> activeToolSelector = new ComboBox<>();

    // Application State
    private Path currentImagePath;
    private Image currentJavaFxImage;
    private SegmentationResult groundTruthResult;
    private final Map<String, CheckBox> toolCheckMap = new HashMap<>();
    private final Map<String, Label> toolStatusBadges = new HashMap<>();
    private final Map<String, SegmentationResult> completedResults = new LinkedHashMap<>();

    public MainViewController(Stage stage) {
        this.stage = stage;
        this.toolRegistry = new ToolRegistry();
        this.validationService = new ValidationService();
        this.coordinator = new AsyncSegmentationCoordinator();
        this.normalizer = new ResultNormalizer();

        setupToolRegistry();
        buildUI();
    }

    private void setupToolRegistry() {
        // Register standard factories for PROCESS, HTTP, MOCK
        toolRegistry.registerFactory(ExecutionType.PROCESS, ProcessModelAdapter::new);
        toolRegistry.registerFactory(ExecutionType.HTTP, HttpModelAdapter::new);
        toolRegistry.registerFactory(ExecutionType.MOCK, config ->
                new DeterministicTestAdapter(config.getId(), config.getName(), false));

        // Load configuration from config/tools.json if present
        Path cfgPath = Path.of("config/tools.json");
        if (Files.exists(cfgPath)) {
            try {
                toolRegistry.loadAndRegisterFromConfig(cfgPath);
            } catch (Exception e) {
                System.err.println("Warning: failed to load config/tools.json: " + e.getMessage());
            }
        }

        // Fallback default tools if config had no tools
        if (toolRegistry.size() == 0) {
            toolRegistry.registerTool(new DeterministicTestAdapter());
        }
    }

    private void buildUI() {
        rootLayout = new BorderPane();

        // 1. Top Header Bar
        VBox header = new VBox(2);
        header.getStyleClass().add("header-bar");
        Label title = new Label("360° Panoramic Image Segmentation Framework");
        title.getStyleClass().add("header-title");
        Label subtitle = new Label("CS5013: Modular Multi-Tool Inference, Normalization & Evaluation");
        subtitle.getStyleClass().add("header-subtitle");
        header.getChildren().addAll(title, subtitle);
        rootLayout.setTop(header);

        // 2. Left Sidebar
        VBox sidebar = buildSidebar();
        rootLayout.setLeft(sidebar);

        // 3. Center Content Tabs
        tabPane = new TabPane();
        tabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);

        // Tab 1: Single Panorama View
        singleViewPane = new PanoramaViewPane();
        Tab tab1 = new Tab("Panorama & Mask Overlay", singleViewPane);

        // Tab 2: Comparison View
        comparisonViewPane = new ComparisonViewPane();
        Tab tab2 = new Tab("Cross-Model Comparison", comparisonViewPane);

        // Tab 3: Diagnostics & Metadata
        diagnosticsArea = new TextArea();
        diagnosticsArea.setEditable(false);
        diagnosticsArea.setWrapText(true);
        diagnosticsArea.setStyle("-fx-font-family: monospace; -fx-font-size: 12px;");
        Tab tab3 = new Tab("Metadata & Canonical Result", new VBox(10, diagnosticsArea));

        tabPane.getTabs().addAll(tab1, tab2, tab3);
        rootLayout.setCenter(tabPane);
    }

    private VBox buildSidebar() {
        VBox sidebar = new VBox(14);
        sidebar.setPrefWidth(300);
        sidebar.getStyleClass().add("sidebar");

        // Section A: Input Image
        VBox secImage = new VBox(6);
        Label hdrImg = new Label("1. Panoramic Image");
        hdrImg.getStyleClass().add("section-header");
        Button btnLoadImg = new Button("Select 360° Panorama...");
        btnLoadImg.getStyleClass().add("btn-secondary");
        btnLoadImg.setMaxWidth(Double.MAX_VALUE);
        btnLoadImg.setOnAction(e -> handleSelectImage());
        inputImageLabel = new Label("No panorama loaded");
        inputImageLabel.setStyle("-fx-text-fill: #64748b; -fx-font-size: 11px;");
        inputImageLabel.setWrapText(true);
        secImage.getChildren().addAll(hdrImg, btnLoadImg, inputImageLabel);

        // Section B: Ground Truth (Optional)
        VBox secGt = new VBox(6);
        Label hdrGt = new Label("2. Ground Truth Mask (Optional)");
        hdrGt.getStyleClass().add("section-header");
        Button btnLoadGt = new Button("Load Ground Truth Mask...");
        btnLoadGt.getStyleClass().add("btn-secondary");
        btnLoadGt.setMaxWidth(Double.MAX_VALUE);
        btnLoadGt.setOnAction(e -> handleSelectGroundTruth());
        groundTruthLabel = new Label("No ground truth loaded (visual comparison only)");
        groundTruthLabel.setStyle("-fx-text-fill: #64748b; -fx-font-size: 11px;");
        groundTruthLabel.setWrapText(true);
        secGt.getChildren().addAll(hdrGt, btnLoadGt, groundTruthLabel);

        // Section C: Registered Tools
        VBox secTools = new VBox(6);
        Label hdrTools = new Label("3. Select Segmentation Tools");
        hdrTools.getStyleClass().add("section-header");
        populateToolCheckboxes();
        secTools.getChildren().addAll(hdrTools, toolCheckboxesBox);

        // Section D: Active View Selector (for Tab 1)
        VBox secActive = new VBox(4);
        Label hdrActive = new Label("Active Display Tool:");
        hdrActive.setStyle("-fx-font-weight: bold; -fx-font-size: 11px;");
        activeToolSelector.setMaxWidth(Double.MAX_VALUE);
        activeToolSelector.setOnAction(e -> updateActiveSingleView());
        secActive.getChildren().addAll(hdrActive, activeToolSelector);

        // Section E: Execution Actions
        VBox secAction = new VBox(8);
        runButton = new Button("Run Segmentation");
        runButton.getStyleClass().add("btn-primary");
        runButton.setMaxWidth(Double.MAX_VALUE);
        runButton.setOnAction(e -> handleRunSegmentation());

        progressBar = new ProgressBar(0);
        progressBar.setMaxWidth(Double.MAX_VALUE);
        progressBar.setVisible(false);

        statusMessageLabel = new Label("Ready");
        statusMessageLabel.setStyle("-fx-text-fill: #475569; -fx-font-size: 11px;");

        errorAlertBox = new VBox(4);
        errorAlertBox.getStyleClass().add("error-banner");
        errorAlertLabel = new Label();
        errorAlertLabel.setWrapText(true);
        errorAlertBox.getChildren().add(errorAlertLabel);
        errorAlertBox.setVisible(false);
        errorAlertBox.setManaged(false);

        secAction.getChildren().addAll(runButton, progressBar, statusMessageLabel, errorAlertBox);

        sidebar.getChildren().addAll(secImage, secGt, secTools, secActive, new Separator(), secAction);
        return sidebar;
    }

    private void populateToolCheckboxes() {
        toolCheckboxesBox.getChildren().clear();
        toolCheckMap.clear();
        toolStatusBadges.clear();
        if (activeToolSelector != null) {
            activeToolSelector.getItems().clear();
        }

        for (SegmentationTool tool : toolRegistry.getAllTools()) {
            String toolId = tool.getMetadata().getId();
            HBox row = new HBox(8);
            row.setAlignment(Pos.CENTER_LEFT);

            CheckBox cb = new CheckBox(tool.getMetadata().getName());
            cb.setSelected(true); // default selected
            toolCheckMap.put(toolId, cb);

            Label badge = new Label(tool.isAvailable() ? "Ready" : "Unavailable");
            badge.getStyleClass().add(tool.isAvailable() ? "status-badge-ready" : "status-badge-failed");
            toolStatusBadges.put(toolId, badge);

            row.getChildren().addAll(cb, badge);
            toolCheckboxesBox.getChildren().add(row);
            if (activeToolSelector != null) {
                activeToolSelector.getItems().add(toolId);
            }
        }

        if (activeToolSelector != null && !activeToolSelector.getItems().isEmpty()) {
            activeToolSelector.getSelectionModel().select(0);
        }
    }

    private void handleSelectImage() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Select Equirectangular 360° Panorama");
        fc.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Panoramic Images (*.jpg, *.jpeg, *.png)", "*.jpg", "*.jpeg", "*.png")
        );

        File initialDir = new File("data");
        if (initialDir.exists()) {
            fc.setInitialDirectory(initialDir);
        }

        File chosen = fc.showOpenDialog(stage);
        if (chosen != null) {
            loadImage(chosen.toPath());
        }
    }

    public void loadImage(Path path) {
        ValidationResult val = validationService.validatePanoramicImage(path);
        if (!val.isValid()) {
            showError("Invalid Panoramic Image:\n" + val.getFormattedErrors());
            return;
        }

        try {
            this.currentImagePath = path;
            Image loadedImage = ImageLoaderHelper.loadJavaFxImage(path);
            if (loadedImage == null || loadedImage.isError() || loadedImage.getWidth() <= 0) {
                showError("Failed to decode panoramic image in JavaFX renderer.");
                return;
            }
            this.currentJavaFxImage = loadedImage;
            clearError();

            int w = (int) currentJavaFxImage.getWidth();
            int h = (int) currentJavaFxImage.getHeight();
            String details = String.format("%s (%dx%d, aspect: %.2f:1)", path.getFileName(), w, h, (double) w / h);
            inputImageLabel.setText(details);

            if (!val.getWarnings().isEmpty()) {
                inputImageLabel.setText(details + "\n⚠ " + val.getWarnings().get(0));
            }

            // Update single view
            singleViewPane.setContent(currentJavaFxImage, null);
            comparisonViewPane.updateData(currentJavaFxImage, completedResults, groundTruthResult);
            statusMessageLabel.setText("Loaded panorama: " + path.getFileName());

        } catch (Exception e) {
            showError("Failed to load image: " + e.getMessage());
        }
    }

    private void handleSelectGroundTruth() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Select Ground Truth Mask");
        fc.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Image Mask or JSON (*.png, *.json)", "*.png", "*.json")
        );
        File chosen = fc.showOpenDialog(stage);
        if (chosen != null) {
            loadGroundTruth(chosen.toPath());
        }
    }

    public void loadGroundTruth(Path gtPath) {
        try {
            clearError();
            if (gtPath.toString().toLowerCase().endsWith(".json")) {
                String json = Files.readString(gtPath);
                ToolConfig dummyConfig = new ToolConfig("ground-truth", "Ground Truth", "",
                        ExecutionType.MOCK, List.of(), "", "", 10L, true, null, null);
                this.groundTruthResult = normalizer.normalizeFromJson(json, dummyConfig, -1, -1);
            } else {
                ToolConfig dummyConfig = new ToolConfig("ground-truth", "Ground Truth", "",
                        ExecutionType.MOCK, List.of(), "", "", 10L, true, null, null);
                MaskData mask = normalizer.loadMaskFromImageFile(gtPath, dummyConfig);
                this.groundTruthResult = new SegmentationResult.Builder()
                        .toolId("ground-truth")
                        .toolName("Ground Truth Reference")
                        .maskData(mask)
                        .dimensions(mask.getWidth(), mask.getHeight())
                        .build();
            }

            groundTruthLabel.setText("Loaded: " + gtPath.getFileName() + " (" +
                    groundTruthResult.getWidth() + "x" + groundTruthResult.getHeight() + ")");
            comparisonViewPane.updateData(currentJavaFxImage, completedResults, groundTruthResult);
            statusMessageLabel.setText("Loaded Ground Truth reference.");

        } catch (Exception e) {
            showError("Failed to load ground truth: " + e.getMessage());
        }
    }

    private void handleRunSegmentation() {
        clearError();

        if (currentImagePath == null) {
            showError("Please load a panoramic image first.");
            return;
        }

        List<SegmentationTool> selected = new ArrayList<>();
        for (Map.Entry<String, CheckBox> entry : toolCheckMap.entrySet()) {
            if (entry.getValue().isSelected()) {
                toolRegistry.getTool(entry.getKey()).ifPresent(selected::add);
            }
        }

        ValidationResult val = validationService.validateToolsSelection(selected);
        if (!val.isValid()) {
            showError(val.getFormattedErrors());
            return;
        }

        // Prepare UI state for execution
        runButton.setDisable(true);
        progressBar.setVisible(true);
        progressBar.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
        statusMessageLabel.setText("Running inference on " + selected.size() + " tool(s)...");

        for (SegmentationTool t : selected) {
            Label badge = toolStatusBadges.get(t.getMetadata().getId());
            if (badge != null) {
                badge.setText("Running");
                badge.getStyleClass().setAll("status-badge-running");
            }
        }

        ToolExecutionListener listener = new ToolExecutionListener() {
            @Override
            public void onToolStarted(String toolId) {
                Platform.runLater(() -> statusMessageLabel.setText("Tool " + toolId + " started..."));
            }

            @Override
            public void onToolProgress(String toolId, String message, double progress) {
                Platform.runLater(() -> statusMessageLabel.setText(toolId + ": " + message));
            }

            @Override
            public void onToolSuccess(String toolId, SegmentationResult result) {
                Platform.runLater(() -> {
                    Label badge = toolStatusBadges.get(toolId);
                    if (badge != null) {
                        badge.setText("Success");
                        badge.getStyleClass().setAll("status-badge-success");
                    }
                    completedResults.put(toolId, result);
                    updateActiveSingleView();
                    comparisonViewPane.updateData(currentJavaFxImage, completedResults, groundTruthResult);
                    updateDiagnostics(result);
                });
            }

            @Override
            public void onToolFailure(String toolId, String errorMessage) {
                Platform.runLater(() -> {
                    Label badge = toolStatusBadges.get(toolId);
                    if (badge != null) {
                        badge.setText("Failed");
                        badge.getStyleClass().setAll("status-badge-failed");
                    }
                    SegmentationResult failRes = SegmentationResult.failure(toolId, toolId, errorMessage);
                    completedResults.put(toolId, failRes);
                    comparisonViewPane.updateData(currentJavaFxImage, completedResults, groundTruthResult);
                    showError("Execution failed for tool '" + toolId + "':\n" + errorMessage);
                });
            }
        };

        coordinator.executeAllAsync(currentImagePath, selected, listener)
                .whenComplete((results, throwable) -> Platform.runLater(() -> {
                    runButton.setDisable(false);
                    progressBar.setVisible(false);
                    if (throwable != null) {
                        showError("Coordinator error: " + throwable.getMessage());
                        statusMessageLabel.setText("Inference aborted due to error");
                    } else {
                        statusMessageLabel.setText("Inference completed for all selected tools.");
                    }
                }));
    }

    private void updateActiveSingleView() {
        String selectedToolId = activeToolSelector.getValue();
        if (selectedToolId != null) {
            SegmentationResult res = completedResults.get(selectedToolId);
            singleViewPane.setContent(currentJavaFxImage, res);
            if (res != null) {
                updateDiagnostics(res);
            }
        }
    }

    private void updateDiagnostics(SegmentationResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== CANONICAL RESULT DIAGNOSTICS ===\n");
        sb.append("Tool: ").append(result.getToolName()).append(" (ID: ").append(result.getToolId()).append(")\n");
        sb.append("Status: ").append(result.getStatus()).append("\n");
        sb.append("Dimensions: ").append(result.getWidth()).append(" x ").append(result.getHeight()).append("\n");
        sb.append("Duration: ").append(result.getExecutionDurationMs()).append(" ms\n");
        sb.append("Detected Categories: ").append(result.getDetectedClasses().size()).append("\n");
        for (var c : result.getDetectedClasses()) {
            sb.append("  • ").append(c).append("\n");
        }
        sb.append("\n=== PRESERVED UNMAPPED / UNSUPPORTED ATTRIBUTES ===\n");
        if (result.getUnsupportedAttributes().isEmpty()) {
            sb.append("None (all attributes mapped)\n");
        } else {
            result.getUnsupportedAttributes().forEach((k, v) ->
                    sb.append("  • ").append(k).append(": ").append(v).append("\n"));
        }
        if (!result.getWarnings().isEmpty()) {
            sb.append("\n=== WARNINGS ===\n");
            for (String w : result.getWarnings()) {
                sb.append("  ⚠ ").append(w).append("\n");
            }
        }
        diagnosticsArea.setText(sb.toString());
    }

    private void showError(String msg) {
        errorAlertLabel.setText(msg);
        errorAlertBox.setVisible(true);
        errorAlertBox.setManaged(true);
    }

    private void clearError() {
        errorAlertBox.setVisible(false);
        errorAlertBox.setManaged(false);
        errorAlertLabel.setText("");
    }

    public Parent getRoot() {
        return rootLayout;
    }

    public ToolRegistry getToolRegistry() {
        return toolRegistry;
    }

    public Map<String, SegmentationResult> getCompletedResults() {
        return completedResults;
    }
}
