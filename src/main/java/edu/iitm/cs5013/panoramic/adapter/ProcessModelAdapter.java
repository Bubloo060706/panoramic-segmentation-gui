package edu.iitm.cs5013.panoramic.adapter;

import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.execution.ProcessExecutionException;
import edu.iitm.cs5013.panoramic.execution.ProcessExecutionResult;
import edu.iitm.cs5013.panoramic.execution.ProcessExecutor;
import edu.iitm.cs5013.panoramic.model.*;
import edu.iitm.cs5013.panoramic.normalization.ResultNormalizer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Adapter for external CLI/Python-based segmentation models executed via ProcessBuilder.
 */
public class ProcessModelAdapter implements SegmentationAdapter<String> {

    private final ToolMetadata metadata;
    private final ToolConfig config;
    private final ProcessExecutor executor;
    private final ResultNormalizer normalizer;
    private int cachedInputWidth = -1;
    private int cachedInputHeight = -1;
    private Path tempOutputFile;

    public ProcessModelAdapter(ToolMetadata metadata, ToolConfig config, ProcessExecutor executor, ResultNormalizer normalizer) {
        this.metadata = Objects.requireNonNull(metadata, "metadata cannot be null");
        this.config = Objects.requireNonNull(config, "config cannot be null");
        this.executor = executor != null ? executor : new ProcessExecutor();
        this.normalizer = normalizer != null ? normalizer : new ResultNormalizer();
    }

    public ProcessModelAdapter(ToolConfig config) {
        this(
                new ToolMetadata(
                        config.getId(),
                        config.getName(),
                        config.getDescription(),
                        "1.0.0",
                        Set.of(SegmentationType.SEMANTIC),
                        Set.of("jpg", "jpeg", "png")
                ),
                config,
                new ProcessExecutor(),
                new ResultNormalizer()
        );
    }

    @Override
    public ToolMetadata getMetadata() {
        return metadata;
    }

    @Override
    public ToolConfig getConfiguration() {
        return config;
    }

    @Override
    public boolean isAvailable() {
        if (!config.isEnabled() || config.getCommandTemplate().isEmpty()) {
            return false;
        }
        String executable = config.getCommandTemplate().get(0);
        // Basic check if executable exists or command is runnable
        try {
            ProcessExecutionResult res = executor.execute(List.of(executable, "--version"), null, null, 3);
            return res.isSuccess() || res.getExitCode() == 0;
        } catch (Exception e) {
            // Check if it's a file path
            return Files.exists(Path.of(executable));
        }
    }

    @Override
    public void preprocess(SegmentationRequest request) throws SegmentationException {
        request.validate();
        try {
            BufferedImage img = ImageIO.read(request.getImageNioPath().toFile());
            if (img == null) {
                throw new SegmentationException(config.getId(), "Unable to decode panoramic image: " + request.getImagePath());
            }
            this.cachedInputWidth = img.getWidth();
            this.cachedInputHeight = img.getHeight();

            // Create temporary file for model output JSON
            this.tempOutputFile = Files.createTempFile("panoseg_out_", ".json");
        } catch (IOException e) {
            throw new SegmentationException(config.getId(), "Failed to preprocess image: " + e.getMessage(), e);
        }
    }

    @Override
    public String executeRaw(SegmentationRequest request) throws SegmentationException {
        Map<String, String> vars = new HashMap<>();
        vars.put("IMAGE_PATH", request.getImageNioPath().toAbsolutePath().toString());
        vars.put("OUTPUT_PATH", tempOutputFile.toAbsolutePath().toString());
        vars.put("TOOL_DIR", config.getWorkingDirectory());

        List<String> command = ProcessExecutor.buildCommand(config.getCommandTemplate(), vars);
        Path workDir = config.getWorkingDirectory().isBlank() ? null : Path.of(config.getWorkingDirectory());

        try {
            ProcessExecutionResult execRes = executor.execute(command, workDir, null, config.getTimeoutSeconds());
            if (!Files.exists(tempOutputFile) || Files.size(tempOutputFile) == 0) {
                // If the model printed output to stdout instead of file
                if (!execRes.getStdout().isBlank()) {
                    return execRes.getStdout();
                }
                throw new SegmentationException(config.getId(), "Model completed but generated empty output file: " + tempOutputFile);
            }
            return Files.readString(tempOutputFile);
        } catch (ProcessExecutionException pe) {
            throw translateError(pe, request);
        } catch (IOException e) {
            throw new SegmentationException(config.getId(), "Failed to read output file: " + e.getMessage(), e);
        } finally {
            try {
                if (tempOutputFile != null) {
                    Files.deleteIfExists(tempOutputFile);
                }
            } catch (IOException ignored) {}
        }
    }

    @Override
    public SegmentationResult postprocess(String rawOutput, SegmentationRequest request) throws SegmentationException {
        return normalizer.normalizeFromJson(rawOutput, config, cachedInputWidth, cachedInputHeight);
    }

    @Override
    public SegmentationException translateError(Throwable t, SegmentationRequest request) {
        if (t instanceof SegmentationException se) {
            return se;
        }
        if (t instanceof ProcessExecutionException pe) {
            ExecutionStatus status = pe.isTimedOut() ? ExecutionStatus.TIMED_OUT : ExecutionStatus.FAILED;
            String msg = "External model process failed (exit code " + pe.getExitCode() + "): " + pe.getMessage();
            if (!pe.getStderr().isBlank()) {
                msg += " Stderr: " + pe.getStderr().trim();
            }
            return new SegmentationException(config.getId(), msg, status, pe);
        }
        return new SegmentationException(config.getId(), "Unexpected execution error: " + t.getMessage(), ExecutionStatus.FAILED, t);
    }
}
