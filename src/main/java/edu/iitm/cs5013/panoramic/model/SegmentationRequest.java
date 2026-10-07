package edu.iitm.cs5013.panoramic.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Encapsulates a request for running segmentation on an equirectangular panoramic image.
 */
public final class SegmentationRequest {
    private final String imagePath;
    private final String toolId;
    private final SegmentationType requestedType;
    private final Map<String, Object> parameters;
    private final long timeoutSeconds;

    @JsonCreator
    public SegmentationRequest(
            @JsonProperty("imagePath") String imagePath,
            @JsonProperty("toolId") String toolId,
            @JsonProperty("requestedType") SegmentationType requestedType,
            @JsonProperty("parameters") Map<String, Object> parameters,
            @JsonProperty("timeoutSeconds") Long timeoutSeconds) {
        this.imagePath = Objects.requireNonNull(imagePath, "imagePath must not be null");
        this.toolId = Objects.requireNonNull(toolId, "toolId must not be null");
        this.requestedType = requestedType != null ? requestedType : SegmentationType.SEMANTIC;
        this.parameters = parameters != null ? Collections.unmodifiableMap(new HashMap<>(parameters)) : Collections.emptyMap();
        this.timeoutSeconds = (timeoutSeconds != null && timeoutSeconds > 0) ? timeoutSeconds : 60L;
    }

    public SegmentationRequest(String imagePath, String toolId) {
        this(imagePath, toolId, SegmentationType.SEMANTIC, Collections.emptyMap(), 60L);
    }

    public String getImagePath() {
        return imagePath;
    }

    public Path getImageNioPath() {
        return Paths.get(imagePath);
    }

    public String getToolId() {
        return toolId;
    }

    public SegmentationType getRequestedType() {
        return requestedType;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public long getTimeoutSeconds() {
        return timeoutSeconds;
    }

    /**
     * Validates that the input image path exists, is readable, and has an allowed format.
     */
    public void validate() {
        if (imagePath.isBlank()) {
            throw new IllegalArgumentException("Image path cannot be blank");
        }
        Path path = getImageNioPath();
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Panoramic image file does not exist: " + imagePath);
        }
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("Panoramic image path is not a file: " + imagePath);
        }
        if (!Files.isReadable(path)) {
            throw new IllegalArgumentException("Panoramic image file is not readable: " + imagePath);
        }
        String fileName = path.getFileName().toString().toLowerCase();
        if (!fileName.endsWith(".jpg") && !fileName.endsWith(".jpeg") && !fileName.endsWith(".png")) {
            throw new IllegalArgumentException("Unsupported image format: " + fileName + ". Supported: .jpg, .jpeg, .png");
        }
    }

    @Override
    public String toString() {
        return "SegmentationRequest[tool=" + toolId + ", image=" + imagePath + ", type=" + requestedType + "]";
    }
}
