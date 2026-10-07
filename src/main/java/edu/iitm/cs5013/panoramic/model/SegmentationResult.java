package edu.iitm.cs5013.panoramic.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.*;

/**
 * Canonical representation of a segmentation result consumed by the GUI and comparison layers.
 * Preserves dimensions, masks, class mappings, execution status, and explicitly records
 * any unsupported or unmapped tool attributes.
 */
public final class SegmentationResult {
    private final String toolId;
    private final String toolName;
    private final ExecutionStatus status;
    private final String errorMessage;
    private final SegmentationType segmentationType;
    private final int width;
    private final int height;
    private final MaskData maskData;
    private final List<SemanticClass> detectedClasses;
    private final Map<String, Object> metadata;
    private final Map<String, Object> unsupportedAttributes;
    private final List<String> warnings;
    private final long executionDurationMs;
    private final Instant timestamp;

    @JsonCreator
    public SegmentationResult(
            @JsonProperty("toolId") String toolId,
            @JsonProperty("toolName") String toolName,
            @JsonProperty("status") ExecutionStatus status,
            @JsonProperty("errorMessage") String errorMessage,
            @JsonProperty("segmentationType") SegmentationType segmentationType,
            @JsonProperty("width") int width,
            @JsonProperty("height") int height,
            @JsonProperty("maskData") MaskData maskData,
            @JsonProperty("detectedClasses") List<SemanticClass> detectedClasses,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("unsupportedAttributes") Map<String, Object> unsupportedAttributes,
            @JsonProperty("warnings") List<String> warnings,
            @JsonProperty("executionDurationMs") long executionDurationMs,
            @JsonProperty("timestamp") Instant timestamp) {
        this.toolId = toolId != null ? toolId : "unknown";
        this.toolName = toolName != null ? toolName : this.toolId;
        this.status = status != null ? status : ExecutionStatus.SUCCESS;
        this.errorMessage = errorMessage;
        this.segmentationType = segmentationType != null ? segmentationType : SegmentationType.SEMANTIC;
        this.width = width;
        this.height = height;
        this.maskData = maskData;
        this.detectedClasses = detectedClasses != null ? Collections.unmodifiableList(new ArrayList<>(detectedClasses)) : Collections.emptyList();
        this.metadata = metadata != null ? Collections.unmodifiableMap(new HashMap<>(metadata)) : Collections.emptyMap();
        this.unsupportedAttributes = unsupportedAttributes != null ? Collections.unmodifiableMap(new HashMap<>(unsupportedAttributes)) : Collections.emptyMap();
        this.warnings = warnings != null ? Collections.unmodifiableList(new ArrayList<>(warnings)) : Collections.emptyList();
        this.executionDurationMs = executionDurationMs;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    public static SegmentationResult failure(String toolId, String toolName, String errorMessage, ExecutionStatus status) {
        return new Builder()
                .toolId(toolId)
                .toolName(toolName)
                .status(status != null ? status : ExecutionStatus.FAILED)
                .errorMessage(errorMessage)
                .build();
    }

    public static SegmentationResult failure(String toolId, String toolName, String errorMessage) {
        return failure(toolId, toolName, errorMessage, ExecutionStatus.FAILED);
    }

    public String getToolId() {
        return toolId;
    }

    public String getToolName() {
        return toolName;
    }

    public ExecutionStatus getStatus() {
        return status;
    }

    @JsonIgnore
    public boolean isSuccess() {
        return status == ExecutionStatus.SUCCESS && maskData != null;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public SegmentationType getSegmentationType() {
        return segmentationType;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public MaskData getMaskData() {
        return maskData;
    }

    public List<SemanticClass> getDetectedClasses() {
        return detectedClasses;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Map<String, Object> getUnsupportedAttributes() {
        return unsupportedAttributes;
    }

    public List<String> getWarnings() {
        return warnings;
    }

    public long getExecutionDurationMs() {
        return executionDurationMs;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public Optional<SemanticClass> findClassById(int id) {
        return detectedClasses.stream().filter(c -> c.getId() == id).findFirst();
    }

    public static class Builder {
        private String toolId = "unknown";
        private String toolName = "Unknown Tool";
        private ExecutionStatus status = ExecutionStatus.SUCCESS;
        private String errorMessage;
        private SegmentationType segmentationType = SegmentationType.SEMANTIC;
        private int width;
        private int height;
        private MaskData maskData;
        private List<SemanticClass> detectedClasses = new ArrayList<>();
        private Map<String, Object> metadata = new HashMap<>();
        private Map<String, Object> unsupportedAttributes = new HashMap<>();
        private List<String> warnings = new ArrayList<>();
        private long executionDurationMs;
        private Instant timestamp = Instant.now();

        public Builder toolId(String toolId) {
            this.toolId = toolId;
            return this;
        }

        public Builder toolName(String toolName) {
            this.toolName = toolName;
            return this;
        }

        public Builder status(ExecutionStatus status) {
            this.status = status;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public Builder segmentationType(SegmentationType type) {
            this.segmentationType = type;
            return this;
        }

        public Builder dimensions(int width, int height) {
            this.width = width;
            this.height = height;
            return this;
        }

        public Builder maskData(MaskData maskData) {
            this.maskData = maskData;
            if (maskData != null) {
                this.width = maskData.getWidth();
                this.height = maskData.getHeight();
            }
            return this;
        }

        public Builder detectedClasses(List<SemanticClass> classes) {
            this.detectedClasses = classes != null ? new ArrayList<>(classes) : new ArrayList<>();
            return this;
        }

        public Builder addClass(SemanticClass sc) {
            if (sc != null) {
                this.detectedClasses.add(sc);
            }
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata != null ? new HashMap<>(metadata) : new HashMap<>();
            return this;
        }

        public Builder addMetadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder addUnsupportedAttribute(String key, Object value) {
            this.unsupportedAttributes.put(key, value);
            return this;
        }

        public Builder unsupportedAttributes(Map<String, Object> unmapped) {
            this.unsupportedAttributes = unmapped != null ? new HashMap<>(unmapped) : new HashMap<>();
            return this;
        }

        public Builder addWarning(String warning) {
            if (warning != null && !warning.isBlank()) {
                this.warnings.add(warning);
            }
            return this;
        }

        public Builder executionDurationMs(long durationMs) {
            this.executionDurationMs = durationMs;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public SegmentationResult build() {
            return new SegmentationResult(
                    toolId, toolName, status, errorMessage, segmentationType,
                    width, height, maskData, detectedClasses, metadata,
                    unsupportedAttributes, warnings, executionDurationMs, timestamp
            );
        }
    }
}
