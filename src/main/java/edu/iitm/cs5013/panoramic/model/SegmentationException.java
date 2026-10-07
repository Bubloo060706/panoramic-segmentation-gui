package edu.iitm.cs5013.panoramic.model;

/**
 * Exception thrown when segmentation request, execution, or normalization fails.
 */
public class SegmentationException extends Exception {
    private final String toolId;
    private final ExecutionStatus status;

    public SegmentationException(String toolId, String message, ExecutionStatus status, Throwable cause) {
        super(message, cause);
        this.toolId = toolId;
        this.status = status != null ? status : ExecutionStatus.FAILED;
    }

    public SegmentationException(String toolId, String message, Throwable cause) {
        this(toolId, message, ExecutionStatus.FAILED, cause);
    }

    public SegmentationException(String toolId, String message) {
        this(toolId, message, ExecutionStatus.FAILED, null);
    }

    public String getToolId() {
        return toolId;
    }

    public ExecutionStatus getStatus() {
        return status;
    }
}
