package edu.iitm.cs5013.panoramic.segmentation;

import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.model.SegmentationException;
import edu.iitm.cs5013.panoramic.model.SegmentationRequest;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.ToolMetadata;

/**
 * Common abstraction contract for all integrated segmentation tools.
 * Exposes tool metadata, runtime configuration, and a controlled segmentation execution method.
 */
public interface SegmentationTool {

    /**
     * @return Metadata describing this tool (identifier, name, supported tasks, format).
     */
    ToolMetadata getMetadata();

    /**
     * @return Current tool configuration.
     */
    ToolConfig getConfiguration();

    /**
     * Executes segmentation on the panoramic input specified in the request.
     *
     * @param request The segmentation request.
     * @return Normalized canonical segmentation result.
     * @throws SegmentationException if execution or validation fails.
     */
    SegmentationResult segment(SegmentationRequest request) throws SegmentationException;

    /**
     * Checks whether the underlying model runtime or service is currently reachable/available.
     *
     * @return true if available, false otherwise.
     */
    boolean isAvailable();
}
