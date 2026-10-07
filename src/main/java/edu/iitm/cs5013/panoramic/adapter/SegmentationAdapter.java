package edu.iitm.cs5013.panoramic.adapter;

import edu.iitm.cs5013.panoramic.config.ToolConfig;
import edu.iitm.cs5013.panoramic.model.SegmentationException;
import edu.iitm.cs5013.panoramic.model.SegmentationRequest;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;

/**
 * Common adapter contract for model-specific preprocessing, external execution,
 * post-processing, and error translation.
 *
 * @param <RAW_OUTPUT> The model-specific raw output type before normalization.
 */
public interface SegmentationAdapter<RAW_OUTPUT> extends SegmentationTool {

    /**
     * Performs model-specific input preprocessing (e.g. verifying input format,
     * staging input files, preparing CLI flags or JSON payloads).
     *
     * @param request the segmentation request
     * @throws SegmentationException if preprocessing fails
     */
    void preprocess(SegmentationRequest request) throws SegmentationException;

    /**
     * Executes the external model process or HTTP service and returns raw output.
     *
     * @param request the segmentation request
     * @return raw model output
     * @throws SegmentationException if invocation fails
     */
    RAW_OUTPUT executeRaw(SegmentationRequest request) throws SegmentationException;

    /**
     * Converts model-specific raw output into the canonical SegmentationResult.
     *
     * @param rawOutput the raw output from the external tool
     * @param request the original request
     * @return canonical SegmentationResult
     * @throws SegmentationException if post-processing fails
     */
    SegmentationResult postprocess(RAW_OUTPUT rawOutput, SegmentationRequest request) throws SegmentationException;

    /**
     * Translates runtime exceptions or process failures into domain-specific exceptions.
     *
     * @param t the caught exception
     * @param request the request being processed
     * @return a structured SegmentationException
     */
    SegmentationException translateError(Throwable t, SegmentationRequest request);

    @Override
    default SegmentationResult segment(SegmentationRequest request) throws SegmentationException {
        try {
            preprocess(request);
            RAW_OUTPUT raw = executeRaw(request);
            return postprocess(raw, request);
        } catch (Throwable t) {
            throw translateError(t, request);
        }
    }
}
