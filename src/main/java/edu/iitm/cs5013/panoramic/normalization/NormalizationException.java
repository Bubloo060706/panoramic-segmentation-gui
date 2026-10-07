package edu.iitm.cs5013.panoramic.normalization;

import edu.iitm.cs5013.panoramic.model.SegmentationException;

/**
 * Thrown when raw model outputs cannot be converted into canonical SegmentationResult.
 */
public class NormalizationException extends SegmentationException {
    public NormalizationException(String message) {
        super("normalizer", message);
    }

    public NormalizationException(String message, Throwable cause) {
        super("normalizer", message, cause);
    }
}
