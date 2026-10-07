package edu.iitm.cs5013.panoramic.comparison;

import java.util.Collections;
import java.util.Map;

/**
 * Encapsulates quantitative evaluation metrics between a prediction and ground truth.
 */
public final class EvaluationResult {
    private final boolean quantitativeValid;
    private final String invalidationReason;
    private final double pixelAccuracy;
    private final double meanIoU;
    private final double meanDice;
    private final Map<Integer, Double> perClassIoU;
    private final Map<Integer, Double> perClassDice;
    private final int totalEvaluatedPixels;

    public EvaluationResult(
            boolean quantitativeValid,
            String invalidationReason,
            double pixelAccuracy,
            double meanIoU,
            double meanDice,
            Map<Integer, Double> perClassIoU,
            Map<Integer, Double> perClassDice,
            int totalEvaluatedPixels) {
        this.quantitativeValid = quantitativeValid;
        this.invalidationReason = invalidationReason != null ? invalidationReason : "";
        this.pixelAccuracy = pixelAccuracy;
        this.meanIoU = meanIoU;
        this.meanDice = meanDice;
        this.perClassIoU = perClassIoU != null ? Collections.unmodifiableMap(perClassIoU) : Collections.emptyMap();
        this.perClassDice = perClassDice != null ? Collections.unmodifiableMap(perClassDice) : Collections.emptyMap();
        this.totalEvaluatedPixels = totalEvaluatedPixels;
    }

    public static EvaluationResult invalid(String reason) {
        return new EvaluationResult(false, reason, 0.0, 0.0, 0.0, Map.of(), Map.of(), 0);
    }

    public boolean isQuantitativeValid() {
        return quantitativeValid;
    }

    public String getInvalidationReason() {
        return invalidationReason;
    }

    public double getPixelAccuracy() {
        return pixelAccuracy;
    }

    public double getMeanIoU() {
        return meanIoU;
    }

    public double getMeanDice() {
        return meanDice;
    }

    public Map<Integer, Double> getPerClassIoU() {
        return perClassIoU;
    }

    public Map<Integer, Double> getPerClassDice() {
        return perClassDice;
    }

    public int getTotalEvaluatedPixels() {
        return totalEvaluatedPixels;
    }

    public String getFormattedSummary() {
        if (!quantitativeValid) {
            return "Quantitative evaluation unavailable: " + invalidationReason;
        }
        return String.format(
                "Evaluation Metrics:\n" +
                "• Pixel Accuracy: %.2f%%\n" +
                "• Mean IoU (mIoU): %.4f (%.2f%%)\n" +
                "• Mean Dice (F1):  %.4f\n" +
                "• Evaluated Classes: %d\n" +
                "• Total Pixels: %d",
                pixelAccuracy * 100.0,
                meanIoU, meanIoU * 100.0,
                meanDice,
                perClassIoU.size(),
                totalEvaluatedPixels
        );
    }

    @Override
    public String toString() {
        return getFormattedSummary();
    }
}
