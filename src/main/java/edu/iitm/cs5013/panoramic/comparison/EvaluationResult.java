package edu.iitm.cs5013.panoramic.comparison;

import java.util.Collections;
import java.util.Map;

/**
 * Encapsulates quantitative evaluation metrics between a prediction and ground truth.
 * Supports mIoU, mAcc, Boundary IoU, Boundary F-score, Pixel Accuracy, and Mean Dice.
 */
public final class EvaluationResult {
    private final boolean quantitativeValid;
    private final String invalidationReason;
    private final double pixelAccuracy;
    private final double meanIoU;
    private final double meanAcc;
    private final double boundaryIoU;
    private final double boundaryFScore;
    private final double meanDice;
    private final Map<Integer, Double> perClassIoU;
    private final Map<Integer, Double> perClassAcc;
    private final Map<Integer, Double> perClassBoundaryIoU;
    private final Map<Integer, Double> perClassBoundaryFScore;
    private final Map<Integer, Double> perClassDice;
    private final int totalEvaluatedPixels;

    /**
     * Legacy constructor for backwards compatibility with earlier tests.
     */
    public EvaluationResult(
            boolean quantitativeValid,
            String invalidationReason,
            double pixelAccuracy,
            double meanIoU,
            double meanDice,
            Map<Integer, Double> perClassIoU,
            Map<Integer, Double> perClassDice,
            int totalEvaluatedPixels) {
        this(quantitativeValid, invalidationReason, pixelAccuracy, meanIoU,
                pixelAccuracy, meanIoU, meanDice, meanDice,
                perClassIoU, Map.of(), Map.of(), Map.of(), perClassDice, totalEvaluatedPixels);
    }

    /**
     * Full constructor including all standard and boundary-aware metrics.
     */
    public EvaluationResult(
            boolean quantitativeValid,
            String invalidationReason,
            double pixelAccuracy,
            double meanIoU,
            double meanAcc,
            double boundaryIoU,
            double boundaryFScore,
            double meanDice,
            Map<Integer, Double> perClassIoU,
            Map<Integer, Double> perClassAcc,
            Map<Integer, Double> perClassBoundaryIoU,
            Map<Integer, Double> perClassBoundaryFScore,
            Map<Integer, Double> perClassDice,
            int totalEvaluatedPixels) {
        this.quantitativeValid = quantitativeValid;
        this.invalidationReason = invalidationReason != null ? invalidationReason : "";
        this.pixelAccuracy = pixelAccuracy;
        this.meanIoU = meanIoU;
        this.meanAcc = meanAcc;
        this.boundaryIoU = boundaryIoU;
        this.boundaryFScore = boundaryFScore;
        this.meanDice = meanDice;
        this.perClassIoU = perClassIoU != null ? Collections.unmodifiableMap(perClassIoU) : Collections.emptyMap();
        this.perClassAcc = perClassAcc != null ? Collections.unmodifiableMap(perClassAcc) : Collections.emptyMap();
        this.perClassBoundaryIoU = perClassBoundaryIoU != null ? Collections.unmodifiableMap(perClassBoundaryIoU) : Collections.emptyMap();
        this.perClassBoundaryFScore = perClassBoundaryFScore != null ? Collections.unmodifiableMap(perClassBoundaryFScore) : Collections.emptyMap();
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

    public double getPixelAccuracyPercent() {
        return pixelAccuracy * 100.0;
    }

    public double getMeanIoU() {
        return meanIoU;
    }

    public double getMeanIoUPercent() {
        return meanIoU * 100.0;
    }

    public double getMeanAcc() {
        return meanAcc;
    }

    public double getMeanAccPercent() {
        return meanAcc * 100.0;
    }

    public double getBoundaryIoU() {
        return boundaryIoU;
    }

    public double getBoundaryIoUPercent() {
        return boundaryIoU * 100.0;
    }

    public double getBoundaryFScore() {
        return boundaryFScore;
    }

    public double getBoundaryFScorePercent() {
        return boundaryFScore * 100.0;
    }

    public double getMeanDice() {
        return meanDice;
    }

    public Map<Integer, Double> getPerClassIoU() {
        return perClassIoU;
    }

    public Map<Integer, Double> getPerClassAcc() {
        return perClassAcc;
    }

    public Map<Integer, Double> getPerClassBoundaryIoU() {
        return perClassBoundaryIoU;
    }

    public Map<Integer, Double> getPerClassBoundaryFScore() {
        return perClassBoundaryFScore;
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
                "• Mean IoU (mIoU):       %.2f%% (%.4f)\n" +
                "• Mean Accuracy (mAcc):  %.2f%%\n" +
                "• Boundary IoU:          %.2f%%\n" +
                "• Boundary F-score (F1): %.2f%%\n" +
                "• Pixel Accuracy:        %.2f%%\n" +
                "• Mean Dice:             %.4f\n" +
                "• Evaluated Classes:     %d\n" +
                "• Total Evaluated Pixels:%d",
                meanIoU * 100.0, meanIoU,
                meanAcc * 100.0,
                boundaryIoU * 100.0,
                boundaryFScore * 100.0,
                pixelAccuracy * 100.0,
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
