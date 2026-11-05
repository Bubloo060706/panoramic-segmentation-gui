package edu.iitm.cs5013.panoramic.comparison;

import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;

import java.util.*;

/**
 * Calculates quantitative segmentation evaluation metrics:
 * <ul>
 *   <li><b>mIoU:</b> Mean Intersection over Union</li>
 *   <li><b>mAcc:</b> Mean Class Accuracy</li>
 *   <li><b>Boundary IoU:</b> Morphological boundary band IoU</li>
 *   <li><b>Boundary F-score:</b> Precision/Recall harmonic mean on boundaries</li>
 *   <li><b>Pixel Accuracy:</b> Overall correct pixel ratio</li>
 *   <li><b>Mean Dice (F1):</b> Macro-averaged Dice coefficient</li>
 * </ul>
 * Strictly guards against computing invalid metrics when ground truth is missing, dimensions mismatch,
 * or label definitions are incompatible.
 */
public class MetricsCalculator {

    private final double boundaryDilationRatio;
    private final int ignoreIndex;

    public MetricsCalculator() {
        this(0.0075, 255);
    }

    public MetricsCalculator(double boundaryDilationRatio, int ignoreIndex) {
        this.boundaryDilationRatio = boundaryDilationRatio;
        this.ignoreIndex = ignoreIndex;
    }

    /**
     * Evaluates a prediction result against a reference ground-truth result.
     *
     * @param prediction The model segmentation result to evaluate.
     * @param groundTruth The ground-truth reference segmentation result.
     * @return EvaluationResult containing quantitative metrics or invalidation rationale.
     */
    public EvaluationResult evaluate(SegmentationResult prediction, SegmentationResult groundTruth) {
        if (groundTruth == null) {
            return EvaluationResult.invalid("Ground truth segmentation mask is missing");
        }
        if (prediction == null) {
            return EvaluationResult.invalid("Prediction segmentation result is missing");
        }
        if (!prediction.isSuccess() || prediction.getMaskData() == null) {
            return EvaluationResult.invalid("Prediction execution was not successful or produced no mask");
        }
        if (!groundTruth.isSuccess() || groundTruth.getMaskData() == null) {
            return EvaluationResult.invalid("Ground truth mask data is unavailable");
        }

        MaskData predMask = prediction.getMaskData();
        MaskData gtMask = groundTruth.getMaskData();

        // 1. Dimension validation
        if (predMask.getWidth() != gtMask.getWidth() || predMask.getHeight() != gtMask.getHeight()) {
            return EvaluationResult.invalid(String.format(
                    "Mismatched dimensions: prediction is %dx%d but ground truth is %dx%d",
                    predMask.getWidth(), predMask.getHeight(), gtMask.getWidth(), gtMask.getHeight()
            ));
        }

        // 2. Compatible label set validation
        Set<Integer> predClasses = new HashSet<>();
        for (SemanticClass c : prediction.getDetectedClasses()) {
            predClasses.add(c.getId());
        }
        Set<Integer> gtClasses = new HashSet<>();
        for (SemanticClass c : groundTruth.getDetectedClasses()) {
            gtClasses.add(c.getId());
        }

        // Check if there is common label space
        Set<Integer> commonClasses = new HashSet<>(predClasses);
        commonClasses.retainAll(gtClasses);
        if (commonClasses.isEmpty() && !predClasses.isEmpty() && !gtClasses.isEmpty()) {
            return EvaluationResult.invalid(
                    "Incompatible label sets: prediction classes " + predClasses +
                    " do not overlap with ground truth classes " + gtClasses
            );
        }

        int[] predLabels = predMask.getLabels();
        int[] gtLabels = gtMask.getLabels();
        int totalPixels = predLabels.length;

        // Collect all distinct classes that appear in either ground truth or prediction
        Set<Integer> activeClasses = new TreeSet<>();
        int maxClassId = 0;
        for (int l : predLabels) {
            if (l != ignoreIndex && l >= 0) {
                activeClasses.add(l);
                if (l > maxClassId) maxClassId = l;
            }
        }
        for (int l : gtLabels) {
            if (l != ignoreIndex && l >= 0) {
                activeClasses.add(l);
                if (l > maxClassId) maxClassId = l;
            }
        }

        if (activeClasses.isEmpty()) {
            return EvaluationResult.invalid("No segmentation classes found in masks");
        }

        // Calculate Pixel Accuracy and Dice
        long correctPixels = 0;
        long validPixels = 0;
        Map<Integer, Long> tpMap = new HashMap<>();
        Map<Integer, Long> fpMap = new HashMap<>();
        Map<Integer, Long> fnMap = new HashMap<>();

        for (int c : activeClasses) {
            tpMap.put(c, 0L);
            fpMap.put(c, 0L);
            fnMap.put(c, 0L);
        }

        for (int i = 0; i < totalPixels; i++) {
            int p = predLabels[i];
            int g = gtLabels[i];

            if (g == ignoreIndex || g < 0) {
                continue;
            }

            validPixels++;
            if (p == g) {
                correctPixels++;
                if (tpMap.containsKey(p)) {
                    tpMap.put(p, tpMap.get(p) + 1L);
                }
            } else {
                if (fpMap.containsKey(p)) {
                    fpMap.put(p, fpMap.get(p) + 1L);
                }
                if (fnMap.containsKey(g)) {
                    fnMap.put(g, fnMap.get(g) + 1L);
                }
            }
        }

        double pixelAccuracy = validPixels > 0 ? ((double) correctPixels / validPixels) : 0.0;

        Map<Integer, Double> perClassDice = new TreeMap<>();
        double diceSum = 0.0;
        int evaluatedDiceClasses = 0;

        for (int c : activeClasses) {
            long tp = tpMap.getOrDefault(c, 0L);
            long fp = fpMap.getOrDefault(c, 0L);
            long fn = fnMap.getOrDefault(c, 0L);
            long denom = 2 * tp + fp + fn;

            if (denom > 0) {
                double dice = (double) (2 * tp) / denom;
                perClassDice.put(c, dice);
                diceSum += dice;
                evaluatedDiceClasses++;
            }
        }
        double mDice = evaluatedDiceClasses > 0 ? (diceSum / evaluatedDiceClasses) : 0.0;

        // Run full boundary-aware segmentation metrics
        int numClasses = Math.max(maxClassId + 1, 21);
        SegmentationMetrics metrics = new SegmentationMetrics(numClasses, ignoreIndex, boundaryDilationRatio);
        metrics.update(predLabels, gtLabels, gtMask.getHeight(), gtMask.getWidth());
        SegmentationMetrics.MetricsSummary summary = metrics.compute();

        return new EvaluationResult(
                true,
                null,
                pixelAccuracy,
                summary.mIoU(),
                summary.mAcc(),
                summary.boundaryIoU(),
                summary.boundaryFScore(),
                mDice,
                summary.classIoU(),
                summary.classAcc(),
                summary.classBoundaryIoU(),
                summary.classBoundaryFScore(),
                perClassDice,
                (int) validPixels
        );
    }
}
