package edu.iitm.cs5013.panoramic.comparison;

import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;

import java.util.*;

/**
 * Calculates quantitative segmentation evaluation metrics (mIoU, Pixel Accuracy, Dice/F1, Per-class IoU).
 * Strictly guards against computing invalid metrics when ground truth is missing, dimensions mismatch,
 * or label definitions are incompatible.
 */
public class MetricsCalculator {

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
        for (int l : predLabels) activeClasses.add(l);
        for (int l : gtLabels) activeClasses.add(l);

        if (activeClasses.isEmpty()) {
            return EvaluationResult.invalid("No segmentation classes found in masks");
        }

        long correctPixels = 0;
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

            if (p == g) {
                correctPixels++;
                tpMap.put(p, tpMap.get(p) + 1L);
            } else {
                fpMap.put(p, fpMap.get(p) + 1L);
                fnMap.put(g, fnMap.get(g) + 1L);
            }
        }

        double pixelAccuracy = (double) correctPixels / totalPixels;

        Map<Integer, Double> perClassIoU = new TreeMap<>();
        Map<Integer, Double> perClassDice = new TreeMap<>();
        double iouSum = 0.0;
        double diceSum = 0.0;
        int evaluatedClassCount = 0;

        for (int c : activeClasses) {
            long tp = tpMap.get(c);
            long fp = fpMap.get(c);
            long fn = fnMap.get(c);
            long union = tp + fp + fn;

            if (union > 0) {
                double iou = (double) tp / union;
                double dice = (double) (2 * tp) / (2 * tp + fp + fn);
                perClassIoU.put(c, iou);
                perClassDice.put(c, dice);
                iouSum += iou;
                diceSum += dice;
                evaluatedClassCount++;
            }
        }

        double mIoU = evaluatedClassCount > 0 ? (iouSum / evaluatedClassCount) : 0.0;
        double mDice = evaluatedClassCount > 0 ? (diceSum / evaluatedClassCount) : 0.0;

        return new EvaluationResult(
                true,
                null,
                pixelAccuracy,
                mIoU,
                mDice,
                perClassIoU,
                perClassDice,
                totalPixels
        );
    }
}
