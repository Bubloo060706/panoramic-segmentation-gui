package edu.iitm.cs5013.panoramic.comparison;

import java.util.*;

/**
 * Boundary-aware semantic segmentation metrics engine, mathematically identical to
 * the reference Python/scipy implementation.
 *
 * Computes:
 * <ul>
 *   <li><b>mIoU:</b> Mean Intersection over Union across active classes.</li>
 *   <li><b>mAcc:</b> Mean Class Accuracy (intersection / gt_area) across active classes.</li>
 *   <li><b>Boundary IoU:</b> Morphological boundary band intersection over union with dilation band.</li>
 *   <li><b>Boundary F-score:</b> Harmonic mean of boundary precision and recall within spatial tolerance.</li>
 * </ul>
 */
public class SegmentationMetrics {

    private final int numClasses;
    private final int ignoreIndex;
    private final double boundaryDilationRatio;

    private final double[] intersection;
    private final double[] union;
    private final double[] predArea;
    private final double[] gtArea;

    private final double[] boundaryIntersection;
    private final double[] boundaryUnion;

    private final double[] predBoundaryCount;
    private final double[] gtBoundaryCount;
    private final double[] predBoundaryMatched;
    private final double[] gtBoundaryMatched;

    public SegmentationMetrics(int numClasses) {
        this(numClasses, 255, 0.0075);
    }

    public SegmentationMetrics(int numClasses, int ignoreIndex, double boundaryDilationRatio) {
        this.numClasses = Math.max(1, numClasses);
        this.ignoreIndex = ignoreIndex;
        this.boundaryDilationRatio = boundaryDilationRatio;

        this.intersection = new double[this.numClasses];
        this.union = new double[this.numClasses];
        this.predArea = new double[this.numClasses];
        this.gtArea = new double[this.numClasses];

        this.boundaryIntersection = new double[this.numClasses];
        this.boundaryUnion = new double[this.numClasses];

        this.predBoundaryCount = new double[this.numClasses];
        this.gtBoundaryCount = new double[this.numClasses];
        this.predBoundaryMatched = new double[this.numClasses];
        this.gtBoundaryMatched = new double[this.numClasses];
    }

    /**
     * Extracts the boundary band of a binary mask via 8-connected morphological erosion with 0-padding.
     * Matches scipy: ndimage.binary_erosion(padded, structure=np.ones((3,3)), iterations=dilation, border_value=0)
     */
    public static boolean[] maskToBoundary(boolean[] mask, int h, int w, double dilationRatio) {
        double imageDiagonal = Math.sqrt((double) h * h + (double) w * w);
        int dilation = Math.max(1, (int) Math.round(dilationRatio * imageDiagonal));

        int padW = w + 2;
        int padH = h + 2;
        boolean[] padded = new boolean[padH * padW];

        // Pad with 1 pixel of false (constant 0) on all borders
        for (int y = 0; y < h; y++) {
            int srcRow = y * w;
            int dstRow = (y + 1) * padW + 1;
            System.arraycopy(mask, srcRow, padded, dstRow, w);
        }

        boolean[] curr = padded;
        for (int iter = 0; iter < dilation; iter++) {
            boolean[] next = new boolean[padH * padW];
            for (int y = 1; y <= h; y++) {
                int yPad = y * padW;
                int yAbove = (y - 1) * padW;
                int yBelow = (y + 1) * padW;
                for (int x = 1; x <= w; x++) {
                    int idx = yPad + x;
                    if (!curr[idx]) continue;

                    boolean all8 =
                            curr[yAbove + (x - 1)] && curr[yAbove + x] && curr[yAbove + (x + 1)] &&
                            curr[yPad + (x - 1)]                       && curr[yPad + (x + 1)] &&
                            curr[yBelow + (x - 1)] && curr[yBelow + x] && curr[yBelow + (x + 1)];

                    next[idx] = all8;
                }
            }
            curr = next;
        }

        // Unpad and perform mask & (~eroded)
        boolean[] result = new boolean[h * w];
        for (int y = 0; y < h; y++) {
            int resRow = y * w;
            int padRow = (y + 1) * padW + 1;
            for (int x = 0; x < w; x++) {
                result[resRow + x] = mask[resRow + x] && !curr[padRow + x];
            }
        }
        return result;
    }

    /**
     * Extracts the 1-pixel boundary of a binary mask.
     * Matches scipy: mask & (~ndimage.binary_erosion(mask, structure=np.ones((3,3)), iterations=1, border_value=0))
     */
    public static boolean[] onePixelBoundary(boolean[] mask, int h, int w) {
        boolean[] result = new boolean[h * w];
        for (int y = 0; y < h; y++) {
            int yRow = y * w;
            for (int x = 0; x < w; x++) {
                int idx = yRow + x;
                if (!mask[idx]) continue;

                // Pixels on outer image border have at least one border-0 neighbor
                if (y == 0 || y == h - 1 || x == 0 || x == w - 1) {
                    result[idx] = true;
                    continue;
                }

                int yAbove = (y - 1) * w;
                int yBelow = (y + 1) * w;
                boolean all8 =
                        mask[yAbove + (x - 1)] && mask[yAbove + x] && mask[yAbove + (x + 1)] &&
                        mask[yRow + (x - 1)]                       && mask[yRow + (x + 1)] &&
                        mask[yBelow + (x - 1)] && mask[yBelow + x] && mask[yBelow + (x + 1)];

                result[idx] = !all8;
            }
        }
        return result;
    }

    /**
     * 4-connected morphological binary dilation.
     * Matches scipy: ndimage.binary_dilation(mask, structure=generate_binary_structure(2, 1), iterations=tolerance)
     */
    public static boolean[] binaryDilation4(boolean[] mask, int h, int w, int iterations) {
        boolean[] curr = Arrays.copyOf(mask, mask.length);
        for (int iter = 0; iter < iterations; iter++) {
            boolean[] next = Arrays.copyOf(curr, curr.length);
            for (int y = 0; y < h; y++) {
                int yRow = y * w;
                for (int x = 0; x < w; x++) {
                    if (!curr[yRow + x]) continue;
                    if (y > 0) next[(y - 1) * w + x] = true;
                    if (y < h - 1) next[(y + 1) * w + x] = true;
                    if (x > 0) next[yRow + (x - 1)] = true;
                    if (x < w - 1) next[yRow + (x + 1)] = true;
                }
            }
            curr = next;
        }
        return curr;
    }

    /**
     * Updates metrics for a prediction and ground-truth pair.
     *
     * @param prediction Flattened class-ID array for prediction.
     * @param groundTruth Flattened class-ID array for ground truth.
     * @param h Height of the image.
     * @param w Width of the image.
     */
    public void update(int[] prediction, int[] groundTruth, int h, int w) {
        if (prediction == null || groundTruth == null || prediction.length != groundTruth.length) {
            return;
        }

        int totalPixels = h * w;
        boolean[] valid = new boolean[totalPixels];
        for (int i = 0; i < totalPixels; i++) {
            valid[i] = (groundTruth[i] != ignoreIndex && groundTruth[i] >= 0);
        }

        double imageDiagonal = Math.sqrt((double) h * h + (double) w * w);
        int tolerance = Math.max(1, (int) Math.round(boundaryDilationRatio * imageDiagonal));

        for (int cls = 0; cls < numClasses; cls++) {
            boolean[] predMask = new boolean[totalPixels];
            boolean[] gtMask = new boolean[totalPixels];
            double predAreaCount = 0;
            double gtAreaCount = 0;
            double interCount = 0;

            for (int i = 0; i < totalPixels; i++) {
                if (valid[i]) {
                    boolean isPred = (prediction[i] == cls);
                    boolean isGt = (groundTruth[i] == cls);
                    if (isPred) {
                        predMask[i] = true;
                        predAreaCount++;
                    }
                    if (isGt) {
                        gtMask[i] = true;
                        gtAreaCount++;
                    }
                    if (isPred && isGt) {
                        interCount++;
                    }
                }
            }

            double unionCount = predAreaCount + gtAreaCount - interCount;
            intersection[cls] += interCount;
            union[cls] += unionCount;
            predArea[cls] += predAreaCount;
            gtArea[cls] += gtAreaCount;

            // If neither pred nor gt has this class, skip morphological operations
            if (predAreaCount == 0 && gtAreaCount == 0) {
                continue;
            }

            // Boundary IoU
            boolean[] predBand = maskToBoundary(predMask, h, w, boundaryDilationRatio);
            boolean[] gtBand = maskToBoundary(gtMask, h, w, boundaryDilationRatio);

            double bInter = 0;
            double bUnion = 0;
            for (int i = 0; i < totalPixels; i++) {
                if (valid[i]) {
                    boolean pb = predBand[i];
                    boolean gb = gtBand[i];
                    if (pb && gb) bInter++;
                    if (pb || gb) bUnion++;
                }
            }
            boundaryIntersection[cls] += bInter;
            boundaryUnion[cls] += bUnion;

            // Boundary F-score
            boolean[] predBoundary = onePixelBoundary(predMask, h, w);
            boolean[] gtBoundary = onePixelBoundary(gtMask, h, w);

            double predCount = 0;
            double gtCount = 0;
            for (int i = 0; i < totalPixels; i++) {
                if (valid[i]) {
                    if (predBoundary[i]) predCount++;
                    if (gtBoundary[i]) gtCount++;
                }
            }

            predBoundaryCount[cls] += predCount;
            gtBoundaryCount[cls] += gtCount;

            if (predCount > 0 && gtCount > 0) {
                boolean[] gtDilated = binaryDilation4(gtBoundary, h, w, tolerance);
                boolean[] predDilated = binaryDilation4(predBoundary, h, w, tolerance);

                double pMatched = 0;
                double gMatched = 0;
                for (int i = 0; i < totalPixels; i++) {
                    if (valid[i]) {
                        if (predBoundary[i] && gtDilated[i]) pMatched++;
                        if (gtBoundary[i] && predDilated[i]) gMatched++;
                    }
                }
                predBoundaryMatched[cls] += pMatched;
                gtBoundaryMatched[cls] += gMatched;
            }
        }
    }

    /**
     * Computes the aggregated metrics across classes where values are defined.
     * Matches Python compute() dictionary keys and percentages.
     */
    public MetricsSummary compute() {
        Map<Integer, Double> classIoU = new TreeMap<>();
        Map<Integer, Double> classAcc = new TreeMap<>();
        Map<Integer, Double> classBoundaryIoU = new TreeMap<>();
        Map<Integer, Double> classBoundaryFScore = new TreeMap<>();

        int validIouCount = 0;
        double sumIou = 0.0;

        int validAccCount = 0;
        double sumAcc = 0.0;

        int validBIouCount = 0;
        double sumBIou = 0.0;

        int validBFCount = 0;
        double sumBF = 0.0;

        for (int cls = 0; cls < numClasses; cls++) {
            if (union[cls] > 0) {
                double val = intersection[cls] / union[cls];
                classIoU.put(cls, val);
                sumIou += val;
                validIouCount++;
            }
            if (gtArea[cls] > 0) {
                double val = intersection[cls] / gtArea[cls];
                classAcc.put(cls, val);
                sumAcc += val;
                validAccCount++;
            }
            if (boundaryUnion[cls] > 0) {
                double val = boundaryIntersection[cls] / boundaryUnion[cls];
                classBoundaryIoU.put(cls, val);
                sumBIou += val;
                validBIouCount++;
            }
            if (predBoundaryCount[cls] > 0 && gtBoundaryCount[cls] > 0) {
                double precision = predBoundaryMatched[cls] / predBoundaryCount[cls];
                double recall = gtBoundaryMatched[cls] / gtBoundaryCount[cls];
                double denom = precision + recall;
                if (denom > 0) {
                    double fscore = (2.0 * precision * recall) / denom;
                    classBoundaryFScore.put(cls, fscore);
                    sumBF += fscore;
                    validBFCount++;
                }
            }
        }

        double mIoU = validIouCount > 0 ? (sumIou / validIouCount) : 0.0;
        double mAcc = validAccCount > 0 ? (sumAcc / validAccCount) : 0.0;
        double boundaryIoU = validBIouCount > 0 ? (sumBIou / validBIouCount) : 0.0;
        double boundaryFScore = validBFCount > 0 ? (sumBF / validBFCount) : 0.0;

        return new MetricsSummary(
                mIoU,
                mAcc,
                boundaryIoU,
                boundaryFScore,
                classIoU,
                classAcc,
                classBoundaryIoU,
                classBoundaryFScore
        );
    }

    public record MetricsSummary(
            double mIoU,
            double mAcc,
            double boundaryIoU,
            double boundaryFScore,
            Map<Integer, Double> classIoU,
            Map<Integer, Double> classAcc,
            Map<Integer, Double> classBoundaryIoU,
            Map<Integer, Double> classBoundaryFScore
    ) {
        public double mIoUPercent() { return mIoU * 100.0; }
        public double mAccPercent() { return mAcc * 100.0; }
        public double boundaryIoUPercent() { return boundaryIoU * 100.0; }
        public double boundaryFScorePercent() { return boundaryFScore * 100.0; }
    }
}

