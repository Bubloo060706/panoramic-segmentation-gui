package edu.iitm.cs5013.panoramic.comparison;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SegmentationMetricsTest {

    @Test
    void testExactParityWithReferencePython4x4Fixture() {
        // 4x4 fixture matching Python test:
        // gt:
        // [0, 0, 1, 1]
        // [0, 0, 1, 1]
        // [2, 2, 3, 3]
        // [2, 2, 3, 3]
        //
        // pred:
        // [0, 0, 1, 1]
        // [0, 0, 1, 2]
        // [2, 2, 3, 3]
        // [2, 2, 3, 3]
        int[] gt = {
                0, 0, 1, 1,
                0, 0, 1, 1,
                2, 2, 3, 3,
                2, 2, 3, 3
        };
        int[] pred = {
                0, 0, 1, 1,
                0, 0, 1, 2,
                2, 2, 3, 3,
                2, 2, 3, 3
        };

        SegmentationMetrics metrics = new SegmentationMetrics(5, 255, 0.0075);
        metrics.update(pred, gt, 4, 4);
        SegmentationMetrics.MetricsSummary summary = metrics.compute();

        // Parity with Python reference outputs:
        // mIoU: 88.75%
        // mAcc: 93.75%
        // Boundary IoU: 88.75%
        // Boundary F-score: 97.2222...%
        assertEquals(88.75, summary.mIoUPercent(), 1e-4);
        assertEquals(93.75, summary.mAccPercent(), 1e-4);
        assertEquals(88.75, summary.boundaryIoUPercent(), 1e-4);
        assertEquals(97.2222, summary.boundaryFScorePercent(), 1e-3);
    }

    @Test
    void testPerfectPredictionYields100Percent() {
        int[] gt = {
                0, 1, 2,
                0, 1, 2,
                0, 1, 2
        };
        SegmentationMetrics metrics = new SegmentationMetrics(3, 255, 0.0075);
        metrics.update(gt, gt, 3, 3);
        SegmentationMetrics.MetricsSummary summary = metrics.compute();

        assertEquals(100.0, summary.mIoUPercent(), 1e-4);
        assertEquals(100.0, summary.mAccPercent(), 1e-4);
        assertEquals(100.0, summary.boundaryIoUPercent(), 1e-4);
        assertEquals(100.0, summary.boundaryFScorePercent(), 1e-4);
    }

    @Test
    void testIgnoreIndexIsExcluded() {
        int[] gt = {
                0, 255,
                1, 255
        };
        int[] pred = {
                0, 2,
                1, 2
        };

        SegmentationMetrics metrics = new SegmentationMetrics(3, 255, 0.0075);
        metrics.update(pred, gt, 2, 2);
        SegmentationMetrics.MetricsSummary summary = metrics.compute();

        // Only classes 0 and 1 were valid in GT, both perfectly predicted
        assertEquals(100.0, summary.mIoUPercent(), 1e-4);
        assertEquals(100.0, summary.mAccPercent(), 1e-4);
    }

    @Test
    void testMorphologyHelperMethods() {
        boolean[] mask = {
                false, false, false, false,
                false, true,  true,  false,
                false, true,  true,  false,
                false, false, false, false
        };

        boolean[] boundary = SegmentationMetrics.onePixelBoundary(mask, 4, 4);
        // All 4 center pixels touch non-mask neighbors, so all 4 are boundary pixels
        int count = 0;
        for (boolean b : boundary) {
            if (b) count++;
        }
        assertEquals(4, count);

        // Dilate by 1 iteration (4-connected)
        boolean[] dilated = SegmentationMetrics.binaryDilation4(boundary, 4, 4, 1);
        int dilatedCount = 0;
        for (boolean b : dilated) {
            if (b) dilatedCount++;
        }
        // Original 4 plus 8 orthogonal neighbors within bounds = 12 pixels
        assertEquals(12, dilatedCount);
    }
}

