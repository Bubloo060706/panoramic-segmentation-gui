package edu.iitm.cs5013.panoramic.comparison;

import edu.iitm.cs5013.panoramic.model.MaskData;
import edu.iitm.cs5013.panoramic.model.SegmentationResult;
import edu.iitm.cs5013.panoramic.model.SemanticClass;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetricsCalculatorTest {

    private MetricsCalculator calculator;

    @BeforeEach
    void setUp() {
        calculator = new MetricsCalculator();
    }

    @Test
    void testIndependentlyCalculatedReferenceValues() {
        // Hand-calculated reference fixture:
        // 4 pixels (2x2):
        // Ground Truth: [1, 1, 2, 2]
        // Prediction:   [1, 2, 2, 2]
        MaskData gtMask = new MaskData(2, 2, new int[]{1, 1, 2, 2});
        MaskData predMask = new MaskData(2, 2, new int[]{1, 2, 2, 2});

        List<SemanticClass> classes = List.of(
                new SemanticClass(1, "Class1", 255, 0, 0),
                new SemanticClass(2, "Class2", 0, 255, 0)
        );

        SegmentationResult gt = new SegmentationResult.Builder()
                .toolId("ground-truth")
                .dimensions(2, 2)
                .maskData(gtMask)
                .detectedClasses(classes)
                .build();

        SegmentationResult pred = new SegmentationResult.Builder()
                .toolId("tool-test")
                .dimensions(2, 2)
                .maskData(predMask)
                .detectedClasses(classes)
                .build();

        EvaluationResult res = calculator.evaluate(pred, gt);

        assertTrue(res.isQuantitativeValid());
        assertEquals(4, res.getTotalEvaluatedPixels());

        // Pixel accuracy: 3 / 4 = 0.75
        assertEquals(0.75, res.getPixelAccuracy(), 1e-6);

        // Class 1 IoU: 1 / 2 = 0.5
        assertEquals(0.5, res.getPerClassIoU().get(1), 1e-6);
        // Class 1 Dice: 2 / 3 = 0.666666...
        assertEquals(2.0 / 3.0, res.getPerClassDice().get(1), 1e-6);

        // Class 2 IoU: 2 / 3 = 0.666666...
        assertEquals(2.0 / 3.0, res.getPerClassIoU().get(2), 1e-6);
        // Class 2 Dice: 4 / 5 = 0.8
        assertEquals(0.8, res.getPerClassDice().get(2), 1e-6);

        // mIoU = (0.5 + 2/3) / 2 = 7/12 = 0.583333...
        assertEquals(7.0 / 12.0, res.getMeanIoU(), 1e-6);

        // mDice = (2/3 + 4/5) / 2 = 11/15 = 0.733333...
        assertEquals(11.0 / 15.0, res.getMeanDice(), 1e-6);
    }

    @Test
    void testMissingGroundTruthRejectsQuantitativeMetrics() {
        SegmentationResult pred = new SegmentationResult.Builder()
                .toolId("pred")
                .dimensions(2, 2)
                .maskData(new MaskData(2, 2, new int[]{0, 0, 0, 0}))
                .build();

        EvaluationResult res = calculator.evaluate(pred, null);
        assertFalse(res.isQuantitativeValid());
        assertTrue(res.getInvalidationReason().contains("missing"));
    }

    @Test
    void testIncompatibleDimensionsRejectsQuantitativeMetrics() {
        SegmentationResult gt = new SegmentationResult.Builder()
                .toolId("gt")
                .dimensions(4, 4)
                .maskData(new MaskData(4, 4, new int[16]))
                .build();

        SegmentationResult pred = new SegmentationResult.Builder()
                .toolId("pred")
                .dimensions(2, 2)
                .maskData(new MaskData(2, 2, new int[4]))
                .build();

        EvaluationResult res = calculator.evaluate(pred, gt);
        assertFalse(res.isQuantitativeValid());
        assertTrue(res.getInvalidationReason().contains("Mismatched dimensions"));
    }

    @Test
    void testIncompatibleLabelsRejectsQuantitativeMetrics() {
        SegmentationResult gt = new SegmentationResult.Builder()
                .toolId("gt")
                .dimensions(1, 2)
                .maskData(new MaskData(1, 2, new int[]{10, 11}))
                .detectedClasses(List.of(new SemanticClass(10, "A", 0, 0, 0), new SemanticClass(11, "B", 0, 0, 0)))
                .build();

        SegmentationResult pred = new SegmentationResult.Builder()
                .toolId("pred")
                .dimensions(1, 2)
                .maskData(new MaskData(1, 2, new int[]{20, 21}))
                .detectedClasses(List.of(new SemanticClass(20, "X", 0, 0, 0), new SemanticClass(21, "Y", 0, 0, 0)))
                .build();

        EvaluationResult res = calculator.evaluate(pred, gt);
        assertFalse(res.isQuantitativeValid());
        assertTrue(res.getInvalidationReason().contains("Incompatible label sets"));
    }
}
