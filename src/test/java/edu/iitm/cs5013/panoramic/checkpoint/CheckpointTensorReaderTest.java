package edu.iitm.cs5013.panoramic.checkpoint;

import org.junit.jupiter.api.Test;
import java.io.File;

import static org.junit.jupiter.api.Assertions.*;

class CheckpointTensorReaderTest {

    @Test
    void testLoadSegformerSafetensors() {
        File sf = new File("PAI Segmentation Checkpoints/Segformer/model.safetensors");
        if (!sf.exists()) return;

        CheckpointTensorReader.ClassifierWeights cw = CheckpointTensorReader.loadClassifierWeights(sf);
        assertNotNull(cw);
        assertEquals(21, cw.numClasses);
        assertEquals(768, cw.featureDim);
        assertEquals(21, cw.weights.length);
        assertEquals(768, cw.weights[0].length);
        assertEquals(21, cw.bias.length);
        // Verify exact values match Python output
        assertEquals(0.008461f, cw.weights[0][0], 1e-4f);
        assertEquals(-0.001892f, cw.bias[0], 1e-4f);
    }

    @Test
    void testLoadSamCnnClassifierPth() {
        File pth = new File("PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth");
        if (!pth.exists()) return;

        CheckpointTensorReader.ClassifierWeights cw = CheckpointTensorReader.loadClassifierWeights(pth);
        assertNotNull(cw);
        assertEquals(21, cw.numClasses);
        assertEquals(512, cw.featureDim);
        assertEquals(21, cw.weights.length);
        assertEquals(512, cw.weights[0].length);
        assertEquals(21, cw.bias.length);
        // Verify exact values match Python output
        assertEquals(-0.016444f, cw.weights[0][0], 1e-4f);
        assertEquals(0.044284f, cw.bias[0], 1e-4f);
    }
}
