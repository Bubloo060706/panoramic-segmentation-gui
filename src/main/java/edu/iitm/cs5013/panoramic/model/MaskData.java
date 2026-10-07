package edu.iitm.cs5013.panoramic.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Encapsulates 2D segmentation mask data with dimensions and class/instance mappings.
 */
public final class MaskData {
    private final int width;
    private final int height;
    private final int[] labels;
    private final int[] instanceIds;

    @JsonCreator
    public MaskData(
            @JsonProperty("width") int width,
            @JsonProperty("height") int height,
            @JsonProperty("labels") int[] labels,
            @JsonProperty("instanceIds") int[] instanceIds) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Width and height must be positive. Given: " + width + "x" + height);
        }
        this.width = width;
        this.height = height;
        int expectedLen = width * height;
        if (labels == null || labels.length != expectedLen) {
            throw new IllegalArgumentException("Labels array length (" + (labels == null ? "null" : labels.length) +
                    ") does not match dimensions (" + expectedLen + ")");
        }
        this.labels = labels;
        this.instanceIds = (instanceIds != null && instanceIds.length == expectedLen) ? instanceIds : null;
    }

    public MaskData(int width, int height, int[] labels) {
        this(width, height, labels, null);
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int[] getLabels() {
        return labels;
    }

    public int[] getInstanceIds() {
        return instanceIds;
    }

    @JsonIgnore
    public int getLabel(int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw new IndexOutOfBoundsException("Coordinates (" + x + ", " + y + ") out of bounds for " + width + "x" + height);
        }
        return labels[y * width + x];
    }

    @JsonIgnore
    public int getInstanceId(int x, int y) {
        if (instanceIds == null) return -1;
        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw new IndexOutOfBoundsException("Coordinates (" + x + ", " + y + ") out of bounds for " + width + "x" + height);
        }
        return instanceIds[y * width + x];
    }

    @JsonIgnore
    public boolean hasInstanceIds() {
        return instanceIds != null;
    }

    /**
     * Computes the frequency count of each label occurring in the mask.
     */
    @JsonIgnore
    public Map<Integer, Long> computeLabelCounts() {
        Map<Integer, Long> counts = new HashMap<>();
        for (int label : labels) {
            counts.put(label, counts.getOrDefault(label, 0L) + 1L);
        }
        return counts;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MaskData maskData)) return false;
        return width == maskData.width &&
                height == maskData.height &&
                Arrays.equals(labels, maskData.labels) &&
                Arrays.equals(instanceIds, maskData.instanceIds);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(width, height);
        result = 31 * result + Arrays.hashCode(labels);
        result = 31 * result + Arrays.hashCode(instanceIds);
        return result;
    }

    @Override
    public String toString() {
        return "MaskData[" + width + "x" + height + ", labels=" + labels.length + ", instances=" + (instanceIds != null) + "]";
    }
}
