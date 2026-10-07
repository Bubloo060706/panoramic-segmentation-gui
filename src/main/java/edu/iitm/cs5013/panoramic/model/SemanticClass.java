package edu.iitm.cs5013.panoramic.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;

/**
 * Represents a semantic or instance category in the segmentation result.
 */
public final class SemanticClass {
    private final int id;
    private final String name;
    private final int red;
    private final int green;
    private final int blue;
    private final long pixelCount;
    private final double confidence;

    @JsonCreator
    public SemanticClass(
            @JsonProperty("id") int id,
            @JsonProperty("name") String name,
            @JsonProperty("red") int red,
            @JsonProperty("green") int green,
            @JsonProperty("blue") int blue,
            @JsonProperty("pixelCount") long pixelCount,
            @JsonProperty("confidence") double confidence) {
        this.id = id;
        this.name = (name != null && !name.isBlank()) ? name : "Class_" + id;
        this.red = Math.clamp(red, 0, 255);
        this.green = Math.clamp(green, 0, 255);
        this.blue = Math.clamp(blue, 0, 255);
        this.pixelCount = Math.max(0, pixelCount);
        this.confidence = Math.clamp(confidence, 0.0, 1.0);
    }

    public SemanticClass(int id, String name, int red, int green, int blue) {
        this(id, name, red, green, blue, 0, 1.0);
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public int getRed() {
        return red;
    }

    public int getGreen() {
        return green;
    }

    public int getBlue() {
        return blue;
    }

    public long getPixelCount() {
        return pixelCount;
    }

    public double getConfidence() {
        return confidence;
    }

    public int getRgb() {
        return ((red & 0xFF) << 16) | ((green & 0xFF) << 8) | (blue & 0xFF);
    }

    public String getHexColor() {
        return String.format("#%02X%02X%02X", red, green, blue);
    }

    public SemanticClass withPixelCount(long newCount) {
        return new SemanticClass(this.id, this.name, this.red, this.green, this.blue, newCount, this.confidence);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SemanticClass that)) return false;
        return id == that.id && Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, name);
    }

    @Override
    public String toString() {
        return String.format("%s (id=%d, rgb=[%d,%d,%d], pixels=%d)", name, id, red, green, blue, pixelCount);
    }
}
