package edu.iitm.cs5013.panoramic.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.Objects;
import java.util.Set;

/**
 * Metadata describing a registered segmentation tool.
 */
public final class ToolMetadata {
    private final String id;
    private final String name;
    private final String description;
    private final String version;
    private final Set<SegmentationType> supportedTypes;
    private final Set<String> supportedExtensions;

    @JsonCreator
    public ToolMetadata(
            @JsonProperty("id") String id,
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("version") String version,
            @JsonProperty("supportedTypes") Set<SegmentationType> supportedTypes,
            @JsonProperty("supportedExtensions") Set<String> supportedExtensions) {
        this.id = Objects.requireNonNull(id, "Tool ID cannot be null");
        this.name = Objects.requireNonNull(name, "Tool name cannot be null");
        this.description = description != null ? description : "";
        this.version = version != null ? version : "1.0.0";
        this.supportedTypes = supportedTypes != null ? Collections.unmodifiableSet(supportedTypes) : Set.of(SegmentationType.SEMANTIC);
        this.supportedExtensions = supportedExtensions != null ? Collections.unmodifiableSet(supportedExtensions) : Set.of("jpg", "jpeg", "png");
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public String getVersion() {
        return version;
    }

    public Set<SegmentationType> getSupportedTypes() {
        return supportedTypes;
    }

    public Set<String> getSupportedExtensions() {
        return supportedExtensions;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ToolMetadata that)) return false;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return name + " (" + id + " v" + version + ")";
    }
}
