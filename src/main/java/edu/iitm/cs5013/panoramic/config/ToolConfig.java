package edu.iitm.cs5013.panoramic.config;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import edu.iitm.cs5013.panoramic.model.SemanticClass;

import java.util.*;

/**
 * Configuration for an external or registered segmentation model tool.
 */
public final class ToolConfig {
    private final String id;
    private final String name;
    private final String description;
    private final ExecutionType executionType;
    private final List<String> commandTemplate;
    private final String workingDirectory;
    private final String endpointUrl;
    private final long timeoutSeconds;
    private final boolean enabled;
    private final Map<String, Object> parameters;
    private final List<SemanticClass> supportedClasses;

    @JsonCreator
    public ToolConfig(
            @JsonProperty("id") String id,
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("executionType") ExecutionType executionType,
            @JsonProperty("commandTemplate") List<String> commandTemplate,
            @JsonProperty("workingDirectory") String workingDirectory,
            @JsonProperty("endpointUrl") String endpointUrl,
            @JsonProperty("timeoutSeconds") Long timeoutSeconds,
            @JsonProperty("enabled") Boolean enabled,
            @JsonProperty("parameters") Map<String, Object> parameters,
            @JsonProperty("supportedClasses") List<SemanticClass> supportedClasses) {
        this.id = Objects.requireNonNull(id, "Tool ID cannot be null");
        this.name = (name != null && !name.isBlank()) ? name : id;
        this.description = description != null ? description : "";
        this.executionType = executionType != null ? executionType : ExecutionType.PROCESS;
        this.commandTemplate = commandTemplate != null ? Collections.unmodifiableList(new ArrayList<>(commandTemplate)) : Collections.emptyList();
        this.workingDirectory = workingDirectory != null ? workingDirectory : "";
        this.endpointUrl = endpointUrl != null ? endpointUrl : "";
        this.timeoutSeconds = (timeoutSeconds != null && timeoutSeconds > 0) ? timeoutSeconds : 60L;
        this.enabled = enabled != null ? enabled : true;
        this.parameters = parameters != null ? Collections.unmodifiableMap(new HashMap<>(parameters)) : Collections.emptyMap();
        this.supportedClasses = supportedClasses != null ? Collections.unmodifiableList(new ArrayList<>(supportedClasses)) : Collections.emptyList();
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

    public ExecutionType getExecutionType() {
        return executionType;
    }

    public List<String> getCommandTemplate() {
        return commandTemplate;
    }

    public String getWorkingDirectory() {
        return workingDirectory;
    }

    public String getEndpointUrl() {
        return endpointUrl;
    }

    public long getTimeoutSeconds() {
        return timeoutSeconds;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public List<SemanticClass> getSupportedClasses() {
        return supportedClasses;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ToolConfig that)) return false;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "ToolConfig[" + id + " (" + executionType + ")]";
    }
}
