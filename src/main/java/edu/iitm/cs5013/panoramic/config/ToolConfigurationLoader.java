package edu.iitm.cs5013.panoramic.config;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Loads and validates tool configurations from JSON files or input streams using Jackson.
 */
public class ToolConfigurationLoader {
    private final ObjectMapper objectMapper;

    public ToolConfigurationLoader() {
        this.objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule())
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }

    public List<ToolConfig> loadFromFile(Path path) throws ConfigurationException {
        if (path == null) {
            throw new ConfigurationException("Configuration file path cannot be null");
        }
        if (!Files.exists(path)) {
            throw new ConfigurationException("Configuration file does not exist: " + path);
        }
        if (!Files.isReadable(path)) {
            throw new ConfigurationException("Configuration file is not readable: " + path);
        }

        try (InputStream in = Files.newInputStream(path)) {
            return loadFromStream(in);
        } catch (IOException e) {
            throw new ConfigurationException("Failed to read configuration file: " + path, e);
        }
    }

    public List<ToolConfig> loadFromString(String jsonContent) throws ConfigurationException {
        if (jsonContent == null || jsonContent.isBlank()) {
            throw new ConfigurationException("Configuration JSON string cannot be empty");
        }
        try {
            List<ToolConfig> configs = objectMapper.readValue(jsonContent, new TypeReference<List<ToolConfig>>() {});
            validateConfigs(configs);
            return configs;
        } catch (Exception e) {
            throw new ConfigurationException("Failed to parse tool configurations from JSON: " + e.getMessage(), e);
        }
    }

    public List<ToolConfig> loadFromStream(InputStream in) throws ConfigurationException {
        if (in == null) {
            throw new ConfigurationException("Configuration input stream cannot be null");
        }
        try {
            List<ToolConfig> configs = objectMapper.readValue(in, new TypeReference<List<ToolConfig>>() {});
            validateConfigs(configs);
            return configs;
        } catch (Exception e) {
            throw new ConfigurationException("Failed to parse tool configurations from stream: " + e.getMessage(), e);
        }
    }

    private void validateConfigs(List<ToolConfig> configs) throws ConfigurationException {
        if (configs == null || configs.isEmpty()) {
            throw new ConfigurationException("No tool configurations found in file/stream");
        }

        Set<String> seenIds = new HashSet<>();
        for (int i = 0; i < configs.size(); i++) {
            ToolConfig c = configs.get(i);
            if (c.getId() == null || c.getId().isBlank()) {
                throw new ConfigurationException("Tool configuration at index " + i + " has an empty or null 'id'");
            }
            if (seenIds.contains(c.getId())) {
                throw new ConfigurationException("Duplicate tool ID detected: '" + c.getId() + "'");
            }
            seenIds.add(c.getId());

            if (c.getName() == null || c.getName().isBlank()) {
                throw new ConfigurationException("Tool configuration for '" + c.getId() + "' has an empty 'name'");
            }
            if (c.getExecutionType() == null) {
                throw new ConfigurationException("Tool configuration for '" + c.getId() + "' has a missing 'executionType'");
            }
            if (c.getExecutionType() == ExecutionType.PROCESS && c.getCommandTemplate().isEmpty()) {
                throw new ConfigurationException("PROCESS tool '" + c.getId() + "' must specify a non-empty 'commandTemplate'");
            }
            if (c.getExecutionType() == ExecutionType.HTTP && (c.getEndpointUrl() == null || c.getEndpointUrl().isBlank())) {
                throw new ConfigurationException("HTTP tool '" + c.getId() + "' must specify an 'endpointUrl'");
            }
        }
    }

    public ObjectMapper getObjectMapper() {
        return objectMapper;
    }
}
