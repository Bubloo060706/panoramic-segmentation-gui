package edu.iitm.cs5013.panoramic.config;

import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Central registry for discovering, registering, and managing segmentation tools.
 * Enables adding new segmentation models without modifying the core GUI or workflow layer.
 */
public class ToolRegistry {
    private final Map<String, SegmentationTool> registeredTools = new ConcurrentHashMap<>();
    private final Map<ExecutionType, Function<ToolConfig, SegmentationTool>> adapterFactories = new ConcurrentHashMap<>();
    private final ToolConfigurationLoader loader;

    public ToolRegistry(ToolConfigurationLoader loader) {
        this.loader = loader != null ? loader : new ToolConfigurationLoader();
    }

    public ToolRegistry() {
        this(new ToolConfigurationLoader());
    }

    /**
     * Registers an adapter factory for a given execution type.
     */
    public void registerFactory(ExecutionType type, Function<ToolConfig, SegmentationTool> factory) {
        if (type != null && factory != null) {
            adapterFactories.put(type, factory);
        }
    }

    /**
     * Registers a pre-instantiated tool.
     */
    public void registerTool(SegmentationTool tool) {
        if (tool == null) {
            throw new IllegalArgumentException("Cannot register null tool");
        }
        String id = tool.getMetadata().getId();
        registeredTools.put(id, tool);
    }

    /**
     * Unregisters a tool by ID.
     */
    public boolean unregisterTool(String toolId) {
        if (toolId == null) return false;
        return registeredTools.remove(toolId) != null;
    }

    /**
     * Looks up a tool by ID.
     */
    public Optional<SegmentationTool> getTool(String toolId) {
        if (toolId == null) return Optional.empty();
        return Optional.ofNullable(registeredTools.get(toolId));
    }

    /**
     * @return List of all registered tools.
     */
    public List<SegmentationTool> getAllTools() {
        return new ArrayList<>(registeredTools.values());
    }

    /**
     * @return List of registered tools that are currently available/reachable.
     */
    public List<SegmentationTool> getAvailableTools() {
        List<SegmentationTool> available = new ArrayList<>();
        for (SegmentationTool tool : registeredTools.values()) {
            if (tool.isAvailable()) {
                available.add(tool);
            }
        }
        return available;
    }

    /**
     * Loads tools from a configuration file and registers them using registered factories.
     */
    public List<SegmentationTool> loadAndRegisterFromConfig(Path configPath) throws ConfigurationException {
        List<ToolConfig> configs = loader.loadFromFile(configPath);
        List<SegmentationTool> loaded = new ArrayList<>();
        for (ToolConfig config : configs) {
            if (!config.isEnabled()) {
                continue;
            }
            Function<ToolConfig, SegmentationTool> factory = adapterFactories.get(config.getExecutionType());
            if (factory != null) {
                SegmentationTool tool = factory.apply(config);
                registerTool(tool);
                loaded.add(tool);
            } else {
                throw new ConfigurationException("No adapter factory registered for execution type: " + config.getExecutionType());
            }
        }
        return loaded;
    }

    public void clear() {
        registeredTools.clear();
    }

    public int size() {
        return registeredTools.size();
    }
}
