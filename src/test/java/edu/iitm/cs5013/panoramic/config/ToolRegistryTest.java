package edu.iitm.cs5013.panoramic.config;

import edu.iitm.cs5013.panoramic.model.*;
import edu.iitm.cs5013.panoramic.segmentation.SegmentationTool;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ToolRegistryTest {

    private ToolRegistry registry;
    private ToolConfigurationLoader loader;

    @BeforeEach
    void setUp() {
        loader = new ToolConfigurationLoader();
        registry = new ToolRegistry(loader);
    }

    @Test
    void testRegisterAndRetrieveTool() {
        ToolMetadata meta = new ToolMetadata("tool-a", "Tool A", "Sample", "1.0",
                Set.of(SegmentationType.SEMANTIC), Set.of("jpg", "png"));
        ToolConfig config = new ToolConfig("tool-a", "Tool A", "Desc", ExecutionType.MOCK,
                List.of(), "", "", 30L, true, null, null);

        SegmentationTool mockTool = new SegmentationTool() {
            @Override
            public ToolMetadata getMetadata() { return meta; }
            @Override
            public ToolConfig getConfiguration() { return config; }
            @Override
            public SegmentationResult segment(SegmentationRequest request) {
                return new SegmentationResult.Builder().toolId("tool-a").build();
            }
            @Override
            public boolean isAvailable() { return true; }
        };

        registry.registerTool(mockTool);

        assertEquals(1, registry.size());
        Optional<SegmentationTool> retrieved = registry.getTool("tool-a");
        assertTrue(retrieved.isPresent());
        assertEquals("Tool A", retrieved.get().getMetadata().getName());

        assertTrue(registry.unregisterTool("tool-a"));
        assertEquals(0, registry.size());
        assertTrue(registry.getTool("tool-a").isEmpty());
    }

    @Test
    void testLoadValidConfigurationString() throws ConfigurationException {
        String json = """
                [
                  {
                    "id": "hoho-net",
                    "name": "HoHoNet Panoramic Segmenter",
                    "description": "Horizon-driven panoramic segmentation",
                    "executionType": "PROCESS",
                    "commandTemplate": ["python", "run_hohonet.py", "--input", "{IMAGE_PATH}"],
                    "timeoutSeconds": 45,
                    "enabled": true
                  },
                  {
                    "id": "remote-panoseg",
                    "name": "Remote PanoSeg API",
                    "description": "Cloud-based inference service",
                    "executionType": "HTTP",
                    "endpointUrl": "http://localhost:8000/segment",
                    "timeoutSeconds": 60,
                    "enabled": true
                  }
                ]
                """;

        List<ToolConfig> configs = loader.loadFromString(json);
        assertEquals(2, configs.size());
        assertEquals("hoho-net", configs.get(0).getId());
        assertEquals(ExecutionType.PROCESS, configs.get(0).getExecutionType());
        assertEquals(45, configs.get(0).getTimeoutSeconds());

        assertEquals("remote-panoseg", configs.get(1).getId());
        assertEquals(ExecutionType.HTTP, configs.get(1).getExecutionType());
        assertEquals("http://localhost:8000/segment", configs.get(1).getEndpointUrl());
    }

    @Test
    void testRejectInvalidConfigurations() {
        // Missing commandTemplate for PROCESS
        String invalidProcess = """
                [
                  {
                    "id": "bad-proc",
                    "name": "Bad Process",
                    "executionType": "PROCESS",
                    "commandTemplate": []
                  }
                ]
                """;
        assertThrows(ConfigurationException.class, () -> loader.loadFromString(invalidProcess));

        // Missing endpointUrl for HTTP
        String invalidHttp = """
                [
                  {
                    "id": "bad-http",
                    "name": "Bad HTTP",
                    "executionType": "HTTP"
                  }
                ]
                """;
        assertThrows(ConfigurationException.class, () -> loader.loadFromString(invalidHttp));

        // Duplicate IDs
        String duplicateIds = """
                [
                  {
                    "id": "dup",
                    "name": "First",
                    "executionType": "HTTP",
                    "endpointUrl": "http://localhost/1"
                  },
                  {
                    "id": "dup",
                    "name": "Second",
                    "executionType": "HTTP",
                    "endpointUrl": "http://localhost/2"
                  }
                ]
                """;
        assertThrows(ConfigurationException.class, () -> loader.loadFromString(duplicateIds));
    }

    @Test
    void testLoadFromFileWithFactories(@TempDir Path tempDir) throws IOException, ConfigurationException {
        Path cfg = tempDir.resolve("tools.json");
        String json = """
                [
                  {
                    "id": "test-mock",
                    "name": "Test Mock",
                    "executionType": "MOCK",
                    "enabled": true
                  }
                ]
                """;
        Files.writeString(cfg, json);

        registry.registerFactory(ExecutionType.MOCK, config -> new SegmentationTool() {
            @Override
            public ToolMetadata getMetadata() {
                return new ToolMetadata(config.getId(), config.getName(), "", "1.0", Set.of(), Set.of());
            }
            @Override
            public ToolConfig getConfiguration() { return config; }
            @Override
            public SegmentationResult segment(SegmentationRequest request) { return null; }
            @Override
            public boolean isAvailable() { return true; }
        });

        List<SegmentationTool> loaded = registry.loadAndRegisterFromConfig(cfg);
        assertEquals(1, loaded.size());
        assertEquals("test-mock", loaded.get(0).getMetadata().getId());
        assertEquals(1, registry.getAllTools().size());
        assertEquals(1, registry.getAvailableTools().size());
    }
}
