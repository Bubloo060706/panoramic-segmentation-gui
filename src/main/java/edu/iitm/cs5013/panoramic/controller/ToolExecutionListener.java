package edu.iitm.cs5013.panoramic.controller;

import edu.iitm.cs5013.panoramic.model.SegmentationResult;

/**
 * Callback listener for observing asynchronous tool execution events.
 */
public interface ToolExecutionListener {
    void onToolStarted(String toolId);
    void onToolProgress(String toolId, String message, double progress);
    void onToolSuccess(String toolId, SegmentationResult result);
    void onToolFailure(String toolId, String errorMessage);
}
