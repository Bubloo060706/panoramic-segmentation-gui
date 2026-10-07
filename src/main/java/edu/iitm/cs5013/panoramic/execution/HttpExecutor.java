package edu.iitm.cs5013.panoramic.execution;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * Handles controlled execution of remote model inference services via HTTP.
 */
public class HttpExecutor {
    private final HttpClient httpClient;

    public HttpExecutor(HttpClient httpClient) {
        this.httpClient = httpClient != null ? httpClient : HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    public HttpExecutor() {
        this(null);
    }

    public HttpExecutionResult postJson(String endpointUrl, String jsonPayload, Map<String, String> headers, long timeoutSeconds)
            throws HttpExecutionException {
        if (endpointUrl == null || endpointUrl.isBlank()) {
            throw new HttpExecutionException("Endpoint URL cannot be empty", -1, "");
        }

        try {
            URI uri = URI.create(endpointUrl);
            HttpRequest.Builder builder = HttpRequest.newBuilder(uri)
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload != null ? jsonPayload : "{}"))
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json")
                    .timeout(Duration.ofSeconds(timeoutSeconds > 0 ? timeoutSeconds : 60));

            if (headers != null) {
                headers.forEach(builder::header);
            }

            long startTime = System.currentTimeMillis();
            HttpResponse<String> response = httpClient.send(builder.build(), HttpResponse.BodyHandlers.ofString());
            long durationMs = System.currentTimeMillis() - startTime;

            if (response.statusCode() >= 400) {
                throw new HttpExecutionException(
                        "HTTP inference endpoint returned error code " + response.statusCode() + ": " + response.body(),
                        response.statusCode(),
                        response.body()
                );
            }

            return new HttpExecutionResult(response.statusCode(), response.body(), response.headers().map(), durationMs);

        } catch (HttpExecutionException he) {
            throw he;
        } catch (Exception e) {
            throw new HttpExecutionException("Failed to send HTTP request to " + endpointUrl + ": " + e.getMessage(), e);
        }
    }

    public boolean ping(String endpointUrl, long timeoutSeconds) {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(endpointUrl))
                    .method("HEAD", HttpRequest.BodyPublishers.noBody())
                    .timeout(Duration.ofSeconds(timeoutSeconds > 0 ? timeoutSeconds : 3))
                    .build();
            HttpResponse<Void> resp = httpClient.send(req, HttpResponse.BodyHandlers.discarding());
            return resp.statusCode() >= 200 && resp.statusCode() < 400;
        } catch (Exception e) {
            return false;
        }
    }
}
