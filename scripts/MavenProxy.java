/*
 * Maven Local Forward Proxy (Pure Java Edition)
 * Replaces maven_proxy.py with zero third-party dependencies using built-in JDK HttpServer & HttpClient.
 *
 * Standalone single-file Java program executable directly via:
 *   java scripts/MavenProxy.java [--port <port>]
 */

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

public class MavenProxy {

    private static final int DEFAULT_PORT = 8888;
    private static final String TARGET_HOST = "https://repo1.maven.org";

    public static void main(String[] args) throws Exception {
        int port = DEFAULT_PORT;
        for (int i = 0; i < args.length; i++) {
            if ("--port".equals(args[i]) && i + 1 < args.length) {
                port = Integer.parseInt(args[++i]);
            }
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        server.createContext("/", new ProxyHandler(client));
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();

        System.out.println("Maven proxy running on http://127.0.0.1:" + port + "/ -> " + TARGET_HOST);
        System.out.flush();

        // Keep running until interrupted
        Thread.currentThread().join();
    }

    private static class ProxyHandler implements HttpHandler {
        private final HttpClient client;

        ProxyHandler(HttpClient client) {
            this.client = client;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                String method = exchange.getRequestMethod();
                URI requestUri = exchange.getRequestURI();
                String targetUrl = TARGET_HOST + requestUri.getRawPath();
                if (requestUri.getRawQuery() != null) {
                    targetUrl += "?" + requestUri.getRawQuery();
                }

                HttpRequest.Builder reqBuilder = HttpRequest.newBuilder()
                        .uri(URI.create(targetUrl))
                        .timeout(Duration.ofSeconds(30))
                        .header("Host", "repo1.maven.org")
                        .header("User-Agent", "Apache-Maven/3.9.9");

                // Forward incoming request headers
                Headers inHeaders = exchange.getRequestHeaders();
                for (Map.Entry<String, List<String>> entry : inHeaders.entrySet()) {
                    String name = entry.getKey();
                    if (!name.equalsIgnoreCase("Host") && !name.equalsIgnoreCase("User-Agent") && !name.equalsIgnoreCase("Content-Length")) {
                        for (String val : entry.getValue()) {
                            try {
                                reqBuilder.header(name, val);
                            } catch (Exception ignored) {
                            }
                        }
                    }
                }

                if ("HEAD".equalsIgnoreCase(method)) {
                    reqBuilder.method("HEAD", HttpRequest.BodyPublishers.noBody());
                } else if ("POST".equalsIgnoreCase(method) || "PUT".equalsIgnoreCase(method)) {
                    byte[] bodyBytes = exchange.getRequestBody().readAllBytes();
                    reqBuilder.method(method, HttpRequest.BodyPublishers.ofByteArray(bodyBytes));
                } else {
                    reqBuilder.GET();
                }

                HttpResponse<InputStream> resp = client.send(reqBuilder.build(), HttpResponse.BodyHandlers.ofInputStream());

                // Set response headers
                Headers outHeaders = exchange.getResponseHeaders();
                for (Map.Entry<String, List<String>> entry : resp.headers().map().entrySet()) {
                    String name = entry.getKey();
                    if (!name.equalsIgnoreCase("Transfer-Encoding") && !name.equalsIgnoreCase("Connection") && !name.equalsIgnoreCase("Content-Length")) {
                        outHeaders.put(name, entry.getValue());
                    }
                }

                if ("HEAD".equalsIgnoreCase(method)) {
                    exchange.sendResponseHeaders(resp.statusCode(), -1);
                } else {
                    exchange.sendResponseHeaders(resp.statusCode(), 0); // Chunked response
                    try (InputStream is = resp.body(); OutputStream os = exchange.getResponseBody()) {
                        byte[] buffer = new byte[65536];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        os.flush();
                    }
                }
            } catch (Exception e) {
                String errorMsg = "Proxy error: " + e.getMessage();
                exchange.sendResponseHeaders(502, errorMsg.length());
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(errorMsg.getBytes());
                }
            } finally {
                exchange.close();
            }
        }
    }
}

