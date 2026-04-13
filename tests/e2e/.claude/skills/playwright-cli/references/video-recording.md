# Video Recording

Capture browser automation sessions as video for debugging, documentation, or verification. Produces WebM (VP8/VP9 codec).

## Basic Recording

```bash
# Start recording
playwright-cli video-start

# Perform actions
playwright-cli open https://example.com
playwright-cli snapshot
playwright-cli click e1
playwright-cli fill e2 "test input"

# Stop and save
playwright-cli video-stop demo.webm
```

## Viewing Recordings

WebM files can be opened in any modern browser or video player (VLC, QuickTime, Chrome, Firefox):

```bash
# Open in browser directly
open recordings/login-flow.webm

# Or serve locally
npx serve recordings/
```

## Common Patterns

### Record a failing test flow for bug reports

```bash
playwright-cli open https://app.example.com
playwright-cli video-start
playwright-cli fill e1 "user@example.com"
playwright-cli fill e2 "wrongpassword"
playwright-cli click e3
playwright-cli video-stop recordings/login-failure.webm
playwright-cli close
```

### Record a complete feature demo

```bash
playwright-cli open https://app.example.com
playwright-cli video-start
# ... walk through the full feature ...
playwright-cli video-stop recordings/feature-demo-$(date +%Y%m%d).webm
playwright-cli close
```

## Best Practices

### 1. Use Descriptive Filenames

```bash
# Include context in filename
playwright-cli video-stop recordings/login-flow-2024-01-15.webm
playwright-cli video-stop recordings/checkout-test-run-42.webm
```

### 2. Clean Up Large Recordings

```bash
# Remove recordings older than 30 days
find recordings/ -name "*.webm" -mtime +30 -delete
```

## Tracing vs Video

| Feature  | Video                | Tracing                                  |
| -------- | -------------------- | ---------------------------------------- |
| Output   | WebM file            | Trace file (viewable in Trace Viewer)    |
| Shows    | Visual recording     | DOM snapshots, network, console, actions |
| Use case | Demos, documentation | Debugging, analysis                      |
| Size     | Larger               | Smaller                                  |

## Limitations

- Recording adds slight overhead to automation
- Large recordings can consume significant disk space
- WebM (VP8/VP9) — ensure your video player supports this codec
