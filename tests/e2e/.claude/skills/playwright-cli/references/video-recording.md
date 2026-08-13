# Video Recording

Capture browser automation sessions as video for debugging, documentation, or verification. Produces WebM (VP8/VP9 codec).

## Basic Recording

```bash
# Open a blank browser before recording starts (captures navigation in the video)
playwright-cli open

# Start recording (filename goes to video-start, not video-stop)
playwright-cli video-start demo.webm

# Add a chapter marker for section transitions
playwright-cli video-chapter "Getting Started" --description="Opening the homepage" --duration=2000

# Navigate and perform actions
playwright-cli goto https://example.com
playwright-cli snapshot
playwright-cli click e1

# Add another chapter
playwright-cli video-chapter "Filling Form" --description="Entering test data" --duration=2000
playwright-cli fill e2 "test input"

# Stop and save
playwright-cli video-stop
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
playwright-cli video-start recordings/login-failure.webm
playwright-cli fill e1 "user@example.com"
playwright-cli fill e2 "wrongpassword"
playwright-cli click e3
playwright-cli video-stop
playwright-cli close
```

### Record a complete feature demo

```bash
playwright-cli open https://app.example.com
playwright-cli video-start recordings/feature-demo-$(date +%Y%m%d).webm
# ... walk through the full feature ...
playwright-cli video-stop
playwright-cli close
```

## Best Practices

### 1. Use Descriptive Filenames

```bash
# Include context in filename
playwright-cli video-start recordings/login-flow-2024-01-15.webm
playwright-cli video-start recordings/checkout-test-run-42.webm
```

### 2. Record Hero Scripts with the Screencast API

When recording a video for a user or as proof of work, create a script and execute it with `run-code`. This allows precise pauses between actions and rich annotations. Use `page.screencast.*` APIs for chapter cards and overlays.

1. Perform the scenario using the CLI and note all locators and actions.
2. Create a script file with `pressSequentially` (for visible typing) and reasonable pauses.
3. Run: `playwright-cli run-code --filename your-script.js`

**Important**: Overlays are `pointer-events: none` — they do not interfere with clicks or fills.

**Note**: Raw `page.*` API calls in this script are only valid inside `run-code`. In page objects, use `pressSequentially()` from `@anaconda/playwright-utils` instead of `locator().pressSequentially()`.

```js
async page => {
  await page.screencast.start({ path: 'video.webm', size: { width: 1280, height: 800 } });
  await page.goto('https://demo.playwright.dev/todomvc');

  // Chapter card — blurs the page and shows a dialog, auto-removes after duration
  await page.screencast.showChapter('Adding Todo Items', {
    description: 'We will add several items to the todo list.',
    duration: 2000,
  });

  await page.getByRole('textbox', { name: 'What needs to be done?' }).pressSequentially('Walk the dog', { delay: 60 });
  await page.getByRole('textbox', { name: 'What needs to be done?' }).press('Enter');
  await page.waitForTimeout(1000);

  // Sticky annotation — stays until disposed
  const annotation = await page.screencast.showOverlay(`
    <div style="position: absolute; top: 8px; right: 8px;
      padding: 6px 12px; background: rgba(0,0,0,0.7);
      border-radius: 8px; font-size: 13px; color: white;">
      ✓ Item added successfully
    </div>
  `);

  await page.getByRole('textbox', { name: 'What needs to be done?' }).pressSequentially('Buy groceries', { delay: 60 });
  await page.getByRole('textbox', { name: 'What needs to be done?' }).press('Enter');
  await page.waitForTimeout(1500);
  await annotation.dispose();

  // Highlight a specific element with a bounding-box overlay
  const bounds = await page.getByText('Walk the dog').boundingBox();
  await page.screencast.showOverlay(
    `
    <div style="position: absolute;
      top: ${bounds.y}px;
      left: ${bounds.x}px;
      width: ${bounds.width}px;
      height: ${bounds.height}px;
      border: 1px solid red;">
    </div>
    <div style="position: absolute;
      top: ${bounds.y + bounds.height + 5}px;
      left: ${bounds.x + bounds.width / 2}px;
      transform: translateX(-50%);
      padding: 6px;
      background: #808080;
      border-radius: 10px;
      font-size: 14px;
      color: white;">Check it out, it is right above this text
    </div>
  `,
    { duration: 2000 },
  );

  await page.screencast.stop();
};
```

#### Overlay API Summary

| Method                                                                         | Use Case                                                                       |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `page.screencast.showChapter(title, { description?, duration?, styleSheet? })` | Full-screen chapter card with blurred backdrop — ideal for section transitions |
| `page.screencast.showOverlay(html, { duration? })`                             | Custom HTML overlay — use for callouts, labels, highlights                     |
| `disposable.dispose()`                                                         | Remove a sticky overlay added without duration                                 |
| `page.screencast.hideOverlays()` / `page.screencast.showOverlays()`            | Temporarily hide/show all overlays                                             |

### 3. Clean Up Large Recordings

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
