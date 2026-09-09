# Propaganda Filter for X

A Chrome browser extension that helps users filter tweets containing nationalist propaganda content on X (Twitter) based on customizable keywords.

## Features

- ✅ **Manifest V3** - Uses the latest Chrome extension manifest version
- ✅ **Keyword-based filtering** - Add/remove custom keywords to filter tweets
- ✅ **Real-time filtering** - Automatically filters tweets as you scroll (infinite scroll support)
- ✅ **Enable/Disable toggle** - Quickly turn filtering on or off
- ✅ **Popup UI** - Easy-to-use interface for managing keywords
- ✅ **Lightweight** - Pure vanilla JavaScript, no external dependencies
- ✅ **Privacy-focused** - No data collection, no tracking, keywords stored locally

## Ethical Guidelines

This extension:
- Filters content based on **keywords only** - no location or nationality-based detection
- Does not collect personal data or track users
- Complies with platform terms of service
- Requires responsible use by the user

## Installation

### Step 1: Download/Clone the Extension

If you haven't already, download or clone this extension to your local machine.

### Step 2: Prepare Icon Files

The extension requires icon files for the Chrome extension interface. You'll need to create three icon PNG files:

- `icon16.png` (16x16 pixels)
- `icon48.png` (48x48 pixels)
- `icon128.png` (128x128 pixels)

**Quick Option:** You can use any image editor or online tool to create simple icons, or use placeholder images. The extension will work without icons, but Chrome may show warnings.

### Step 3: Load the Extension in Chrome

1. Open Google Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the folder containing the extension files
5. The extension should now appear in your extensions list

### Step 4: Pin the Extension (Optional)

1. Click the puzzle piece icon (extensions icon) in Chrome's toolbar
2. Find "Propaganda Filter for X"
3. Click the pin icon to keep it visible in your toolbar

## Usage

### Adding Keywords

1. Click the extension icon in your Chrome toolbar
2. In the popup, type a keyword or phrase in the input field
3. Click "Add" or press Enter
4. The keyword will be added to your filter list

### Removing Keywords

1. Open the extension popup
2. Find the keyword in the list
3. Click the "Remove" button next to it

### Using Default Keywords

1. Open the extension popup
2. Expand the "Add Default Keywords" section
3. Click any default keyword button to add it to your filter list

### Enabling/Disabling the Filter

- Toggle the "Enable Filter" switch in the popup to turn filtering on or off

## Testing

### Test 1: Basic Functionality

1. Load the extension in Chrome
2. Navigate to https://x.com or https://twitter.com
3. Open the extension popup
4. Add a test keyword (e.g., "test")
5. Find a tweet that contains "test" in its text
6. The tweet should be hidden automatically
7. Toggle the filter off - the tweet should reappear
8. Toggle the filter back on - the tweet should hide again

### Test 2: Multiple Keywords

1. Add multiple keywords (e.g., "test1", "test2", "test3")
2. Find tweets containing any of these keywords
3. All matching tweets should be filtered

### Test 3: Dynamic Loading (Infinite Scroll)

1. Add a keyword that appears in tweets
2. Scroll down on X/Twitter to load more tweets
3. New tweets containing the keyword should automatically be filtered as they load

### Test 4: Case Insensitivity

1. Add keyword "TEST" (uppercase)
2. Tweets containing "test", "Test", or "TEST" should all be filtered

### Test 5: Partial Matches

1. Add keyword "test"
2. Tweets containing "testing", "contest", or "attest" should all be filtered

### Test 6: Settings Persistence

1. Add several keywords
2. Close and reopen the extension popup
3. Keywords should still be there
4. Refresh the X/Twitter page
5. Filter should still be active

### Test 7: Navigation Handling

1. Add keywords and verify filtering works
2. Navigate to different pages on X (Home, Explore, Profile, etc.)
3. Filtering should continue to work on all pages

## Troubleshooting

### Extension Not Filtering Tweets

- **Check if filter is enabled**: Open the popup and verify the toggle is on
- **Check if keywords are added**: Make sure you have at least one keyword in the list
- **Refresh the page**: Reload the X/Twitter page
- **Check console**: Open DevTools (F12) and check for any error messages
- **Check permissions**: Ensure the extension has permission to access twitter.com/x.com

### Keywords Not Persisting

- **Check Chrome sync**: Ensure you're signed into Chrome and sync is enabled
- **Check storage limits**: Chrome.storage.sync has limits; if you have many keywords, consider using local storage instead

### Performance Issues

- **Too many keywords**: If you have a very large number of keywords, performance may degrade. Consider consolidating keywords.
- **Page lag**: The extension debounces filtering to avoid excessive processing. If pages are very slow, reduce the number of keywords.

### Tweets Not Being Detected

- **Selector changes**: X/Twitter frequently updates their HTML structure. If filtering stops working, the selectors may need to be updated.
- **Check tweet format**: Some tweet types (e.g., ads, promoted tweets) may use different HTML structures.

## File Structure

```
propaganda-filter-x/
├── manifest.json       # Extension manifest (Manifest V3)
├── popup.html          # Popup UI HTML
├── popup.js            # Popup logic
├── popup.css           # Popup styling
├── content.js          # Content script for filtering tweets
├── background.js       # Background service worker
├── icon16.png          # 16x16 icon (you need to create)
├── icon48.png          # 48x48 icon (you need to create)
├── icon128.png         # 128x128 icon (you need to create)
└── README.md           # This file
```

## Development

### Key Components

- **manifest.json**: Defines extension permissions, content scripts, and popup
- **content.js**: Injected into X/Twitter pages, handles tweet filtering
- **popup.js**: Manages keywords and settings via chrome.storage.sync
- **background.js**: Service worker for handling extension lifecycle events

### Modifying Keywords

Keywords are stored in `chrome.storage.sync` and are automatically synced across devices when Chrome sync is enabled.

### Adding Custom Selectors

If X/Twitter changes their HTML structure, update the `TWEET_SELECTORS` array in `content.js`.

## Privacy & Security

- **No data collection**: The extension does not collect or transmit any data
- **Local storage only**: Keywords are stored locally in Chrome's sync storage
- **No external requests**: The extension makes no network requests
- **Open source**: Code is available for review

## Limitations

- Keyword-based filtering may have false positives/negatives
- Filtering is case-insensitive and partial match (e.g., "test" matches "testing")
- Some tweet formats (ads, promoted content) may not be filtered
- HTML structure changes by X/Twitter may require selector updates

## License

This extension is provided as-is for educational and personal use. Use responsibly and in compliance with X (Twitter) terms of service.

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review browser console for errors
3. Verify all files are present and correctly formatted
4. Ensure you're using Manifest V3 compatible Chrome version

## Version

Version: 1.0.0
Manifest Version: 3
