// Content script for filtering tweets on X/Twitter

const KEYWORDS_KEY = 'filterKeywords';
const ENABLED_KEY = 'filterEnabled';
const FILTERED_CLASS = 'propaganda-filter-hidden';

let filterEnabled = true;
let keywords = [];
let observer = null;
let isProcessing = false;

// Tweet selectors (X/Twitter uses various selectors)
const TWEET_SELECTORS = [
  'article[data-testid="tweet"]',
  'div[data-testid="tweet"]',
  'article[role="article"]'
];

// Initialize the filter
async function init() {
  await loadSettings();
  await loadKeywords();
  startFiltering();
  setupMessageListener();
}

// Load filter enabled state
async function loadSettings() {
  try {
    const result = await chrome.storage.sync.get([ENABLED_KEY]);
    filterEnabled = result[ENABLED_KEY] !== false; // default to true
  } catch (error) {
    console.error('[Propaganda Filter] Error loading settings:', error);
    filterEnabled = true; // Default to enabled
  }
}

// Load keywords from storage
async function loadKeywords() {
  try {
    const result = await chrome.storage.sync.get([KEYWORDS_KEY]);
    keywords = result[KEYWORDS_KEY] || [];
  } catch (error) {
    console.error('[Propaganda Filter] Error loading keywords:', error);
    keywords = [];
  }
}

// Start filtering tweets
function startFiltering() {
  if (!filterEnabled || keywords.length === 0) {
    return;
  }

  // Filter existing tweets
  filterTweets();

  // Setup MutationObserver for dynamically loaded tweets
  setupObserver();
}

// Setup MutationObserver to handle infinite scroll
function setupObserver() {
  if (observer) {
    observer.disconnect();
  }

  observer = new MutationObserver((mutations) => {
    // Debounce processing to avoid excessive filtering
    if (isProcessing) return;
    
    isProcessing = true;
    setTimeout(() => {
      filterTweets();
      isProcessing = false;
    }, 500);
  });

  // Observe the main timeline container
  const timelineSelectors = [
    '[aria-label="Timeline: Your Home Timeline"]',
    '[aria-label="Timeline: Search timeline"]',
    'main[role="main"]',
    '[data-testid="primaryColumn"]',
    'body'
  ];

  let targetElement = null;
  for (const selector of timelineSelectors) {
    targetElement = document.querySelector(selector);
    if (targetElement) break;
  }

  if (targetElement) {
    observer.observe(targetElement, {
      childList: true,
      subtree: true
    });
  } else {
    // Fallback: observe body
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }
}

// Filter tweets based on keywords
function filterTweets() {
  if (!filterEnabled || keywords.length === 0) {
    showHiddenTweets();
    return;
  }

  let tweetsFound = 0;

  // Try different selectors to find tweets
  for (const selector of TWEET_SELECTORS) {
    const tweets = document.querySelectorAll(selector);
    
    tweets.forEach(tweet => {
      // Skip if already processed
      if (tweet.classList.contains(FILTERED_CLASS)) {
        return;
      }

      // Get tweet text content
      const tweetText = getTweetText(tweet);
      
      if (tweetText && containsKeywords(tweetText)) {
        hideTweet(tweet);
        tweetsFound++;
      }
    });
  }

  if (tweetsFound > 0) {
    console.log(`[Propaganda Filter] Filtered ${tweetsFound} tweet(s)`);
  }
}

// Get text content from a tweet element
function getTweetText(tweetElement) {
  try {
    // Try to get text from tweet text container
    const textSelectors = [
      '[data-testid="tweetText"]',
      '.tweet-text',
      '[lang]',
      'span'
    ];

    let text = '';
    
    // Primary selector for tweet text
    const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]');
    if (tweetTextElement) {
      text = tweetTextElement.textContent || tweetTextElement.innerText || '';
    } else {
      // Fallback: get all text from tweet, but exclude UI elements
      const allTextNodes = tweetElement.querySelectorAll('span[lang]');
      text = Array.from(allTextNodes)
        .map(node => node.textContent || '')
        .join(' ')
        .trim();
    }

    // Also check alt text from images in the tweet
    const images = tweetElement.querySelectorAll('img[alt]');
    images.forEach(img => {
      const altText = img.getAttribute('alt');
      if (altText && altText !== 'Image') {
        text += ' ' + altText;
      }
    });

    return text.toLowerCase();
  } catch (error) {
    console.error('[Propaganda Filter] Error extracting tweet text:', error);
    return '';
  }
}

// Check if text contains any of the keywords
function containsKeywords(text) {
  if (!text) return false;

  return keywords.some(keyword => {
    if (!keyword) return false;
    // Case-insensitive partial match
    return text.includes(keyword.toLowerCase());
  });
}

// Hide a tweet
function hideTweet(tweetElement) {
  try {
    tweetElement.classList.add(FILTERED_CLASS);
    tweetElement.style.display = 'none';
  } catch (error) {
    console.error('[Propaganda Filter] Error hiding tweet:', error);
  }
}

// Show previously hidden tweets (when filter is disabled)
function showHiddenTweets() {
  const hiddenTweets = document.querySelectorAll(`.${FILTERED_CLASS}`);
  hiddenTweets.forEach(tweet => {
    tweet.classList.remove(FILTERED_CLASS);
    tweet.style.display = '';
  });
}

// Setup message listener for popup communication
function setupMessageListener() {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'refresh') {
      // Reload settings and keywords
      init().then(() => {
        sendResponse({ success: true });
      }).catch(error => {
        console.error('[Propaganda Filter] Error refreshing:', error);
        sendResponse({ success: false, error: error.message });
      });
      return true; // Indicates we will send a response asynchronously
    }
  });
}

// Error handling for selector changes
window.addEventListener('error', (event) => {
  if (event.message && event.message.includes('querySelector')) {
    console.warn('[Propaganda Filter] Selector may have changed. Extension will continue attempting to filter.');
  }
});

// Re-initialize on navigation (SPA navigation)
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    // Small delay to let page load
    setTimeout(() => {
      init();
    }, 1000);
  }
}).observe(document, { subtree: true, childList: true });

// Initialize when script loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  // Page already loaded
  setTimeout(init, 1000); // Give page time to render
}
