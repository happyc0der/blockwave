// Background service worker for Propaganda Filter for X

// Install/update handler
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('[Propaganda Filter] Extension installed');
    
    // Set default values
    chrome.storage.sync.set({
      filterEnabled: true,
      filterKeywords: []
    });
  } else if (details.reason === 'update') {
    console.log('[Propaganda Filter] Extension updated');
  }
});

// Listen for storage changes to sync across tabs
chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === 'sync') {
    // Notify all content scripts about changes
    chrome.tabs.query({ url: ['https://twitter.com/*', 'https://x.com/*'] }, (tabs) => {
      tabs.forEach(tab => {
        chrome.tabs.sendMessage(tab.id, { action: 'refresh' }).catch(() => {
          // Content script might not be ready, ignore error
        });
      });
    });
  }
});
