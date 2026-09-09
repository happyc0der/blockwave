// Popup script for managing keywords and settings

const KEYWORDS_KEY = 'filterKeywords';
const ENABLED_KEY = 'filterEnabled';

// DOM elements
const filterEnabledCheckbox = document.getElementById('filterEnabled');
const keywordInput = document.getElementById('keywordInput');
const addKeywordBtn = document.getElementById('addKeywordBtn');
const keywordsList = document.getElementById('keywordsList');
const keywordCount = document.getElementById('keywordCount');
const statusDiv = document.getElementById('status');
const defaultKeywordBtns = document.querySelectorAll('.btn-default-keyword');

// Initialize popup
async function init() {
  await loadSettings();
  await loadKeywords();
  setupEventListeners();
}

// Load filter enabled state
async function loadSettings() {
  try {
    const result = await chrome.storage.sync.get([ENABLED_KEY]);
    filterEnabledCheckbox.checked = result[ENABLED_KEY] !== false; // default to true
  } catch (error) {
    console.error('Error loading settings:', error);
    showStatus('Error loading settings', 'error');
  }
}

// Load keywords from storage
async function loadKeywords() {
  try {
    const result = await chrome.storage.sync.get([KEYWORDS_KEY]);
    const keywords = result[KEYWORDS_KEY] || [];
    renderKeywords(keywords);
    updateKeywordCount(keywords.length);
  } catch (error) {
    console.error('Error loading keywords:', error);
    showStatus('Error loading keywords', 'error');
  }
}

// Render keywords list
function renderKeywords(keywords) {
  keywordsList.innerHTML = '';

  if (keywords.length === 0) {
    keywordsList.innerHTML = '<div class="empty-state">No keywords added yet. Add keywords to start filtering tweets.</div>';
    return;
  }

  keywords.forEach((keyword, index) => {
    const keywordItem = document.createElement('div');
    keywordItem.className = 'keyword-item';
    keywordItem.innerHTML = `
      <span class="keyword-text">${escapeHtml(keyword)}</span>
      <button class="btn-remove" data-index="${index}">Remove</button>
    `;
    keywordsList.appendChild(keywordItem);
  });

  // Add remove button listeners
  keywordsList.querySelectorAll('.btn-remove').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const index = parseInt(e.target.dataset.index);
      await removeKeyword(index);
    });
  });
}

// Update keyword count display
function updateKeywordCount(count) {
  keywordCount.textContent = `${count} keyword${count !== 1 ? 's' : ''}`;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Add keyword
async function addKeyword(keyword) {
  if (!keyword || !keyword.trim()) {
    showStatus('Please enter a keyword', 'error');
    return;
  }

  const trimmedKeyword = keyword.trim().toLowerCase();

  try {
    const result = await chrome.storage.sync.get([KEYWORDS_KEY]);
    const keywords = result[KEYWORDS_KEY] || [];

    if (keywords.includes(trimmedKeyword)) {
      showStatus('Keyword already exists', 'error');
      return;
    }

    keywords.push(trimmedKeyword);
    await chrome.storage.sync.set({ [KEYWORDS_KEY]: keywords });
    
    await loadKeywords();
    keywordInput.value = '';
    showStatus('Keyword added successfully', 'success');

    // Notify content script to refresh
    notifyContentScript();
  } catch (error) {
    console.error('Error adding keyword:', error);
    showStatus('Error adding keyword', 'error');
  }
}

// Remove keyword
async function removeKeyword(index) {
  try {
    const result = await chrome.storage.sync.get([KEYWORDS_KEY]);
    const keywords = result[KEYWORDS_KEY] || [];

    if (index >= 0 && index < keywords.length) {
      keywords.splice(index, 1);
      await chrome.storage.sync.set({ [KEYWORDS_KEY]: keywords });
      
      await loadKeywords();
      showStatus('Keyword removed', 'success');

      // Notify content script to refresh
      notifyContentScript();
    }
  } catch (error) {
    console.error('Error removing keyword:', error);
    showStatus('Error removing keyword', 'error');
  }
}

// Setup event listeners
function setupEventListeners() {
  // Toggle filter enabled
  filterEnabledCheckbox.addEventListener('change', async (e) => {
    try {
      await chrome.storage.sync.set({ [ENABLED_KEY]: e.target.checked });
      showStatus(e.target.checked ? 'Filter enabled' : 'Filter disabled', 'success');
      notifyContentScript();
    } catch (error) {
      console.error('Error saving settings:', error);
      showStatus('Error saving settings', 'error');
    }
  });

  // Add keyword button
  addKeywordBtn.addEventListener('click', () => {
    addKeyword(keywordInput.value);
  });

  // Add keyword on Enter key
  keywordInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      addKeyword(keywordInput.value);
    }
  });

  // Default keyword buttons
  defaultKeywordBtns.forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const keyword = e.target.dataset.keyword;
      await addKeyword(keyword);
    });
  });
}

// Notify content script to refresh
async function notifyContentScript() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && (tab.url?.includes('twitter.com') || tab.url?.includes('x.com'))) {
      chrome.tabs.sendMessage(tab.id, { action: 'refresh' }).catch(() => {
        // Content script might not be ready, ignore error
      });
    }
  } catch (error) {
    // Ignore errors
  }
}

// Show status message
function showStatus(message, type = 'success') {
  statusDiv.textContent = message;
  statusDiv.className = `status ${type} show`;
  
  setTimeout(() => {
    statusDiv.classList.remove('show');
  }, 3000);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
